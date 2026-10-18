"""SketchMemory: a bounded "hot" memory maintained online over a chunk stream.

Hot memory (searched at query time, size fixed by the config, independent of
stream length):
  * at most `max_cells` semantic cells. Each cell is a clustering feature (embedding
    sum + count, which gives the centroid and radius), a time span, a per-cell
    Count-Min Sketch over content terms, and an extractive digest of at most
    `digest_words` words
  * a global Count-Min Sketch of document frequencies (streaming IDF)
  * a global Misra-Gries heavy-hitter table (stream topics)
  * optionally, a bounded reservoir of raw-chunk pointers per cell

Cold storage (never searched, only read on an explicit fallback; grows with the
stream): the raw chunk text and the cell -> chunk pointer lists.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .sketches.countmin import CountMinSketch
from .sketches.misra_gries import MisraGries
from .sketches.reservoir import Reservoir
from .stream import Chunk
from .text import split_sentences, terms


@dataclass
class SketchConfig:
    max_cells: int = 64
    join_threshold: float = 0.90   # cosine sim needed to join an existing cell directly;
                                   # otherwise open a cell and let merge-and-reduce enforce the budget
    recency_bonus: float = 0.05    # event-style continuity: prefer the current cell
    digest_words: int = 120
    max_digest_sentence_words: int = 40
    cms_width: int = 512
    cms_depth: int = 3
    df_width: int = 1 << 13
    df_depth: int = 3
    mg_k: int = 256
    pointer_mode: str = "cold"     # "cold": full pointer lists in cold store; "reservoir": bounded hot sample
    reservoir_size: int = 16
    exemplars: int = 8             # diverse member embeddings kept per cell for ranking (0 = centroid only)
    seed: int = 0


@dataclass
class Cell:
    id: int
    vsum: np.ndarray
    n: int
    t_first: int
    t_last: int
    cms: CountMinSketch
    digest: list[tuple[float, int, str]] = field(default_factory=list)  # (score, t, sentence)
    reservoir: Reservoir | None = None
    exemplars: np.ndarray | None = None  # (E, d) float16, farthest-point diverse members

    @property
    def centroid(self) -> np.ndarray:
        return self.vsum / (np.linalg.norm(self.vsum) + 1e-12)

    @property
    def radius(self) -> float:
        """RMS distance of member embeddings to their mean. For unit vectors,
        E||x - mu||^2 = 1 - ||mu||^2, so the clustering feature (sum, n) is enough."""
        mu = self.vsum / self.n
        return math.sqrt(max(0.0, 1.0 - float(mu @ mu)))

    def digest_text(self) -> str:
        return " ".join(s for _, _, s in sorted(self.digest, key=lambda x: x[1]))

    def digest_terms(self) -> set[str]:
        return set(terms(self.digest_text()))


def _diverse(X: np.ndarray, k: int) -> np.ndarray:
    """Keep at most k rows of X, repeatedly dropping the most redundant one (the row whose
    nearest other row is most similar). This keeps a spread-out set of representatives,
    in the spirit of EM-LLM's representative tokens per event."""
    X = X.astype(np.float16)
    while len(X) > k:
        Xf = X.astype(np.float32)
        S = Xf @ Xf.T
        np.fill_diagonal(S, -np.inf)
        X = np.delete(X, int(np.argmax(S.max(axis=1))), axis=0)
    return X


class ColdStore:
    """Append-only raw evidence. Only touched by fetch_raw()."""

    def __init__(self):
        self.chunks: dict[int, Chunk] = {}
        self.embs: dict[int, np.ndarray] = {}  # stored next to the text so a fallback can rerank inside a cell
        self.cell_ptrs: dict[int, list[int]] = {}

    def add(self, chunk: Chunk, cell_id: int, emb: np.ndarray) -> None:
        self.chunks[chunk.id] = chunk
        self.embs[chunk.id] = emb.astype(np.float16)
        self.cell_ptrs.setdefault(cell_id, []).append(chunk.id)

    def merge_cells(self, keep: int, drop: int) -> None:
        self.cell_ptrs.setdefault(keep, []).extend(self.cell_ptrs.pop(drop, []))

    @property
    def nbytes(self) -> int:
        text = sum(len(c.text.encode("utf-8")) for c in self.chunks.values())
        ptrs = 8 * sum(len(p) for p in self.cell_ptrs.values())
        return text + ptrs + sum(e.nbytes for e in self.embs.values())


class SketchMemory:
    def __init__(self, cfg: SketchConfig | None = None, dim: int = 384):
        self.cfg = cfg or SketchConfig()
        self.dim = dim
        self.cells: list[Cell] = []
        self._next_id = 0
        self._current: int | None = None  # id of the cell that received the last chunk
        self.df = CountMinSketch(self.cfg.df_width, self.cfg.df_depth)
        self.mg = MisraGries(self.cfg.mg_k)
        self.n_chunks = 0
        self.cold = ColdStore()
        self.rng = random.Random(self.cfg.seed)
        self.n_merges = 0

    # ---------- streaming statistics ----------
    def idf(self, term: str) -> float:
        df = self.df.query(term)
        return math.log((self.n_chunks + 1) / (df + 1)) + 1.0

    # ---------- ingest ----------
    def ingest(self, chunk: Chunk, emb: np.ndarray) -> int:
        """Absorb one chunk. Amortised O(K*d + chunk terms); returns the cell id used."""
        chunk_terms = terms(chunk.text)
        counts = Counter(chunk_terms)
        self.n_chunks += 1
        for t in counts:
            self.df.update(t)
        for t, c in counts.items():
            self.mg.update(t, c)

        cell = self._assign(emb, chunk.id)
        cell.vsum += emb
        cell.n += 1
        cell.t_last = max(cell.t_last, chunk.id)
        cell.t_first = min(cell.t_first, chunk.id)
        cell.cms.update_many(counts)
        if self.cfg.exemplars:
            cell.exemplars = _diverse(emb[None, :] if cell.exemplars is None
                                      else np.vstack([cell.exemplars, emb[None, :].astype(np.float16)]),
                                      self.cfg.exemplars)
        if cell.reservoir is not None:
            cell.reservoir.add(chunk.id)
        self._add_to_digest(cell, chunk, emb)
        self.cold.add(chunk, cell.id, emb)
        self._current = cell.id

        if len(self.cells) > self.cfg.max_cells:
            self._merge_closest_pair()
        return self._current if self._current is not None else cell.id

    def _new_cell(self, t: int) -> Cell:
        cell = Cell(
            id=self._next_id, vsum=np.zeros(self.dim, np.float32), n=0, t_first=t, t_last=t,
            cms=CountMinSketch(self.cfg.cms_width, self.cfg.cms_depth),
            reservoir=Reservoir(self.cfg.reservoir_size, self.rng) if self.cfg.pointer_mode == "reservoir" else None,
        )
        self._next_id += 1
        self.cells.append(cell)
        return cell

    def _assign(self, emb: np.ndarray, t: int) -> Cell:
        if not self.cells:
            return self._new_cell(t)
        sims = self.centroids() @ emb
        if self._current is not None:
            for i, c in enumerate(self.cells):
                if c.id == self._current:
                    sims[i] += self.cfg.recency_bonus
        best = int(np.argmax(sims))
        if sims[best] >= self.cfg.join_threshold:
            return self.cells[best]
        return self._new_cell(t)

    def _add_to_digest(self, cell: Cell, chunk: Chunk, emb: np.ndarray) -> None:
        rep = float(cell.centroid @ emb)  # how typical this chunk is of the cell
        for sent in split_sentences(chunk.text):
            words = sent.split()
            if len(words) < 4:
                continue
            words = words[: self.cfg.max_digest_sentence_words]
            ts = terms(" ".join(words))
            if not ts:
                continue
            salience = sum(self.idf(t) for t in set(ts)) / math.sqrt(len(words))
            cell.digest.append((salience * max(rep, 0.0), chunk.id, " ".join(words)))
        self._trim_digest(cell)

    def _trim_digest(self, cell: Cell) -> None:
        cell.digest.sort(key=lambda x: -x[0])
        kept, used = [], 0
        for item in cell.digest:
            w = len(item[2].split())
            if used + w > self.cfg.digest_words:
                continue
            kept.append(item)
            used += w
        cell.digest = kept

    def _merge_closest_pair(self) -> None:
        """Ward linkage on cell means: merge the pair whose union increases the total
        within-cell variance least, n_a*n_b/(n_a+n_b) * ||mu_a - mu_b||^2. Unlike plain
        centroid similarity this resists one cell swallowing the whole stream."""
        n = np.array([c.n for c in self.cells], dtype=np.float64)
        M = np.stack([c.vsum / c.n for c in self.cells]).astype(np.float64)
        sq = (M * M).sum(1)
        D = np.maximum(sq[:, None] + sq[None, :] - 2 * M @ M.T, 0.0)
        cost = D * (n[:, None] * n[None, :]) / (n[:, None] + n[None, :])
        np.fill_diagonal(cost, np.inf)
        i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
        a, b = self.cells[min(i, j)], self.cells[max(i, j)]
        a.vsum += b.vsum
        a.n += b.n
        a.t_first = min(a.t_first, b.t_first)
        a.t_last = max(a.t_last, b.t_last)
        a.cms.merge(b.cms)
        if a.exemplars is not None and b.exemplars is not None:
            a.exemplars = _diverse(np.vstack([a.exemplars, b.exemplars]), self.cfg.exemplars)
        a.digest.extend(b.digest)
        self._trim_digest(a)
        if a.reservoir is not None and b.reservoir is not None:
            a.reservoir.merge(b.reservoir)
        self.cold.merge_cells(a.id, b.id)
        if self._current == b.id:
            self._current = a.id
        self.cells.remove(b)
        self.n_merges += 1

    # ---------- views ----------
    def cell_scores(self, q: np.ndarray) -> np.ndarray:
        """Query-to-cell similarity: max over the cell's centroid and its exemplars."""
        cent = self.centroids() @ q
        if not self.cfg.exemplars:
            return cent
        ex = np.array([float((c.exemplars.astype(np.float32) @ q).max()) if c.exemplars is not None else -1.0
                       for c in self.cells])
        return np.maximum(cent, ex)

    def centroids(self) -> np.ndarray:
        if not self.cells:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([c.centroid for c in self.cells])

    def cell(self, cell_id: int) -> Cell:
        for c in self.cells:
            if c.id == cell_id:
                return c
        raise KeyError(cell_id)

    def pointers(self, cell_id: int) -> list[int]:
        c = self.cell(cell_id)
        if c.reservoir is not None:
            return sorted(c.reservoir.items)
        return self.cold.cell_ptrs.get(cell_id, [])

    def hot_nbytes(self) -> int:
        per_cell = 0
        for c in self.cells:
            per_cell += c.vsum.nbytes + c.cms.nbytes + 32  # vsum, sketch, n/t_first/t_last/id
            per_cell += c.exemplars.nbytes if c.exemplars is not None else 0
            per_cell += sum(len(s.encode("utf-8")) + 12 for _, _, s in c.digest)
            if c.reservoir is not None:
                per_cell += 8 * c.reservoir.capacity
        return per_cell + self.df.nbytes + self.mg.nbytes

    def hot_words(self) -> int:
        return sum(len(s.split()) for c in self.cells for _, _, s in c.digest)

    def stats(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "n_cells": len(self.cells),
            "n_merges": self.n_merges,
            "hot_bytes": self.hot_nbytes(),
            "hot_digest_words": self.hot_words(),
            "cold_bytes": self.cold.nbytes,
            "mean_radius": float(np.mean([c.radius for c in self.cells])) if self.cells else 0.0,
        }
