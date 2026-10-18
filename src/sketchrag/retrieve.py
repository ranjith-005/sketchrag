"""Sketch-aware retrieval over SketchMemory, plus the error-bound fallback gate."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .memory import Cell, SketchMemory
from .text import terms


@dataclass
class RetrieveConfig:
    top_k: int = 4
    w_lex: float = 0.0          # CMS evidence in cell *ranking*; dev tuning found it hurts (0.5 = ablation).
                                # CMS is still used by the gate.
    contiguity: int = 1         # extra temporal neighbours of the top cell (0 => "-temporal" ablation)
    rare_idf: float = 2.5       # query terms with idf >= this are "rare keys"
    max_radius: float = 0.75    # cells looser than this have lossy digests
    confidence: str = "bound"   # "bound": est > eps*N (guaranteed, conservative);
                                # "noise": est > N/w + z*sqrt(N/w) (empirical noise floor)
    z: float = 3.0
    raw_per_cell: int = 2       # raw chunks returned per fetched cell
    replace_digest: bool = True # a fetched cell's raw evidence replaces its digest instead of being added
    cooccur: bool = True        # trust a digest only if one of its sentences holds the rare query terms together
    n_topics: int = 15          # heavy hitters shown as global stream context


@dataclass
class CellHit:
    cell: Cell
    score: float
    dense: float
    lex: float


@dataclass
class GateDecision:
    fallback_cells: list[int] = field(default_factory=list)
    reasons: dict[int, str] = field(default_factory=dict)

    @property
    def fallback(self) -> bool:
        return bool(self.fallback_cells)


def _query_terms(mem: SketchMemory, query: str) -> dict[str, float]:
    return {t: mem.idf(t) for t in dict.fromkeys(terms(query))}


def search(mem: SketchMemory, query: str, q_emb: np.ndarray, cfg: RetrieveConfig) -> list[CellHit]:
    if not mem.cells:
        return []
    dense = mem.cell_scores(q_emb)
    qt = _query_terms(mem, query)
    lex = np.zeros(len(mem.cells))
    if qt and cfg.w_lex > 0:
        z = sum(qt.values())
        for i, c in enumerate(mem.cells):
            noise = c.cms.expected_noise()
            s = 0.0
            for t, w in qt.items():
                s += w * math.log1p(max(0.0, c.cms.query(t) - noise))
            lex[i] = s / z
        if lex.max() > 0:
            lex = lex / lex.max()
    score = dense + cfg.w_lex * lex
    order = list(np.argsort(-score))
    chosen = order[: cfg.top_k]
    if cfg.contiguity and chosen:
        top = mem.cells[chosen[0]]
        mid = (top.t_first + top.t_last) / 2
        rest = [i for i in order if i not in chosen]

        def gap(i: int) -> float:
            c = mem.cells[i]
            if c.t_last < top.t_first:
                return top.t_first - c.t_last
            if c.t_first > top.t_last:
                return c.t_first - top.t_last
            return abs((c.t_first + c.t_last) / 2 - mid) * 1e-3  # overlapping spans
        chosen += sorted(rest, key=gap)[: cfg.contiguity]
    return [CellHit(mem.cells[i], float(score[i]), float(dense[i]), float(lex[i])) for i in chosen]


def gate(mem: SketchMemory, query: str, hits: list[CellHit], cfg: RetrieveConfig) -> GateDecision:
    """Decide which retrieved cells cannot be trusted as summaries.

    A cell is escalated to raw evidence when
      (lossy)      a rare query term is present in the cell beyond the sketch noise floor
                   but missing from its digest, so the summary dropped the fact; or
      (loose)      the cell radius is above max_radius, so its digest averages too much; or
      (unresolved) no retrieved cell holds a rare query term above the guaranteed CMS
                   bound eps*N. The sketch cannot vouch for any location, so the top
                   two cells are read raw.
    Questions with no rare terms (global/summary questions) are trusted to the sketch.
    """
    dec = GateDecision()
    rare = [t for t, w in _query_terms(mem, query).items() if w >= cfg.rare_idf]
    any_confident = False
    for h in hits:
        c = h.cell
        if c.radius > cfg.max_radius:
            dec.reasons[c.id] = "loose"
        if not rare:
            continue
        dterms = c.digest_terms()
        if cfg.cooccur and len(rare) >= 2:
            need = min(2, len(rare))
            present = [t for t in rare if c.cms.query(t) - c.cms.expected_noise() >= 1.0]
            if len(present) >= need and not any(
                    sum(t in set(terms(sent)) for t in present) >= need for _, _, sent in c.digest):
                dec.reasons.setdefault(c.id, "lossy")
        noise = c.cms.expected_noise()
        if cfg.confidence == "bound":
            bound = c.cms.error_bound()
        else:
            bound = noise + cfg.z * math.sqrt(noise)
        for t in rare:
            est = c.cms.query(t)
            if est > bound:
                any_confident = True
            if est - noise >= 1.0 and t not in dterms:
                dec.reasons.setdefault(c.id, "lossy")
    if rare and not any_confident:
        for h in hits[:2]:
            dec.reasons.setdefault(h.cell.id, "unresolved")
    dec.fallback_cells = [h.cell.id for h in hits if h.cell.id in dec.reasons]
    return dec


def fetch_raw(mem: SketchMemory, cell_id: int, query: str, k: int,
              q_emb: np.ndarray | None = None) -> list[tuple[int, str]]:
    """Read the cell's raw chunks from cold storage and return the k best, in stream order.
    Score = cosine(query, chunk) (if a query embedding is given) + idf-weighted term overlap,
    with the lexical part scaled to [0, 1] within the cell."""
    qt = _query_terms(mem, query)
    ids = list(mem.pointers(cell_id))
    if not ids:
        return []
    lex = []
    for cid in ids:
        tc = Counter(terms(mem.cold.chunks[cid].text))
        lex.append(sum(w * math.log1p(tc[t]) for t, w in qt.items()))
    lex = np.array(lex)
    if lex.max() > 0:
        lex = lex / lex.max()
    dense = np.zeros(len(ids))
    if q_emb is not None:
        dense = np.array([float(mem.cold.embs[cid].astype(np.float32) @ q_emb) for cid in ids])
    score = dense + lex
    best = sorted(np.argsort(-score)[:k], key=lambda i: ids[i])
    return [(ids[i], mem.cold.chunks[ids[i]].text) for i in best]


def render_cells(hits: list[CellHit], skip: set[int] | frozenset = frozenset()) -> str:
    blocks = []
    for h in sorted(hits, key=lambda h: h.cell.t_first):
        if h.cell.id in skip:
            continue
        c = h.cell
        blocks.append(f"[memory cell {c.id} | stream positions {c.t_first}-{c.t_last} | {c.n} chunks]\n{c.digest_text()}")
    return "\n\n".join(blocks)


def render_topics(mem: SketchMemory, n: int) -> str:
    return ", ".join(t for t, _ in mem.mg.top(n))


def build_context(mem: SketchMemory, query: str, q_emb: np.ndarray, cfg: RetrieveConfig,
                  policy: str = "gate") -> tuple[str, dict]:
    """Assemble the LLM context for the non-interactive policies.

    policy: "sketch" (digests only), "raw" (raw chunks of every retrieved cell),
            "gate" (digests, plus raw chunks for the cells the gate escalates).
    """
    hits = search(mem, query, q_emb, cfg)
    dec = gate(mem, query, hits, cfg)
    if policy == "sketch":
        raw_cells = []
    elif policy == "raw":
        raw_cells = [h.cell.id for h in hits]
    elif policy == "gate":
        raw_cells = dec.fallback_cells
    else:
        raise ValueError(policy)
    skip = set(raw_cells) if cfg.replace_digest else set()
    parts = [f"Stream topics (heavy hitters): {render_topics(mem, cfg.n_topics)}", render_cells(hits, skip)]
    raw_ids: list[int] = []
    for cid in raw_cells:
        for chunk_id, text in fetch_raw(mem, cid, query, cfg.raw_per_cell, q_emb):
            raw_ids.append(chunk_id)
            parts.append(f"[raw evidence from cell {cid}, stream position {chunk_id}]\n{text}")
    info = {
        "cells": [h.cell.id for h in hits],
        "fallback_cells": raw_cells,
        "gate_reasons": {str(k): v for k, v in dec.reasons.items()},
        "raw_chunk_ids": raw_ids,
        "cell_spans": [[h.cell.t_first, h.cell.t_last] for h in hits],
    }
    return "\n\n".join(parts), info
