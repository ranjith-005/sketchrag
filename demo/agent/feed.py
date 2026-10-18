"""The live feed the demo agent monitors, and the agent's memory state.

The feed is the first ~400k words of the stress-test stream (the same fixed shuffled order of
English LongBench documents as src/sketchrag/data/long_stream.py), so every chunk embedding
is already in cache/emb.sqlite. A background thread ingests it chunk by chunk into a
SketchMemory at a controllable rate; the server reads snapshots of this state.
"""
from __future__ import annotations

import hashlib
import random
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sketchrag.data.loaders import Example, load_task  # noqa: E402
from sketchrag.data.long_stream import QA_TASKS, STREAM_TASKS  # noqa: E402
from sketchrag.embed import Embedder  # noqa: E402
from sketchrag.memory import SketchConfig, SketchMemory  # noqa: E402
from sketchrag.stream import Chunk, chunk_text  # noqa: E402
from sketchrag.text import terms  # noqa: E402

FEED_WORDS = 400_000
CHUNKS_PER_SEC = 25  # at 1x

KIND = {  # task -> (icon, label) shown on the feed cards
    "narrativeqa": ("📖", "Story"), "qasper": ("📄", "Paper"), "multifieldqa_en": ("🗂️", "Article"),
    "hotpotqa": ("🌐", "Wiki pages"), "2wikimqa": ("🌐", "Wiki pages"), "musique": ("🌐", "Wiki pages"),
    "gov_report": ("🏛️", "Gov report"), "qmsum": ("🗣️", "Meeting"),
}


@dataclass
class Doc:
    key: str
    task: str
    title: str
    n_words: int
    first_chunk: int
    n_chunks: int

    @property
    def kind(self) -> str:
        return KIND[self.task][1]

    @property
    def icon(self) -> str:
        return KIND[self.task][0]


def _title(text: str) -> str:
    line = next((l.strip() for l in text.splitlines() if len(l.strip()) > 20), text.strip())
    for prefix in ("Passage 1:", "Passage 1"):
        line = line.removeprefix(prefix).strip()
    words = line.split()
    return " ".join(words[:9]) + ("…" if len(words) > 9 else "")


def build_feed(max_words: int = FEED_WORDS, seed: int = 0):
    """Documents in stream order (as long_stream.build_stream, but keeping each document's task),
    the chunk list, and the questions asked about each document."""
    texts, task_of, questions = {}, {}, {}
    for task in STREAM_TASKS:
        for ex in load_task(task):
            key = hashlib.sha1(ex.context.encode("utf-8")).hexdigest()[:16]
            if key not in texts:
                texts[key], task_of[key] = ex.context, task
            if task in QA_TASKS:
                questions.setdefault(key, []).append(ex)
    order = sorted(texts)
    random.Random(seed).shuffle(order)
    docs, chunks, total = [], [], 0
    for key in order:
        if total >= max_words:
            break
        pieces = chunk_text(texts[key], 150)
        docs.append(Doc(key, task_of[key], _title(texts[key]), len(texts[key].split()), len(chunks), len(pieces)))
        chunks += [Chunk(len(chunks) + i, key, p) for i, p in enumerate(pieces)]
        total += docs[-1].n_words
    return docs, chunks, {d.key: questions.get(d.key, []) for d in docs}


class Feed:
    def __init__(self, max_cells: int = 64):
        self.docs, self.chunks, self.questions = build_feed()
        self.doc_of = {d.key: d for d in self.docs}
        self.embedder = Embedder(cache_path=ROOT / "cache" / "emb.sqlite")
        self.embs = self.embedder.embed([c.text for c in self.chunks])
        self.words = np.cumsum([len(c.text.split()) for c in self.chunks])
        self.text_bytes = np.cumsum([len(c.text.encode("utf-8")) for c in self.chunks])
        self.lock = threading.RLock()
        self.speed = 1
        self.playing = True
        self.thinking = False
        self.last_tick = time.time()
        self.cfg = SketchConfig(max_cells=max_cells)
        self.default_cfg = SketchConfig(max_cells=max_cells)
        self.events: deque = deque(maxlen=200)
        self.event_id = 0
        self._reset_memory()
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def max_cells(self) -> int:
        return self.cfg.max_cells

    def _reset_memory(self, cfg: SketchConfig | None = None):
        with self.lock:
            if cfg is not None:
                self.cfg = cfg
            self.mem = SketchMemory(self.cfg, dim=self.embs.shape[1])
            self.pos = 0
            self.arrived_at: dict[str, float] = {}
            self.history: list[dict] = []
            self.event("info", "config", f"memory reset: {self.cfg.max_cells} cells, CMS width {self.cfg.cms_width}, "
                                          f"join threshold {self.cfg.join_threshold:.2f}")

    def event(self, level: str, kind: str, text: str, detail: str = ""):
        """Event log entry (the server also logs query events here)."""
        if kind == "merge" and self.events and self.events[0]["kind"] == "merge":  # collapse runs of merges
            e = self.events[0]
            e["count"] = e.get("count", 1) + 1
            e["text"] = f"{text} x{e['count']}"
            e["ts"] = time.time()
            self.event_id += 1
            e["id"] = self.event_id
            return
        self.event_id += 1
        self.events.appendleft({"id": self.event_id, "ts": time.time(), "level": level, "kind": kind,
                                "text": text, "detail": detail, "at": self.pos})

    def ingest_one(self):
        """Absorb the next chunk and log what happened to the memory."""
        c, e = self.chunks[self.pos], self.embs[self.pos]
        doc = self.doc_of[c.doc_id]
        if c.id == doc.first_chunk:
            self.arrived_at[doc.key] = time.time()
            self.event("info", "doc", f"{doc.kind} arriving: {doc.title}", f"{doc.n_words:,} words, {doc.n_chunks} chunks")
        n_cells, merges = len(self.mem.cells), self.mem.n_merges
        cell_id = self.mem.ingest(c, e)
        if self.mem.n_merges > merges:
            self.event("info", "merge", "cell budget full: merged the two closest cells (Ward)",
                       f"{self.max_cells}/{self.max_cells} cells")
        elif len(self.mem.cells) > n_cells:
            self.event("info", "cell", f"opened cell {cell_id} for a new topic", f"{len(self.mem.cells)}/{self.max_cells} cells")
        self.pos += 1
        if self.pos % 16 == 0 or self.pos == len(self.chunks):
            self.history.append(self._sizes())

    def _sizes(self) -> dict:
        n = self.pos
        return {"chunks": n, "words": int(self.words[n - 1]) if n else 0, "hot": self.mem.hot_nbytes(),
                "cold": self.mem.cold.nbytes if n else 0, "text": int(self.text_bytes[n - 1]) if n else 0,
                "dense": int(self.text_bytes[n - 1] + n * 384 * 4) if n else 0}

    def _run(self):
        tick = 0.04
        acc = 0.0
        while True:
            time.sleep(tick)
            self.last_tick = time.time()
            if not self.playing or self.thinking or self.pos >= len(self.chunks):
                continue
            acc += CHUNKS_PER_SEC * self.speed * tick
            with self.lock:
                while acc >= 1 and self.pos < len(self.chunks):
                    self.ingest_one()
                    acc -= 1
            acc = min(acc, 1.0)

    # ---------------- views (all live, read under the lock) ----------------
    def kind_of_chunk(self, i: int) -> str:
        return self.doc_of[self.chunks[i].doc_id].kind

    def cell_kinds(self) -> dict[int, str]:
        """Dominant document type of each cell, from its pointer list in cold storage."""
        out = {}
        for c in self.mem.cells:
            kinds = Counter(self.kind_of_chunk(i) for i in self.mem.pointers(c.id))
            out[c.id] = kinds.most_common(1)[0][0] if kinds else "?"
        return out

    def status(self) -> dict:
        with self.lock:
            n = self.pos
            words = int(self.words[n - 1]) if n else 0
            state = ("thinking" if self.thinking else "paused" if not self.playing
                     else "finished" if n >= len(self.chunks) else "ingesting")
            sizes = self._sizes()
            return {"pos": n, "total": len(self.chunks), "words": words, "total_words": int(self.words[-1]),
                    "progress": n / len(self.chunks), "docs_arrived": sum(d.first_chunk < n for d in self.docs),
                    "docs_total": len(self.docs), "tokens_full": int(round(words * 1.33)), "state": state,
                    "playing": self.playing, "speed": self.speed, "cells": len(self.mem.cells),
                    "max_cells": self.max_cells, "merges": self.mem.n_merges,
                    "hot_bytes": sizes["hot"], "cold_bytes": sizes["cold"], "dense_bytes": sizes["dense"],
                    "text_bytes": sizes["text"]}

    def documents(self) -> list[dict]:
        with self.lock:
            n = self.pos
            out = []
            for d in self.docs:
                done = min(max(0, n - d.first_chunk), d.n_chunks)
                status = "queued" if done == 0 else "ingested" if done == d.n_chunks else "ingesting"
                out.append({"key": d.key, "kind": d.kind, "title": d.title, "words": d.n_words, "chunks": d.n_chunks,
                            "first_chunk": d.first_chunk, "progress": done / d.n_chunks, "status": status,
                            "arrived_at": self.arrived_at.get(d.key)})
            return out

    def cells(self) -> list[dict]:
        with self.lock:
            kinds = self.cell_kinds()
            return [{"id": c.id, "n": c.n, "span": [c.t_first, c.t_last], "kind": kinds.get(c.id, "?"),
                     "radius": round(c.radius, 3), "digest_words": len(c.digest_text().split())} for c in self.mem.cells]

    def cell_detail(self, cell_id: int) -> dict | None:
        with self.lock:
            try:
                c = self.mem.cell(cell_id)
            except KeyError:
                return None
            ptrs = list(self.mem.pointers(cell_id))
            docs = Counter(self.chunks[i].doc_id for i in ptrs)
            dterms = Counter(terms(c.digest_text()))
            top = sorted(dterms, key=lambda t: -c.cms.query(t))[:12]
            return {"id": c.id, "n": c.n, "span": [c.t_first, c.t_last], "radius": round(c.radius, 3),
                    "loose": c.radius > 0.75, "digest": c.digest_text(), "digest_words": len(c.digest_text().split()),
                    "digest_limit": self.cfg.digest_words, "exemplars": 0 if c.exemplars is None else len(c.exemplars),
                    "max_exemplars": self.cfg.exemplars, "dim": int(c.vsum.shape[0]),
                    "cms": {"width": c.cms.width, "depth": c.cms.depth, "total": int(c.cms.total),
                            "eps": round(c.cms.eps, 5), "bound": round(c.cms.error_bound(), 2),
                            "noise": round(c.cms.expected_noise(), 3), "kb": round(c.cms.nbytes / 1024, 1),
                            "top_terms": [{"term": t, "est": int(c.cms.query(t))} for t in top]},
                    "pointers": ptrs[:400], "n_pointers": len(ptrs),
                    "documents": [{"kind": self.doc_of[k].kind, "title": self.doc_of[k].title, "chunks": v}
                                  for k, v in docs.most_common(8)],
                    "samples": [{"chunk": i, "kind": self.kind_of_chunk(i), "text": self.chunks[i].text[:500]}
                                for i in ptrs[:3]]}

    def topics(self, n: int = 15) -> list[dict]:
        with self.lock:
            return [{"term": t, "count": int(c)} for t, c in self.mem.mg.top(n)]
