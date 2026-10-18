"""All compared systems behind one interface.

A system turns an Example into the context that the (shared) LLM answers from.
Non-agentic systems do this without any LLM call (`prepare`), so they can be sent
through the Batches API. Agentic systems implement `run_live` and drive their own
Claude loop.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .data.loaders import Example
from .embed import Embedder
from .memory import SketchConfig, SketchMemory
from .retrieve import RetrieveConfig, build_context, fetch_raw, gate, render_cells, render_topics, search
from .stream import Chunk, chunk_text
from .text import est_tokens, split_sentences, terms


@dataclass
class DocIndex:
    """Chunks and chunk embeddings of one example's document (shared by all systems)."""
    chunks: list[Chunk]
    embs: np.ndarray
    q_emb: np.ndarray
    embed_s: float

    @classmethod
    def build(cls, ex: Example, embedder: Embedder, chunk_words: int) -> "DocIndex":
        t0 = time.perf_counter()
        texts = chunk_text(ex.context, chunk_words)
        embs = embedder.embed(texts)
        q = embedder.embed_queries([ex.question.strip() or "summary of the main points"])[0]
        chunks = [Chunk(i, ex.id, t) for i, t in enumerate(texts)]
        return cls(chunks, embs, q, time.perf_counter() - t0)


@dataclass
class Prepared:
    context: str
    info: dict = field(default_factory=dict)


def _in_order(chunks: list[Chunk], ids) -> str:
    return "\n\n".join(f"[passage at stream position {c.id}]\n{c.text}" for c in sorted((chunks[i] for i in ids), key=lambda c: c.id))


class System:
    name = "base"
    agentic = False

    def prepare(self, ex: Example, idx: DocIndex) -> Prepared:  # pragma: no cover - interface
        raise NotImplementedError


class FullContext(System):
    """The whole document, middle-truncated to fit the model window when it is too long
    (LongBench's protocol: keep the head and the tail, since both carry instructions and answers)."""
    name = "full_context"

    def __init__(self, max_chars: int | None = None):
        self.max_chars = max_chars

    def prepare(self, ex, idx):
        truncated = self.max_chars is not None and len(ex.context) > self.max_chars
        ctx = ex.context
        if truncated:
            half = self.max_chars // 2
            ctx = ex.context[:half] + " [...] " + ex.context[-half:]
        return Prepared(ctx, {"hot_bytes": len(ex.context.encode("utf-8")), "stored_tokens": est_tokens(ex.context),
                              "truncated": truncated})


class BM25RAG(System):
    name = "bm25_rag"

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def prepare(self, ex, idx):
        from rank_bm25 import BM25Okapi

        t0 = time.perf_counter()
        bm = BM25Okapi([terms(c.text) or ["_"] for c in idx.chunks])
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        scores = bm.get_scores(terms(ex.question) or ["_"])
        top = list(np.argsort(-scores)[: self.top_k])
        ret_s = time.perf_counter() - t0
        text_bytes = sum(len(c.text.encode("utf-8")) for c in idx.chunks)
        postings = sum(len(d) for d in bm.doc_freqs) * 12  # term id + tf per posting
        return Prepared(_in_order(idx.chunks, top), {
            "chunk_ids": [int(i) for i in top], "hot_bytes": text_bytes + postings,
            "stored_tokens": sum(est_tokens(c.text) for c in idx.chunks),
            "ingest_s": build_s, "retrieve_s": ret_s})


class DenseRAG(System):
    name = "dense_rag"

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def prepare(self, ex, idx):
        import faiss

        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(idx.embs.shape[1])
        index.add(idx.embs)
        build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, I = index.search(idx.q_emb[None, :], min(self.top_k, len(idx.chunks)))
        ret_s = time.perf_counter() - t0
        top = [int(i) for i in I[0] if i >= 0]
        text_bytes = sum(len(c.text.encode("utf-8")) for c in idx.chunks)
        return Prepared(_in_order(idx.chunks, top), {
            "chunk_ids": top, "hot_bytes": text_bytes + idx.embs.nbytes,
            "stored_tokens": sum(est_tokens(c.text) for c in idx.chunks),
            "ingest_s": build_s, "retrieve_s": ret_s})


class SARALite(System):
    """SARA-inspired: retrieve generously, then compress post-retrieval to a fixed word
    budget by sentence-level selection (query-term idf overlap + chunk relevance).
    SARA's learned compression vectors are not reproduced; that deviation is in the report."""
    name = "sara_lite"

    def __init__(self, candidates: int = 20, budget_words: int = 600):
        self.candidates = candidates
        self.budget_words = budget_words

    def prepare(self, ex, idx):
        t0 = time.perf_counter()
        sims = idx.embs @ idx.q_emb
        cand = list(np.argsort(-sims)[: self.candidates])
        df = Counter(t for c in idx.chunks for t in set(terms(c.text)))
        n = len(idx.chunks)
        qt = set(terms(ex.question))
        sents = []
        for rank, ci in enumerate(cand):
            for si, s in enumerate(split_sentences(idx.chunks[ci].text)):
                st = set(terms(s))
                lex = sum(math.log((n + 1) / (df[t] + 1)) + 1 for t in qt & st)
                sents.append((lex + 2.0 * float(sims[ci]), int(ci), si, s))
        sents.sort(key=lambda x: -x[0])
        kept, used = [], 0
        for item in sents:
            w = len(item[3].split())
            if used + w > self.budget_words:
                continue
            kept.append(item)
            used += w
        kept.sort(key=lambda x: (x[1], x[2]))
        ctx = "\n".join(f"[position {ci}] {s}" for _, ci, _, s in kept)
        text_bytes = sum(len(c.text.encode("utf-8")) for c in idx.chunks)
        return Prepared(ctx, {"chunk_ids": sorted({k[1] for k in kept}), "hot_bytes": text_bytes + idx.embs.nbytes,
                              "stored_tokens": sum(est_tokens(c.text) for c in idx.chunks),
                              "retrieve_s": time.perf_counter() - t0})


class SketchRAG(System):
    """The proposed method with a non-interactive policy: 'sketch', 'raw' or 'gate'."""

    def __init__(self, policy: str = "gate", mem_cfg: SketchConfig | None = None,
                 ret_cfg: RetrieveConfig | None = None, name: str | None = None):
        self.policy = policy
        self.mem_cfg = mem_cfg or SketchConfig()
        self.ret_cfg = ret_cfg or RetrieveConfig()
        self.name = name or f"sketchrag_{policy}"

    def build_memory(self, idx: DocIndex) -> tuple[SketchMemory, float]:
        mem = SketchMemory(self.mem_cfg, dim=idx.embs.shape[1])
        t0 = time.perf_counter()
        for ch, e in zip(idx.chunks, idx.embs):
            mem.ingest(ch, e)
        return mem, time.perf_counter() - t0

    def prepare(self, ex, idx):
        mem, ingest_s = self.build_memory(idx)
        t0 = time.perf_counter()
        ctx, info = build_context(mem, ex.question, idx.q_emb, self.ret_cfg, self.policy)
        info.update(mem.stats())
        info.update({"ingest_s": ingest_s, "ingest_ms_per_chunk": 1000 * ingest_s / max(1, len(idx.chunks)),
                     "retrieve_s": time.perf_counter() - t0, "hot_bytes": mem.hot_nbytes(),
                     "stored_tokens": est_tokens(" ".join(c.digest_text() for c in mem.cells)),
                     "chunk_ids": info["raw_chunk_ids"]})
        # for the answer-recall proxy, fallback evidence counts only if it is actually in context
        return Prepared(ctx, info)
