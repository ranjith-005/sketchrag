"""Shared helpers for the SketchRAG demo notebook and the Streamlit app.

Everything here calls the real package code (SketchMemory, search, gate, build_context).
The only extra logic is `explain_gate`, which recomputes the numbers the gate looks at
(CMS estimate, noise floor, εN bound, digest membership) so they can be shown on
screen; the decision itself always comes from `retrieve.gate`.

Saved answers are the qwen2.5:7b-instruct answers from the main Kaggle run
(results/runs/main). Live answers use the local Ollama server.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# bge-small is already in the local Hugging Face cache; skip the hub checks that add ~40 s on load
os.environ.setdefault("HF_HUB_OFFLINE", "1")
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sketchrag.data.loaders import Example, load_task  # noqa: E402
from sketchrag.embed import Embedder  # noqa: E402
from sketchrag.llm import LLM  # noqa: E402
from sketchrag.memory import SketchConfig, SketchMemory  # noqa: E402
from sketchrag.prompts import MAX_TOKENS, SYSTEM, user_prompt  # noqa: E402
from sketchrag.retrieve import RetrieveConfig, build_context, gate, search, _query_terms  # noqa: E402
from sketchrag.stream import Chunk, chunk_text  # noqa: E402
from sketchrag.text import terms  # noqa: E402
from sketchrag.eval.metrics import answer_in_context, score  # noqa: E402

RUN_DIR = ROOT / "results" / "runs" / "main"
CHUNK_WORDS = 150
SAVED_MODEL = "qwen2.5:7b-instruct"

# Hand-picked test questions (seed 0 split), one per behaviour worth showing.
SHOWCASE = [
    {"id": "hotpotqa/9336554e511e88766228ffa1acd1c6f82c9a41046514c253",
     "title": "Summary is enough (gate stays closed)",
     "story": "The digest already holds the answer, so the gate trusts the sketch and reads nothing raw."},
    {"id": "hotpotqa/1ceb1c91662d241b3421853112a91b5868d1b81fe3b88ce4",
     "title": "Lossy digest (gate fires: lossy)",
     "story": "The cell's Count-Min Sketch says the rare query terms are in the cell, but the digest dropped "
              "them, so the gate reads those cells' raw chunks."},
    {"id": "kv_retrieval/130",
     "title": "Exact key lookup (gate fires: unresolved)",
     "story": "A UUID key never survives summarisation. No cell can vouch for it above the εN bound, so "
              "the gate reads raw evidence. Digest-only answers with a wrong fragment."},
    {"id": "narrativeqa/a5e44b1e2b98f80aabab5f1b8296d0fe66d53b31c1f3e332",
     "title": "Long story, specific fact (gate fires: unresolved)",
     "story": "Digest-only answers 'not enough information'; with raw evidence the model names the victim."},
    {"id": "passkey/62",
     "title": "Needle in a haystack (full context fails)",
     "story": "About 100k words of repeated filler with one pass key. Full context must be middle-truncated to fit "
              "the model window, loses the needle and answers with filler text. SketchRAG finds the key."},
    {"id": "gov_report/81e845f35acd55c1f6b4edda428a0a52fd22bade5be9cf13",
     "title": "Global summary (no rare terms)",
     "story": "A summarisation request has no rare query terms, so the gate never fires and the digests are the context."},
]

SYSTEM_LABELS = {
    "sketchrag_sketch": "SketchRAG, digest only",
    "sketchrag_gate": "SketchRAG + gate",
    "sketchrag_raw": "SketchRAG, always raw",
    "full_context": "Full context",
    "bm25_rag": "BM25 RAG",
    "dense_rag": "Dense RAG",
    "sara_lite": "SARA-lite",
    "amber_lite": "AMBER-lite",
    "sketchrag_agent": "SketchRAG LLM agent",
    "sketchrag_hybrid": "SketchRAG gate + agent",
}


# ---------------- data ----------------
def find_example(ex_id: str) -> Example:
    task = ex_id.split("/")[0]
    n = 20 if task == "longbook_qa_eng" else 30  # the main run's sample sizes
    for ex in load_task(task, n, 0):
        if ex.id == ex_id:
            return ex
    raise KeyError(ex_id)


def text_example(text: str, question: str, answer: str = "") -> Example:
    """Wrap user-pasted text as an Example so it runs through the same code."""
    return Example(id="custom/pasted", task="custom", context=text, question=question,
                   answers=[answer] if answer else [])


_EMB: Embedder | None = None


def embedder() -> Embedder:
    """The same bge-small embedder and cache as the experiments; cached chunks load instantly."""
    global _EMB
    if _EMB is None:
        _EMB = Embedder(cache_path=ROOT / "cache" / "emb.sqlite")
    return _EMB


def chunk_and_embed(ex: Example) -> tuple[list[Chunk], np.ndarray]:
    texts = chunk_text(ex.context, CHUNK_WORDS)
    embs = embedder().embed(texts)
    return [Chunk(i, ex.id, t) for i, t in enumerate(texts)], embs


def embed_question(question: str) -> np.ndarray:
    return embedder().embed_queries([question.strip() or "summary of the main points"])[0]


# ---------------- streaming ----------------
def stream(chunks: list[Chunk], embs: np.ndarray, cfg: SketchConfig | None = None,
           every: int = 1) -> tuple[SketchMemory, list[dict]]:
    """Ingest chunk by chunk, recording memory statistics every `every` chunks."""
    mem = SketchMemory(cfg or SketchConfig(), dim=embs.shape[1])
    trace = []
    t0 = time.perf_counter()
    for i, (ch, e) in enumerate(zip(chunks, embs), 1):
        mem.ingest(ch, e)
        if i % every == 0 or i == len(chunks):
            trace.append({"chunks": i, "cells": len(mem.cells), "merges": mem.n_merges,
                          "hot_kb": mem.hot_nbytes() / 1024, "cold_kb": mem.cold.nbytes / 1024,
                          "ms": 1000 * (time.perf_counter() - t0)})
    return mem, trace


def cells_table(mem: SketchMemory) -> list[dict]:
    rows = []
    for c in sorted(mem.cells, key=lambda c: c.t_first):
        rows.append({"cell": c.id, "positions": f"{c.t_first}-{c.t_last}", "chunks": c.n,
                     "radius": round(c.radius, 3), "digest words": len(c.digest_text().split()),
                     "digest": c.digest_text()})
    return rows


# ---------------- the gate, explained ----------------
def explain_gate(mem: SketchMemory, question: str, q_emb: np.ndarray, cfg: RetrieveConfig | None = None) -> dict:
    cfg = cfg or RetrieveConfig()
    hits = search(mem, question, q_emb, cfg)
    dec = gate(mem, question, hits, cfg)  # the real decision
    qt = _query_terms(mem, question)
    rare = [t for t, w in qt.items() if w >= cfg.rare_idf]
    hit_rows, term_rows = [], []
    for rank, h in enumerate(hits, 1):
        c = h.cell
        hit_rows.append({"rank": rank, "cell": c.id, "positions": f"{c.t_first}-{c.t_last}", "chunks": c.n,
                         "similarity": round(h.dense, 3), "radius": round(c.radius, 3),
                         "εN bound": round(c.cms.error_bound(), 1), "noise N/w": round(c.cms.expected_noise(), 2),
                         "gate": dec.reasons.get(c.id, "trusted")})
        dterms = c.digest_terms()
        for t in rare:
            est = c.cms.query(t)
            term_rows.append({"term": t, "idf": round(qt[t], 2), "cell": c.id, "CMS estimate": est,
                              "above noise": est - c.cms.expected_noise() >= 1.0,
                              "above εN": est > c.cms.error_bound(), "in digest": t in dterms})
    return {"hits": hits, "decision": dec, "hit_rows": hit_rows, "term_rows": term_rows, "rare": rare,
            "all_terms": {t: round(w, 2) for t, w in qt.items()}}


REASON_TEXT = {
    "trusted": "digest is trusted",
    "lossy": "the sketch says a rare query term is in this cell but the digest dropped it",
    "loose": "the cell is too spread out (radius above 0.75) for its digest to be faithful",
    "unresolved": "no retrieved cell holds the rare terms above the guaranteed εN bound, so the top cells are read raw",
}


def contexts(mem: SketchMemory, question: str, q_emb: np.ndarray, cfg: RetrieveConfig | None = None) -> dict:
    """The context each SketchRAG policy would send to the LLM."""
    cfg = cfg or RetrieveConfig()
    out = {}
    for policy in ("sketch", "gate", "raw"):
        ctx, info = build_context(mem, question, q_emb, cfg, policy)
        out[policy] = {"context": ctx, "info": info, "words": len(ctx.split())}
    return out


# ---------------- answers ----------------
_SAVED: dict[str, dict[str, dict]] | None = None


def saved_answers(ex_id: str) -> dict[str, dict]:
    """Every system's recorded answer for this question in the main run (qwen2.5:7b on Kaggle)."""
    global _SAVED
    if _SAVED is None:
        _SAVED = {}
        for path in RUN_DIR.glob("*.jsonl"):
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    _SAVED.setdefault(r["id"], {})[r["system"]] = r
    return _SAVED.get(ex_id, {})


def ollama_models() -> list[str]:
    import urllib.request
    try:
        host = LLM().host.replace("//localhost", "//127.0.0.1")  # localhost tries IPv6 first: +2 s on Windows
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r)["models"]]
    except Exception:
        return []


def live_answer(ex: Example, context: str, model: str) -> dict:
    """Ask a local Ollama model with the experiments' exact prompt."""
    comp = LLM(model).complete(SYSTEM, user_prompt(ex.kind if ex.task != "custom" else "qa", context, ex.question),
                               MAX_TOKENS.get(ex.kind if ex.task != "custom" else "qa", 128))
    row = {"pred": comp.text, "context_tokens": comp.usage.input_tokens, "seconds": round(comp.latency_s, 1)}
    if ex.answers:
        row.update(score(ex.kind if ex.task != "custom" else "qa", comp.text, ex.answers))
    return row


# ---------------- whole-run numbers ----------------
def load_run(run: str = "main") -> dict[str, list[dict]]:
    out = {}
    for path in (ROOT / "results" / "runs" / run).glob("*.jsonl"):
        out[path.stem] = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    return out


def gate_calibration(run: dict[str, list[dict]]) -> dict:
    """Split questions by whether the gate fired and compare policies on each side (mean score x100)."""
    by = {s: {r["id"]: r for r in run[s]} for s in ("sketchrag_sketch", "sketchrag_gate", "sketchrag_raw")}
    ids = list(by["sketchrag_gate"])
    fired = [i for i in ids if by["sketchrag_gate"][i]["fallback"]]
    trusted = [i for i in ids if not by["sketchrag_gate"][i]["fallback"]]

    def mean(system, group):
        return 100 * sum(by[system][i]["score"] for i in group) / max(1, len(group))
    return {"n": len(ids), "fired": len(fired), "trusted": len(trusted),
            "fired_sketch": mean("sketchrag_sketch", fired), "fired_gate": mean("sketchrag_gate", fired),
            "trusted_gate": mean("sketchrag_gate", trusted), "trusted_raw": mean("sketchrag_raw", trusted)}
