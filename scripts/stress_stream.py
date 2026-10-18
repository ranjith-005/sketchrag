"""Long-stream stress test (RQ3/RQ5 at scale): one ~11M-token stream of all English LongBench
documents is ingested once by SketchRAG (bounded) and by dense / BM25 RAG (unbounded).

Measures, at checkpoints along the stream: hot-memory bytes, ingest latency per chunk,
retrieval latency. At the end it asks probe questions whose source documents are
spread over the stream, and records answer recall (gold answer string in context), plus
LLM answers from the shared Ollama model when --llm is given.

  python scripts/stress_stream.py --max-words 2000000 --probes 40           # quick
  python scripts/stress_stream.py --probes 80 --llm --model qwen2.5:7b-instruct  # full (Ollama)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sketchrag.data.long_stream import build_stream, probe_questions  # noqa: E402
from sketchrag.embed import Embedder  # noqa: E402
from sketchrag.eval.metrics import answer_in_context, score  # noqa: E402
from sketchrag.memory import SketchMemory  # noqa: E402
from sketchrag.registry import _apply  # noqa: E402
from sketchrag.retrieve import build_context  # noqa: E402
from sketchrag.stream import Chunk, chunk_text  # noqa: E402
from sketchrag.text import terms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words", type=int, default=None, help="truncate the stream (quick runs)")
    ap.add_argument("--probes", type=int, default=60)
    ap.add_argument("--set", nargs="*", default=["mem.max_cells=256"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--checkpoints", type=int, default=10)
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--out", default=str(ROOT / "results" / "stress"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stream, questions = build_stream()
    if args.max_words:
        kept, total = [], 0
        for d in stream:
            if total >= args.max_words:
                break
            kept.append(d)
            total += d.n_words
        stream = kept
    probes = probe_questions(stream, questions, args.probes)
    print(f"stream: {len(stream)} docs, {sum(d.n_words for d in stream):,} words; {len(probes)} probes", flush=True)

    emb = Embedder(cache_path=ROOT / "cache" / "emb.sqlite")
    mem_cfg, ret_cfg, _ = _apply(args.set)
    ret_cfg.top_k = max(1, args.top_k - ret_cfg.contiguity)
    mem = SketchMemory(mem_cfg, dim=384)

    chunks: list[Chunk] = []
    doc_first_chunk = []
    for d in stream:
        doc_first_chunk.append(len(chunks))
        for piece in chunk_text(d.text, 150):
            chunks.append(Chunk(len(chunks), d.doc_key, piece))
    n = len(chunks)
    marks = set(int(n * (i + 1) / args.checkpoints) - 1 for i in range(args.checkpoints))
    curve, embs, ingest_s, t_emb = [], [], 0.0, 0.0
    B = 2048
    for s in range(0, n, B):
        t0 = time.perf_counter()
        e = emb.embed([c.text for c in chunks[s:s + B]])
        t_emb += time.perf_counter() - t0
        embs.append(e)
        for c, v in zip(chunks[s:s + B], e):
            t0 = time.perf_counter()
            mem.ingest(c, v)
            ingest_s += time.perf_counter() - t0
            if c.id in marks:
                text_bytes = sum(len(x.text.encode("utf-8")) for x in chunks[: c.id + 1])
                row = {"chunks": c.id + 1, "words": sum(len(x.text.split()) for x in chunks[: c.id + 1]),
                       "sketch_hot_bytes": mem.hot_nbytes(), "sketch_cells": len(mem.cells),
                       "dense_bytes": text_bytes + (c.id + 1) * 384 * 4, "text_bytes": text_bytes,
                       "ingest_ms_per_chunk": 1000 * ingest_s / (c.id + 1)}
                curve.append(row)
                print(json.dumps(row), flush=True)
    E = np.concatenate(embs)

    import faiss
    from rank_bm25 import BM25Okapi

    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    t0 = time.perf_counter()
    bm = BM25Okapi([terms(c.text) or ["_"] for c in chunks])
    bm_build = time.perf_counter() - t0
    q_embs = emb.embed_queries([ex.question for ex, _ in probes])

    rows, prompts, ages = [], {}, {}
    for (ex, doc_i), q in zip(probes, q_embs):
        age = 1 - doc_first_chunk[doc_i] / n  # 1.0 = arrived at the very start of the stream
        ages[ex.id] = age
        t0 = time.perf_counter()
        _, I = index.search(q[None, :], args.top_k)
        dense_ms = 1000 * (time.perf_counter() - t0)
        dense_ctx = "\n\n".join(chunks[i].text for i in sorted(I[0]))
        t0 = time.perf_counter()
        bm_top = np.argsort(-bm.get_scores(terms(ex.question) or ["_"]))[: args.top_k]
        bm_ms = 1000 * (time.perf_counter() - t0)
        bm_ctx = "\n\n".join(chunks[i].text for i in sorted(bm_top))
        res = {}
        for policy in ("sketch", "gate", "raw"):
            t0 = time.perf_counter()
            ctx, info = build_context(mem, ex.question, q, ret_cfg, policy)
            res[policy] = (ctx, info, 1000 * (time.perf_counter() - t0))
        row = {"id": ex.id, "task": ex.task, "age": age, "dense_ms": dense_ms, "bm25_ms": bm_ms,
               "recall": {"dense_rag": answer_in_context(dense_ctx, ex.answers),
                          "bm25_rag": answer_in_context(bm_ctx, ex.answers),
                          **{f"sketchrag_{p}": answer_in_context(c, ex.answers) for p, (c, _, _) in res.items()}},
               "ctx_words": {"dense_rag": len(dense_ctx.split()), "bm25_rag": len(bm_ctx.split()),
                             **{f"sketchrag_{p}": len(c.split()) for p, (c, _, _) in res.items()}},
               "sketch_ms": {p: ms for p, (_, _, ms) in res.items()},
               "gate_fallback": bool(res["gate"][1]["fallback_cells"])}
        rows.append(row)
        for name, ctx in [("dense_rag", dense_ctx), ("bm25_rag", bm_ctx)] + [(f"sketchrag_{p}", c) for p, (c, _, _) in res.items()]:
            prompts[(ex.id, name)] = (ex, ctx)

    summary = {"n_chunks": n, "n_words": curve[-1]["words"], "embed_s_total": t_emb, "bm25_build_s": bm_build,
               "final": curve[-1], "config": args.set, "probes": len(rows)}
    for name in rows[0]["recall"]:
        summary[f"recall_{name}"] = float(np.mean([r["recall"][name] for r in rows]))
        summary[f"ctx_words_{name}"] = float(np.mean([r["ctx_words"][name] for r in rows]))
    summary["gate_fallback_rate"] = float(np.mean([r["gate_fallback"] for r in rows]))
    print(json.dumps(summary, indent=1))
    tag = f"{curve[-1]['words'] // 1000}kw"
    (out / f"curve_{tag}.json").write_text(json.dumps(curve, indent=1))
    (out / f"probes_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))

    if args.llm:
        from sketchrag.llm import LLM
        from sketchrag.prompts import MAX_TOKENS, SYSTEM, user_prompt

        llm = LLM(args.model, num_ctx=args.num_ctx) if args.model else LLM(num_ctx=args.num_ctx)
        with open(out / f"answers_{tag}.jsonl", "w", encoding="utf-8") as f:
            for (eid, name), (ex, ctx) in prompts.items():
                comp = llm.complete(SYSTEM, user_prompt(ex.kind, ctx, ex.question), MAX_TOKENS[ex.kind])
                f.write(json.dumps({"id": eid, "system": name, "age": ages[eid], "pred": comp.text,
                                    **score(ex.kind, comp.text, ex.answers),
                                    "context_tokens": comp.usage.input_tokens, "llm_s": comp.usage.llm_s}) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
