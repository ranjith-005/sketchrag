"""Choose the demo's suggested questions honestly.

Replays the feed offline. After every document finishes arriving, every question about an
already-arrived document is checked: is its gold answer in the context SketchRAG's gate would
send (answer_in_context)? A question qualifies as a preset only if that holds at every checkpoint
after its document arrived, so it works whenever it is asked during the demo. Dense top-5
retrieval over all chunks is scored the same way for reference.

With --llm the qualifying questions are also answered by the local Ollama model at the end of
the feed, and only those it answers correctly (score >= 0.5) are kept.

  python demo/agent/pick_presets.py --llm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feed import ROOT, build_feed  # noqa: E402
from sketchrag.embed import Embedder  # noqa: E402
from sketchrag.eval.metrics import answer_in_context, score  # noqa: E402
from sketchrag.memory import SketchConfig, SketchMemory  # noqa: E402
from sketchrag.retrieve import RetrieveConfig, build_context  # noqa: E402

OUT = Path(__file__).resolve().parent / "presets.json"


def replay(docs, chunks, embs, qs, q_emb, max_cells):
    mem = SketchMemory(SketchConfig(max_cells=max_cells), dim=embs.shape[1])
    cfg = RetrieveConfig()
    ends = {d.first_chunk + d.n_chunks - 1: d for d in docs}
    hist = {ex.id: [] for ex in qs}  # per question: gate recall at each checkpoint after its doc arrived
    ctxs = {ex.id: [] for ex in qs}  # and the gated context at each of those checkpoints
    arrived = []
    for c, e in zip(chunks, embs):
        mem.ingest(c, e)
        if c.id in ends:
            arrived.append(ends[c.id].key)
            for ex in qs:
                if ex.meta["doc"] in arrived:
                    ctx, info = build_context(mem, ex.question, q_emb[ex.id], cfg, "gate")
                    hist[ex.id].append(answer_in_context(ctx, ex.answers))
                    ctxs[ex.id].append(ctx)
    final = {}
    for ex in qs:
        out = {}
        for policy in ("sketch", "gate"):
            ctx, info = build_context(mem, ex.question, q_emb[ex.id], cfg, policy)
            out[policy] = (ctx, info)
        sims = embs @ q_emb[ex.id]
        dense_ctx = "\n\n".join(chunks[i].text for i in sorted(np.argsort(-sims)[:5]))
        final[ex.id] = {"gate_ctx": out["gate"][0], "gate_info": out["gate"][1],
                        "gate": answer_in_context(out["gate"][0], ex.answers),
                        "sketch": answer_in_context(out["sketch"][0], ex.answers),
                        "dense": answer_in_context(dense_ctx, ex.answers), "ctxs": ctxs[ex.id]}
    return hist, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--keep", type=int, default=8)
    args = ap.parse_args()

    docs, chunks, questions = build_feed()
    emb = Embedder(cache_path=ROOT / "cache" / "emb.sqlite")
    embs = emb.embed([c.text for c in chunks])
    qs = []
    for d in docs:
        for ex in questions[d.key]:
            ex.meta["doc"] = d.key
            qs.append(ex)
    q_emb = dict(zip([ex.id for ex in qs], emb.embed_queries([ex.question for ex in qs])))
    print(f"feed: {len(docs)} docs, {len(chunks)} chunks, {len(qs)} questions", flush=True)

    results = {}
    for cells in (64, 256):
        t0 = time.time()
        hist, final = replay(docs, chunks, embs, qs, q_emb, cells)
        robust = [ex for ex in qs if hist[ex.id] and all(h >= 1.0 for h in hist[ex.id])]
        rate = {k: sum(final[ex.id][k] for ex in qs) / len(qs) for k in ("gate", "sketch", "dense")}
        print(f"{cells} cells: {len(robust)} robust questions; end-of-feed recall {rate} ({time.time() - t0:.0f}s)", flush=True)
        results[cells] = (hist, final, robust, rate)
    cells = max(results, key=lambda k: (len(results[k][2]), results[k][3]["gate"]))
    hist, final, robust, rate = results[cells]
    print(f"using {cells} cells", flush=True)

    doc_of = {d.key: d for d in docs}
    picks = []
    for ex in robust:
        f = final[ex.id]
        d = doc_of[ex.meta["doc"]]
        row = {"id": ex.id, "question": ex.question.strip(), "answers": ex.answers[:3], "kind": ex.kind,
               "doc": d.key, "doc_kind": d.kind, "doc_icon": d.icon, "doc_title": d.title,
               "available_at": d.first_chunk + d.n_chunks, "gate_fired": bool(f["gate_info"]["fallback_cells"]),
               "digest_only_has_answer": bool(f["sketch"]), "dense_has_answer": bool(f["dense"])}
        if args.llm:
            # rehearse at three moments: end of feed, right after the document arrived, and halfway between;
            # a preset must be answered correctly at all of them, since the memory differs over time
            from sketchrag.llm import LLM
            from sketchrag.prompts import MAX_TOKENS, SYSTEM, user_prompt
            cs = f["ctxs"]
            moments = [("end", cs[-1]), ("arrival", cs[0]), ("mid", cs[len(cs) // 2])]
            row["rehearsal"], ok = [], True
            for name, ctx in moments:
                if any(r["context"] == ctx for r in row["rehearsal"]):
                    continue  # same memory state as a moment already rehearsed
                comp = LLM(args.model).complete(SYSTEM, user_prompt(ex.kind, ctx, ex.question), MAX_TOKENS[ex.kind])
                sc = score(ex.kind, comp.text, ex.answers)["score"]
                row["rehearsal"].append({"moment": name, "pred": comp.text, "score": sc, "context": ctx})
                print(f"  [{name}] {ex.question[:55]!r} -> {comp.text[:40]!r} score={sc:.2f}", flush=True)
                if sc < 0.5:
                    ok = False
                    break
            for r in row["rehearsal"]:
                r.pop("context")
            if not ok:
                continue
        picks.append(row)
    # spread over arrival time, and prefer a mix of gate-fired and trusted questions
    picks.sort(key=lambda r: r["available_at"])
    if len(picks) > args.keep:
        idx = np.linspace(0, len(picks) - 1, args.keep).round().astype(int)
        picks = [picks[i] for i in sorted(set(idx))]
    OUT.write_text(json.dumps({"max_cells": cells, "n_questions": len(qs), "end_of_feed_recall": rate,
                               "robust": len(robust), "presets": picks}, indent=1), encoding="utf-8")
    print(f"wrote {len(picks)} presets -> {OUT}")


if __name__ == "__main__":
    main()
