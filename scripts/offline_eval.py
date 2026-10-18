"""LLM-free evaluation: does the context each system would send contain the gold answer?

Reports answer recall (gold answer string inside the context), context size in words,
hot-memory bytes, and for SketchRAG the gate's fallback rate. Used to tune
SketchRAG on a dev split before spending API money.

  python scripts/offline_eval.py --tasks hotpotqa passkey --n 20 --systems dense_rag sketchrag_gate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sketchrag.data.loaders import INFBENCH_TASKS, LONGBENCH_TASKS, load_task  # noqa: E402
from sketchrag.embed import Embedder  # noqa: E402
from sketchrag.eval.metrics import answer_in_context  # noqa: E402
from sketchrag.registry import make_system  # noqa: E402
from sketchrag.systems import DocIndex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=LONGBENCH_TASKS + list(INFBENCH_TASKS))
    ap.add_argument("--systems", nargs="+", default=["bm25_rag", "dense_rag", "sara_lite", "sketchrag_sketch",
                                                     "sketchrag_gate", "sketchrag_raw"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="dev", help="dev uses seed+1000 so tuning never touches test questions")
    ap.add_argument("--chunk-words", type=int, default=150)
    ap.add_argument("--set", nargs="*", default=[], help="override config, e.g. mem.max_cells=32 ret.w_lex=0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed = args.seed + (1000 if args.split == "dev" else 0)
    emb = Embedder(cache_path=ROOT / "cache" / "emb.sqlite")
    systems = [make_system(s, args.set) for s in args.systems]
    agg = defaultdict(lambda: defaultdict(list))
    rows = []
    for task in args.tasks:
        if task in ("gov_report", "qmsum"):  # summaries have no answer string to look for
            continue
        exs = load_task(task, args.n, seed)
        t0 = time.time()
        for ex in exs:
            idx = DocIndex.build(ex, emb, args.chunk_words)
            for sys_ in systems:
                p = sys_.prepare(ex, idx)
                rec = answer_in_context(p.context, ex.answers)
                row = {"id": ex.id, "task": task, "system": sys_.name, "answer_recall": rec,
                       "ctx_words": len(p.context.split()), "n_chunks": len(idx.chunks),
                       "hot_bytes": p.info.get("hot_bytes"), "fallback": bool(p.info.get("fallback_cells")),
                       "gate_reasons": p.info.get("gate_reasons"), "n_cells": p.info.get("n_cells")}
                rows.append(row)
                a = agg[(task, sys_.name)]
                a["rec"].append(rec)
                a["words"].append(row["ctx_words"])
                a["fb"].append(row["fallback"])
        print(f"{task}: {len(exs)} examples in {time.time() - t0:.0f}s", flush=True)

    print(f"\n{'task':22s} {'system':22s} {'ans_recall':>10s} {'ctx_words':>9s} {'fallback':>8s}")
    for (task, s), a in sorted(agg.items()):
        n = len(a["rec"])
        print(f"{task:22s} {s:22s} {sum(a['rec']) / n:10.2f} {sum(a['words']) / n:9.0f} {sum(a['fb']) / n:8.2f}")
    by_sys = defaultdict(list)
    for (task, s), a in agg.items():
        by_sys[s].append(sum(a["rec"]) / len(a["rec"]))
    print("\nmacro answer recall:", {s: round(sum(v) / len(v), 3) for s, v in by_sys.items()})
    for s in by_sys:
        words = [w for (t, s2), a in agg.items() if s2 == s for w in a["words"]]
        fb = [x for (t, s2), a in agg.items() if s2 == s for x in a["fb"]]
        print(f"SUMMARY {s} set={' '.join(args.set) or 'default'} recall={sum(by_sys[s]) / len(by_sys[s]):.3f} "
              f"words={sum(words) / len(words):.0f} fallback={sum(fb) / len(fb):.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
