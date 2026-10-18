"""Score the official EM-LLM predictions (kaggle/emllm_repro output) with SketchRAG's metrics,
so the reproduction sits in the same tables. Writes results/runs/emllm/em_llm_phi35.jsonl.

  python scripts/score_emllm.py kaggle/emllm_repro/out_v6/emllm_results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sketchrag.data.loaders import load_task  # noqa: E402
from sketchrag.eval.metrics import normalize_answer, score  # noqa: E402

N = {"passkey": 10, "kv_retrieval": 10}  # LongBench tasks: 20

src = Path(sys.argv[1])
out = ROOT / "results" / "runs" / "emllm"
out.mkdir(parents=True, exist_ok=True)
rows = []
for f in sorted(src.glob("*.jsonl")):
    task = f.stem.replace("__long", "")
    preds = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
    if "id" in preds[0]:
        preds.sort(key=lambda r: r["id"])
    exs = load_task(task, N.get(task, 20), 0)
    if len(preds) != len(exs):
        print(f"warning: {task}: {len(preds)} predictions for {len(exs)} examples", flush=True)
    for ex, p in zip(exs, preds):
        gold = p.get("answers") or p.get("answer")
        gold = gold if isinstance(gold, list) else [gold]
        if not {normalize_answer(str(a)) for a in gold} & {normalize_answer(a) for a in ex.answers}:
            raise SystemExit(f"order mismatch in {task}: {gold} vs {ex.answers}")
        rows.append({"system": "em_llm_phi35", "id": ex.id, "task": task, "kind": ex.kind, "pred": p["pred"],
                     "answers": ex.answers[:3], **score(ex.kind, p["pred"], ex.answers),
                     "model": "Phi-3.5-mini-instruct (EM-LLM official)"})
with open(out / "em_llm_phi35.jsonl", "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"{len(rows)} rows -> {out / 'em_llm_phi35.jsonl'}")
