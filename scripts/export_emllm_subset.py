"""Write the exact test questions (seed 0, nested subsample) for the EM-LLM Kaggle run."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sketchrag.data.loaders import load_task  # noqa: E402

LB = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "passage_retrieval_en"]
IB = ["passkey", "kv_retrieval"]
out = ROOT / "kaggle" / "emllm_repro" / "data"
(out / "longbench").mkdir(parents=True, exist_ok=True)
(out / "infinite-bench").mkdir(parents=True, exist_ok=True)
for t in LB:
    with open(out / "longbench" / f"{t}.jsonl", "w", encoding="utf-8") as f:
        for ex in load_task(t, 20, 0):
            f.write(json.dumps({"input": ex.question, "context": ex.context, "answers": ex.answers,
                                "length": ex.meta.get("length"), "dataset": t, "language": "en",
                                "all_classes": ex.meta.get("all_classes"), "_id": ex.id.split("/", 1)[1]}) + "\n")
for t in IB:
    with open(out / "infinite-bench" / f"{t}.jsonl", "w", encoding="utf-8") as f:
        for ex in load_task(t, 10, 0):
            f.write(json.dumps({"id": int(ex.id.split("/")[1]), "context": ex.context, "input": ex.question,
                                "answer": ex.answers, "options": []}) + "\n")
print("wrote", sorted(p.name for p in out.rglob("*.jsonl")))
