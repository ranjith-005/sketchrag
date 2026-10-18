"""Aggregate results/runs/<run>/*.jsonl into the report's tables (markdown + JSON).
Every number in the report must come from here.

  python scripts/make_tables.py --run main
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASK_ORDER = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
              "passage_retrieval_en", "passkey", "kv_retrieval", "longbook_qa_eng"]
SHORT = {"narrativeqa": "NrtvQA", "qasper": "Qasper", "multifieldqa_en": "MFQA", "hotpotqa": "HotpotQA",
         "2wikimqa": "2WikiMQA", "musique": "MuSiQue", "gov_report": "GovRep", "qmsum": "QMSum",
         "passage_retrieval_en": "PRetr", "passkey": "PassKey", "kv_retrieval": "KV", "longbook_qa_eng": "En.QA"}


def load(run: str) -> list[dict]:
    rows = []
    for p in sorted((ROOT / "results" / "runs" / run).glob("*.jsonl")):
        rows += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    rows = load(args.run)
    by = defaultdict(list)
    for r in rows:
        by[(r["system"], r["task"])].append(r)
    systems = sorted({r["system"] for r in rows}, key=lambda s: (not s.startswith(("full", "bm25", "dense", "sara", "amber")), s))
    tasks = [t for t in TASK_ORDER if any(r["task"] == t for r in rows)]

    out = {"quality": {}, "efficiency": {}}
    lines = ["| System | " + " | ".join(SHORT[t] for t in tasks) + " | Avg |", "|" + "---|" * (len(tasks) + 2)]
    for s in systems:
        vals = [100 * mean([r["score"] for r in by[(s, t)]]) if by[(s, t)] else float("nan") for t in tasks]
        avg = mean([v for v in vals if v == v])
        out["quality"][s] = dict(zip(tasks, vals), avg=avg)
        lines.append(f"| {s} | " + " | ".join(f"{v:.1f}" if v == v else "-" for v in vals) + f" | **{avg:.1f}** |")
    quality_md = "\n".join(lines)

    eff_cols = [("context_tokens", "LLM ctx tok", "{:.0f}"), ("output_tokens", "out tok", "{:.0f}"),
                ("llm_calls", "calls", "{:.2f}"), ("llm_s", "LLM s / q", "{:.1f}"), ("hot_bytes", "hot KB", "{:.0f}"),
                ("ingest_ms_per_chunk", "ingest ms/chunk", "{:.2f}"), ("retrieve_s", "retrieve ms", "{:.1f}"),
                ("fallback", "fallback rate", "{:.2f}"), ("answer_recall", "ans. recall", "{:.2f}")]
    lines = ["| System | " + " | ".join(c[1] for c in eff_cols) + " |", "|" + "---|" * (len(eff_cols) + 1)]
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        vals = {}
        for key, _, _ in eff_cols:
            v = mean([r.get(key) for r in rs if key in r])
            if key == "hot_bytes":
                v = v / 1024
            if key == "retrieve_s":
                v = v * 1000
            if key == "fallback":
                v = mean([float(r.get("fallback", False)) for r in rs])
            vals[key] = v
        out["efficiency"][s] = vals
        lines.append(f"| {s} | " + " | ".join(fmt.format(vals[k]) if vals[k] == vals[k] else "-" for k, _, fmt in eff_cols) + " |")
    eff_md = "\n".join(lines)

    dest = ROOT / "results" / "tables"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{args.run}.md").write_text(f"## Answer quality (task score x100)\n\n{quality_md}\n\n## Efficiency (mean per question)\n\n{eff_md}\n", encoding="utf-8")
    (dest / f"{args.run}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(quality_md, "\n\n", eff_md, sep="")


if __name__ == "__main__":
    main()
