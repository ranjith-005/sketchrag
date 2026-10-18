"""Collect every number the slide deck shows from results/ into deck_data.json (no hand-typed figures).

  python report/slides/extract_data.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
LABEL = {"sketchrag_sketch": "SketchRAG digest only", "sketchrag_gate": "SketchRAG + gate", "sketchrag_raw": "SketchRAG always raw",
         "full_context": "Full context", "bm25_rag": "BM25 RAG", "dense_rag": "Dense RAG", "sara_lite": "SARA-lite",
         "amber_lite": "AMBER-lite", "sketchrag_agent": "SketchRAG LLM agent", "sketchrag_hybrid": "SketchRAG gate + agent",
         "em_llm_phi35": "EM-LLM (official)"}


def rows(run):
    out = []
    for p in sorted((RUNS / run).glob("*.jsonl")):
        out += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return out


def macro(rs):
    by = defaultdict(list)
    for r in rs:
        by[r["task"]].append(r["score"])
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def group(rs, key=lambda r: r["system"]):
    g = defaultdict(list)
    for r in rs:
        g[key(r)].append(r)
    return g


main = rows("main")
systems = []
for s, rs in group(main).items():
    systems.append({"key": s, "label": LABEL[s], "score": round(macro(rs), 1), "tokens": round(mean(r["context_tokens"] for r in rs)),
                    "llm_s": round(mean(r["llm_s"] for r in rs), 1), "fallback": round(mean(bool(r.get("fallback")) for r in rs), 2),
                    "calls": round(mean(r["llm_calls"] for r in rs), 2)})
systems.sort(key=lambda x: -x["score"])

by = {s: {r["id"]: r for r in rs} for s, rs in group(main).items()}
ids = list(by["sketchrag_gate"])
fired = [i for i in ids if by["sketchrag_gate"][i]["fallback"]]
trusted = [i for i in ids if not by["sketchrag_gate"][i]["fallback"]]
m = lambda s, g: round(100 * mean(by[s][i]["score"] for i in g), 1)
calibration = {"n": len(ids), "fired": len(fired), "trusted": len(trusted),
               "fired_sketch": m("sketchrag_sketch", fired), "fired_gate": m("sketchrag_gate", fired),
               "trusted_gate": m("sketchrag_gate", trusted), "trusted_raw": m("sketchrag_raw", trusted),
               "agent_tool_calls": sum(1 for r in by["sketchrag_agent"].values() if r.get("llm_calls", 1) > 1)}

per_task = defaultdict(dict)
for s, rs in group(main).items():
    for t, trs in group(rs, lambda r: r["task"]).items():
        per_task[s][t] = round(100 * mean(r["score"] for r in trs), 1)

ab_labels = {"ret.w_lex=0.5": "CMS evidence in ranking", "ret.contiguity=0": "No temporal neighbour",
             "ret.cooccur=false": "No co-occurrence check", "ret.replace_digest=false": "Raw added, digest kept",
             "mem.pointer_mode=reservoir": "16-pointer reservoir per cell", "ret.confidence=noise": "Noise-floor confidence"}
frozen = [r for r in rows("sweep_cells") if r["system"] == "sketchrag_gate@mem.max_cells=64"]
ablations = [{"label": "Frozen configuration", "score": round(macro(frozen), 1), "fallback": round(mean(bool(r.get("fallback")) for r in frozen), 2)}]
for s, rs in group(rows("ablate")).items():
    ablations.append({"label": ab_labels[s.split("@")[1]], "score": round(macro(rs), 1),
                      "fallback": round(mean(bool(r.get("fallback")) for r in rs), 2)})


def sweep(run, key):
    out = defaultdict(list)
    for s, rs in group(rows(run)).items():
        base, setting = s.split("@")
        out[LABEL[base]].append({key: int(re.search(rf"{key}=(\d+)", setting).group(1)), "score": round(macro(rs), 1),
                                 "fallback": round(mean(bool(r.get("fallback")) for r in rs), 2),
                                 "hot_kb": round(mean(r["hot_bytes"] for r in rs) / 1024)})
    return {k: sorted(v, key=lambda x: x[key]) for k, v in out.items()}


stress = {}
for cells in (64, 256):
    stress[cells] = json.loads((ROOT / "results" / "stress" / f"cells_{cells}" / "curve_8488kw.json").read_text())
summary = json.loads((ROOT / "results" / "stress" / "cells_256" / "summary_8488kw.json").read_text())

em = rows("emllm")
em_tasks = sorted({r["task"] for r in em})
matched = [r for r in rows("emllm_match") if r["task"] in em_tasks]
emllm = {"tasks": em_tasks, "questions": len(em),
         "systems": sorted([{"label": LABEL[s], "avg": round(macro(rs), 1)} for s, rs in group(em + matched).items()], key=lambda x: -x["avg"])}

data = {"systems": systems, "calibration": calibration, "per_task": per_task, "ablations": ablations,
        "sweep_cells": sweep("sweep_cells", "max_cells"), "sweep_cms": sweep("sweep_cms", "cms_width"),
        "stress": {"curve_64": stress[64], "curve_256": stress[256],
                   "recall": {k.replace("recall_", ""): v for k, v in summary.items() if k.startswith("recall_")},
                   "n_words": summary["n_words"], "n_chunks": summary["n_chunks"], "gate_fallback_rate": summary["gate_fallback_rate"]},
        "emllm": emllm,
        "presets_feed": json.loads((ROOT / "demo" / "agent" / "presets.json").read_text(encoding="utf-8"))["end_of_feed_recall"]}
out = Path(__file__).with_name("deck_data.json")
out.write_text(json.dumps(data, indent=1), encoding="utf-8")
print(json.dumps({k: data[k] for k in ("calibration", "ablations")}, indent=1))
print([(s["label"], s["score"], s["tokens"], s["llm_s"]) for s in systems])
print(data["sweep_cells"]["SketchRAG + gate"])
print(data["emllm"], data["stress"]["recall"], data["stress"]["curve_64"][-1]["sketch_hot_bytes"], data["stress"]["curve_256"][-1])
