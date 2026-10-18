"""Report figures, all regenerated from results/.

  python scripts/make_figs.py --run main --sweep sweep_cells --stress results/stress
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
BASE = {"full_context": "#6b7280", "bm25_rag": "#d97706", "dense_rag": "#2563eb", "sara_lite": "#7c3aed",
        "amber_lite": "#db2777"}
SKETCH = "#059669"


def load(run):
    rows = []
    for p in sorted((ROOT / "results" / "runs" / run).glob("*.jsonl")):
        rows += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def pareto(rows, name):
    """Answer quality vs LLM context tokens per question (macro over tasks)."""
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["system"]][r["task"]].append(r)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for s, tasks in agg.items():
        q = mean([100 * mean([r["score"] for r in rs]) for rs in tasks.values()])
        tok = mean([r["context_tokens"] for rs in tasks.values() for r in rs])
        c = BASE.get(s.split("@")[0], SKETCH)
        ax.scatter(tok, q, s=60, color=c, zorder=3)
        off = (6, -11) if s == "sara_lite" else (6, 4)  # keeps it clear of SketchRAG-hybrid at the same score
        ax.annotate(s.replace("sketchrag_", "SketchRAG-"), (tok, q), textcoords="offset points", xytext=off, fontsize=8)
    ax.set_xscale("log")
    style(ax, "Answer quality vs context cost", "LLM context tokens per question (log)", "macro task score")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}_pareto.png", dpi=200)


def sweep(rows, name, key="max_cells"):
    """Quality, fallback rate and hot memory as the budget varies (RQ3/RQ5)."""
    pts = defaultdict(list)
    for r in rows:
        m = re.search(rf"{key}=(\d+)", r["system"])
        if m:
            pts[(r["system"].split("@")[0], int(m.group(1)))].append(r)
    if not pts:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for pol in sorted({p for p, _ in pts}):
        xs = sorted(b for p, b in pts if p == pol)
        rs = [pts[(pol, b)] for b in xs]
        axes[0].plot(xs, [100 * mean([r["score"] for r in g]) for g in rs], marker="o", label=pol)
        axes[1].plot(xs, [mean([float(r.get("fallback", False)) for r in g]) for g in rs], marker="o", label=pol)
        axes[2].plot(xs, [mean([r.get("hot_bytes") for r in g]) / 1024 for g in rs], marker="o", label=pol)
    style(axes[0], "Answer quality", key, "macro score")
    style(axes[1], "Fallback rate", key, "fraction of questions")
    style(axes[2], "Hot memory", key, "KB")
    for ax in axes:
        ax.set_xscale("log", base=2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}_sweep_{key}.png", dpi=200)


def stress(path: Path):
    curves = sorted(path.glob("curve_*.json"), key=lambda p: int(re.search(r"(\d+)kw", p.name).group(1)))
    if not curves:
        return
    curve = json.loads(curves[-1].read_text())
    words = [c["words"] / 1e6 for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(words, [c["dense_bytes"] / 2**20 for c in curve], marker="o", color=BASE["dense_rag"], label="dense RAG index (text + vectors)")
    axes[0].plot(words, [c["text_bytes"] / 2**20 for c in curve], marker="o", color=BASE["bm25_rag"], label="raw text only")
    axes[0].plot(words, [c["sketch_hot_bytes"] / 2**20 for c in curve], marker="o", color=SKETCH, label="SketchRAG hot memory")
    style(axes[0], "Searchable memory vs stream length", "stream length (M words)", "MB")
    axes[0].legend(fontsize=8)
    axes[1].plot(words, [c["ingest_ms_per_chunk"] for c in curve], marker="o", color=SKETCH)
    style(axes[1], "SketchRAG ingest cost", "stream length (M words)", "ms per chunk (cumulative mean)")
    fig.tight_layout()
    fig.savefig(FIG / "stress_memory.png", dpi=200)

    probes = sorted(path.glob("probes_*.jsonl"), key=lambda p: int(re.search(r"(\d+)kw", p.name).group(1)))
    rows = [json.loads(l) for l in probes[-1].read_text().splitlines() if l.strip()]
    bins = [0, 0.25, 0.5, 0.75, 1.0001]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for sysname in rows[0]["recall"]:
        ys = []
        for lo, hi in zip(bins, bins[1:]):
            g = [r["recall"][sysname] for r in rows if lo <= r["age"] < hi]
            ys.append(mean(g))
        c = BASE.get(sysname, SKETCH)
        ls = {"sketchrag_sketch": ":", "sketchrag_raw": "--"}.get(sysname, "-")
        ax.plot([0.125, 0.375, 0.625, 0.875], ys, marker="o", color=c, ls=ls, label=sysname)
    style(ax, "Answer recall by age of the evidence", "age (1 = arrived at stream start)", "answer recall")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "stress_age.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--sweep", nargs="*", default=[])
    ap.add_argument("--stress", default=None)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if args.run:
        pareto(load(args.run), args.run)
    for s in args.sweep:
        rows = load(s)
        for key in ("max_cells", "cms_width"):
            sweep(rows, s, key)
    if args.stress:
        stress(Path(args.stress))
    print("figures in", FIG)


if __name__ == "__main__":
    main()
