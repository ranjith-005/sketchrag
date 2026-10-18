"""Generate every table, figure and number used in the paper from results/.

  python report/paper/make_assets.py      ->  report/paper/{tables,figs}/ and numbers.tex

Nothing in the paper's tables or quoted numbers is typed by hand: the prose uses the macros
written to numbers.tex, and the tables are \\input from tables/.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "results" / "runs"
TAB, FIG = HERE / "tables", HERE / "figs"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
         "passage_retrieval_en", "passkey", "kv_retrieval", "longbook_qa_eng"]
SHORT = {"narrativeqa": "NQA", "qasper": "Qasp", "multifieldqa_en": "MFQA", "hotpotqa": "HQA", "2wikimqa": "2WQA",
         "musique": "MuSQ", "gov_report": "GovR", "qmsum": "QMS", "passage_retrieval_en": "PRet", "passkey": "PKey",
         "kv_retrieval": "KV", "longbook_qa_eng": "EnQA"}
LABEL = {"full_context": "Full context", "bm25_rag": "BM25 RAG", "dense_rag": "Dense RAG", "sara_lite": "SARA-lite",
         "amber_lite": "AMBER-lite", "sketchrag_sketch": "SketchRAG-sketch", "sketchrag_raw": "SketchRAG-raw",
         "sketchrag_gate": "SketchRAG-gate", "sketchrag_agent": "SketchRAG-agent", "sketchrag_hybrid": "SketchRAG-hybrid",
         "em_llm_phi35": "EM-LLM (official)"}
ORDER = ["full_context", "bm25_rag", "dense_rag", "sara_lite", "amber_lite", "sketchrag_sketch", "sketchrag_raw",
         "sketchrag_gate", "sketchrag_agent", "sketchrag_hybrid"]
COL = {"full_context": "#6b7280", "bm25_rag": "#d97706", "dense_rag": "#2563eb", "sara_lite": "#7c3aed",
       "amber_lite": "#db2777"}
SKETCH = "#0d9488"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def rows(run):
    out = []
    for p in sorted((RUNS / run).glob("*.jsonl")):
        out += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return out


def group(rs, key=lambda r: r["system"]):
    g = defaultdict(list)
    for r in rs:
        g[key(r)].append(r)
    return g


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def task_means(rs):
    return {t: 100 * mean(r["score"] for r in trs) for t, trs in group(rs, lambda r: r["task"]).items()}


def macro(rs):
    return mean(task_means(rs).values())


N = {}  # macros for numbers.tex


def num(name, value, fmt="{:.1f}"):
    N[name] = fmt.format(value) if not isinstance(value, str) else value


# ---------------- main run ----------------
main = group(rows("main"))
per = {s: task_means(rs) for s, rs in main.items()}
avg = {s: macro(rs) for s, rs in main.items()}

lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Answer quality on the 350 test questions (task score $\times$100; F1 for QA, "
         r"ROUGE-L for GovR/QMS, accuracy for PRet/PKey/KV). All systems use qwen2.5:7b-instruct with the same prompt; "
         r"only the context differs. Best per column in bold.}", r"\label{tab:main}", r"\setlength{\tabcolsep}{4.2pt}",
         r"\begin{tabular}{l" + "r" * (len(TASKS) + 1) + "}", r"\toprule",
         "System & " + " & ".join(SHORT[t] for t in TASKS) + r" & \textbf{Avg} \\", r"\midrule"]
best = {t: max(per[s][t] for s in ORDER) for t in TASKS}
best_avg = max(avg[s] for s in ORDER)
for i, s in enumerate(ORDER):
    if i == 5:
        lines.append(r"\midrule")
    cells = [(r"\textbf{%.1f}" if abs(per[s][t] - best[t]) < 1e-9 else "%.1f") % per[s][t] for t in TASKS]
    a = (r"\textbf{%.1f}" if abs(avg[s] - best_avg) < 1e-9 else "%.1f") % avg[s]
    lines.append(f"{LABEL[s]} & " + " & ".join(cells) + f" & {a} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(TAB / "main.tex").write_text("\n".join(lines), encoding="utf-8")

eff = {}
for s, rs in main.items():
    eff[s] = {"tok": mean(r["context_tokens"] for r in rs), "calls": mean(r["llm_calls"] for r in rs),
              "llm": mean(r["llm_s"] for r in rs), "fb": mean(float(bool(r.get("fallback"))) for r in rs),
              "rec": mean(r.get("answer_recall") for r in rs), "hot": mean(r.get("hot_bytes") for r in rs) / 1024,
              "ing": mean(r.get("ingest_ms_per_chunk") for r in rs)}
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Efficiency per question (means over 350 questions). Tok = LLM context "
         r"tokens; Rec = answer recall of the context (gold string present); Fb = gate fallback rate.}", r"\label{tab:eff}",
         r"\setlength{\tabcolsep}{3.4pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"System & Tok & Calls & LLM s & Rec & Fb \\", r"\midrule"]
for s in ORDER:
    e = eff[s]
    lines.append(f"{LABEL[s]} & {e['tok']:,.0f} & {e['calls']:.2f} & {e['llm']:.1f} & {e['rec']:.2f} & {e['fb']:.2f} \\\\".replace(",", "{,}"))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "efficiency.tex").write_text("\n".join(lines), encoding="utf-8")

g, f, b, d, sk, rw, ag = (avg[k] for k in ("sketchrag_gate", "full_context", "bm25_rag", "dense_rag", "sketchrag_sketch",
                                            "sketchrag_raw", "sketchrag_agent"))
for k, v in [("GateScore", g), ("FullScore", f), ("BMScore", b), ("DenseScore", d), ("SketchScore", sk), ("RawScore", rw),
             ("AgentScore", ag), ("SaraScore", avg["sara_lite"]), ("AmberScore", avg["amber_lite"]),
             ("HybridScore", avg["sketchrag_hybrid"])]:
    num(k, v)
num("GateTok", eff["sketchrag_gate"]["tok"], "{:,.0f}")
num("FullTok", eff["full_context"]["tok"], "{:,.0f}")
num("RawTok", eff["sketchrag_raw"]["tok"], "{:,.0f}")
num("GateTokPct", 100 * eff["sketchrag_gate"]["tok"] / eff["full_context"]["tok"])
num("GateLLMs", eff["sketchrag_gate"]["llm"])
num("FullLLMs", eff["full_context"]["llm"])
num("AmberCalls", eff["amber_lite"]["calls"])
num("GateHotKB", eff["sketchrag_gate"]["hot"], "{:.0f}")
num("DenseHotKB", eff["dense_rag"]["hot"], "{:.0f}")
num("GateIngest", eff["sketchrag_gate"]["ing"])
num("KVSketch", per["sketchrag_sketch"]["kv_retrieval"])
num("KVGate", per["sketchrag_gate"]["kv_retrieval"])
num("PKFull", per["full_context"]["passkey"])
num("WQAGate", per["sketchrag_gate"]["2wikimqa"])
num("WQASara", per["sara_lite"]["2wikimqa"])
num("WQAFull", per["full_context"]["2wikimqa"])
num("PRetBM", per["bm25_rag"]["passage_retrieval_en"])
num("PRetGate", per["sketchrag_gate"]["passage_retrieval_en"])
num("GovMin", min(per[s]["gov_report"] for s in ("sketchrag_sketch", "sketchrag_gate", "sketchrag_raw")))
num("GovMax", max(per[s]["gov_report"] for s in ("sketchrag_sketch", "sketchrag_gate", "sketchrag_raw")))
num("GainPct", 100 * (g - sk) / (rw - sk), "{:.0f}")
num("TokSavePct", 100 * (1 - eff["sketchrag_gate"]["tok"] / eff["sketchrag_raw"]["tok"]), "{:.0f}")
num("TruncatedFull", str(sum(bool(r.get("truncated")) for r in main["full_context"])))

# gate calibration
by = {s: {r["id"]: r for r in rs} for s, rs in main.items()}
ids = list(by["sketchrag_gate"])
fired = [i for i in ids if by["sketchrag_gate"][i]["fallback"]]
trusted = [i for i in ids if not by["sketchrag_gate"][i]["fallback"]]
m = lambda s, grp: 100 * mean(by[s][i]["score"] for i in grp)
num("NQ", str(len(ids)))
num("NFired", str(len(fired)))
num("NTrusted", str(len(trusted)))
num("FiredPct", 100 * len(fired) / len(ids), "{:.0f}")
num("FiredSketch", m("sketchrag_sketch", fired))
num("FiredGate", m("sketchrag_gate", fired))
num("TrustedGate", m("sketchrag_gate", trusted))
num("TrustedRaw", m("sketchrag_raw", trusted))
fire_task = {t: mean(float(bool(by["sketchrag_gate"][i]["fallback"])) for i in ids if by["sketchrag_gate"][i]["task"] == t) for t in TASKS}
num("FireKV", 100 * fire_task["kv_retrieval"], "{:.0f}")
num("FirePK", 100 * fire_task["passkey"], "{:.0f}")
num("FireGov", 100 * fire_task["gov_report"], "{:.0f}")
num("AgentToolQ", str(sum(1 for r in by["sketchrag_agent"].values() if r["llm_calls"] > 1)))

# ---------------- ablations and sweeps (110 questions) ----------------
ab_name = {"ret.w_lex=0.5": "CMS evidence in ranking ($w_{lex}{=}0.5$)", "ret.contiguity=0": "No temporal neighbour",
           "ret.cooccur=false": "No co-occurrence check", "ret.replace_digest=false": "Raw added, digest kept",
           "mem.pointer_mode=reservoir": "16-pointer reservoir per cell", "ret.confidence=noise": "Noise-floor confidence"}
frozen = [r for r in rows("sweep_cells") if r["system"] == "sketchrag_gate@mem.max_cells=64"]
fz = macro(frozen)
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Ablations of SketchRAG-gate (110 questions, 11 tasks $\times$ 10). "
         r"$\Delta$ is relative to the frozen configuration.}", r"\label{tab:ablate}", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrr}", r"\toprule", r"Variant & Score & $\Delta$ & Fb \\", r"\midrule",
         f"Frozen configuration & {fz:.1f} & -- & {mean(float(bool(r.get('fallback'))) for r in frozen):.2f} \\\\"]
for s, rs in group(rows("ablate")).items():
    setting = s.split("@")[1]
    v = macro(rs)
    lines.append(f"{ab_name[setting]} & {v:.1f} & {v - fz:+.1f} & {mean(float(bool(r.get('fallback'))) for r in rs):.2f} \\\\")
    if setting == "mem.pointer_mode=reservoir":
        num("ResScore", v)
    if setting == "ret.contiguity=0":
        num("NoContig", fz - v)
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "ablate.tex").write_text("\n".join(lines), encoding="utf-8")
num("FrozenScore", fz)


def sweep(run, key):
    out = defaultdict(list)
    for s, rs in group(rows(run)).items():
        base, setting = s.split("@")
        out[base].append((int(re.search(rf"{key}=(\d+)", setting).group(1)), macro(rs),
                          mean(float(bool(r.get("fallback"))) for r in rs), mean(r["hot_bytes"] for r in rs) / 1024,
                          mean(r["answer_recall"] for r in rs)))
    return {k: sorted(v) for k, v in out.items()}


cells = sweep("sweep_cells", "max_cells")
cms = sweep("sweep_cms", "cms_width")
gs = cells["sketchrag_gate"]
num("SweepMin", min(p[1] for p in gs))
num("SweepMax", max(p[1] for p in gs))
num("FbEight", gs[0][2], "{:.2f}")
num("FbTwoFiftySix", gs[-1][2], "{:.2f}")
num("HotEight", gs[0][3], "{:.0f}")
num("HotTwoFiftySix", gs[-1][3], "{:.0f}")
sks = cells["sketchrag_sketch"]
num("RecSketchEight", sks[0][4], "{:.2f}")
num("RecSketchTwoFiftySix", sks[-1][4], "{:.2f}")
cg = cms["sketchrag_gate"]
num("FrozenRerun", [p for p in cg if p[0] == 512][0][1])
num("CmsMin", min(p[1] for p in cg))
num("CmsMax", max(p[1] for p in cg))
num("CmsFbSixtyFour", cg[0][2], "{:.2f}")
num("CmsFbFiveTwelve", [p for p in cg if p[0] == 512][0][2], "{:.2f}")
num("CmsHotTwoK", cg[-1][3], "{:,.0f}")
num("CmsHotFiveTwelve", [p for p in cg if p[0] == 512][0][3], "{:.0f}")

fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.55))
for base, lab, c, ls in [("sketchrag_gate", "gate", SKETCH, "-"), ("sketchrag_sketch", "sketch", "#94a3b8", "--")]:
    xs = [p[0] for p in cells[base]]
    ax[0].plot(xs, [p[1] for p in cells[base]], marker="o", ms=3, color=c, ls=ls, label=lab)
    ax[1].plot(xs, [p[2] for p in cells[base]], marker="o", ms=3, color=c, ls=ls, label=lab)
ax2 = ax[1]
for a, t, yl in [(ax[0], "Score", "task score"), (ax2, "Fallback rate", "fraction")]:
    a.set_xscale("log", base=2)
    a.set_xticks([8, 32, 128])
    a.set_xticklabels(["8", "32", "128"])
    a.set_title(t)
    a.set_xlabel("max cells")
    a.grid(alpha=0.25)
ax[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.52))
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "sweep_cells.pdf")
plt.close(fig)

# ---------------- accuracy vs cost ----------------
fig, ax = plt.subplots(figsize=(3.45, 2.15))
lab_off = {"full_context": (-4, 4, "right"), "bm25_rag": (5, -2, "left"), "sketchrag_raw": (5, 2, "left"),
           "sketchrag_gate": (5, -7, "left"), "sketchrag_hybrid": (5, 3, "left"), "sara_lite": (5, -6, "left"),
           "dense_rag": (5, -3, "left"), "amber_lite": (5, -3, "left"), "sketchrag_agent": (5, 2, "left"),
           "sketchrag_sketch": (5, 1, "left")}
for s in ORDER:
    c = COL.get(s, SKETCH)
    ax.scatter(eff[s]["tok"], avg[s], s=24, color=c, zorder=3, edgecolor="white", linewidth=0.5)
    dx, dy, ha = lab_off[s]
    ax.annotate(LABEL[s].replace("SketchRAG-", "SR-"), (eff[s]["tok"], avg[s]), textcoords="offset points",
                xytext=(dx, dy), fontsize=6.3, ha=ha)
ax.set_xscale("log")
ax.set_xticks([1000, 2000, 5000, 10000, 15000])
ax.set_xticklabels(["1k", "2k", "5k", "10k", "15k"])
ax.minorticks_off()
ax.set_xlim(900, 24000)
ax.set_ylim(28.5, 47.5)
ax.set_xlabel("LLM context tokens per question (log)")
ax.set_ylabel("average score")
ax.grid(alpha=0.25)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "pareto.pdf")
plt.close(fig)

# ---------------- stress test ----------------
curves = {c: json.loads((ROOT / "results" / "stress" / f"cells_{c}" / "curve_8488kw.json").read_text()) for c in (64, 256)}
summ = json.loads((ROOT / "results" / "stress" / "cells_256" / "summary_8488kw.json").read_text())
c256, c64 = curves[256], curves[64]
MB = 1024 * 1024
fig, ax = plt.subplots(figsize=(3.45, 1.75))
w = [c["words"] / 1e6 for c in c256]
ax.plot(w, [c["dense_bytes"] / MB for c in c256], color=COL["dense_rag"], marker="o", ms=2.5, label="dense index (text + vectors)")
ax.plot(w, [c["text_bytes"] / MB for c in c256], color=COL["bm25_rag"], marker="o", ms=2.5, label="raw text")
ax.plot(w, [c["sketch_hot_bytes"] / MB for c in c256], color="#5eead4", marker="o", ms=2.5, label="SketchRAG hot, 256 cells")
ax.plot([c["words"] / 1e6 for c in c64], [c["sketch_hot_bytes"] / MB for c in c64], color=SKETCH, marker="o", ms=2.5,
        label="SketchRAG hot, 64 cells")
ax.set_xlabel("stream length (million words)")
ax.set_ylabel("MB")
ax.grid(alpha=0.25)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "stress.pdf")
plt.close(fig)
num("StressWords", summ["n_words"], "{:,}")
num("StressMWords", summ["n_words"] / 1e6)
num("StressChunks", summ["n_chunks"], "{:,}")
num("StressHotSixtyFour", c64[-1]["sketch_hot_bytes"] / MB)
num("StressHotTwoFiftySix", c256[-1]["sketch_hot_bytes"] / MB)
num("StressDense", c256[-1]["dense_bytes"] / MB, "{:.0f}")
num("StressText", c256[-1]["text_bytes"] / MB, "{:.0f}")
num("StressIngest", c256[-1]["ingest_ms_per_chunk"])
for k, name in [("dense_rag", "RecDense"), ("bm25_rag", "RecBM"), ("sketchrag_sketch", "RecSRSketch"),
                ("sketchrag_gate", "RecSRGate"), ("sketchrag_raw", "RecSRRaw")]:
    num(name, summ[f"recall_{k}"], "{:.2f}")
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from sketchrag.data.long_stream import build_stream  # noqa: E402
num("StressDocs", len(build_stream()[0]), "{:,}")
demo = json.loads((ROOT / "demo" / "agent" / "presets.json").read_text(encoding="utf-8"))
num("DemoGate", 100 * demo["end_of_feed_recall"]["gate"], "{:.0f}")
num("DemoDense", 100 * demo["end_of_feed_recall"]["dense"], "{:.0f}")
num("DemoQ", str(demo["n_questions"]))
num("StressFb", summ["gate_fallback_rate"], "{:.2f}")
num("StressProbes", str(summ["probes"]))

# ---------------- EM-LLM matched comparison ----------------
em = rows("emllm")
em_tasks = [t for t in TASKS if t in {r["task"] for r in em}]
matched = [r for r in rows("emllm_match") if r["task"] in em_tasks]
emg = group(em + matched)
em_avg = {s: macro(rs) for s, rs in emg.items()}
em_order = sorted(emg, key=lambda s: -em_avg[s])
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Matched comparison with EM-LLM's official code: same base model "
         r"(Phi-3.5-mini) and the same %d questions on the %d LongBench tasks EM-LLM completed.}" % (len(em), len(em_tasks)),
         r"\label{tab:emllm}", r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
         r"\begin{tabular}{l" + "r" * (len(em_tasks) + 1) + "}", r"\toprule",
         "System & " + " & ".join(SHORT[t] for t in em_tasks) + r" & Avg \\", r"\midrule"]
for s in em_order:
    tm = task_means(emg[s])
    lines.append(f"{LABEL[s]} & " + " & ".join(f"{tm[t]:.1f}" for t in em_tasks) + f" & {em_avg[s]:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "emllm.tex").write_text("\n".join(lines), encoding="utf-8")
num("EMGate", em_avg["sketchrag_gate"])
num("EMLLM", em_avg["em_llm_phi35"])
num("EMFull", em_avg["full_context"])
num("EMQ", str(len(em)))
num("EMTasks", str(len(em_tasks)))

# ---------------- gate fire rate per task ----------------
fig, ax = plt.subplots(figsize=(3.45, 1.45))
order_t = sorted(TASKS, key=lambda t: -fire_task[t])
ax.bar([SHORT[t] for t in order_t], [100 * fire_task[t] for t in order_t], color="#7c3aed", width=0.65)
ax.set_ylabel("% escalated")
ax.set_ylim(0, 105)
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fire_rate.pdf")
plt.close(fig)

# ---------------- qualitative cases (real predictions from the main run) ----------------
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from sketchrag.data.loaders import load_task  # noqa: E402


def tex_escape(t):
    return (t.replace("\\", "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace("$", r"\$").replace("{", "").replace("}", ""))


CASES = [("narrativeqa/a5e44b1e2b98f80aabab5f1b8296d0fe66d53b31c1f3e332", "NQA"),
         ("kv_retrieval/130", "KV"),
         ("musique/fe8275738f337cc423752bea9b254da45cecdf2559f38352", "MuSQ"),
         ("2wikimqa/0a64d8873482d91efc595a508218c6ce881c13c95028039e", "2WQA")]
lines = [r"\begin{table*}[t]", r"\centering", r"\caption{Real answers from the main run (qwen2.5:7b). The gate's escalation "
         r"reason is shown per question. The last row is a failure: the gate escalated, but the evidence was not in the "
         r"retrieved cells.}", r"\label{tab:cases}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{l p{4.0cm} l l l l l}", r"\toprule",
         r"Task & Question (abridged) & Gold & Digest only & SketchRAG-gate & Full context & Gate \\", r"\midrule"]
for cid, short in CASES:
    task = cid.split("/")[0]
    ex = next(e for e in load_task(task, 30, 0) if e.id == cid)
    q = " ".join(ex.question.split())
    if task == "kv_retrieval":
        q = "Value for key 35b9fdc1-c2c2-\\ldots?"
    elif len(q) < 95:
        q = tex_escape(q)
    else:
        q = tex_escape(q[:92].rsplit(" ", 1)[0]) + r"\ldots"
    cut = lambda s, n=19: tex_escape(s if len(s) <= n else s[: n - 1] + "…").replace("…", r"\ldots")
    reason = ", ".join(sorted(set(by["sketchrag_gate"][cid]["gate_reasons"].values())))
    lines.append(f"{short} & {q} &{cut(by['sketchrag_gate'][cid]['answers'][0], 16)} & "
                 f"{cut(by['sketchrag_sketch'][cid]['pred'])} & {cut(by['sketchrag_gate'][cid]['pred'])} & "
                 f"{cut(by['full_context'][cid]['pred'])} & {reason} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(TAB / "cases.tex").write_text("\n".join(lines), encoding="utf-8")

# ---------------- hot-memory accounting (measured on a real document stream) ----------------
sys.path.insert(0, str(ROOT / "demo"))
import numpy as np  # noqa: E402
import demo_core as dc  # noqa: E402

ex = dc.find_example("narrativeqa/a5e44b1e2b98f80aabab5f1b8296d0fe66d53b31c1f3e332")
chunks_, embs_ = dc.chunk_and_embed(ex)
mem, _ = dc.stream(chunks_, embs_)
c0 = mem.cells[0]
exm = np.mean([c.exemplars.nbytes for c in mem.cells])
dig = np.mean([sum(len(s.encode()) + 12 for _, _, s in c.digest) for c in mem.cells])
rows_m = [("Centroid sum (384 $\\times$ fp32)", c0.vsum.nbytes, "per cell"),
          ("Count-Min Sketch ($3\\times512$ int32)", c0.cms.nbytes, "per cell"),
          ("Exemplars ($\\le$8 $\\times$ 384 fp16; mean here)", exm, "per cell"),
          ("Extractive digest ($\\le$120 words; mean)", dig, "per cell"),
          ("Document-frequency CMS ($3\\times8192$)", mem.df.nbytes, "global"),
          ("Misra--Gries table ($k{=}256$)", mem.mg.nbytes, "global")]
lines = [r"\begin{table}[t]", r"\centering", r"\caption{Hot-memory accounting, measured on a %d-chunk NarrativeQA stream with "
         r"$K{=}64$ cells (total %.0f~KB). Cold storage is not included.}" % (len(chunks_), mem.hot_nbytes() / 1024),
         r"\label{tab:memory}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lrl}", r"\toprule",
         r"Component & Bytes & Scope \\", r"\midrule"]
for name, b_, scope in rows_m:
    lines.append(f"{name} & {b_:,.0f} & {scope} \\\\".replace(",", "{,}"))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(TAB / "memory.tex").write_text("\n".join(lines), encoding="utf-8")
per_cell_max = c0.vsum.nbytes + c0.cms.nbytes + 8 * 384 * 2 + dig + 32
num("CellKBMax", per_cell_max / 1024)
num("CellKBMean", (c0.vsum.nbytes + c0.cms.nbytes + exm + dig + 32) / 1024)

# ---------------- write macros ----------------
out = ["% generated by make_assets.py from results/; do not edit by hand"]
for k, v in N.items():
    out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(HERE / "numbers.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(N)} numbers, tables: {[p.name for p in TAB.iterdir()]}, figs: {[p.name for p in FIG.iterdir()]}")
print({k: N[k] for k in ("GateScore", "BMScore", "GateTok", "FiredSketch", "FiredGate", "TrustedGate", "TrustedRaw",
                         "GainPct", "TokSavePct", "AgentToolQ", "ResScore", "FrozenScore", "EMGate", "EMLLM", "RecSRGate", "RecDense", "StressDocs", "DemoGate", "FrozenRerun")})
