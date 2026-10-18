"""Build report/SketchRAG_Report.html (and .pdf via headless Chrome) from results/.

Sections whose results are not there yet are rendered as an explicit "pending" note, so the
report can never silently show stale or invented numbers.

  python report/build_report.py            # html + pdf
"""
from __future__ import annotations

import base64
import html
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "report"))
import content as C  # noqa: E402

RUNS = ROOT / "results" / "runs"
FIG = ROOT / "report" / "figures"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
         "passage_retrieval_en", "passkey", "kv_retrieval", "longbook_qa_eng"]
SHORT = {"narrativeqa": "NrtvQA", "qasper": "Qasper", "multifieldqa_en": "MFQA", "hotpotqa": "HotpotQA",
         "2wikimqa": "2Wiki", "musique": "MuSiQue", "gov_report": "GovRep", "qmsum": "QMSum",
         "passage_retrieval_en": "PRetr", "passkey": "PassKey", "kv_retrieval": "KV", "longbook_qa_eng": "En.QA"}
NAMES = {"full_context": "Full context", "bm25_rag": "BM25 RAG", "dense_rag": "Dense RAG", "sara_lite": "SARA-lite",
         "amber_lite": "AMBER-lite", "sketchrag_sketch": "SketchRAG &middot; sketch", "sketchrag_gate": "SketchRAG &middot; gate",
         "sketchrag_raw": "SketchRAG &middot; raw", "sketchrag_agent": "SketchRAG &middot; agent",
         "sketchrag_hybrid": "SketchRAG &middot; hybrid", "em_llm_phi35": "EM-LLM (official)"}
ORDER = list(NAMES)


def load(run: str) -> list[dict]:
    rows = []
    for p in sorted((RUNS / run).glob("*.jsonl")) if (RUNS / run).exists() else []:
        rows += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return rows


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f1(x, nd=1, scale=1.0):
    return "&ndash;" if x is None else f"{x * scale:.{nd}f}"


def pending(what: str) -> str:
    return f'<p class="pending">Pending: {html.escape(what)} (no result files yet).</p>'


def img(name: str, caption: str) -> str:
    p = FIG / name
    if not p.exists():
        return pending(f"figure {name}")
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}"><figcaption>{caption}</figcaption></figure>'


def quality_table(rows: list[dict], tasks=None, systems=None) -> str:
    by = defaultdict(list)
    for r in rows:
        by[(r["system"], r["task"])].append(r["score"])
    tasks = tasks or [t for t in TASKS if any(r["task"] == t for r in rows)]
    systems = systems or sorted({r["system"] for r in rows}, key=lambda s: ORDER.index(s) if s in ORDER else 99)
    avgs = {}
    body = []
    for s in systems:
        vals = [mean(by[(s, t)]) for t in tasks]
        avg = mean([v for v in vals if v is not None])
        avgs[s] = avg
        body.append((s, vals, avg))
    best = max(a for a in avgs.values() if a is not None) if avgs else None
    out = ['<table><tr><th>System</th>' + "".join(f"<th>{SHORT[t]}</th>" for t in tasks) + "<th>Avg</th></tr>"]
    for s, vals, avg in body:
        cls = ' class="ours"' if s.startswith("sketchrag") else ""
        bold = "<b>" if avg == best else ""
        out.append(f"<tr{cls}><td>{NAMES.get(s, html.escape(s))}</td>" + "".join(f"<td>{f1(v, 1, 100)}</td>" for v in vals)
                   + f"<td>{bold}{f1(avg, 1, 100)}{'</b>' if bold else ''}</td></tr>")
    out.append("</table>")
    return "".join(out)


def efficiency_table(rows: list[dict]) -> str:
    systems = sorted({r["system"] for r in rows}, key=lambda s: ORDER.index(s) if s in ORDER else 99)
    cols = [("context_tokens", "LLM ctx tokens", 0, 1), ("llm_calls", "LLM calls", 2, 1), ("llm_s", "LLM s", 1, 1),
            ("hot_bytes", "searchable memory (KB)", 0, 1 / 1024), ("answer_recall", "answer recall", 2, 1),
            ("fallback", "fallback rate", 2, 1)]
    out = ["<table><tr><th>System</th>" + "".join(f"<th>{c[1]}</th>" for c in cols) + "</tr>"]
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        cells = []
        for key, _, nd, sc in cols:
            if key == "fallback":
                v = mean([float(r.get("fallback", False)) for r in rs]) if s.startswith(("sketchrag", "amber")) else None
            else:
                v = mean([r.get(key) for r in rs if key in r])
            cells.append(f1(v, nd, sc))
        cls = ' class="ours"' if s.startswith("sketchrag") else ""
        out.append(f"<tr{cls}><td>{NAMES.get(s, html.escape(s))}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def ablation_table(main: list[dict], abl: list[dict]) -> str:
    base = [r for r in main if r["system"] == "sketchrag_gate"]
    ids = {r["id"] for r in abl}
    base = [r for r in base if r["id"] in ids] or base
    labels = {"ret.w_lex=0.5": "+ CMS evidence in cell ranking (w=0.5)", "ret.contiguity=0": "&minus; temporal contiguity",
              "ret.cooccur=false": "&minus; co-occurrence check in gate", "ret.replace_digest=false": "raw appended instead of replacing digest",
              "mem.pointer_mode=reservoir": "bounded reservoir of 16 pointers per cell", "ret.confidence=noise": "gate uses empirical noise floor instead of &epsilon;N bound"}
    rows = [("SketchRAG &middot; gate (frozen config)", base)]
    for s in sorted({r["system"] for r in abl}):
        ov = s.split("@", 1)[1] if "@" in s else s
        rows.append((labels.get(ov, html.escape(ov)), [r for r in abl if r["system"] == s]))
    out = ["<table><tr><th>Variant</th><th>Avg score</th><th>LLM ctx tokens</th><th>fallback rate</th><th>n</th></tr>"]
    for name, rs in rows:
        out.append(f"<tr><td>{name}</td><td>{f1(mean([r['score'] for r in rs]), 1, 100)}</td>"
                   f"<td>{f1(mean([r['context_tokens'] for r in rs]), 0)}</td>"
                   f"<td>{f1(mean([float(r.get('fallback', False)) for r in rs]), 2)}</td><td>{len(rs)}</td></tr>")
    out.append("</table>")
    return "".join(out)


def stress_section() -> str:
    parts = []
    rows = []
    for d in sorted((ROOT / "results" / "stress").glob("cells_*"), key=lambda p: int(p.name.split("_")[1])):
        sm = sorted(d.glob("summary_*.json"))
        if not sm:
            continue
        s = json.loads(sm[-1].read_text())
        rows.append((int(d.name.split("_")[1]), s))
    if not rows:
        return pending("long-stream stress test")
    s0 = rows[0][1]
    parts.append(f"<p>Stream: {s0['n_chunks']:,} chunks, {s0['n_words']:,} words, {s0['probes']} probe questions. "
                 f"Final dense-RAG index (text + float32 vectors): <b>{s0['final']['dense_bytes'] / 2**20:.0f} MB</b>; "
                 f"raw text alone: {s0['final']['text_bytes'] / 2**20:.0f} MB.</p>")
    t = ["<table><tr><th>max cells</th><th>hot memory (MB)</th><th>ingest ms/chunk</th><th>answer recall: sketch</th>"
         "<th>gate</th><th>raw</th><th>dense RAG</th><th>BM25 RAG</th><th>gate fallback</th></tr>"]
    for c, s in rows:
        t.append(f"<tr><td>{c}</td><td>{s['final']['sketch_hot_bytes'] / 2**20:.2f}</td><td>{s['final']['ingest_ms_per_chunk']:.1f}</td>"
                 f"<td>{s['recall_sketchrag_sketch']:.2f}</td><td>{s['recall_sketchrag_gate']:.2f}</td><td>{s['recall_sketchrag_raw']:.2f}</td>"
                 f"<td>{s['recall_dense_rag']:.2f}</td><td>{s['recall_bm25_rag']:.2f}</td><td>{s['gate_fallback_rate']:.2f}</td></tr>")
    t.append("</table>")
    parts.append("".join(t))
    parts.append(img("stress_memory.png", "Searchable memory and ingest cost as the stream grows (256 cells)."))
    parts.append(img("stress_age.png", "Answer recall by how long ago the evidence arrived (256 cells)."))
    return "".join(parts)


def emllm_section() -> str:
    rows = load("emllm")
    if not rows:
        return pending("EM-LLM reproduction")
    paper = json.loads((ROOT / "results" / "emllm_raw" / "paper_phi35.json").read_text())
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r["score"])
    out = ["<table><tr><th>Category</th><th>Tasks run</th><th>Ours (official code, subset)</th><th>Paper (full LongBench)</th></tr>"]
    for cat, tasks in paper["category_tasks"].items():
        have = [t for t in tasks if by.get(t)]
        ours = mean([mean(by[t]) for t in have]) if have else None
        out.append(f"<tr><td>{cat}</td><td>{', '.join(SHORT[t] for t in have) or '&ndash;'}"
                   f"{' (of ' + str(len(tasks)) + ')' if len(have) < len(tasks) else ''}</td>"
                   f"<td>{f1(ours, 1, 100)}</td><td>{paper['categories'][cat]}</td></tr>")
    out.append("</table>")
    matched = load("emllm_match")
    if matched:
        tasks = [t for t in TASKS if t in by]
        out.append("<p>Same base model (Phi-3.5-mini), same questions:</p>")
        out.append(quality_table(rows + [r for r in matched if r["task"] in tasks], tasks))
    else:
        out.append(pending("Phi-3.5-matched SketchRAG / RAG runs"))
    info = ROOT / "results" / "emllm_raw"
    changes = []
    for p in sorted(info.glob("*/run_info.json")):
        changes += json.loads(p.read_text())["changes"]
    out.append("<p><b>Changes forced by the hardware</b> (2&times;T4, 29 GB RAM):</p><ul>"
               + "".join(f"<li>{html.escape(c)}</li>" for c in dict.fromkeys(changes)) + "</ul>")
    return "".join(out)


CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;font-size:10.5pt;line-height:1.45;color:#111;max-width:180mm;margin:12mm auto}
h1{font-size:17pt;margin:0 0 4px}h2{font-size:13pt;border-bottom:1px solid #999;margin-top:22px}h3{font-size:11pt;margin:14px 0 4px}
.authors{color:#333;margin-bottom:2px}.inst{color:#555;font-size:9.5pt;margin-bottom:14px}
table{border-collapse:collapse;margin:8px 0;font-size:8.6pt;width:100%}th,td{border:1px solid #bbb;padding:3px 5px;text-align:center}
td:first-child,th:first-child{text-align:left}tr.ours td{background:#eef7f1}th{background:#f1f1f1}
figure{margin:10px 0;text-align:center}figure img{max-width:88%;max-height:105mm}figcaption{font-size:9pt;color:#444}
.pending{color:#a33;font-style:italic}code{font-size:9pt;background:#f4f4f4;padding:0 2px}
@page{size:A4;margin:10mm}
"""


def main():
    main_rows = load("main")
    parts = [f"<html><head><meta charset='utf-8'><title>SketchRAG Report</title><style>{CSS}</style></head><body>",
             f"<h1>{C.TITLE}</h1>",
             "<div class='authors'>" + " &middot; ".join(f"{n} ({r})" for n, r in C.AUTHORS) + "</div>",
             f"<div class='inst'>{C.INSTITUTE}</div>"]
    abstract = (ROOT / "report" / "abstract.html")
    parts.append("<h2>Abstract</h2>" + (abstract.read_text(encoding="utf-8") if abstract.exists() else pending("abstract")))
    parts += ["<h2>1. Introduction</h2>", C.INTRO, "<h3>Contributions</h3>", C.NOVELTY,
              "<h2>2. Method</h2>", C.METHOD, "<h2>3. Experimental setup</h2>", C.SETUP,
              "<h2>4. Results</h2>", "<h3>4.1 Answer quality (test split, task score &times;100)</h3>",
              quality_table(main_rows) if main_rows else pending("main run"),
              "<h3>4.2 Efficiency (mean per question)</h3>", efficiency_table(main_rows) if main_rows else pending("main run"),
              img("main_pareto.png", "Answer quality versus LLM context tokens per question."),
              "<h3>4.3 Ablations of SketchRAG &middot; gate</h3>",
              ablation_table(main_rows, load("ablate")) if load("ablate") and main_rows else pending("ablation runs"),
              "<h3>4.4 Memory budget (RQ3, RQ5)</h3>",
              img("sweep_cells_sweep_max_cells.png", "Score, fallback rate and hot memory as the cell budget varies."),
              img("sweep_cms_sweep_cms_width.png", "Effect of the Count-Min width (the &epsilon; of the gate's bound)."),
              "<h3>4.5 Long-stream stress test (~11M tokens)</h3>", stress_section(),
              "<h3>4.6 EM-LLM reproduction</h3>", emllm_section()]
    discussion = ROOT / "report" / "discussion.html"
    parts.append("<h2>5. Answers to the research questions</h2>" + (discussion.read_text(encoding="utf-8") if discussion.exists() else pending("discussion")))
    parts += ["<h2>6. Limitations</h2>", C.LIMITATIONS,
              "<h2>7. Reproducibility</h2><p>Code: <code>sketchrag/</code> (unit tests: <code>pytest</code>). Every table "
              "and figure in this report is regenerated from <code>results/</code> by <code>report/build_report.py</code>; the "
              "dev-split tuning log is in <code>results/tuning_dev.md</code>.</p></body></html>"]
    out = ROOT / "report" / "SketchRAG_Report.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    pdf = out.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "--virtual-time-budget=20000", out.resolve().as_uri()], capture_output=True)
    print("wrote", out, "and", pdf if pdf.exists() else "(pdf failed)")


if __name__ == "__main__":
    main()
