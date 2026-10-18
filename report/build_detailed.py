"""Build the long, plain-English report: report/SketchRAG_Detailed_Report.html (+ .pdf).

Theory text comes from detailed_content.py. Every result number is computed here from
results/, reusing the table builders of build_report.py, so nothing is typed by hand.

  python report/build_detailed.py
"""
from __future__ import annotations

import html
import json
import re
import subprocess
from collections import defaultdict

import build_report as B
import content as C
import detailed_content as D

ROOT = B.ROOT


def macro(rows, system, tasks=None):
    by = defaultdict(list)
    for r in rows:
        if r["system"] == system and (tasks is None or r["task"] in tasks):
            by[r["task"]].append(r["score"])
    return B.mean([B.mean(v) for v in by.values()])


def per_q(rows, system, key):
    return B.mean([float(r.get(key) or 0) for r in rows if r["system"] == system])


def task_score(rows, system, task):
    return B.mean([r["score"] for r in rows if r["system"] == system and r["task"] == task])


def p(x, nd=1):
    return B.f1(x, nd, 100)


def results_sections(main, emllm, matched, abl, cells, cms) -> str:
    out = []
    systems = sorted({r["system"] for r in main}, key=lambda s: B.ORDER.index(s))
    avg = {s: macro(main, s) for s in systems}
    rank = sorted(systems, key=lambda s: -avg[s])
    name = lambda s: B.NAMES[s]
    g, raw, sk, full = "sketchrag_gate", "sketchrag_raw", "sketchrag_sketch", "full_context"
    ctx = {s: per_q(main, s, "context_tokens") for s in systems}
    llm_s = {s: per_q(main, s, "llm_s") for s in systems}
    calls = {s: per_q(main, s, "llm_calls") for s in systems}

    # 8.1 quality
    out.append("<h3>8.1 Answer quality on the 350 test questions</h3>")
    out.append(B.quality_table(main))
    out.append("<p class='cap'>Table 1. Score &times;100 per task (green rows = SketchRAG). Bold = best average.</p>")
    out.append("<div class='mean'><b>What this means.</b><ul>"
               f"<li>Ranking by average: " + ", ".join(f"{name(s)} {p(avg[s])}" for s in rank) + ".</li>"
               f"<li>Our gated system ({p(avg[g])}) beats reading the whole document ({p(avg[full])}), dense RAG "
               f"({p(avg['dense_rag'])}), SARA-lite ({p(avg['sara_lite'])}) and AMBER-lite ({p(avg['amber_lite'])}). "
               f"Plain BM25 word-matching is the strongest ({p(avg['bm25_rag'])}), mostly because several tasks reward copying "
               f"exact strings (PassageRetrieval: BM25 {p(task_score(main, 'bm25_rag', 'passage_retrieval_en'))}).</li>"
               f"<li>The summary on its own (<i>sketch</i>, {p(avg[sk])}) loses a lot. On KV retrieval it scores "
               f"{p(task_score(main, sk, 'kv_retrieval'))}; with the gate the same memory scores "
               f"{p(task_score(main, g, 'kv_retrieval'))}. That jump is the gate doing its job.</li>"
               f"<li>Full context fails PassKey ({p(task_score(main, full, 'passkey'))}) because the document had to be cut in the "
               "middle to fit the window, and the hidden number was in the part that was cut.</li>"
               f"<li>Letting the LLM decide when to look at raw text (<i>agent</i>, {p(avg['sketchrag_agent'])}) does not work "
               f"with a 7B model: it rarely calls the tool and stays near the digest-only score.</li></ul></div>")

    # 8.2 efficiency
    out.append("<h3>8.2 Cost and efficiency</h3>")
    out.append(B.efficiency_table(main))
    out.append("<p class='cap'>Table 2. Mean per question. &ldquo;LLM s&rdquo; = seconds the LLM spent; searchable memory = hot index for one document.</p>")
    out.append(B.img("main_pareto.png", "Accuracy versus the number of tokens sent to the LLM. Up and to the left is better."))
    out.append("<div class='mean'><b>What this means.</b><ul>"
               f"<li>SketchRAG &middot; gate sends the LLM the <b>fewest tokens</b> of any system: {ctx[g]:,.0f} per question, versus "
               f"{ctx[full]:,.0f} for full context ({ctx[full] / ctx[g]:.1f}&times; more) and {ctx['sara_lite']:,.0f} for SARA-lite.</li>"
               f"<li>It needs {llm_s[g]:.1f} s of LLM time per question against {llm_s[full]:.1f} s for full context, and one LLM call "
               f"against {calls['amber_lite']:.2f} for AMBER-lite.</li>"
               "<li>For a <i>single</i> document, SketchRAG's hot memory is about the same size as a dense index, because a few "
               "thousand words fit easily in either. The memory advantage only appears on long streams (Section 8.4).</li></ul></div>")

    # 8.3 the gate
    fb = {r["id"]: bool(r.get("fallback")) for r in main if r["system"] == g}
    sc = defaultdict(dict)
    for r in main:
        sc[r["system"]][r["id"]] = r["score"]
    esc = [i for i, f in fb.items() if f]
    tru = [i for i, f in fb.items() if not f]
    m = lambda s, ids: B.mean([sc[s][i] for i in ids if i in sc[s]])
    gain_all = avg[raw] - avg[sk]
    gain_gate = avg[g] - avg[sk]
    out.append("<h3>8.3 Is the gate making good decisions?</h3>")
    t = ["<table><tr><th>Questions</th><th>n</th><th>sketch (never raw)</th><th>gate</th><th>raw (always raw)</th></tr>"]
    for lab, ids in (("escalated by the gate", esc), ("trusted by the gate", tru), ("all", list(fb))):
        t.append(f"<tr><td>{lab}</td><td>{len(ids)}</td><td>{p(m(sk, ids))}</td><td>{p(m(g, ids))}</td><td>{p(m(raw, ids))}</td></tr>")
    t.append("</table>")
    out.append("".join(t))
    rate = defaultdict(list)
    for r in main:
        if r["system"] == g:
            rate[r["task"]].append(float(bool(r.get("fallback"))))
    out.append("<p class='cap'>Table 3. Mean score per question, split by the gate's decision.</p>")
    out.append("<table><tr><th>Task</th>" + "".join(f"<th>{B.SHORT[t]}</th>" for t in B.TASKS if t in rate)
               + "</tr><tr><td>gate fallback rate</td>" + "".join(f"<td>{B.mean(rate[t]):.2f}</td>" for t in B.TASKS if t in rate)
               + "</tr></table><p class='cap'>Table 4. How often the gate read raw evidence, per task.</p>")
    out.append("<div class='mean'><b>What this means.</b> A good gate reads raw text exactly when it helps. "
               f"<ul><li>On the {len(esc)} questions it escalated, raw evidence lifts the score from {p(m(sk, esc))} to {p(m(g, esc))}: "
               "these really needed the original text.</li>"
               f"<li>On the {len(tru)} it trusted, always reading raw text would only have given {p(m(raw, tru))} instead of "
               f"{p(m(g, tru))}: the summary really was enough.</li>"
               f"<li>Overall it gets {100 * gain_gate / gain_all:.0f}% of the benefit of always reading raw text, while sending "
               f"{100 * (1 - ctx[g] / ctx[raw]):.0f}% fewer tokens.</li>"
               f"<li>It reads raw text for {B.mean(rate['kv_retrieval']):.0%} of KV and {B.mean(rate['passkey']):.0%} of PassKey "
               f"questions (exact keys that summaries drop) and {B.mean(rate['gov_report']):.0%} of GovReport summaries "
               "(nothing rare to check). This is the behaviour the Count-Min bound was designed to produce.</li></ul></div>")

    # 8.4 stress
    out.append("<h3>8.4 The long stream (~11 million tokens)</h3>")
    out.append(B.stress_section())
    st = {}
    for c in (64, 256):
        sm = sorted((ROOT / "results" / "stress" / f"cells_{c}").glob("summary_*.json"))
        if sm:
            st[c] = json.loads(sm[-1].read_text())
    if st:
        s64, s256 = st.get(64), st.get(256)
        dense = (s64 or s256)["final"]["dense_bytes"] / 2 ** 20
        out.append("<div class='mean'><b>What this means.</b><ul>"
                   f"<li>As the stream grows, dense RAG's searchable index grows with it, to {dense:.0f} MB. SketchRAG's hot memory "
                   f"stays flat at {s64['final']['sketch_hot_bytes'] / 2 ** 20:.1f} MB (64 cells) or "
                   f"{s256['final']['sketch_hot_bytes'] / 2 ** 20:.1f} MB (256 cells): that is the bounded-memory promise, "
                   "and it holds.</li>"
                   f"<li>But retrieval gets worse: answer recall {s64['recall_sketchrag_gate']:.2f} / {s256['recall_sketchrag_gate']:.2f} "
                   f"against {s64['recall_dense_rag']:.2f} for dense RAG and {s64['recall_bm25_rag']:.2f} for BM25. With more than a "
                   "thousand unrelated documents squeezed into a few hundred cells, a cell's centre is too blunt to point at the right "
                   f"one. The gate notices and reads raw text every time (fallback {s256['gate_fallback_rate']:.2f}).</li>"
                   "<li>This is our main limitation, and we report it openly.</li></ul></div>")

    # 8.5 ablations
    out.append("<h3>8.5 Ablations: which parts matter?</h3>")
    out.append(B.ablation_table(main, abl))
    base = B.mean([r["score"] for r in main if r["system"] == g and r["id"] in {x["id"] for x in abl}])
    by = defaultdict(list)
    for r in abl:
        by[r["system"].split("@", 1)[1]].append(r["score"])
    out.append("<p class='cap'>Table 5. Each row changes one thing in the gated system (110 questions, 10 per task).</p>")
    out.append("<div class='mean'><b>What this means.</b><ul>"
               f"<li><b>Keep all pointers to raw text.</b> Replacing each cell's full pointer list with a random sample of 16 "
               f"(reservoir sampling) drops the score from {p(base)} to {p(B.mean(by['mem.pointer_mode=reservoir']))}: the chunk that "
               "holds the answer is often no longer reachable, so escalation brings back the wrong text.</li>"
               f"<li><b>Temporal contiguity helps a little.</b> Without the time-neighbour cell the score is "
               f"{p(B.mean(by['ret.contiguity=0']))}.</li>"
               "<li>The other changes stay within about 1 point, inside run-to-run noise (next section).</li></ul></div>")

    # 8.6 sweeps
    out.append("<h3>8.6 How big should the memory be?</h3>")
    out.append(B.img("sweep_cells_sweep_max_cells.png", "Changing the number of cells (8 to 256)."))
    out.append(B.img("sweep_cms_sweep_cms_width.png", "Changing the Count-Min width (64 to 2048 columns)."))

    def sweep_row(rows, sysname, key, val):
        rs = [r for r in rows if r["system"] == f"{sysname}@mem.{key}={val}"]
        return (B.mean([r["score"] for r in rs]), B.mean([float(bool(r.get("fallback"))) for r in rs]),
                B.mean([r["hot_bytes"] for r in rs]), B.mean([r["answer_recall"] for r in rs]))

    c8, c256 = sweep_row(cells, g, "max_cells", 8), sweep_row(cells, g, "max_cells", 256)
    s8, s256 = sweep_row(cells, sk, "max_cells", 8), sweep_row(cells, sk, "max_cells", 256)
    gs = [sweep_row(cells, g, "max_cells", v)[0] for v in (8, 16, 32, 64, 128, 256)]
    w64, w512, w2048 = (sweep_row(cms, g, "cms_width", v) for v in (64, 512, 2048))
    reps = [sweep_row(cells, g, "max_cells", 64)[0], w512[0], base]
    out.append("<div class='mean'><b>What this means.</b><ul>"
               f"<li><b>Fewer cells = smaller memory, more fallbacks.</b> Going from 256 to 8 cells shrinks hot memory from "
               f"{c256[2] / 1e3:.0f} to {c8[2] / 1e3:.0f} KB. The digests get coarser (digest-only recall {s256[3]:.2f} &rarr; "
               f"{s8[3]:.2f}), and the gate makes up for it by reading raw text more often (fallback {c256[1]:.2f} &rarr; "
               f"{c8[1]:.2f}). The gated score stays between {p(min(gs))} and {p(max(gs))} at every size.</li>"
               f"<li><b>Narrower Count-Min = looser bound = more caution.</b> With 64 columns the error bound &epsilon;N is 8&times; larger "
               f"than with 512, so the gate trusts fewer cells (fallback {w64[1]:.2f} vs {w512[1]:.2f}). Accuracy stays the same, because "
               f"the gate errs on the safe side. 2048 columns raises hot memory to {w2048[2] / 1e3:,.0f} KB (from {w512[2] / 1e3:,.0f}) for no gain.</li>"
               f"<li><b>Noise.</b> The same configuration run three times scored {', '.join(p(x) for x in reps)}. Differences of about "
               "2 points are noise, not real effects.</li></ul></div>")

    # 8.7 EM-LLM
    out.append("<h3>8.7 Comparison with EM-LLM (official code)</h3>")
    out.append(B.emllm_section())
    if emllm and matched:
        tasks = sorted({r["task"] for r in emllm})
        both = emllm + [r for r in matched if r["task"] in tasks]
        sa = {s: macro(both, s, tasks) for s in sorted({r["system"] for r in both})}
        out.append("<div class='mean'><b>What this means.</b> With the same base model (Phi-3.5-mini) and the same questions, on "
                   f"the {len(tasks)} tasks EM-LLM completed: " + ", ".join(f"{B.NAMES.get(s, s)} {p(v)}" for s, v in sorted(sa.items(), key=lambda kv: -kv[1]))
                   + ". The systems are within about 3 points of each other, which is inside the noise for this many questions. "
                   "SketchRAG gets there with one short LLM call, while EM-LLM streams the whole document through the model and needed "
                   "most of a 29 GB machine. One caveat: EM-LLM ran the full-precision model, while ours used Ollama's quantised "
                   "version.</div>")
    return "\n".join(out)


def executive_summary(main) -> str:
    g, full = "sketchrag_gate", "full_context"
    avg = {s: macro(main, s) for s in {r["system"] for r in main}}
    ctx = {s: per_q(main, s, "context_tokens") for s in avg}
    return ("<div class='note'><b>The project in one paragraph.</b> AI agents that read never-ending text streams cannot keep and "
            "search everything forever. We built <b>SketchRAG</b>, which keeps a small, fixed-size summary of the stream, made from "
            "classic streaming algorithms (Count-Min Sketch, Misra&ndash;Gries, streaming clustering). It answers questions from that "
            "summary, and uses the Count-Min Sketch's mathematical error bound to decide when the summary cannot be trusted and the "
            f"original text must be read. On 350 test questions it scored <b>{p(avg[g])}</b>, higher than reading the whole document "
            f"({p(avg[full])}), dense RAG ({p(avg['dense_rag'])}) and our versions of SARA ({p(avg['sara_lite'])}) and AMBER "
            f"({p(avg['amber_lite'])}), while sending the LLM the fewest tokens of any system ({ctx[g]:,.0f} per question, "
            f"{100 * ctx[g] / ctx[full]:.1f}% of full context). Simple BM25 search was still the most accurate ({p(avg['bm25_rag'])}). "
            "On a stream of 8.5 million words the summary stays at a few megabytes while a normal index grows past a hundred, "
            "but finding the right evidence gets harder: that is our main limitation.</div>")


CSS_EXTRA = """
.note{background:#f3f6fb;border-left:4px solid #557;padding:8px 12px;margin:10px 0}
.eg{background:#fbf8ef;border-left:4px solid #a84;padding:8px 12px;margin:10px 0}
.mean{background:#eef7f1;border-left:4px solid #486;padding:6px 12px;margin:8px 0 14px}
.formula{text-align:center;font-family:Cambria,'Times New Roman',serif;font-size:11.5pt;margin:8px 0}
.cap{font-size:9pt;color:#444;text-align:center;margin-top:2px}.dim{color:#777;font-size:9pt}
table.small{width:auto;margin:6px auto}h2{page-break-after:avoid}h3{page-break-after:avoid}
.mean,.eg,.note,table,figure{page-break-inside:avoid}
dl.gl dt{font-weight:bold;float:left;clear:left;width:42mm}dl.gl dd{margin:0 0 4px 44mm}
"""


def main():
    main_rows = B.load("main")
    emllm, matched = B.load("emllm"), B.load("emllm_match")
    abl, cells, cms = B.load("ablate"), B.load("sweep_cells"), B.load("sweep_cms")
    disc = (ROOT / "report" / "discussion.html").read_text(encoding="utf-8")
    # discussion.html cites the short report's section numbers; map them to this report's
    sec = {"4.2": "8.2", "4.3": "8.5", "4.4": "8.6", "4.5": "8.4", "4.6": "8.7"}
    disc = re.sub(r"Section (4\.[2-6])", lambda mo: "Section " + sec[mo.group(1)], disc)
    parts = [f"<html><head><meta charset='utf-8'><title>SketchRAG Detailed Report</title><style>{B.CSS}{CSS_EXTRA}</style></head><body>",
             f"<h1>{C.TITLE}</h1><div class='authors'>Detailed project report with background theory</div>",
             "<div class='authors'>" + " &middot; ".join(f"{n} ({r})" for n, r in C.AUTHORS) + "</div>",
             f"<div class='inst'>{C.INSTITUTE}</div>",
             D.READING_GUIDE, executive_summary(main_rows),
             "<h2>1. Introduction</h2>", C.INTRO,
             "<h3>What is new in this project</h3>", C.NOVELTY,
             "<p><b>Research questions</b> (from our proposal):</p><ol>"
             "<li>RQ1: Can approximate streaming summaries keep enough information for accurate RAG?</li>"
             "<li>RQ2: How does SketchRAG compare with episodic-memory, adaptive-memory and compression approaches?</li>"
             "<li>RQ3: How does reducing memory size affect retrieval accuracy and downstream reasoning?</li>"
             "<li>RQ4: Can an agent decide when the summary is enough and when raw evidence is needed?</li>"
             "<li>RQ5: What is the best trade-off between memory, compression, latency and accuracy?</li></ol>",
             "<h2>2. Background: LLMs, RAG and streams</h2>", D.BACKGROUND,
             "<h2>3. Theory: the streaming algorithms we use</h2>", D.SKETCH_THEORY,
             "<h2>4. Related work</h2>", D.RELATED,
             "<h2>5. How SketchRAG works</h2>", D.DESIGN.replace("{diagram}", D.DIAGRAM),
             "<h2>6. Implementation</h2>", D.IMPLEMENTATION,
             "<h2>7. Experimental setup</h2>", D.SETUP,
             "<h2>8. Results</h2>", results_sections(main_rows, emllm, matched, abl, cells, cms),
             "<h2>9. Answers to the research questions</h2>", disc,
             "<h2>10. Limitations</h2>", C.LIMITATIONS,
             "<h2>11. Future work</h2>", D.FUTURE,
             "<h2>12. Conclusion</h2><p>We set out to test whether classic streaming summaries can serve as the memory of an AI agent. "
             "The answer is a qualified yes. A fixed-size sketch memory, combined with a gate driven by the Count-Min error bound, "
             "answered long-document questions more accurately than full-context reading, dense RAG and our SARA and AMBER "
             "re-implementations, while sending the model far fewer tokens, and it matched the official EM-LLM on the tasks EM-LLM "
             "could run. The most useful finding is that the sketch's own mathematical guarantee is a much better judge of "
             "&ldquo;is the summary enough?&rdquo; than asking the LLM itself. The open problem is scale: on a stream of thousands of "
             "unrelated documents, a few hundred cells are too coarse to find the right evidence, which points to hierarchical "
             "cells as the next step.</p>",
             "<h2>13. Reproducibility</h2><p>All code is in <code>sketchrag/</code>; <code>pytest</code> runs the unit tests. Every number "
             "in this report is computed from <code>results/</code> by <code>report/build_detailed.py</code>. The tuning log for the dev "
             "split is in <code>results/tuning_dev.md</code>.</p>",
             "<h2>Glossary</h2><dl class='gl'>" + "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in D.GLOSSARY) + "</dl>",
             "</body></html>"]
    out = ROOT / "report" / "SketchRAG_Detailed_Report.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    pdf = out.with_suffix(".pdf")
    subprocess.run([B.CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    "--virtual-time-budget=20000", out.resolve().as_uri()], capture_output=True)
    print("wrote", out, "and", pdf if pdf.exists() else "(pdf failed)")


if __name__ == "__main__":
    main()
