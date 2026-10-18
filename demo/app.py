"""SketchRAG interactive demo.

  python -m streamlit run demo/app.py

Pick a test question (or paste your own text), stream the document into bounded memory,
and watch the error-bound gate decide which cells to read raw. Saved answers are the
qwen2.5:7b-instruct answers from the main run; "Ask live" uses the local Ollama server.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_core as d  # noqa: E402

st.set_page_config(page_title="SketchRAG demo", layout="wide")


@st.cache_resource(show_spinner=False)
def prepared(ex_id: str):
    ex = d.find_example(ex_id)
    chunks, embs = d.chunk_and_embed(ex)
    return ex, chunks, embs


@st.cache_data(show_spinner=False)
def run_summary():
    run = d.load_run("main")
    rows = []
    for name, recs in run.items():
        df = pd.DataFrame(recs)
        rows.append({"system": d.SYSTEM_LABELS.get(name, name),
                     "avg score": round(df.groupby("task")["score"].mean().mean() * 100, 1),
                     "LLM context tokens": round(df.context_tokens.mean()),
                     "LLM s / question": round(df.llm_s.mean(), 1)})
    table = pd.DataFrame(rows).sort_values("avg score", ascending=False).set_index("system")
    return table, d.gate_calibration(run)


# ---------------- sidebar ----------------
st.sidebar.title("SketchRAG")
st.sidebar.caption("Approximate streaming summaries as retrieval memory for LLM agents")
source = st.sidebar.radio("Document", ["Showcase question", "Paste your own text"])
if source == "Showcase question":
    titles = [c["title"] for c in d.SHOWCASE]
    pick = st.sidebar.selectbox("Question", titles)
    case = d.SHOWCASE[titles.index(pick)]
    ex, chunks, embs = prepared(case["id"])
    question = ex.question
else:
    case = None
    text = st.sidebar.text_area("Document text", height=200,
                                placeholder="Paste a long article, transcript or log (a few thousand words works best)")
    question = st.sidebar.text_input("Question")
    gold = st.sidebar.text_input("Gold answer (optional)")
    if not text.strip() or not question.strip():
        st.info("Paste a document and a question in the sidebar.")
        st.stop()
    ex = d.text_example(text, question, gold)
    with st.spinner("Embedding chunks with bge-small (CPU)..."):
        chunks, embs = d.chunk_and_embed(ex)

st.sidebar.divider()
max_cells = st.sidebar.select_slider("Memory budget: max cells", [8, 16, 32, 64, 128, 256], value=64)
cms_width = st.sidebar.select_slider("Count-Min width", [64, 128, 512, 2048], value=512)
st.sidebar.caption("The frozen configuration is 64 cells, width 512.")

mem_cfg = d.SketchConfig(max_cells=max_cells, cms_width=cms_width)

# ---------------- header ----------------
st.title("SketchRAG: when can an agent trust its summary?")
if case:
    st.markdown(f"**{case['title']}**. {case['story']}")
c1, c2, c3 = st.columns(3)
c1.metric("Document", f"{len(ex.context.split()):,} words")
c2.metric("Chunks (150 words)", len(chunks))
c3.metric("Task", ex.task)
st.markdown(f"**Question:** {question.strip()[:400] or '(summarise the report)'}")
if ex.answers:
    st.markdown(f"**Gold answer:** `{ex.answers[0][:200]}`")

tab_stream, tab_gate, tab_answer, tab_results = st.tabs(
    ["1. Stream into memory", "2. The gate", "3. Answers", "4. Results (350 questions)"])

# ---------------- 1. stream ----------------
with tab_stream:
    st.markdown("Chunks arrive one at a time. Each joins its nearest cell or opens a new one; above the budget, "
                "the closest pair of cells is merged. Raw chunks go to cold storage.")
    animate = st.button("Stream it (animated)")
    every = max(1, len(chunks) // 60)
    mem, trace = d.stream(chunks, embs, mem_cfg, every=every)
    tr = pd.DataFrame(trace).set_index("chunks")
    if animate:
        m1, m2, m3, m4 = st.columns(4)
        box1, box2 = st.columns(2)
        ph = [m1.empty(), m2.empty(), m3.empty(), m4.empty()]
        chart1, chart2 = box1.empty(), box2.empty()
        for i in range(1, len(tr) + 1):
            part = tr.iloc[:i]
            row = part.iloc[-1]
            ph[0].metric("Chunks streamed", int(part.index[-1]))
            ph[1].metric("Cells", int(row.cells))
            ph[2].metric("Hot memory", f"{row.hot_kb:.0f} KB")
            ph[3].metric("Cold storage", f"{row.cold_kb:.0f} KB")
            chart1.line_chart(part[["cells"]], height=260)
            chart2.line_chart(part[["hot_kb", "cold_kb"]], height=260)
            time.sleep(0.05)
    else:
        s = mem.stats()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Chunks streamed", s["n_chunks"])
        m2.metric("Cells", s["n_cells"], help=f"{s['n_merges']} merges")
        m3.metric("Hot memory", f"{s['hot_bytes'] / 1024:.0f} KB")
        m4.metric("Cold storage", f"{s['cold_bytes'] / 1024:.0f} KB")
        b1, b2 = st.columns(2)
        b1.line_chart(tr[["cells"]], height=260)
        b2.line_chart(tr[["hot_kb", "cold_kb"]], height=260)
    st.caption(f"Ingest: {trace[-1]['ms'] / len(chunks):.1f} ms per chunk. Hot memory has a fixed ceiling set by the "
               "config (cells x digest + exemplars + sketches), so on one short document it can exceed the raw text; "
               "on the 8.5M-word stress stream it stays at 1-4 MB against a "
               "146 MB dense index.")
    st.markdown("**Stream topics** (Misra-Gries heavy hitters): " + ", ".join(t for t, _ in mem.mg.top(15)))
    st.markdown("**Cells** (in stream order)")
    st.dataframe(pd.DataFrame(d.cells_table(mem)), hide_index=True, height=300)

# ---------------- 2. gate ----------------
q_emb = d.embed_question(question)
g = d.explain_gate(mem, question, q_emb)
ctx = d.contexts(mem, question, q_emb)

with tab_gate:
    dec = g["decision"]
    if dec.fallback:
        st.error(f"Gate fired: {len(dec.fallback_cells)} cell(s) read raw from cold storage")
    else:
        st.success("Gate closed: every retrieved digest is trusted, nothing read raw")
    for cid, why in dec.reasons.items():
        st.markdown(f"- cell **{cid}**: `{why}`, {d.REASON_TEXT[why]}")
    st.markdown("**Retrieved cells** (up to 4 by similarity, plus 1 temporal neighbour)")
    st.dataframe(pd.DataFrame(g["hit_rows"]), hide_index=True)
    st.markdown("**Rare query terms** (idf >= 2.5): " + (", ".join(f"`{t}`" for t in g["rare"]) or
                                                        "none, so the gate has nothing to check"))
    if g["term_rows"]:
        t = pd.DataFrame(g["term_rows"])
        st.markdown("Count-Min estimate of each rare term in each retrieved cell, compared with the cell's bounds:")
        st.dataframe(t.pivot_table(index="term", columns="cell", values="CMS estimate", sort=False))
        with st.expander("Full per-term checks"):
            st.dataframe(t, hide_index=True)
    st.markdown("**Context sent to the LLM under each policy**")
    cols = st.columns(3)
    for col, (policy, label) in zip(cols, [("sketch", "Digest only"), ("gate", "Gate (SketchRAG)"),
                                           ("raw", "Always raw")]):
        hit = d.answer_in_context(ctx[policy]["context"], ex.answers) if ex.answers else None
        col.metric(label, f"{ctx[policy]['words']} words",
                   None if hit is None else ("gold answer present" if hit else "gold answer missing"),
                   delta_color="normal" if hit else "inverse")
        with col.expander("Show context"):
            st.text(ctx[policy]["context"][:6000])

# ---------------- 3. answers ----------------
with tab_answer:
    if case:
        saved = d.saved_answers(ex.id)
        order = ["sketchrag_sketch", "sketchrag_gate", "sketchrag_raw", "full_context", "bm25_rag", "dense_rag",
                 "sara_lite", "amber_lite", "sketchrag_agent"]
        rows = [{"system": d.SYSTEM_LABELS[s], "answer": saved[s]["pred"][:120], "score": round(saved[s]["score"], 2),
                 "LLM context tokens": saved[s]["context_tokens"], "LLM seconds": round(saved[s]["llm_s"], 1)}
                for s in order if s in saved]
        st.markdown(f"**Saved answers from the main run** ({d.SAVED_MODEL} on a Kaggle GPU, same prompt for every system)")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown("**Ask live** with the local Ollama server (CPU, about 20 s to a minute)")
    models = d.ollama_models()
    if not models:
        st.warning("No Ollama server found at OLLAMA_HOST / localhost:11434.")
    else:
        a, b = st.columns(2)
        model = a.selectbox("Model", models, index=models.index("llama3.2:3b") if "llama3.2:3b" in models else 0)
        policy = b.selectbox("Context", ["gate", "sketch", "raw"])
        if st.button("Ask"):
            with st.spinner(f"{model} is answering..."):
                res = d.live_answer(ex, ctx[policy]["context"], model)
            st.markdown(f"**Answer:** {res['pred']}")
            st.caption(f"{res['context_tokens']} context tokens, {res['seconds']} s"
                       + (f", score {res['score']:.2f}" if "score" in res else ""))

# ---------------- 4. results ----------------
with tab_results:
    table, cal = run_summary()
    st.markdown("12 LongBench / ∞-Bench tasks, 350 test questions, qwen2.5:7b-instruct for every system.")
    st.dataframe(table)
    a, b, c = st.columns(3)
    a.metric("Gate fires on", f"{cal['fired'] / cal['n']:.0%} of questions")
    b.metric("Score when it fires", f"{cal['fired_gate']:.1f}", f"{cal['fired_gate'] - cal['fired_sketch']:+.1f} vs digest only")
    c.metric("Left on the table when it trusts", f"{cal['trusted_raw'] - cal['trusted_gate']:.1f} points",
             help="always-raw minus gate, on the questions the gate trusted")
    figs = d.ROOT / "report" / "figures"
    f1, f2 = st.columns(2)
    f1.image(str(figs / "main_pareto.png"), caption="Accuracy vs LLM context tokens")
    f2.image(str(figs / "stress_memory.png"), caption="Memory on the 8.5M-word stream")
    st.caption("BM25 RAG remains the strongest system (46.0). At stream scale, recall drops to 0.29 against 0.44 for dense RAG.")
