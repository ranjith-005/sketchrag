import html

import altair as alt
import pandas as pd
import streamlit as st

import components as ui
import demo_core as d

st.title("🚦 Ask & Gate")
st.markdown('<div class="muted">The core of SketchRAG: find the relevant cells, check each one\'s Count-Min Sketch '
            'against its error bound, read raw evidence only where the digest cannot be trusted, then answer.</div>',
            unsafe_allow_html=True)

doc = ui.current_doc()
if doc is None:
    st.stop()
ex, chunks, embs, mem, trace = doc
case = ui.showcase(st.session_state["doc_id"])

with st.container(border=True):
    q_default = ex.question.strip()
    question = st.text_area("Question", q_default, height=68 if len(q_default) < 160 else 110,
                            key=f"q_{st.session_state['doc_id']}",
                            help="Edit it to try your own question about this document.")
    gold = ex.answers[0] if ex.answers else ""
    edited = question.strip() != q_default
    info = [f"**Document:** {len(ex.context.split()):,} words · {len(chunks)} chunks · {len(mem.cells)} cells"]
    if gold and not edited:
        info.append(f"**Gold answer:** `{gold[:120]}`")
    if case and not edited:
        info.append(f"*{case['story']}*")
    st.markdown("  \n".join(info))

q_emb = d.embed_question(question)
g = d.explain_gate(mem, question, q_emb)
ctx = d.contexts(mem, question, q_emb)
answers = ex.answers if not edited else []
dec = g["decision"]

# ---------------- 1. retrieve ----------------
ui.step(1, "Retrieve cells", "Cells ranked by similarity to the question (centroid or best exemplar), plus one "
        "temporal neighbour of the best cell.")
hits = pd.DataFrame(g["hit_rows"])
hits["verdict"] = hits.gate.map(lambda v: "✅ trusted" if v == "trusted" else f"🔴 {v}")
st.dataframe(hits[["rank", "cell", "positions", "chunks", "similarity", "radius", "εN bound", "verdict"]],
             hide_index=True, use_container_width=True,
             column_config={"similarity": st.column_config.ProgressColumn("similarity", min_value=0, max_value=1,
                                                                          format="%.3f"),
                            "εN bound": st.column_config.NumberColumn("εN bound", help="guaranteed CMS over-count")})

# ---------------- 2. gate ----------------
ui.step(2, "Gate check", "For every rare query term (idf ≥ 2.5): is it in the cell according to the sketch, and did "
        "the digest keep it?")
if dec.fallback:
    ui.banner(True, f"🔴 Gate fired: {len(dec.fallback_cells)} of {len(hits)} cells will be read raw from cold storage")
else:
    ui.banner(False, "🟢 Gate closed: every retrieved digest is trusted, nothing is read raw")
for cid, why in dec.reasons.items():
    st.markdown(f"- **cell {cid}** · `{why}`: {d.REASON_TEXT[why]}")
if not g["rare"]:
    st.caption("The question has no rare terms (typical for summary requests), so there is nothing the digests could "
               "have dropped.")
else:
    t = pd.DataFrame(g["term_rows"])
    t["status"] = [("in digest" if r["in digest"] else
                    "in cell, dropped from digest" if r["above noise"] else "not detected")
                   for _, r in t.iterrows()]
    t["cell"] = "cell " + t.cell.astype(str)
    t["label"] = t["CMS estimate"].astype(str) + t["above εN"].map({True: " ★", False: ""})
    base = alt.Chart(t).encode(x=alt.X("cell:N", sort=list(dict.fromkeys(t.cell)), title=None,
                                       axis=alt.Axis(orient="top", labelAngle=0)),
                               y=alt.Y("term:N", sort=list(dict.fromkeys(t.term)), title=None))
    heat = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color("status:N", scale=alt.Scale(domain=["in digest", "in cell, dropped from digest", "not detected"],
                                                    range=["#a7f3d0", "#fca5a5", "#f3f4f6"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["term", "cell", "idf", "CMS estimate", "above noise", "above εN", "in digest"])
    text = base.mark_text(fontSize=12).encode(text="label:N")
    st.altair_chart((heat + text).properties(height=alt.Step(28)), use_container_width=True)
    st.caption("Numbers are Count-Min estimates; ★ means the count clears the guaranteed εN bound, so the sketch can "
               "vouch for that term in that cell. Red cells are what the digest lost.")

# ---------------- 3. context ----------------
ui.step(3, "Build the LLM context", "Escalated cells have their digest replaced by their 2 most relevant raw chunks.")
cols = st.columns(3)
for col, (policy, label, sub) in zip(cols, [("sketch", "Digest only", "never reads raw"),
                                            ("gate", "SketchRAG gate", "raw only where the gate fires"),
                                            ("raw", "Always raw", "raw for every retrieved cell")]):
    c = ctx[policy]
    with col.container(border=True):
        hit = d.answer_in_context(c["context"], answers) if answers else None
        badge = "" if hit is None else (ui.chip("gold answer present", "ok") if hit else ui.chip("gold answer missing", "bad"))
        st.markdown(f"**{label}** · {c['words']} words  \n{badge}", unsafe_allow_html=True)
        st.caption(sub + (f" · raw chunks {c['info']['raw_chunk_ids']}" if c["info"]["raw_chunk_ids"] else ""))
        st.markdown(ui.context_html(c["context"], answers), unsafe_allow_html=True)

# ---------------- 4. answer ----------------
ui.step(4, "Answer")
saved = d.saved_answers(ex.id) if not edited and ex.task != "custom" else {}
if saved:
    order = ["sketchrag_sketch", "sketchrag_gate", "sketchrag_raw", "full_context", "bm25_rag", "dense_rag",
             "sara_lite", "amber_lite", "sketchrag_agent", "sketchrag_hybrid"]
    rows = [{"system": d.SYSTEM_LABELS[s], "answer": saved[s]["pred"][:140],
             "result": "✅" if saved[s]["score"] >= 0.5 else ("🟡" if saved[s]["score"] > 0 else "❌"),
             "score": saved[s]["score"], "LLM tokens": saved[s]["context_tokens"],
             "LLM seconds": round(saved[s]["llm_s"], 1)} for s in order if s in saved]
    st.markdown(f"Saved answers from the main run: **{d.SAVED_MODEL}** on a Kaggle GPU, same prompt for every system.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1, format="%.2f")})
elif edited:
    st.caption("You edited the question, so there is no saved answer. Ask the local model below.")

models = ui.ollama_models()
with st.container(border=True):
    st.markdown("**Ask live** with the local Ollama server (CPU: about 20 s to a minute)")
    if not models:
        st.caption("⚪ No Ollama server at localhost:11434. Start it with `ollama serve` to enable live answers.")
    else:
        a, b, c = st.columns([2, 2, 1])
        model = a.selectbox("Model", models, index=models.index("llama3.2:3b") if "llama3.2:3b" in models else 0)
        policy = b.selectbox("Context", ["gate", "sketch", "raw"],
                             format_func={"gate": "SketchRAG gate", "sketch": "Digest only", "raw": "Always raw"}.get)
        c.write("")
        if c.button("Ask", type="primary", use_container_width=True):
            ask_ex = ex if not edited else d.text_example(ex.context, question)
            with st.spinner(f"{model} is reading {ctx[policy]['words']} words..."):
                res = d.live_answer(ask_ex, ctx[policy]["context"], model)
            st.session_state["live"] = (st.session_state["doc_id"], question, model, policy, res)
        live = st.session_state.get("live")
        if live and live[0] == st.session_state["doc_id"] and live[1] == question:
            _, _, m, p, res = live
            ok = res.get("score")
            st.markdown(f"**{m}** with *{p}* context: `{html.escape(res['pred'][:300])}`")
            st.caption(f"{res['context_tokens']} context tokens · {res['seconds']} s"
                       + (f" · score {ok:.2f}" if ok is not None else ""))
