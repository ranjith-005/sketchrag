import streamlit as st

import components as ui
import data

st.title("🔍 Question Explorer")
st.markdown('<div class="muted">All 350 test questions with the gate\'s decision and every system\'s score from the '
            'main run. Filter, then open any question in Ask & Gate to see why the gate decided as it did.</div>', unsafe_allow_html=True)

t = data.explorer_table()
with st.container(border=True):
    a, b, c, e = st.columns([2, 2, 2, 3])
    tasks = a.multiselect("Task", list(data.TASK_SHORT.values()), placeholder="all tasks")
    gate = b.segmented_control("Gate", ["all", "fired", "trusted"], default="all")
    story = c.selectbox("Show", ["all questions", "gate fixed the digest", "gate beat full context",
                                 "every system failed", "systems disagree"])
    search = e.text_input("Search question or answer", placeholder="e.g. passkey, Baron, 1990")

v = t.copy()
if tasks:
    v = v[v.task.isin(tasks)]
if gate == "fired":
    v = v[v.gate.str.startswith("fired")]
elif gate == "trusted":
    v = v[v.gate == "trusted"]
score_cols = ["digest only", "gate score", "always raw", "full context", "BM25", "dense"]
if story == "gate fixed the digest":
    v = v[(v["gate score"] >= 0.5) & (v["digest only"] < 0.5)]
elif story == "gate beat full context":
    v = v[(v["gate score"] >= 0.5) & (v["full context"] < 0.5)]
elif story == "every system failed":
    v = v[(v[score_cols] < 0.2).all(axis=1)]
elif story == "systems disagree":
    v = v[(v[score_cols].max(axis=1) - v[score_cols].min(axis=1)) >= 0.5]
if search:
    s = search.lower()
    v = v[v.question.str.lower().str.contains(s, regex=False) | v.gold.str.lower().str.contains(s, regex=False)]

st.caption(f"{len(v)} of {len(t)} questions")
cfg = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="%.2f", width="small") for c in score_cols}
cfg["id"] = None
st.dataframe(v, hide_index=True, use_container_width=True, height=430, column_config=cfg)

if len(v):
    with st.container(border=True):
        a, b = st.columns([5, 1])
        ids = list(v.id)
        label = dict(zip(v.id, v.task + " · " + v.question.str.slice(0, 110) + "  (" + v.gate + ")"))
        pick = a.selectbox("Open a question in Ask & Gate", ids, format_func=label.get, key="explorer_pick")
        b.write("")
        if b.button("🚦 Open", type="primary", use_container_width=True):
            extra = st.session_state.setdefault("extra_docs", [])
            if pick not in ui.doc_options():
                extra.append(pick)
            st.session_state["pending_doc"] = pick
            st.switch_page("views/ask_gate.py")
else:
    st.info("No questions match these filters.")
