import altair as alt
import pandas as pd
import streamlit as st

import components as ui
import data

st.title("📊 Results")
st.markdown('<div class="muted">Every experiment, recomputed from <code>results/</code>. Main run: 12 LongBench / '
            '∞-Bench tasks, 350 test questions, qwen2.5:7b-instruct through Ollama for every system. Scores are task '
            'scores × 100, averaged over tasks.</div>', unsafe_allow_html=True)

tabs = st.tabs(["🏆 Main", "🧩 Per task", "🎯 Gate calibration", "✂️ Ablations", "📏 Memory sweeps",
                "🌊 Stress test", "🧠 vs EM-LLM"])

# ---------------- main ----------------
with tabs[0]:
    lb = data.leaderboard()
    st.markdown("**Is SketchRAG accurate, and what does it cost?** The gate is on the accuracy/cost front: the "
                "fewest LLM tokens of any system for the third-best score.")
    a, b = st.columns([5, 6])
    with a:
        st.dataframe(lb.drop(columns=["key", "LLM calls"]), hide_index=True, use_container_width=True, height=390,
                     column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=50, format="%.1f"),
                                    "context tokens": st.column_config.NumberColumn(format="%d"),
                                    "LLM s / q": st.column_config.NumberColumn(format="%.1f"),
                                    "fallback rate": st.column_config.NumberColumn("reads raw", format="%.2f")})
    with b:
        sc = alt.Chart(lb).encode(
            x=alt.X("context tokens:Q", scale=alt.Scale(type="log"), title="LLM context tokens per question (log)"),
            y=alt.Y("score:Q", scale=alt.Scale(zero=False), title="score"),
            color=alt.Color("system:N", scale=ui.color_scale(lb.system), legend=None),
            tooltip=["system", alt.Tooltip("score:Q", format=".1f"), alt.Tooltip("context tokens:Q", format=",.0f"),
                     alt.Tooltip("LLM s / q:Q", format=".1f")])
        low = alt.datum.system == "SARA-lite"  # sits at the same score as gate + agent; label it below the point
        pts = (sc.mark_circle(size=180, opacity=.9)
               + sc.mark_text(align="left", dx=10, fontSize=11).encode(text="system:N").transform_filter(~low)
               + sc.mark_text(align="left", dx=10, dy=13, fontSize=11).encode(text="system:N").transform_filter(low))
        st.altair_chart(pts.properties(height=390, title="Accuracy vs context cost (up and left is better)"),
                        use_container_width=True)

# ---------------- per task ----------------
with tabs[1]:
    st.markdown("**Where each system wins.** Exact-match tasks (PassKey, KV, PRetr) reward reading raw text; "
                "multi-hop QA rewards wide context; summaries (GovRep, QMSum) are flat for everyone.")
    pt = data.per_task()
    order = list(data.leaderboard().system)
    heat = alt.Chart(pt).encode(
        x=alt.X("task:N", sort=list(data.TASK_SHORT.values()), title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("system:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)))
    st.altair_chart((heat.mark_rect(stroke="white").encode(
        color=alt.Color("score:Q", scale=alt.Scale(scheme="greens", domain=[0, 100]), legend=None),
        tooltip=["system", "task", alt.Tooltip("score:Q", format=".1f")])
        + heat.mark_text(fontSize=11).encode(text=alt.Text("score:Q", format=".0f"),
                                             color=alt.condition("datum.score > 60", alt.value("white"), alt.value("#111"))))
        .properties(height=380), use_container_width=True)

# ---------------- calibration ----------------
with tabs[2]:
    cal = data.calibration()
    st.markdown("**Can the system tell when its summary is not enough?** The gate splits the questions in two. Where "
                "it fires, raw evidence is worth a lot; where it trusts the digest, there is almost nothing left to gain.")
    m = st.columns(4)
    ui.kpi(m[0], f"{cal['fired']}/{cal['n']}", "questions escalated", f"{cal['fired'] / cal['n']:.0%} of the test set")
    ui.kpi(m[1], f"{cal['fired_sketch']:.1f} → {cal['fired_gate']:.1f}", "score on escalated questions",
           "digest only → with raw evidence")
    ui.kpi(m[2], f"{cal['trusted_gate']:.1f} vs {cal['trusted_raw']:.1f}", "score on trusted questions",
           "gate vs always raw")
    lb = data.leaderboard().set_index("key")
    gain_raw = lb.loc["sketchrag_raw"].score - lb.loc["sketchrag_sketch"].score
    gain_gate = lb.loc["sketchrag_gate"].score - lb.loc["sketchrag_sketch"].score
    ui.kpi(m[3], f"{100 * gain_gate / gain_raw:.0f}%", "of the always-raw gain recovered",
           f"with {100 * (1 - lb.loc['sketchrag_gate']['context tokens'] / lb.loc['sketchrag_raw']['context tokens']):.0f}% fewer tokens")
    st.write("")
    a, b = st.columns(2)
    bars = pd.DataFrame([
        {"questions": "gate fired", "policy": "digest only", "score": cal["fired_sketch"]},
        {"questions": "gate fired", "policy": "SketchRAG gate", "score": cal["fired_gate"]},
        {"questions": "gate trusted", "policy": "SketchRAG gate", "score": cal["trusted_gate"]},
        {"questions": "gate trusted", "policy": "always raw", "score": cal["trusted_raw"]}])
    a.altair_chart(alt.Chart(bars).mark_bar().encode(
        x=alt.X("questions:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        xOffset=alt.XOffset("policy:N", sort=["digest only", "SketchRAG gate", "always raw"]),
        y=alt.Y("score:Q", title="score"),
        color=alt.Color("policy:N", sort=["digest only", "SketchRAG gate", "always raw"],
                        scale=alt.Scale(domain=["digest only", "SketchRAG gate", "always raw"],
                                        range=["#a7f3d0", "#047857", "#34d399"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["questions", "policy", alt.Tooltip("score:Q", format=".1f")])
        .properties(height=280, title="Score by gate decision"), use_container_width=True, height=360)
    fr = data.fire_rate_by_task()
    b.altair_chart(alt.Chart(fr).mark_bar(color="#b91c1c", opacity=.8).encode(
        y=alt.Y("task:N", sort="-x", title=None), x=alt.X("fallback:Q", title="share of questions escalated",
                                                          axis=alt.Axis(format="%")),
        tooltip=["task", alt.Tooltip("fallback:Q", format=".0%")]).properties(height=alt.Step(24), title="Gate fire rate per task"),
        use_container_width=True)
    st.caption(f"The LLM-driven agent, which decides for itself whether to fetch raw text, scores "
               f"{lb.loc['sketchrag_agent'].score:.1f}: the calibrated signal is the sketch bound, not the model's self-assessment.")

# ---------------- ablations ----------------
with tabs[3]:
    ab = data.ablations()
    st.markdown("**Which parts matter?** Each variant changes one thing in the gated system (110 questions, 11 tasks "
                "× 10). Capping the raw pointers per cell hurts most; dropping the temporal neighbour costs 1.7 points; the "
                "other changes are within about ±1 point, the size of run-to-run noise.")
    ab["kind"] = ["frozen"] + ["big drop" if v < -5 else "within noise" if abs(v) <= 1.5 else "small change"
                               for v in ab["vs frozen"][1:]]
    ch = alt.Chart(ab).encode(y=alt.Y("variant:N", sort=list(ab.variant), title=None, axis=alt.Axis(labelLimit=260)),
                              x=alt.X("score:Q", scale=alt.Scale(domain=[0, 50])))
    st.altair_chart((ch.mark_bar().encode(
        color=alt.Color("kind:N", scale=alt.Scale(domain=["frozen", "within noise", "small change", "big drop"],
                                                  range=["#047857", "#9ca3af", "#f59e0b", "#b91c1c"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["variant", alt.Tooltip("score:Q", format=".1f"), alt.Tooltip("vs frozen:Q", format="+.1f"),
                 alt.Tooltip("fallback rate:Q", format=".2f")])
        + ch.mark_text(align="left", dx=5).encode(text=alt.Text("score:Q", format=".1f")))
        .properties(height=300), use_container_width=True)
    st.dataframe(ab.drop(columns=["kind", "n"]), hide_index=True, use_container_width=True,
                 column_config={"score": st.column_config.NumberColumn(format="%.1f"),
                                "vs frozen": st.column_config.NumberColumn(format="%+.1f"),
                                "fallback rate": st.column_config.NumberColumn(format="%.2f")})

# ---------------- sweeps ----------------
with tabs[4]:
    st.markdown("**How small can the memory get?** Fewer cells mean coarser digests: digest-only recall falls, "
                "and the gate compensates by escalating more often, so the gated score holds up.")
    which = st.segmented_control("Sweep", ["Max cells", "Count-Min width"], default="Max cells")
    run, key = ("sweep_cells", "max_cells") if which != "Count-Min width" else ("sweep_cms", "cms_width")
    sw = data.sweep(run, key)
    cols = st.columns(3)
    for col, (y, title, fmt) in zip(cols, [("score", "Answer score", ".1f"), ("fallback rate", "Gate fire rate", ".2f"),
                                           ("hot KB", "Hot memory (KB)", ".0f")]):
        col.altair_chart(alt.Chart(sw).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2.5).encode(
            x=alt.X(f"{key}:Q", scale=alt.Scale(type="log", base=2), title=key.replace("_", " ")),
            y=alt.Y(f"{y}:Q", title=None, scale=alt.Scale(zero=y != "score")),
            color=alt.Color("policy:N", scale=ui.color_scale(sw.policy), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["policy", key, alt.Tooltip(f"{y}:Q", format=fmt)]).properties(height=240, title=title),
            use_container_width=True, height=330)
    st.caption("110 questions per point. Re-running the frozen configuration gives 42.4 and 43.3, so differences of "
               "about 2 points are noise. Width 2048 more than doubles hot memory without lowering the fire rate.")

# ---------------- stress ----------------
with tabs[5]:
    curves = data.stress_curves()
    probes, summ = data.stress_probes()
    st.markdown("**Does the memory stay bounded on a really long stream?** All English LongBench QA and summary documents "
                "concatenated into one 8.5M-word stream (65k chunks), with 100 probe questions spread over its length.")
    a, b = st.columns(2)
    a.altair_chart(alt.Chart(curves).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("M words:Q", title="stream length (million words)"),
        y=alt.Y("MB:Q", title="searchable memory (MB, log scale)", scale=alt.Scale(type="log")),
        color=alt.Color("series:N", scale=alt.Scale(
            domain=["Dense RAG index", "Raw text", "SketchRAG (256 cells)", "SketchRAG (64 cells)"],
            range=["#2563eb", "#d97706", "#34d399", "#047857"]), legend=alt.Legend(orient="bottom", columns=2, title=None)),
        tooltip=["series", alt.Tooltip("M words:Q", format=".1f"), alt.Tooltip("MB:Q", format=".1f")])
        .properties(height=300, title="Memory vs stream length"), use_container_width=True, height=400)
    b.altair_chart(alt.Chart(probes).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("evidence age:N", sort=["newest quarter", "2nd", "3rd", "oldest quarter"], title="where the evidence sits",
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("answer recall:Q", title="answer recall"),
        color=alt.Color("system:N", scale=ui.color_scale(probes.system), legend=alt.Legend(orient="bottom", columns=3, title=None)),
        tooltip=["system", "evidence age", alt.Tooltip("answer recall:Q", format=".2f")])
        .properties(height=300, title="Answer recall by age of the evidence (256 cells)"), use_container_width=True, height=400)
    m = st.columns(4)
    ui.kpi(m[0], f"{summ['recall_sketchrag_gate']:.2f}", "SketchRAG gate recall", "256 cells")
    ui.kpi(m[1], f"{summ['recall_dense_rag']:.2f}", "dense RAG recall", "")
    ui.kpi(m[2], f"{summ['recall_bm25_rag']:.2f}", "BM25 recall", "")
    ui.kpi(m[3], f"{summ['gate_fallback_rate']:.0%}", "gate fire rate", "every probe reads raw")
    st.warning("**Main limitation.** The memory stays bounded, but recall falls well below dense RAG. With thousands of "
               "unrelated documents sharing 64-256 cells, a cell centroid is too blunt a key, and SketchRAG acts like a "
               "two-level router over cold storage.")

# ---------------- EM-LLM ----------------
with tabs[6]:
    t, paper = data.emllm()
    st.markdown("**Against EM-LLM (official code).** Same base model (Phi-3.5-mini) and the same questions, on the six "
                "LongBench tasks EM-LLM completed on Kaggle's 2×T4 GPUs (120 questions).")
    gate_avg, em_avg = t.loc["SketchRAG + gate", "Avg"], t.loc["EM-LLM (official code)", "Avg"]
    st.markdown(f"SketchRAG + gate **{gate_avg:.1f}** vs EM-LLM **{em_avg:.1f}**: within a point, while SketchRAG makes "
                "one LLM call per question instead of streaming the whole document through the model.")
    long = t.drop(columns="Avg").reset_index(names="system").melt("system", var_name="task", value_name="score")
    a, b = st.columns([3, 2])
    a.altair_chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("task:N", title=None, sort=[c for c in data.TASK_SHORT.values() if c in t.columns],
                axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("system:N", sort=list(t.index)), y="score:Q",
        color=alt.Color("system:N", scale=ui.color_scale(t.index), sort=list(t.index),
                        legend=alt.Legend(orient="bottom", columns=3, title=None, labelLimit=200)),
        tooltip=["system", "task", alt.Tooltip("score:Q", format=".1f")]).properties(height=300), use_container_width=True, height=380)
    b.dataframe(t[["Avg"]].round(1), use_container_width=True)
    b.caption("EM-LLM ran in fp16 from its repository; our systems used Ollama's quantised Phi-3.5. Same questions, "
              f"not an identical model. Paper average on full LongBench: {paper['categories']['Avg']}.")
