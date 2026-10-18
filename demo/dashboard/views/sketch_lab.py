import random
from collections import Counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import components as ui
from sketchrag.sketches.countmin import CountMinSketch
from sketchrag.sketches.misra_gries import MisraGries
from sketchrag.sketches.reservoir import Reservoir

st.title("🧪 Sketch Lab")
st.markdown('<div class="muted">The three streaming sketches SketchRAG is built on, written by hand in '
            '<code>src/sketchrag/sketches/</code>. Change the stream and the sketch sizes and check that the '
            'guarantees hold.</div>', unsafe_allow_html=True)

with st.container(border=True):
    a, b, c, e, f = st.columns(5)
    skew = a.slider("Zipf skew", 1.1, 2.0, 1.3, 0.1, help="Higher = a few terms dominate the stream (like real text).")
    length = b.select_slider("Stream length", [20_000, 50_000, 100_000, 200_000], value=100_000)
    width = c.select_slider("CMS width w", [64, 128, 256, 512, 1024, 2048], value=512)
    depth = e.select_slider("CMS depth d", [1, 2, 3, 4, 5], value=3)
    k = f.select_slider("Misra-Gries counters k", [16, 32, 64, 128, 256], value=64)


@st.cache_data(show_spinner="Running the sketches over the stream...")
def run(skew, length, width, depth, k):
    rng = np.random.default_rng(0)
    toks = [f"t{x}" for x in rng.zipf(skew, length) if x < 100_000]
    cms, mg = CountMinSketch(width, depth), MisraGries(k)
    for t in toks:
        cms.update(t)
        mg.update(t)
    true = Counter(toks)
    over = pd.DataFrame({"term": list(true), "true": list(true.values())})
    over["estimate"] = [cms.query(t) for t in over.term]
    over["over-count"] = over.estimate - over.true
    top_true = true.most_common(15)
    mg_est = dict(mg.top(len(true)))
    exact_kb = sum(len(t) + 8 for t in true) / 1024
    return {"N": len(toks), "distinct": len(true), "over": over, "eps": cms.eps, "delta": cms.delta,
            "bound": cms.error_bound(), "cms_kb": cms.nbytes / 1024, "mg_kb": mg.nbytes / 1024, "exact_kb": exact_kb,
            "top": [{"term": t, "true": c, "Misra-Gries": mg_est.get(t, 0)} for t, c in top_true],
            "mg_top": [t for t, _ in mg.top(10)], "true_top": [t for t, _ in top_true[:10]],
            "mg_threshold": len(toks) / (k + 1), "mg_bound": mg.error_bound(), "true_counts": true, "mg_all": mg_est,
            "cms": cms}


@st.cache_data(show_spinner=False)
def reservoir_hits(cap, trials, stream_len):
    hits = np.zeros(stream_len)
    rng = random.Random(0)
    for _ in range(trials):
        res = Reservoir(cap, rng)
        for i in range(stream_len):
            res.add(i)
        for i in res.items:
            hits[i] += 1
    return hits


r = run(skew, length, width, depth, k)
over = r["over"]
frac_over = float((over["over-count"] > r["bound"]).mean())
st.write("")
m = st.columns(4)
ui.kpi(m[0], f"{r['N']:,}", "tokens in the stream", f"{r['distinct']:,} distinct terms")
ui.kpi(m[1], f"{r['cms_kb']:.0f} KB", "Count-Min memory", f"exact counting table ≈ {r['exact_kb']:.0f} KB")
ui.kpi(m[2], "✔" if (over["over-count"] >= 0).all() else "✘", "never under-counts",
       "Count-Min only over-estimates")
ui.kpi(m[3], f"{frac_over:.2%}", "terms over the εN bound", f"guarantee: at most δ = {r['delta']:.1%}")
st.write("")

left, right = st.columns(2)
with left.container(border=True):
    st.markdown(f"**Count-Min Sketch**: over-count per term, with the bound εN = {r['bound']:.0f}")
    hist = alt.Chart(over).mark_bar(color=ui.GREEN).encode(
        x=alt.X("over-count:Q", bin=alt.Bin(maxbins=40), title="estimate − true count"),
        y=alt.Y("count():Q", title="terms", scale=alt.Scale(type="symlog")))
    rule = alt.Chart(pd.DataFrame({"x": [r["bound"]]})).mark_rule(color="#b91c1c", strokeDash=[6, 4], size=2).encode(x="x:Q")
    st.altair_chart((hist + rule).properties(height=300), use_container_width=True)
    st.caption(f"ε = e / w = {r['eps']:.4f}, δ = e^(−d) = {r['delta']:.3f}. Red line: the bound. Wider sketches "
               "lower the bound; deeper sketches make exceeding it rarer.")

with right.container(border=True):
    st.markdown(f"**Misra-Gries**: top-15 terms, true count vs the {k}-counter estimate")
    top = pd.DataFrame(r["top"]).melt("term", var_name="count of", value_name="count")
    chart = alt.Chart(top).mark_bar().encode(
        y=alt.Y("term:N", sort=[t["term"] for t in r["top"]], title=None),
        x=alt.X("count:Q", title="count"), yOffset="count of:N",
        color=alt.Color("count of:N", scale=alt.Scale(domain=["true", "Misra-Gries"], range=["#9ca3af", ui.GREEN]),
                        legend=alt.Legend(orient="bottom", title=None))).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
    overlap = len(set(r["mg_top"]) & set(r["true_top"]))
    st.caption(f"Top-10 overlap {overlap}/10. Every term more frequent than N/(k+1) = {r['mg_threshold']:.0f} is "
               f"guaranteed to be kept; estimates are at most {r['mg_bound']:.0f} too low.")

with st.container(border=True):
    st.markdown("**Look up a term**")
    a, b = st.columns([1, 3])
    term = a.text_input("Term", "t7", help="Terms are named t1, t2, ... by frequency rank.")
    true_c = r["true_counts"].get(term, 0)
    with b:
        x = st.columns(3)
        x[0].metric("True count", f"{true_c:,}")
        x[1].metric("Count-Min estimate", f"{r['cms'].query(term):,}", f"+{r['cms'].query(term) - true_c:,}",
                    delta_color="off")
        x[2].metric("Misra-Gries estimate", f"{r['mg_all'].get(term, 0):,}",
                    f"{r['mg_all'].get(term, 0) - true_c:,}", delta_color="off")

with st.container(border=True):
    st.markdown("**Reservoir sampling**: a uniform sample of fixed size from a stream of unknown length")
    a, b = st.columns([1, 3])
    cap = a.select_slider("Reservoir size", [8, 16, 32, 64], value=16)
    trials, stream_len = 2000, 200
    hits = reservoir_hits(cap, trials, stream_len)
    df = pd.DataFrame({"stream position": np.arange(stream_len), "kept (%)": 100 * hits / trials})
    expected = 100 * cap / stream_len
    line = alt.Chart(df).mark_bar(color=ui.GREEN, opacity=.8).encode(x="stream position:Q", y="kept (%):Q")
    rule = alt.Chart(pd.DataFrame({"y": [expected]})).mark_rule(color="#b91c1c", strokeDash=[6, 4]).encode(y="y:Q")
    b.altair_chart(line + rule, use_container_width=True)
    a.caption(f"{trials:,} runs over a {stream_len}-item stream. Every position is kept about "
              f"{expected:.0f}% of the time (red line), however early or late it arrived. SketchRAG's "
              "`pointer_mode=reservoir` ablation uses this to cap raw pointers per cell.")

with st.expander("Where these are used in SketchRAG"):
    st.markdown("""
- **Count-Min Sketch, one per cell** (width 512, depth 3) counts the content terms of the cell's chunks. The gate asks
  it whether a rare query term is in the cell, and whether that count clears the guaranteed εN bound.
- **Count-Min Sketch, global** (width 8,192) counts document frequencies, giving a streaming IDF that decides which
  query terms are rare.
- **Misra-Gries** (k = 256) keeps the stream's heavy hitters, shown to the LLM as the stream's topics.
- **Reservoir** is only used in the ablation, where it caps each cell's raw pointers at 16. That drops the score from
  42.4 to 28.9, so full pointer lists in cold storage matter.
""")
