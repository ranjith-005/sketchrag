import time

import altair as alt
import pandas as pd
import streamlit as st

import components as ui
import demo_core as d

st.title("🌊 Stream into memory")
st.markdown('<div class="muted">The document arrives as 150-word chunks. Each chunk joins its most similar cell '
            '(cosine ≥ 0.90) or opens a new one. Once there are more cells than the budget, the two closest cells are '
            'merged (Ward linkage). Raw chunks go to cold storage; only the hot memory is searched.</div>',
            unsafe_allow_html=True)

doc = ui.current_doc()
if doc is None:
    st.stop()
ex, chunks, embs, mem, trace = doc
tr = pd.DataFrame(trace)
s = mem.stats()

st.markdown(f"**{ui.doc_label(st.session_state['doc_id'])}** · {len(ex.context.split()):,} words · "
            f"{len(chunks)} chunks · budget {mem.cfg.max_cells} cells")

ctrl = st.columns([1, 1, 4])
play = ctrl[0].button("▶ Stream it", type="primary", use_container_width=True)
speed = ctrl[1].segmented_control("Speed", ["slow", "fast"], default="fast", label_visibility="collapsed")

kp = st.columns(6)
slots = [c.empty() for c in kp]
ch1, ch2 = st.columns(2)
c_cells, c_mem = ch1.empty(), ch2.empty()


def draw(part: pd.DataFrame) -> None:
    row = part.iloc[-1]
    vals = [(f"{int(row.chunks)}", "chunks streamed", f"of {len(chunks)}"),
            (f"{int(row.cells)}", "cells", f"budget {mem.cfg.max_cells}"),
            (f"{int(row.merges)}", "merges", "Ward linkage"),
            (f"{row.hot_kb:.0f} KB", "hot memory", "searched"),
            (f"{row.cold_kb:.0f} KB", "cold storage", "raw chunks"),
            (f"{row.ms / row.chunks:.1f} ms", "ingest per chunk", "CPU, one pass")]
    for slot, (v, l, sub) in zip(slots, vals):
        slot.markdown(f'<div class="kpi"><div class="v" style="font-size:1.5rem">{v}</div><div class="l">{l}</div>'
                      f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)
    full_x = alt.Scale(domain=[0, len(chunks)])
    cells = alt.Chart(part).mark_line(color=ui.GREEN, strokeWidth=2.5).encode(
        x=alt.X("chunks:Q", scale=full_x, title="chunks streamed"),
        y=alt.Y("cells:Q", scale=alt.Scale(domain=[0, max(mem.cfg.max_cells, int(tr.cells.max())) + 2]), title="cells"))
    cap = alt.Chart(pd.DataFrame({"y": [mem.cfg.max_cells]})).mark_rule(strokeDash=[5, 4], color="#9ca3af").encode(y="y:Q")
    c_cells.altair_chart((cells + cap).properties(height=250, title="Cells (dashed line = budget)"), use_container_width=True, height=320)
    m = part.melt("chunks", ["hot_kb", "cold_kb"], var_name="store", value_name="KB")
    m["store"] = m.store.map({"hot_kb": "hot memory (searched)", "cold_kb": "cold storage (raw)"})
    memc = alt.Chart(m).mark_line(strokeWidth=2.5).encode(
        x=alt.X("chunks:Q", scale=full_x, title="chunks streamed"),
        y=alt.Y("KB:Q", scale=alt.Scale(domain=[0, float(max(tr.hot_kb.max(), tr.cold_kb.max())) * 1.05])),
        color=alt.Color("store:N", scale=alt.Scale(domain=["hot memory (searched)", "cold storage (raw)"],
                                                   range=[ui.GREEN, "#d97706"]),
                        legend=alt.Legend(orient="top", title=None)))
    c_mem.altair_chart(memc.properties(height=230, title="Memory size"), use_container_width=True, height=320)


if play:
    # each frame redraws two charts (~0.1 s), so fast mode skips trace points to finish in a few seconds
    frames, delay = (len(tr), 0.08) if speed == "slow" else (min(len(tr), 25), 0.0)
    for i in sorted({max(1, round(len(tr) * f / frames)) for f in range(1, frames + 1)}):
        draw(tr.iloc[:i])
        time.sleep(delay)
else:
    draw(tr)

st.caption("Hot memory has a fixed ceiling set by the configuration (cells × (digest + 8 exemplar embeddings + "
           "sketch)). On one short document it can be larger than the raw text; on the 8.5M-word stress stream it "
           "stays at 1-4 MB while a dense index grows to 146 MB (Results → Stress test).")

st.divider()
left, right = st.columns([3, 2])
with left:
    st.markdown("**Where each cell sits in the stream**")
    cells = pd.DataFrame([{"cell": c.id, "first": c.t_first, "last": c.t_last + 1, "chunks": c.n,
                           "radius": round(c.radius, 3)} for c in sorted(mem.cells, key=lambda c: c.t_first)])
    cells["row"] = range(len(cells))
    tl = alt.Chart(cells).mark_bar(height=max(3, min(12, 520 // max(1, len(cells))))).encode(
        x=alt.X("first:Q", title="stream position (chunk)"), x2="last:Q",
        y=alt.Y("row:O", axis=None, sort="ascending"),
        color=alt.Color("chunks:Q", scale=alt.Scale(scheme="greens"), title="chunks"),
        tooltip=["cell", "first", "last", "chunks", "radius"])
    st.altair_chart(tl.properties(height=min(520, 14 * len(cells) + 20)), use_container_width=True)
    st.caption("One bar per cell, from its first to its last chunk. Long bars are cells that merged chunks from "
               "far apart in the stream (a recurring topic).")
with right:
    st.markdown("**Stream topics** (Misra-Gries heavy hitters)")
    top = pd.DataFrame(mem.mg.top(15), columns=["term", "count"])
    st.altair_chart(alt.Chart(top).mark_bar(color=ui.GREEN).encode(
        y=alt.Y("term:N", sort="-x", title=None), x=alt.X("count:Q", title="estimated count"))
        .properties(height=min(520, 14 * len(cells) + 20)), use_container_width=True)

st.divider()
st.markdown("**🔎 Cell inspector**")
order = sorted(mem.cells, key=lambda c: -c.n)
pick = st.selectbox("Cell", [c.id for c in order],
                    format_func=lambda i: f"cell {i}  ·  {mem.cell(i).n} chunks  ·  positions "
                                          f"{mem.cell(i).t_first}-{mem.cell(i).t_last}")
c = mem.cell(pick)
a, b = st.columns([3, 2])
with a.container(border=True):
    st.markdown(f"**Digest** ({len(c.digest_text().split())} words, kept in hot memory)")
    st.write(c.digest_text())
    st.caption(f"radius {c.radius:.3f} (loose above 0.75) · {c.cms.total:,} terms counted by its Count-Min Sketch "
               f"(bound εN = {c.cms.error_bound():.1f})")
with b.container(border=True):
    ptrs = mem.pointers(pick)
    st.markdown(f"**Raw chunks in cold storage** ({len(ptrs)})")
    for cid in ptrs[:6]:
        st.markdown(f'<div class="ctx" style="max-height:120px;margin-bottom:6px"><b>position {cid}</b>\n'
                    f'{mem.cold.chunks[cid].text[:600]}</div>', unsafe_allow_html=True)
    if len(ptrs) > 6:
        st.caption(f"... and {len(ptrs) - 6} more")
