import streamlit as st
import streamlit.components.v1 as components

import components as ui
import data

lb = data.leaderboard().set_index("key")
cal = data.calibration()
_, stress = data.stress_probes()
curves = data.stress_curves()
pt = data.per_task().pivot(index="system", columns="task", values="score")
last = curves[curves["M words"] == curves["M words"].max()].set_index("series")["MB"]

st.title("SketchRAG")
st.markdown("#### Approximate streaming summaries as retrieval memory for agentic LLMs")
st.markdown('<div class="muted">An agent reading an endless stream cannot keep and search every chunk. SketchRAG keeps '
            'a small, fixed-size summary of the stream and uses a sketch error bound to decide when that summary can be '
            'trusted and when the raw text must be read.</div>', unsafe_allow_html=True)
st.write("")

g, full = lb.loc["sketchrag_gate"], lb.loc["full_context"]
c1, c2, c3, c4 = st.columns(4)
ui.kpi(c1, f"{g.score:.1f}", "Answer score with the gate",
       f"vs full context {full.score:.1f} · dense RAG {lb.loc['dense_rag'].score:.1f} · BM25 {lb.loc['bm25_rag'].score:.1f}")
ui.kpi(c2, f"{g['context tokens']:,.0f}", "LLM tokens per question",
       f"fewest of any system · {100 * g['context tokens'] / full['context tokens']:.1f}% of full context")
ui.kpi(c3, f"{cal['fired'] / cal['n']:.0%}", "Questions where the gate reads raw",
       f"there: {cal['fired_sketch']:.1f} → {cal['fired_gate']:.1f} · elsewhere only {cal['trusted_raw'] - cal['trusted_gate']:.1f} pts left")
ui.kpi(c4, f"{last['SketchRAG (64 cells)']:.1f} MB", "Searchable memory on 8.5M words",
       f"vs {last['Dense RAG index']:.0f} MB dense index · {last['SketchRAG (256 cells)']:.1f} MB with 256 cells")
st.write("")

st.markdown("##### How it works")
components.html("""
<svg viewBox="0 0 1180 250" width="100%" style="display:block;font-family:Segoe UI,Arial,sans-serif;font-size:13px">
 <defs><marker id="a" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
   <path d="M0,0 L9,4.5 L0,9 z" fill="#6b7280"/></marker></defs>
 <g fill="#f3f4f6" stroke="#9ca3af"><rect x="10" y="85" width="120" height="60" rx="10"/></g>
 <text x="70" y="111" text-anchor="middle" font-weight="600">Stream</text>
 <text x="70" y="129" text-anchor="middle" fill="#6b7280">150-word chunks</text>
 <line x1="130" y1="115" x2="180" y2="115" stroke="#6b7280" stroke-width="2" marker-end="url(#a)"/>
 <rect x="185" y="20" width="330" height="140" rx="12" fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>
 <text x="350" y="42" text-anchor="middle" font-weight="700" fill="#065f46">HOT MEMORY · fixed size</text>
 <text x="200" y="68" fill="#111">● ≤ 64 semantic cells (Ward merge)</text>
 <text x="200" y="90" fill="#111">● Count-Min Sketch per cell</text>
 <text x="200" y="112" fill="#111">● Misra-Gries heavy hitters (topics)</text>
 <text x="200" y="134" fill="#111">● ≤ 120-word extractive digest per cell</text>
 <rect x="185" y="180" width="330" height="55" rx="12" fill="#fff7ed" stroke="#d97706" stroke-width="1.5"/>
 <text x="350" y="203" text-anchor="middle" font-weight="700" fill="#9a3412">COLD STORAGE · grows</text>
 <text x="350" y="222" text-anchor="middle" fill="#6b7280">every raw chunk, read only on fallback</text>
 <line x1="515" y1="90" x2="585" y2="90" stroke="#6b7280" stroke-width="2" marker-end="url(#a)"/>
 <rect x="590" y="60" width="135" height="60" rx="10" fill="#f3f4f6" stroke="#9ca3af"/>
 <text x="657" y="86" text-anchor="middle" font-weight="600">Question</text>
 <text x="657" y="104" text-anchor="middle" fill="#6b7280">rank cells</text>
 <line x1="725" y1="90" x2="775" y2="90" stroke="#6b7280" stroke-width="2" marker-end="url(#a)"/>
 <polygon points="845,45 915,90 845,135 775,90" fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>
 <text x="845" y="86" text-anchor="middle" font-weight="700">GATE</text>
 <text x="845" y="103" text-anchor="middle" fill="#6b7280" font-size="11">CMS vs εN</text>
 <line x1="915" y1="90" x2="1030" y2="90" stroke="#059669" stroke-width="2" marker-end="url(#a)"/>
 <text x="972" y="80" text-anchor="middle" fill="#065f46" font-size="12">trusted: digest</text>
 <path d="M845,135 L845,207 L520,207" fill="none" stroke="#b91c1c" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#a)"/>
 <text x="690" y="198" text-anchor="middle" fill="#b91c1c" font-size="12">escalate: read raw chunks</text>
 <path d="M515,222 L1090,222 L1090,125" fill="none" stroke="#b91c1c" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#a)"/>
 <rect x="1035" y="60" width="120" height="60" rx="10" fill="#1f2937"/>
 <text x="1095" y="86" text-anchor="middle" fill="#fff" font-weight="600">LLM</text>
 <text x="1095" y="104" text-anchor="middle" fill="#d1d5db">one call</text>
</svg>
""", height=255)

st.markdown("##### Walk through the demo")
cols = st.columns(5)
steps = [("views/sketch_lab.py", "🧪", "Sketch Lab", "The two sketches and their error guarantees"),
         ("views/stream.py", "🌊", "Stream", "Watch a document fill bounded memory"),
         ("views/ask_gate.py", "🚦", "Ask & Gate", "See the gate decide, then compare answers"),
         ("views/results.py", "📊", "Results", "All experiments over 350 questions"),
         ("views/explorer.py", "🔍", "Explorer", "Browse every test question")]
for i, (col, (page, icon, name, sub)) in enumerate(zip(cols, steps), 1):
    with col.container(border=True):
        st.markdown(f"**{i}. {name}**")
        st.caption(sub)
        st.page_link(page, label=f"Open {name}", icon=icon)

st.write("")
a, b = st.columns(2)
with a.container(border=True):
    st.markdown("**✅ What works**")
    st.markdown(f"- The gate beats full context, dense RAG, SARA-lite and AMBER-lite with the fewest tokens.\n"
                f"- Its signal is the Count-Min bound, not the LLM: the LLM-driven agent scores "
                f"{lb.loc['sketchrag_agent'].score:.1f}, about the same as digest only ({lb.loc['sketchrag_sketch'].score:.1f}).\n"
                f"- Exact keys: KV retrieval goes from {pt.loc['SketchRAG, digest only', 'KV']:.1f} (digest) to "
                f"{pt.loc['SketchRAG + gate', 'KV']:.1f} (gate).")
with b.container(border=True):
    st.markdown("**⚠️ Honest limits**")
    st.markdown(f"- Plain BM25 RAG is still the best system overall ({lb.loc['bm25_rag'].score:.1f}).\n"
                f"- At 8.5M words, answer recall drops to {stress['recall_sketchrag_gate']:.2f} (256 cells) against "
                f"{stress['recall_dense_rag']:.2f} for dense RAG.\n"
                f"- Multi-hop QA trails: evidence spread over many chunks does not fit in 4 cells.")
st.caption("Benchmark: 12 LongBench / ∞-Bench tasks, 350 test questions, qwen2.5:7b-instruct for every system.")
