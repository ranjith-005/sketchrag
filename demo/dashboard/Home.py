"""SketchRAG dashboard.

  python -m streamlit run demo/dashboard/Home.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

import components as ui  # noqa: E402

st.set_page_config(page_title="SketchRAG dashboard", page_icon="🧭", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

PAGES = HERE / "views"
pages = [
    st.Page(PAGES / "overview.py", url_path="overview", title="Overview", icon="🏠", default=True),
    st.Page(PAGES / "sketch_lab.py", url_path="sketch-lab", title="Sketch Lab", icon="🧪"),
    st.Page(PAGES / "stream.py", url_path="stream", title="Stream into memory", icon="🌊"),
    st.Page(PAGES / "ask_gate.py", url_path="ask-gate", title="Ask & Gate", icon="🚦"),
    st.Page(PAGES / "results.py", url_path="results", title="Results", icon="📊"),
    st.Page(PAGES / "explorer.py", url_path="explorer", title="Question Explorer", icon="🔍"),
]
nav = st.navigation(pages)

# ---------------- sidebar shared by every page ----------------
with st.sidebar:
    st.markdown("### 🧭 SketchRAG")
    st.caption("Bounded streaming memory for LLM agents")
    st.divider()
    if "pending_doc" in st.session_state:  # set by the Explorer before switching page
        st.session_state["doc_id"] = st.session_state.pop("pending_doc")
    opts = ui.doc_options()
    if st.session_state.get("doc_id") not in opts:
        st.session_state["doc_id"] = opts[2]  # the KV example: the clearest gate demo
    st.selectbox("Document & question", opts, key="doc_id", format_func=ui.doc_label,
                 help="Used by the Stream and Ask & Gate pages. Pick a showcase case, a question from the "
                      "Explorer, or paste your own text.")
    case = ui.showcase(st.session_state["doc_id"])
    if case:
        st.caption(case["story"])
    with st.expander("⚙️ Memory settings"):
        st.select_slider("Max cells (memory budget)", [8, 16, 32, 64, 128, 256], value=64, key="max_cells")
        st.select_slider("Count-Min width", [64, 128, 512, 2048], value=512, key="cms_width")
        st.caption("Frozen configuration: 64 cells, width 512.")
    models = ui.ollama_models()
    st.divider()
    if models:
        st.markdown(f"🟢 **Ollama on**: {len(models)} model(s), live answers available")
    else:
        st.markdown("⚪ **Ollama off**: saved answers only")
    st.caption("Ranjith Kumar · Bhuvan S M · Nooh K · Hariprasanth  \nIIIT Kottayam")

nav.run()
