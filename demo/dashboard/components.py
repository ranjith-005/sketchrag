"""Shared UI pieces: colours, cards, the document picker state and the cached memory build."""
from __future__ import annotations

import hashlib
import html
import re

import altair as alt
import streamlit as st

import demo_core as d

GREEN = "#059669"
# same palette as scripts/make_figs.py so charts read like the report; SketchRAG policies in greens
COLORS = {
    "Full context": "#6b7280", "BM25 RAG": "#d97706", "Dense RAG": "#2563eb", "SARA-lite": "#7c3aed",
    "AMBER-lite": "#db2777", "SketchRAG + gate": "#047857", "SketchRAG, always raw": "#34d399",
    "SketchRAG, digest only": "#a7f3d0", "SketchRAG LLM agent": "#65a30d", "SketchRAG gate + agent": "#0d9488",
    "EM-LLM (official code)": "#b91c1c",
}


def color_scale(names) -> alt.Scale:
    names = list(dict.fromkeys(names))
    return alt.Scale(domain=names, range=[COLORS.get(n, "#94a3b8") for n in names])


CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1350px;}
.kpi {border: 1px solid #e5e7eb; border-radius: 14px; padding: 16px 18px; background: #ffffff;
      box-shadow: 0 1px 2px rgba(0,0,0,.04); height: 100%;}
.kpi .v {font-size: 2.0rem; font-weight: 700; color: #065f46; line-height: 1.1;}
.kpi .l {font-size: .92rem; font-weight: 600; color: #111827; margin-top: 4px;}
.kpi .s {font-size: .80rem; color: #6b7280; margin-top: 4px;}
.banner {border-radius: 12px; padding: 14px 18px; font-size: 1.15rem; font-weight: 600; margin: 6px 0 10px;}
.banner.ok {background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0;}
.banner.fire {background: #fef2f2; color: #991b1b; border: 1px solid #fecaca;}
.chip {display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .78rem; font-weight: 600;}
.chip.ok {background: #d1fae5; color: #065f46;} .chip.bad {background: #fee2e2; color: #991b1b;}
.chip.mid {background: #f3f4f6; color: #374151;}
.ctx {max-height: 340px; overflow-y: auto; font-size: .80rem; white-space: pre-wrap; background: #f9fafb;
      border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px;}
.ctx mark {background: #fde68a; padding: 0 2px;}
.ctx .raw {color: #9a3412; font-weight: 600;}
.step {font-size: 1.05rem; font-weight: 700; color: #065f46; margin: 18px 0 2px;}
.muted {color: #6b7280; font-size: .88rem;}
</style>
"""


def kpi(col, value: str, label: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def banner(fired: bool, text: str) -> None:
    st.markdown(f'<div class="banner {"fire" if fired else "ok"}">{text}</div>', unsafe_allow_html=True)


def chip(text: str, kind: str = "mid") -> str:
    return f'<span class="chip {kind}">{html.escape(text)}</span>'


def step(n: int, title: str, sub: str = "") -> None:
    st.markdown(f'<div class="step">{n}. {title}</div>' + (f'<div class="muted">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def context_html(text: str, answers: list[str]) -> str:
    """Escape the context, mark raw-evidence headers and highlight the gold answer."""
    out = html.escape(text)
    out = re.sub(r"(\[raw evidence from[^\]]*\])", r'<span class="raw">\1</span>', out)
    out = re.sub(r"(\[memory cell[^\]]*\])", r'<b>\1</b>', out)
    for a in answers[:1]:
        a = a.strip().strip(".")
        if len(a) >= 2:
            ea = html.escape(a)
            idx = out.lower().find(ea.lower())
            while idx != -1:
                out = out[:idx] + "<mark>" + out[idx:idx + len(ea)] + "</mark>" + out[idx + len(ea):]
                idx = out.lower().find(ea.lower(), idx + len(ea) + 13)
    return f'<div class="ctx">{out}</div>'


# ---------------- document state ----------------
CUSTOM = "custom"


def doc_options() -> list[str]:
    return [c["id"] for c in d.SHOWCASE] + st.session_state.get("extra_docs", []) + [CUSTOM]


def doc_label(doc_id: str) -> str:
    for c in d.SHOWCASE:
        if c["id"] == doc_id:
            return c["title"]
    if doc_id == CUSTOM:
        return "Your own text"
    return f"Explorer: {doc_id[:38]}"


def showcase(doc_id: str) -> dict | None:
    return next((c for c in d.SHOWCASE if c["id"] == doc_id), None)


@st.cache_resource(show_spinner="Chunking and embedding the document...", max_entries=12)
def _prepared(doc_id: str, text: str | None, question: str | None, gold: str | None):
    ex = d.text_example(text, question, gold or "") if doc_id == CUSTOM else d.find_example(doc_id)
    chunks, embs = d.chunk_and_embed(ex)
    return ex, chunks, embs


@st.cache_resource(show_spinner="Streaming into memory...", max_entries=24)
def _memory(doc_id: str, key: str, max_cells: int, cms_width: int, _chunks, _embs):
    cfg = d.SketchConfig(max_cells=max_cells, cms_width=cms_width)
    return d.stream(_chunks, _embs, cfg, every=max(1, len(_chunks) // 80))


def current_doc():
    """(example, chunks, embeddings, memory, trace) for the sidebar's document, or None if the
    user picked 'your own text' and has not filled it in yet."""
    doc_id = st.session_state.get("doc_id", d.SHOWCASE[2]["id"])
    text = question = gold = None
    if doc_id == CUSTOM:
        with st.expander("✏️ Your own document", expanded=not st.session_state.get("custom_text")):
            st.text_area("Document text", key="custom_text", height=180,
                         placeholder="Paste an article, transcript or log. A few thousand words works best.")
            a, b = st.columns([2, 1])
            a.text_input("Question", key="custom_q")
            b.text_input("Gold answer (optional)", key="custom_gold")
        text, question, gold = (st.session_state.get(k, "") for k in ("custom_text", "custom_q", "custom_gold"))
        if not text.strip() or not question.strip():
            st.info("Paste a document and a question above to continue.")
            return None
    ex, chunks, embs = _prepared(doc_id, text, question, gold)
    key = doc_id if doc_id != CUSTOM else hashlib.sha1(text.encode()).hexdigest()
    mem, trace = _memory(doc_id, key, st.session_state.get("max_cells", 64),
                         st.session_state.get("cms_width", 512), chunks, embs)
    return ex, chunks, embs, mem, trace


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models() -> list[str]:
    return d.ollama_models()
