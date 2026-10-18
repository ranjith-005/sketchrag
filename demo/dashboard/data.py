"""Cached loaders for the dashboard. Every number shown comes from results/ or from running the
package code; nothing is typed in by hand."""
from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

import demo_core as d

ROOT = d.ROOT
RES = ROOT / "results"

TASK_SHORT = {"narrativeqa": "NrtvQA", "qasper": "Qasper", "multifieldqa_en": "MFQA", "hotpotqa": "HotpotQA",
              "2wikimqa": "2WikiMQA", "musique": "MuSiQue", "gov_report": "GovRep", "qmsum": "QMSum",
              "passage_retrieval_en": "PRetr", "passkey": "PassKey", "kv_retrieval": "KV",
              "longbook_qa_eng": "En.QA"}
TASK_ORDER = list(TASK_SHORT)


def _rows(run: str) -> list[dict]:
    out = []
    for p in sorted((RES / "runs" / run).glob("*.jsonl")):
        out += [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    return out


def macro(df: pd.DataFrame, col: str = "score") -> float:
    """Mean over tasks of the per-task mean, x100 (the report's 'Avg')."""
    return 100 * df.groupby("task")[col].mean().mean()


@st.cache_data(show_spinner=False)
def main_df() -> pd.DataFrame:
    df = pd.DataFrame(_rows("main"))
    df["fallback"] = df.get("fallback", False)
    df["fallback"] = df["fallback"].fillna(False).astype(bool)
    return df


@st.cache_data(show_spinner=False)
def leaderboard() -> pd.DataFrame:
    df = main_df()
    rows = []
    for s, g in df.groupby("system"):
        rows.append({"key": s, "system": d.SYSTEM_LABELS.get(s, s), "score": macro(g),
                     "context tokens": g.context_tokens.mean(), "LLM s / q": g.llm_s.mean(),
                     "LLM calls": g.llm_calls.mean(), "fallback rate": g.fallback.mean()})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def per_task() -> pd.DataFrame:
    df = main_df()
    t = (df.groupby(["system", "task"]).score.mean() * 100).reset_index()
    t["system"] = t.system.map(lambda s: d.SYSTEM_LABELS.get(s, s))
    t["task"] = t.task.map(TASK_SHORT)
    return t


@st.cache_data(show_spinner=False)
def fire_rate_by_task() -> pd.DataFrame:
    g = main_df().query("system == 'sketchrag_gate'")
    t = g.groupby("task").fallback.mean().reset_index()
    t["task"] = t.task.map(TASK_SHORT)
    return t


@st.cache_data(show_spinner=False)
def calibration() -> dict:
    return d.gate_calibration(d.load_run("main"))


def _variant(system: str) -> tuple[str, str]:
    base, _, setting = system.partition("@")
    return base, setting


@st.cache_data(show_spinner=False)
def ablations() -> pd.DataFrame:
    """Each ablation of sketchrag_gate against the frozen configuration on the same 110 questions
    (the frozen point is the max_cells=64 run of the cell sweep, which uses the frozen config)."""
    rows = _rows("ablate")
    frozen = [r for r in _rows("sweep_cells") if r["system"] == "sketchrag_gate@mem.max_cells=64"]
    labels = {"ret.w_lex=0.5": "CMS evidence in ranking (w_lex 0.5)", "ret.contiguity=0": "no temporal neighbour",
              "ret.cooccur=false": "no co-occurrence check", "ret.replace_digest=false": "raw added, digest kept",
              "mem.pointer_mode=reservoir": "16-pointer reservoir per cell", "ret.confidence=noise": "noise-floor confidence"}
    out = [{"variant": "frozen configuration", "setting": "", "score": macro(pd.DataFrame(frozen)),
            "fallback rate": pd.DataFrame(frozen).fallback.mean(), "n": len(frozen)}]
    df = pd.DataFrame(rows)
    for s, g in df.groupby("system"):
        setting = _variant(s)[1]
        out.append({"variant": labels.get(setting, setting), "setting": setting, "score": macro(g),
                    "fallback rate": g.fallback.fillna(False).astype(float).mean(), "n": len(g)})
    t = pd.DataFrame(out)
    t["vs frozen"] = t.score - t.score.iloc[0]
    return t


@st.cache_data(show_spinner=False)
def sweep(run: str, key: str) -> pd.DataFrame:
    df = pd.DataFrame(_rows(run))
    out = []
    for s, g in df.groupby("system"):
        base, setting = _variant(s)
        m = re.search(rf"{key}=(\d+)", setting)
        if not m:
            continue
        out.append({"policy": d.SYSTEM_LABELS.get(base, base), key: int(m.group(1)), "score": macro(g),
                    "fallback rate": g.fallback.fillna(False).astype(float).mean(),
                    "hot KB": g.hot_bytes.mean() / 1024, "answer recall": g.answer_recall.mean()})
    return pd.DataFrame(out).sort_values(key)


@st.cache_data(show_spinner=False)
def stress_curves() -> pd.DataFrame:
    rows = []
    for cells in (64, 256):
        for c in json.loads((RES / "stress" / f"cells_{cells}" / "curve_8488kw.json").read_text()):
            w = c["words"] / 1e6
            rows.append({"M words": w, "series": f"SketchRAG ({cells} cells)", "MB": c["sketch_hot_bytes"] / 2**20})
            if cells == 256:
                rows.append({"M words": w, "series": "Dense RAG index", "MB": c["dense_bytes"] / 2**20})
                rows.append({"M words": w, "series": "Raw text", "MB": c["text_bytes"] / 2**20})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def stress_probes() -> tuple[pd.DataFrame, dict]:
    path = RES / "stress" / "cells_256"
    summary = json.loads((path / "summary_8488kw.json").read_text())
    probes = [json.loads(l) for l in (path / "probes_8488kw.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    bins = [(0, 0.25, "newest quarter"), (0.25, 0.5, "2nd"), (0.5, 0.75, "3rd"), (0.75, 1.0001, "oldest quarter")]
    for lo, hi, name in bins:
        grp = [p for p in probes if lo <= p["age"] < hi]
        for s in probes[0]["recall"]:
            rows.append({"evidence age": name, "system": d.SYSTEM_LABELS.get(s, s),
                         "answer recall": sum(p["recall"][s] for p in grp) / max(1, len(grp))})
    return pd.DataFrame(rows), summary


@st.cache_data(show_spinner=False)
def emllm() -> tuple[pd.DataFrame, dict]:
    em = _rows("emllm")
    tasks = sorted({r["task"] for r in em}, key=TASK_ORDER.index)
    rows = em + [r for r in _rows("emllm_match") if r["task"] in tasks]
    df = pd.DataFrame(rows)
    t = (df.groupby(["system", "task"]).score.mean() * 100).unstack()[tasks]
    t.columns = [TASK_SHORT[c] for c in t.columns]
    t["Avg"] = t.mean(axis=1)
    t.index = [{"em_llm_phi35": "EM-LLM (official code)"}.get(i, d.SYSTEM_LABELS.get(i, i)) for i in t.index]
    paper = json.loads((RES / "emllm_raw" / "paper_phi35.json").read_text())
    return t.sort_values("Avg", ascending=False), paper


@st.cache_resource(show_spinner="Loading the 350 test questions...")
def questions() -> dict:
    out = {}
    for task in TASK_ORDER:
        for ex in d.load_task(task, 20 if task == "longbook_qa_eng" else 30, 0):
            out[ex.id] = ex
    return out


@st.cache_data(show_spinner=False)
def explorer_table() -> pd.DataFrame:
    df = main_df()
    qs = questions()
    piv = df.pivot_table(index="id", columns="system", values="score")
    gate = df.query("system == 'sketchrag_gate'").set_index("id")
    rows = []
    for qid in gate.index:
        ex = qs.get(qid)
        g = gate.loc[qid]
        reasons = g.get("gate_reasons") if isinstance(g.get("gate_reasons"), dict) else {}
        rows.append({"id": qid, "task": TASK_SHORT[g.task],
                     "question": (ex.question.strip() if ex else "")[:160] or "(summarise)",
                     "gold": (g.answers[0] if len(g.answers) else "")[:60],
                     "gate": "fired: " + ", ".join(sorted(set(reasons.values()))) if g.fallback else "trusted",
                     "digest only": piv.loc[qid, "sketchrag_sketch"], "gate score": piv.loc[qid, "sketchrag_gate"],
                     "always raw": piv.loc[qid, "sketchrag_raw"], "full context": piv.loc[qid, "full_context"],
                     "BM25": piv.loc[qid, "bm25_rag"], "dense": piv.loc[qid, "dense_rag"]})
    return pd.DataFrame(rows)
