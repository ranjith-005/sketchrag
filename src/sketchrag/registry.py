"""Build systems by name, with `section.field=value` overrides for sweeps/ablations.

Names: full_context, bm25_rag, dense_rag, sara_lite, sketchrag_{sketch,raw,gate},
sketchrag_agent (LLM decides fallback), sketchrag_hybrid (gate + LLM may fetch more),
amber_lite. Ablations are expressed as overrides, e.g. ret.w_lex=0 (-CMS),
ret.contiguity=0 (-temporal), mem.pointer_mode=reservoir.
"""
from __future__ import annotations

from dataclasses import fields

from .memory import SketchConfig
from .retrieve import RetrieveConfig
from .systems import BM25RAG, DenseRAG, FullContext, SARALite, SketchRAG


def _apply(overrides: list[str]) -> tuple[SketchConfig, RetrieveConfig, dict]:
    mem, ret, other = SketchConfig(), RetrieveConfig(), {}
    for ov in overrides:
        key, val = ov.split("=", 1)
        sec, name = key.split(".", 1)
        target = {"mem": mem, "ret": ret}.get(sec)
        if target is None:
            other[name] = val
            continue
        ftype = {f.name: f.type for f in fields(target)}[name]
        cast = {"int": int, "float": float, "str": str, "bool": lambda v: v.lower() in ("1", "true")}[ftype if isinstance(ftype, str) else ftype.__name__]
        setattr(target, name, cast(val))
    return mem, ret, other


def make_system(name: str, overrides: list[str] | None = None, llm=None):
    mem, ret, other = _apply(overrides or [])
    suffix = ("@" + ",".join(overrides)) if overrides else ""
    if name == "full_context":
        mc = other.get("max_chars")
        return FullContext(int(mc) if mc else None)
    if name == "bm25_rag":
        return BM25RAG()
    if name == "dense_rag":
        return DenseRAG()
    if name == "sara_lite":
        return SARALite()
    if name.startswith("sketchrag_") and name.split("_", 1)[1] in ("sketch", "raw", "gate"):
        return SketchRAG(name.split("_", 1)[1], mem, ret, name=name + suffix)
    if name in ("sketchrag_agent", "sketchrag_hybrid"):
        from .agent import SketchAgent
        return SketchAgent(llm, mode=name.split("_", 1)[1], mem_cfg=mem, ret_cfg=ret, name=name + suffix)
    if name == "amber_lite":
        from .baselines.amber_lite import AmberLite
        return AmberLite(llm)
    raise KeyError(name)
