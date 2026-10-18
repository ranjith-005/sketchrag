"""Agentic SketchRAG: the LLM reads the sketch and decides whether to fetch raw evidence.

mode="agent"  : the initial context is the sketch only, and the LLM alone decides on fallback (RQ4 policy "LLM-decides")
mode="hybrid" : the initial context is the gate's output (digests plus gated raw evidence),
                and the LLM may fetch more
"""
from __future__ import annotations

import time

from .embed import Embedder
from .llm import LLM, Completion, Usage
from .memory import SketchConfig
from .prompts import INSTRUCTION, MAX_TOKENS, SYSTEM, user_prompt
from .retrieve import RetrieveConfig, build_context, fetch_raw, render_cells, search
from .systems import DocIndex, SketchRAG

TOOLS = [
    {"type": "function", "function": {
        "name": "fetch_raw",
        "description": "Read the original raw text behind one memory cell from cold storage. Use it when a "
                       "cell's summary looks relevant but does not contain the exact fact you need.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_id": {"type": "integer", "description": "id shown in the [memory cell N | ...] header"},
                "focus": {"type": "string", "description": "words to look for inside the cell's raw text"},
            },
            "required": ["cell_id", "focus"],
        },
    }},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search the compressed stream memory with a new query and get back the summaries of "
                       "the best-matching memory cells. Useful for multi-hop questions.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }},
]

AGENT_NOTE = (
    "The context is a bounded compressed memory of the stream: memory cells with extractive summaries. "
    "If the summaries do not contain what you need, call fetch_raw on the most promising cell, or "
    "search_memory with a better query. When you have enough information, reply with the final answer only."
)


class SketchAgent(SketchRAG):
    agentic = True

    def __init__(self, llm: LLM, mode: str = "agent", mem_cfg: SketchConfig | None = None,
                 ret_cfg: RetrieveConfig | None = None, name: str | None = None, max_rounds: int = 3):
        super().__init__("sketch" if mode == "agent" else "gate", mem_cfg, ret_cfg, name or f"sketchrag_{mode}")
        self.llm = llm
        self.mode = mode
        self.max_rounds = max_rounds

    def run_live(self, ex, idx: DocIndex, embedder: Embedder) -> tuple[Completion, dict]:
        mem, ingest_s = self.build_memory(idx)
        ctx, info = build_context(mem, ex.question, idx.q_emb, self.ret_cfg, self.policy)
        usage = Usage()
        fetched: list[int] = list(info["fallback_cells"])
        searches = 0
        t0 = time.perf_counter()
        kind = ex.kind
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_prompt(kind, ctx, ex.question) + "\n\n" + AGENT_NOTE}]
        evidence = [ctx]
        text, reason = "", "max_rounds"
        for rnd in range(self.max_rounds + 1):
            last = rnd == self.max_rounds
            r = self.llm.chat(messages, usage, tools=None if last else TOOLS, max_tokens=MAX_TOKENS[kind])
            if not r.tool_calls or last:
                text, reason = r.text, r.done_reason
                break
            messages.append(r.message)
            for call in r.tool_calls:
                args = call["arguments"]
                try:
                    if call["name"] == "fetch_raw":
                        cid = int(args["cell_id"])
                        hits = fetch_raw(mem, cid, ex.question + " " + str(args.get("focus", "")),
                                         self.ret_cfg.raw_per_cell, idx.q_emb)
                        fetched.append(cid)
                        out = "\n\n".join(f"[raw, stream position {p}]\n{t}" for p, t in hits) or "(empty cell)"
                    elif call["name"] == "search_memory":
                        q = str(args["query"])
                        out = render_cells(search(mem, q, embedder.embed_queries([q])[0], self.ret_cfg))
                        searches += 1
                    else:
                        raise KeyError(call["name"])
                except (KeyError, ValueError, TypeError) as e:
                    out = f"error: {e!r}"
                evidence.append(out)
                messages.append({"role": "tool", "content": out, "tool_name": call["name"]})
            if rnd == self.max_rounds - 1:
                messages.append({"role": "user", "content": "Tool budget exhausted. " + INSTRUCTION[kind]})
        info.update(mem.stats())
        info.update({"ingest_s": ingest_s, "hot_bytes": mem.hot_nbytes(), "fetched_cells": fetched,
                     "llm_fetches": len(fetched) - len(info["fallback_cells"]), "searches": searches,
                     "context": "\n\n".join(evidence)})
        return Completion(text, reason, usage, time.perf_counter() - t0), info
