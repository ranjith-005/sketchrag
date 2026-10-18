"""AMBER-inspired adaptive memory RAG (Qin et al., Findings of ACL 2025), re-implemented
on the shared LLM and dense retriever:

  AMU (agent-based memory updater) -> `notes`, rewritten after every retrieval round
  AIC (adaptive information collector) -> `sufficient` / `next_query`, which decide whether and what to retrieve next
  MCF (multi-granular content filter) -> `relevant_sentences`, a sentence-level filter of each round's chunks

AMBER's multi-agent prompts and trained filters are not reproduced; the report lists this as a deviation.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ..llm import LLM, Completion, Usage
from ..prompts import MAX_TOKENS, SYSTEM, user_prompt

ROUND_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_sentences": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "next_query": {"type": "string"},
    },
    "required": ["relevant_sentences", "notes", "sufficient", "next_query"],
}

ROUND_PROMPT = """Question: {question}

Current memory notes:
{notes}

Newly retrieved passages:
{passages}

Reply in JSON:
- relevant_sentences: sentences copied verbatim from the new passages that help answer the question (may be empty)
- notes: rewritten memory notes holding everything useful so far (under 150 words)
- sufficient: true if the notes are now enough to answer the question
- next_query: if not sufficient, the next search query that would find the missing information, else ""
"""


class AmberLite:
    name = "amber_lite"
    agentic = True

    def __init__(self, llm: LLM, top_k: int = 3, max_rounds: int = 3):
        self.llm = llm
        self.top_k = top_k
        self.max_rounds = max_rounds

    def run_live(self, ex, idx, embedder) -> tuple[Completion, dict]:
        usage = Usage()
        t0 = time.perf_counter()
        notes, kept, seen = "(empty)", [], set()
        q_emb = idx.q_emb
        rounds = 0
        for rounds in range(1, self.max_rounds + 1):
            sims = idx.embs @ q_emb
            top = [int(i) for i in np.argsort(-sims) if int(i) not in seen][: self.top_k]
            seen.update(top)
            passages = "\n\n".join(idx.chunks[i].text for i in sorted(top))
            r = self.llm.chat([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": ROUND_PROMPT.format(question=ex.question, notes=notes,
                                                                               passages=passages)}],
                              usage, fmt=ROUND_SCHEMA, max_tokens=700)
            try:
                out = json.loads(r.text)
            except json.JSONDecodeError:
                break
            notes = str(out.get("notes") or notes)
            kept.extend(str(s) for s in out.get("relevant_sentences") or [])
            nq = str(out.get("next_query") or "").strip()
            if out.get("sufficient") or not nq:
                break
            q_emb = embedder.embed_queries([nq])[0]
        ctx = f"Memory notes:\n{notes}\n\nEvidence:\n" + "\n".join(dict.fromkeys(kept))
        r = self.llm.chat([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": user_prompt(ex.kind, ctx, ex.question)}],
                          usage, max_tokens=MAX_TOKENS[ex.kind])
        text_bytes = sum(len(c.text.encode("utf-8")) for c in idx.chunks)
        return (Completion(r.text, r.done_reason, usage, time.perf_counter() - t0),
                {"rounds": rounds, "chunk_ids": sorted(seen), "context": ctx,
                 "hot_bytes": text_bytes + idx.embs.nbytes})
