"""Turn a long document into a time-ordered chunk stream."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .text import split_sentences


@dataclass(frozen=True)
class Chunk:
    id: int  # global arrival index in the stream (also its timestamp)
    doc_id: str
    text: str


def word_cost(word: str) -> float:
    """Chunk budget used by one whitespace token. Ordinary words cost 1; very long tokens
    (UUIDs, URLs, JSON fragments) cost about one unit per 6 characters, so a chunk of
    machine-generated text holds roughly as many LLM tokens as a chunk of prose."""
    return 1.0 if len(word) <= 12 else len(word) / 6.0


def chunk_text(text: str, chunk_words: int = 150) -> list[str]:
    """Greedy sentence packing into chunks of about `chunk_words` word-units (see word_cost).
    A sentence longer than the budget is hard-split on whitespace."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0.0
    for sent in split_sentences(text):
        words = sent.split()
        costs = [word_cost(w) for w in words]
        while sum(costs) > chunk_words:  # pathological long "sentence" (e.g. JSON)
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_n = [], 0.0
            used, k = 0.0, 0
            while k < len(words) and (k == 0 or used + costs[k] <= chunk_words):
                used += costs[k]
                k += 1
            chunks.append(" ".join(words[:k]))
            words, costs = words[k:], costs[k:]
        n = sum(costs)
        if cur_n + n > chunk_words and cur:
            chunks.append(" ".join(cur))
            cur, cur_n = [], 0.0
        cur.extend(words)
        cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def stream_document(text: str, doc_id: str, chunk_words: int = 150, start_id: int = 0) -> Iterator[Chunk]:
    for i, piece in enumerate(chunk_text(text, chunk_words)):
        yield Chunk(id=start_id + i, doc_id=doc_id, text=piece)
