"""Long-stream stress test: all English LongBench QA/summary documents concatenated into
one ~11M-token stream in a fixed shuffled order. The probe questions come from those
documents, so each has a known position in the stream ("how long ago did this arrive?").
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .loaders import Example, load_task

STREAM_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum"]
QA_TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]


@dataclass
class StreamDoc:
    doc_key: str
    text: str
    n_words: int


def build_stream(seed: int = 0) -> tuple[list[StreamDoc], dict[str, list[Example]]]:
    docs: dict[str, StreamDoc] = {}
    questions: dict[str, list[Example]] = {}
    for task in STREAM_TASKS:
        for ex in load_task(task):
            key = hashlib.sha1(ex.context.encode("utf-8")).hexdigest()[:16]
            if key not in docs:
                docs[key] = StreamDoc(key, ex.context, len(ex.context.split()))
            if task in QA_TASKS:
                questions.setdefault(key, []).append(ex)
    order = sorted(docs)
    random.Random(seed).shuffle(order)
    return [docs[k] for k in order], questions


def probe_questions(stream: list[StreamDoc], questions: dict[str, list[Example]], n: int,
                    seed: int = 0) -> list[tuple[Example, int]]:
    """n probe questions spread evenly over stream position (by document index).
    Returns (example, index of its document in the stream)."""
    rng = random.Random(seed)
    with_q = [i for i, d in enumerate(stream) if d.doc_key in questions]
    picks = [with_q[int(j * (len(with_q) - 1) / max(1, n - 1))] for j in range(n)]
    return [(rng.choice(questions[stream[i].doc_key]), i) for i in picks]
