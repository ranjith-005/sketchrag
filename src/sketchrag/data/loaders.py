"""Benchmark loaders: LongBench (THUDM, data.zip) and InfiniteBench (jsonl).

Every loader returns `Example`s with a fixed-seed subsample, so all systems see
identical questions.
"""
from __future__ import annotations

import json
import random
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

RAW = Path(os.environ.get("SKETCHRAG_RAW", Path(__file__).resolve().parents[3] / "data_raw"))

LONGBENCH_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "passage_retrieval_en",
]
INFBENCH_TASKS = {"passkey": "ib_passkey.jsonl", "kv_retrieval": "ib_kv_retrieval.jsonl",
                  "longbook_qa_eng": "ib_longbook_qa_eng.jsonl"}
TASK_TYPE = {  # how a task is scored / prompted
    "narrativeqa": "qa", "qasper": "qa", "multifieldqa_en": "qa", "hotpotqa": "qa", "2wikimqa": "qa",
    "musique": "qa", "gov_report": "summ", "qmsum": "summ", "passage_retrieval_en": "retrieval",
    "passkey": "exact", "kv_retrieval": "exact", "longbook_qa_eng": "qa",
}


@dataclass
class Example:
    id: str
    task: str
    context: str
    question: str
    answers: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return TASK_TYPE[self.task]


def _subsample(rows: list, n: int | None, seed: int) -> list:
    if n is None or n >= len(rows):
        return rows
    # Nested subsamples: the first n of one fixed shuffle, so a smaller run (e.g. the
    # EM-LLM reproduction) always uses a subset of the questions in a larger one.
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    idx = sorted(order[:n])
    return [rows[i] for i in idx]


def load_longbench(task: str, n: int | None = None, seed: int = 0) -> list[Example]:
    with zipfile.ZipFile(RAW / "longbench_data.zip") as z:
        lines = z.read(f"data/{task}.jsonl").decode("utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    out = []
    for r in _subsample(rows, n, seed):
        out.append(Example(id=f"{task}/{r['_id']}", task=task, context=r["context"], question=r["input"],
                           answers=list(r["answers"]), meta={"length": r.get("length"),
                                                             "all_classes": r.get("all_classes")}))
    return out


def load_infbench(task: str, n: int | None = None, seed: int = 0) -> list[Example]:
    rows = []
    with open(RAW / INFBENCH_TASKS[task], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    out = []
    for r in _subsample(rows, n, seed):
        answers = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
        answers = [str(a).strip('"') for a in answers]
        out.append(Example(id=f"{task}/{r['id']}", task=task, context=r["context"], question=r["input"].strip(),
                           answers=answers))
    return out


def _exported(task: str, n: int | None, seed: int) -> Path:
    return RAW / "examples" / f"{task}_n{n}_s{seed}.jsonl"


def export_examples(task: str, n: int | None, seed: int) -> Path:
    """Freeze a (task, n, seed) sample to a small JSONL so runs elsewhere (Kaggle) need no raw dumps."""
    path = _exported(task, n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = load_task(task, n, seed)  # load before creating the file, or load_task reads it back
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.__dict__) + "\n")
    return path


def load_task(task: str, n: int | None = None, seed: int = 0) -> list[Example]:
    frozen = _exported(task, n, seed)
    if frozen.exists():
        return [Example(**json.loads(l)) for l in open(frozen, encoding="utf-8") if l.strip()]
    if task in INFBENCH_TASKS:
        return load_infbench(task, n, seed)
    if task in LONGBENCH_TASKS:
        return load_longbench(task, n, seed)
    raise KeyError(task)
