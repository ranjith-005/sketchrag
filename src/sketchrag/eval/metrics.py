"""Answer metrics. qa_f1 / normalize_answer follow LongBench's metrics.py (English);
ROUGE-L is a dependency-free LCS implementation of the same F-measure."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def contains(pred: str, gold: str) -> float:
    return float(normalize_answer(gold) in normalize_answer(pred))


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, gold: str) -> float:
    p, g = pred.lower().split(), gold.lower().split()
    lcs = _lcs(p, g)
    if lcs == 0:
        return 0.0
    r, pr = lcs / len(g), lcs / len(p)
    return 2 * r * pr / (r + pr)


def retrieval_score(pred: str, gold: str) -> float:
    """LongBench passage_retrieval_en: fraction of predicted paragraph numbers that are right."""
    m = re.findall(r"Paragraph (\d+)", gold)
    nums = re.findall(r"\d+", pred)
    if not m or not nums:
        return 0.0
    return sum(1 for n in nums if n == m[0]) / len(nums)


def score(kind: str, pred: str, answers: list[str]) -> dict[str, float]:
    """Primary metric per task family (max over gold answers), plus EM/F1 where meaningful."""
    if kind == "summ":
        return {"score": max(rouge_l(pred, a) for a in answers)}
    if kind == "retrieval":
        return {"score": max(retrieval_score(pred, a) for a in answers)}
    if kind == "exact":
        s = max(contains(pred, a) for a in answers)
        return {"score": s, "em": s}
    f1 = max(qa_f1(pred, a) for a in answers)
    return {"score": f1, "f1": f1, "em": max(exact_match(pred, a) for a in answers)}


def answer_in_context(context: str, answers: list[str]) -> float:
    """Retrieval proxy where no gold evidence exists: does any gold answer string occur in
    the retrieved context? (Reported as 'answer recall', not Recall@K.)"""
    ctx = normalize_answer(context)
    return float(any(normalize_answer(a) and normalize_answer(a) in ctx for a in answers))
