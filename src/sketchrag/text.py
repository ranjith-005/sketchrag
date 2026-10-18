"""Shared text utilities: sentence splitting, term extraction, token estimates."""
from __future__ import annotations

import re

_SENT_RE = re.compile(r"(?<=[.!?。])\s+|\n{1,}")
_TERM_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being below
    between both but by can did do does doing down during each few for from further had has have having he
    her here hers herself him himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through to too under until
    up very was we were what when where which while who whom why will with would you your yours yourself
    yourselves also may might must shall could one two said says say like get got us let""".split()
)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s and s.strip()]


def terms(text: str) -> list[str]:
    """Lower-cased content terms. Numbers and hex fragments are kept: those are exactly
    the exact-match keys (passkeys, ids, years) that dense centroids blur."""
    out = []
    for tok in _TERM_RE.findall(text):
        t = tok.lower()
        if t in STOPWORDS:
            continue
        if len(t) < 3 and not t.isdigit():
            continue
        out.append(t)
    return out


def n_words(text: str) -> int:
    return len(text.split())


def est_tokens(text: str) -> int:
    """Rough LLM-token estimate (~1.33 tokens per English word). Only used for
    stored-text accounting; LLM context tokens are taken from API usage."""
    return int(round(n_words(text) * 1.33))
