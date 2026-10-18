"""Misra-Gries frequent-items summary.

Keeps at most k counters. For every item x, f(x) - N/(k+1) <= est(x) <= f(x),
so every item with frequency > N/(k+1) is guaranteed to be tracked.
SketchRAG uses it as a global "what is this stream about" heavy-hitter table,
which helps global/summarisation questions that no single cell answers.
"""
from __future__ import annotations


class MisraGries:
    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be positive")
        self.k = int(k)
        self.counters: dict[str, int] = {}
        self.total = 0

    def update(self, key: str, count: int = 1) -> None:
        self.total += count
        if key in self.counters:
            self.counters[key] += count
            return
        if len(self.counters) < self.k:
            self.counters[key] = count
            return
        # Decrement all counters by the largest amount that fits, then retry.
        while count > 0:
            dec = min(count, min(self.counters.values()))
            for item in list(self.counters):
                self.counters[item] -= dec
                if self.counters[item] == 0:
                    del self.counters[item]
            count -= dec
            if len(self.counters) < self.k:
                if count > 0:
                    self.counters[key] = count
                return

    def query(self, key: str) -> int:
        return self.counters.get(key, 0)

    def error_bound(self) -> float:
        return self.total / (self.k + 1)

    def top(self, n: int) -> list[tuple[str, int]]:
        return sorted(self.counters.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    @property
    def nbytes(self) -> int:
        # 8-byte count + UTF-8 key bytes per counter (what a compact encoding would need)
        return sum(8 + len(k.encode("utf-8")) for k in self.counters)
