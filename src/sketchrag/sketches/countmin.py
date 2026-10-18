"""Count-Min Sketch (Cormode & Muthukrishnan, 2005) with optional conservative update.

Guarantee (point query, non-negative updates): with width w = ceil(e / eps) and
depth d = ceil(ln(1 / delta)),

    f(x) <= est(x) <= f(x) + eps * N    with probability >= 1 - delta,

where N is the total count inserted. SketchRAG uses this bound directly as the
"is the sketch able to resolve this term?" signal for the agentic fallback gate.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

_MASK64 = (1 << 64) - 1


def _base_hashes(key: str) -> tuple[int, int]:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    h1 = int.from_bytes(digest[:8], "little")
    h2 = int.from_bytes(digest[8:], "little") | 1  # odd => full period
    return h1, h2


class CountMinSketch:
    def __init__(self, width: int, depth: int, conservative: bool = True):
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        self.width = int(width)
        self.depth = int(depth)
        self.conservative = conservative
        self.table = np.zeros((self.depth, self.width), dtype=np.int32)
        self.total = 0
        self._rows = np.arange(self.depth)

    @classmethod
    def from_error(cls, eps: float, delta: float, conservative: bool = True) -> "CountMinSketch":
        return cls(math.ceil(math.e / eps), math.ceil(math.log(1.0 / delta)), conservative)

    @property
    def eps(self) -> float:
        return math.e / self.width

    @property
    def delta(self) -> float:
        return math.exp(-self.depth)

    def error_bound(self) -> float:
        """Additive over-estimation bound eps * N (holds w.p. >= 1 - delta)."""
        return self.eps * self.total

    def expected_noise(self) -> float:
        """Mean collision mass in one row, N / w. A tighter, non-guaranteed noise level."""
        return self.total / self.width

    def _cols(self, key: str) -> np.ndarray:
        # Kirsch-Mitzenmacher double hashing: h_i = h1 + i * h2 (mod 2^64) mod w
        h1, h2 = _base_hashes(key)
        return np.array([((h1 + i * h2) & _MASK64) % self.width for i in range(self.depth)], dtype=np.int64)

    def update(self, key: str, count: int = 1) -> None:
        if count < 0:
            raise ValueError("Count-Min Sketch supports non-negative updates only")
        cols = self._cols(key)
        if self.conservative:
            cur = self.table[self._rows, cols]
            target = cur.min() + count
            self.table[self._rows, cols] = np.maximum(cur, target)
        else:
            self.table[self._rows, cols] += count
        self.total += count

    def update_many(self, counts: dict[str, int]) -> None:
        for key, c in counts.items():
            self.update(key, c)

    def query(self, key: str) -> int:
        return int(self.table[self._rows, self._cols(key)].min())

    def merge(self, other: "CountMinSketch") -> None:
        """In-place merge. CMS is linear, so summing tables sketches the union stream.
        (After conservative updates the sum is still a valid over-estimate.)"""
        if (self.width, self.depth) != (other.width, other.depth):
            raise ValueError("cannot merge sketches of different shape")
        self.table += other.table
        self.total += other.total

    @property
    def nbytes(self) -> int:
        return int(self.table.nbytes)
