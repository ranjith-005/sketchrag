"""Reservoir sampling (Vitter's Algorithm R) with a weighted merge.

A Reservoir of capacity R over a stream of n items holds a uniform random
sample of min(n, R) items. Merging two reservoirs that summarise disjoint
streams of sizes n1 and n2 draws each slot from side 1 with probability
proportional to its remaining stream mass, which keeps the union sample uniform.
"""
from __future__ import annotations

import random
from typing import Generic, TypeVar

T = TypeVar("T")


class Reservoir(Generic[T]):
    def __init__(self, capacity: int, rng: random.Random | None = None):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.items: list[T] = []
        self.seen = 0
        self.rng = rng or random.Random(0)

    def add(self, item: T) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.items[j] = item

    def merge(self, other: "Reservoir[T]") -> None:
        a, b = list(self.items), list(other.items)
        self.rng.shuffle(a)
        self.rng.shuffle(b)
        na, nb = self.seen, other.seen
        # Simulate drawing without replacement from the union stream: the number of
        # slots taken from each side is hypergeometric in (na, nb).
        ra, rb = na, nb
        out: list[T] = []
        while len(out) < self.capacity and (a or b):
            take_a = bool(a) and (not b or self.rng.random() * (ra + rb) < ra)
            if take_a:
                out.append(a.pop())
                ra -= 1
            else:
                out.append(b.pop())
                rb -= 1
        self.items = out
        self.seen = na + nb
