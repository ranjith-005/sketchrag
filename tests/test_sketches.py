import random
from collections import Counter

import numpy as np
import pytest

from sketchrag.sketches.countmin import CountMinSketch
from sketchrag.sketches.misra_gries import MisraGries
from sketchrag.sketches.reservoir import Reservoir


def zipf_stream(n, vocab, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.zipf(1.3, size=n) % vocab
    return [f"w{i}" for i in ids]


@pytest.mark.parametrize("conservative", [False, True])
def test_countmin_never_underestimates_and_respects_bound(conservative):
    stream = zipf_stream(50_000, 5_000)
    true = Counter(stream)
    cms = CountMinSketch.from_error(eps=0.002, delta=0.01, conservative=conservative)
    for x in stream:
        cms.update(x)
    bound = cms.error_bound()
    violations = 0
    for x, f in true.items():
        est = cms.query(x)
        assert est >= f
        violations += est - f > bound
    # guarantee is per query w.p. >= 1 - delta; allow 2x slack on the empirical rate
    assert violations / len(true) <= 2 * cms.delta


def test_countmin_merge_equals_sketch_of_union():
    a, b = CountMinSketch(128, 3, conservative=False), CountMinSketch(128, 3, conservative=False)
    u = CountMinSketch(128, 3, conservative=False)
    for x in zipf_stream(5000, 300, seed=1):
        a.update(x); u.update(x)
    for x in zipf_stream(5000, 300, seed=2):
        b.update(x); u.update(x)
    a.merge(b)
    assert np.array_equal(a.table, u.table) and a.total == u.total


def test_countmin_conservative_is_tighter():
    stream = zipf_stream(20_000, 3_000, seed=3)
    plain, cons = CountMinSketch(64, 3, False), CountMinSketch(64, 3, True)
    for x in stream:
        plain.update(x); cons.update(x)
    keys = set(stream)
    assert sum(cons.query(k) for k in keys) <= sum(plain.query(k) for k in keys)


def test_misra_gries_guarantee():
    stream = zipf_stream(30_000, 2_000, seed=4)
    true = Counter(stream)
    mg = MisraGries(50)
    for x in stream:
        mg.update(x)
    assert len(mg.counters) <= 50
    eb = mg.error_bound()
    for x, f in true.items():
        est = mg.query(x)
        assert f - eb <= est <= f
        if f > eb:
            assert x in mg.counters


def test_misra_gries_weighted_updates():
    mg = MisraGries(3)
    for k, c in [("a", 5), ("b", 3), ("c", 2), ("d", 4), ("a", 1)]:
        mg.update(k, c)
    true = {"a": 6, "b": 3, "c": 2, "d": 4}
    for k, f in true.items():
        assert f - mg.error_bound() <= mg.query(k) <= f


def test_reservoir_is_uniform():
    hits = Counter()
    for trial in range(3000):
        r = Reservoir(5, random.Random(trial))
        for i in range(50):
            r.add(i)
        hits.update(r.items)
    freqs = np.array([hits[i] for i in range(50)]) / 3000
    assert abs(freqs.mean() - 0.1) < 1e-9
    assert freqs.min() > 0.07 and freqs.max() < 0.13


def test_reservoir_merge_is_proportional():
    from_a = 0
    for trial in range(2000):
        rng = random.Random(trial)
        a, b = Reservoir(10, rng), Reservoir(10, rng)
        for i in range(300):
            a.add(("a", i))
        for i in range(100):
            b.add(("b", i))
        a.merge(b)
        assert len(a.items) == 10 and a.seen == 400
        from_a += sum(1 for s, _ in a.items if s == "a")
    assert abs(from_a / (2000 * 10) - 0.75) < 0.02
