import numpy as np

from sketchrag.memory import SketchConfig, SketchMemory
from sketchrag.retrieve import RetrieveConfig, build_context, fetch_raw, gate, search
from sketchrag.stream import Chunk, chunk_text


def topic_stream(n_chunks, n_topics=20, dim=32, seed=0):
    """Synthetic stream: each chunk belongs to a topic, embedding = topic vector + noise."""
    rng = np.random.default_rng(seed)
    topics = rng.normal(size=(n_topics, dim))
    topics /= np.linalg.norm(topics, axis=1, keepdims=True)
    for i in range(n_chunks):
        k = (i // 7) % n_topics  # topics arrive in runs, like documents
        e = topics[k] + 0.25 * rng.normal(size=dim)
        e /= np.linalg.norm(e)
        text = f"Topic{k} discussion number {i}. The topic{k} entity appears here with detail {i * 13 % 997}. Filler words fill the rest."
        yield Chunk(i, f"doc{k}", text), e.astype(np.float32), topics[k]


def test_hot_memory_is_bounded_while_cold_grows():
    cfg = SketchConfig(max_cells=8, join_threshold=0.8, cms_width=64, df_width=4096, mg_k=32)
    mem = SketchMemory(cfg, dim=32)
    sizes = []
    for i, (ch, e, _) in enumerate(topic_stream(3000)):
        mem.ingest(ch, e)
        assert len(mem.cells) <= cfg.max_cells
        assert all(sum(len(s.split()) for _, _, s in c.digest) <= cfg.digest_words for c in mem.cells)
        if i in (500, 1500, 2999):
            sizes.append((mem.hot_nbytes(), mem.cold.nbytes))
    hot = [h for h, _ in sizes]
    cold = [c for _, c in sizes]
    assert max(hot) - min(hot) < 0.1 * max(hot)  # flat (only MG key-length jitter)
    assert cold[2] > 4 * cold[0]
    assert sum(len(p) for p in mem.cold.cell_ptrs.values()) == 3000


def test_retrieval_finds_topic_and_raw_fallback_finds_needle():
    cfg = SketchConfig(max_cells=8, join_threshold=0.8, cms_width=64, df_width=4096, mg_k=32)
    mem = SketchMemory(cfg, dim=32)
    topic_vec = {}
    for ch, e, tv in topic_stream(600):
        mem.ingest(ch, e)
        topic_vec[ch.doc_id] = tv
    q = "what is said about topic3 detail 286"  # chunk 22 is topic3 with detail 22*13%997 = 286
    hits = search(mem, q, topic_vec["doc3"].astype(np.float32), RetrieveConfig(top_k=2, contiguity=0))
    assert hits
    best = hits[0].cell
    assert 22 in mem.pointers(best.id)
    raw = fetch_raw(mem, best.id, q, k=1)
    assert raw[0][0] == 22
    ctx, info = build_context(mem, q, topic_vec["doc3"].astype(np.float32), RetrieveConfig(top_k=2), policy="gate")
    dec = gate(mem, q, search(mem, q, topic_vec["doc3"].astype(np.float32), RetrieveConfig(top_k=2)), RetrieveConfig())
    assert info["fallback_cells"] == dec.fallback_cells


def test_reservoir_pointer_mode_is_bounded():
    cfg = SketchConfig(max_cells=4, join_threshold=0.8, pointer_mode="reservoir", reservoir_size=5,
                       cms_width=32, df_width=1024, mg_k=16)
    mem = SketchMemory(cfg, dim=32)
    for ch, e, _ in topic_stream(400):
        mem.ingest(ch, e)
    assert all(len(mem.pointers(c.id)) <= 5 for c in mem.cells)


def test_chunk_text_respects_budget():
    text = ". ".join(f"Sentence number {i} has a few words in it" for i in range(200)) + ". " + "x " * 500
    chunks = chunk_text(text, chunk_words=50)
    assert all(len(c.split()) <= 50 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == len(text.split())
