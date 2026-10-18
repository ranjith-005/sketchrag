"""Collect every chunk and query text any experiment will embed, minus what is already
cached, so the bulk embedding can run once on a Kaggle GPU (kaggle/embed_cache)."""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sketchrag.data.loaders import load_task  # noqa: E402
from sketchrag.embed import QUERY_PREFIX, Embedder  # noqa: E402
from sketchrag.stream import chunk_text  # noqa: E402

FULL = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum",
        "passage_retrieval_en", "longbook_qa_eng"]
SAMPLED = {"passkey": 60, "kv_retrieval": 60}  # test (seed 0) + dev (seed 1000) samples are inside these

emb = Embedder(cache_path=ROOT / "cache" / "emb.sqlite")
texts, seen_ctx = {}, set()
for task in FULL + list(SAMPLED):
    exs = load_task(task) if task in FULL else load_task(task, SAMPLED[task], 0) + load_task(task, SAMPLED[task], 1000)
    for ex in exs:
        q = ex.question.strip() or "summary of the main points"
        texts[emb._key(QUERY_PREFIX + q)] = QUERY_PREFIX + q
        if ex.context in seen_ctx:
            continue
        seen_ctx.add(ex.context)
        for c in chunk_text(ex.context, 150):
            texts[emb._key(c)] = c
    print(task, len(texts), flush=True)
have = {k for (k,) in emb.db.execute("SELECT k FROM emb")}
todo = {k: t for k, t in texts.items() if k not in have}
out = ROOT / "kaggle" / "embed_cache" / "data" / "texts.jsonl.gz"
with gzip.open(out, "wt", encoding="utf-8") as f:
    for k, t in todo.items():
        f.write(json.dumps({"k": k, "t": t}) + "\n")
print(f"{len(todo)} texts to embed ({len(texts) - len(todo)} cached) -> {out}")
