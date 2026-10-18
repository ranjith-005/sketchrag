"""Bulk bge-small embeddings for SketchRAG, written in the same SQLite schema as
src/sketchrag/embed.py (key -> float16 blob of the L2-normalised CLS vector)."""
import glob
import gzip
import json
import sqlite3
import time

import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "BAAI/bge-small-en-v1.5"
src = sorted(glob.glob("/kaggle/input/**/texts.jsonl*", recursive=True))
print("inputs:", src, flush=True)
src = src[0]  # Kaggle may have decompressed the .gz on upload
opener = gzip.open if src.endswith(".gz") else open
rows = [json.loads(l) for l in opener(src, "rt", encoding="utf-8")]
rows.sort(key=lambda r: len(r["t"]))  # length-sorted batches waste less padding
print(len(rows), "texts", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
db = sqlite3.connect("/kaggle/working/emb.sqlite")
db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
t0, B = time.time(), 256
for i in range(0, len(rows), B):
    batch = rows[i:i + B]
    x = tok([r["t"] for r in batch], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
    with torch.no_grad():
        v = torch.nn.functional.normalize(model(**x).last_hidden_state[:, 0].float(), dim=-1)
    v = v.cpu().numpy().astype("float16")
    db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", [(r["k"], e.tobytes()) for r, e in zip(batch, v)])
    if (i // B) % 100 == 0:
        db.commit()
        print(f"{i + len(batch)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
db.commit()
print("done", time.time() - t0, flush=True)
