"""Merge an embedding SQLite produced on Kaggle (kaggle/embed_cache) into cache/emb.sqlite."""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = sys.argv[1]
dst = sqlite3.connect(str(ROOT / "cache" / "emb.sqlite"))
dst.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
dst.execute("ATTACH DATABASE ? AS src", (src,))
before = dst.execute("SELECT COUNT(*) FROM emb").fetchone()[0]
dst.execute("INSERT OR IGNORE INTO emb SELECT k, v FROM src.emb")
dst.commit()
after = dst.execute("SELECT COUNT(*) FROM emb").fetchone()[0]
print(f"merged: {before} -> {after} embeddings")
