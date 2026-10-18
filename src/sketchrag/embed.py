"""bge embeddings with a persistent, content-addressed cache.

The cache is a SQLite file keyed by sha1(model + text), so the same chunk is never
embedded twice (passkey haystacks repeat the same filler thousands of times), and a
cache built on a Kaggle GPU can be copied here and reused unchanged.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_path: str | Path = "cache/emb.sqlite",
                 device: str | None = None, batch_size: int = 64, max_length: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None
        self._tok = None
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._lock = threading.Lock()  # agentic systems embed new queries from worker threads
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self.dim: int | None = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).eval()
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
            if self.device == "cuda":
                self._model.half()

    def _key(self, text: str) -> str:
        return hashlib.sha1((self.model_name + "\x00" + text).encode("utf-8")).hexdigest()

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = self._tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                cls = self._model(**batch).last_hidden_state[:, 0].float()
            cls = torch.nn.functional.normalize(cls, dim=-1)
            out.append(cls.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return self._embed(texts)

    def _embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        uniq = list(dict.fromkeys(keys))
        for i in range(0, len(uniq), 500):
            part = uniq[i:i + 500]
            q = "SELECT k, v FROM emb WHERE k IN (%s)" % ",".join("?" * len(part))
            for k, v in self.db.execute(q, part):
                found[k] = np.frombuffer(v, dtype=np.float16).astype(np.float32)
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in found]
        if missing:
            vecs = self._encode([t for _, t in missing])
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                [(k, v.astype(np.float16).tobytes()) for (k, _), v in zip(missing, vecs)])
            self.db.commit()
            for (k, _), v in zip(missing, vecs):
                found[k] = v
        arr = np.stack([found[k] for k in keys]) if keys else np.zeros((0, self.dim or 384), np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12  # fp16 round-trip
        self.dim = arr.shape[1] if len(arr) else self.dim
        return arr

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self.embed([QUERY_PREFIX + q for q in queries])
