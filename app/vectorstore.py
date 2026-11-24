"""A small persistent vector store: cosine-similarity search over a numpy matrix.

Chunks and vectors are kept in memory and saved to <dir>/store.json + <dir>/vectors.npy.
Brute-force search is exact and plenty fast for thousands of chunks; for millions you
would switch to an ANN index (FAISS / HNSW) or a vector database.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

import numpy as np


class VectorStore:
    def __init__(self, directory, embedder_name: str):
        self.dir = Path(directory)
        self.embedder_name = embedder_name
        self._lock = RLock()
        self.docs: dict = {}                    # doc_id -> {"filename", "chunks", "added_at"}
        self.meta: list[dict] = []              # one entry per chunk, row-aligned with vectors
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self):
        meta_path, vec_path = self.dir / "store.json", self.dir / "vectors.npy"
        if not meta_path.exists():
            return
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if data.get("embedder") != self.embedder_name:
            raise ValueError(
                f"The stored index was built with '{data.get('embedder')}' but the current "
                f"embedder is '{self.embedder_name}'. Delete the data directory to rebuild."
            )
        self.docs, self.meta = data["docs"], data["chunks"]
        if self.meta:
            self.vectors = np.load(vec_path)

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {"embedder": self.embedder_name, "docs": self.docs, "chunks": self.meta}
        tmp_json, tmp_npy = self.dir / "store.json.tmp", self.dir / "vectors.npy.tmp"
        tmp_json.write_text(json.dumps(payload), encoding="utf-8")
        with open(tmp_npy, "wb") as fh:
            np.save(fh, self.vectors)
        os.replace(tmp_npy, self.dir / "vectors.npy")   # atomic swaps: no half-written index
        os.replace(tmp_json, self.dir / "store.json")

    # ---- operations --------------------------------------------------------
    def add_document(self, filename: str, chunks: list[str], vectors: np.ndarray) -> str:
        with self._lock:
            if len(chunks) != len(vectors):
                raise ValueError("chunks and vectors must have the same length")
            doc_id = uuid.uuid4().hex[:12]
            self.docs[doc_id] = {
                "filename": filename,
                "chunks": len(chunks),
                "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self.meta.extend(
                {"doc_id": doc_id, "filename": filename, "chunk_index": i, "text": text}
                for i, text in enumerate(chunks)
            )
            self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
            self._save()
            return doc_id

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self.docs:
                return False
            keep = [i for i, m in enumerate(self.meta) if m["doc_id"] != doc_id]
            self.meta = [self.meta[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            del self.docs[doc_id]
            self._save()
            return True

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[dict]:
        with self._lock:
            if not self.meta:
                return []
            scores = self.vectors @ query_vector           # cosine similarity (unit vectors)
            top = np.argsort(-scores)[:k]
            return [{**self.meta[i], "score": float(scores[i])} for i in top]

    def list_documents(self) -> list[dict]:
        with self._lock:
            return [{"doc_id": doc_id, **info} for doc_id, info in self.docs.items()]
