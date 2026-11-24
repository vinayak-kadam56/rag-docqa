"""Text -> vector. Two interchangeable embedders:

* SentenceTransformerEmbedder - real semantic embeddings (best quality, needs the
  optional `sentence-transformers` package and downloads a model on first use).
* HashingEmbedder - lightweight bag-of-words with the hashing trick. No downloads,
  deterministic, great for tests and for running anywhere; matches keywords, not meaning.

Both return L2-normalised float32 arrays, so a dot product equals cosine similarity.
"""
import hashlib
import logging
import math
import os
import re

import numpy as np

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been it its this that as at by "
    "from what which who how do does did i you we my your can should would".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # crude plural stripping: "volumes" -> "volume"
        tokens.append(tok)
    return tokens


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _slot(self, feature: str):
        digest = hashlib.md5(feature.encode()).digest()   # stable across runs (unlike hash())
        index = int.from_bytes(digest[:4], "big") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0              # signed hashing reduces collision bias
        return index, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]  # unigrams + bigrams
            for feat in features:
                idx, sign = self._slot(feat)
                out[row, idx] += sign
            out[row] = np.sign(out[row]) * np.log1p(np.abs(out[row]))  # dampen repeated words
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = f"sbert:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


def get_embedder():
    """EMBEDDINGS=hash | sbert | auto (default: sbert if installed, else hash)."""
    mode = os.getenv("EMBEDDINGS", "auto").lower()
    if mode == "hash":
        return HashingEmbedder()
    if mode == "sbert":
        return SentenceTransformerEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        log.warning("sentence-transformers not installed; falling back to the hashing embedder.")
        return HashingEmbedder()
