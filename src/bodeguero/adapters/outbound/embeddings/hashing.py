"""Hashing embeddings: a free, offline stand-in for a real embedding model.

Each word is hashed into one of `dimension` buckets (the "hashing trick"),
then the vector is L2-normalized. Texts that share words end up close.

It captures WORD OVERLAP, not MEANING: "gaseosa" and "bebida" are unrelated
here, while a real embedding model would put them close together. That gap
is exactly what you'll see improve when you plug in a real model and rerun
the evals. It's a good story for an interview.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "los", "las", "del", "que", "un", "una", "por", "con",
    "para", "es", "se", "al", "lo", "mi", "me", "cual", "cuanto", "cuantos", "como", "esta",
}


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS and len(t) > 1]


def terms(text: str) -> list[str]:
    """Tokens with crude plural stemming: "gaseosas" and "gaseosa" -> same term."""
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in _tokens(text)]


class HashingEmbeddings:
    def __init__(self, dimension: int = 1024) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in terms(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
