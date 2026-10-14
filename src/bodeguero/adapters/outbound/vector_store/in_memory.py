"""In-memory vector store: brute-force cosine similarity. For tests and quick demos.

Brute force is O(n) per query, which is fine for a few thousand chunks. pgvector
with an HNSW index does approximate nearest-neighbour search instead, which
stays fast at millions of rows.
"""

from __future__ import annotations

import math

from bodeguero.domain.models import Domain, EmbeddedChunk, RetrievedChunk


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, EmbeddedChunk] = {}

    def upsert(self, items: list[EmbeddedChunk]) -> int:
        written = 0
        for item in items:
            key = item.chunk.content_hash
            if key not in self._items:
                self._items[key] = item
                written += 1
        return written

    def search(self, query_embedding: list[float], top_k: int, domain: Domain | None = None) -> list[RetrievedChunk]:
        candidates = (
            i for i in self._items.values() if domain is None or i.chunk.domain == domain
        )
        scored = [RetrievedChunk(chunk=i.chunk, score=cosine(query_embedding, i.embedding)) for i in candidates]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._items)
