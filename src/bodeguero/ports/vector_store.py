"""Vector store port: stores embedded chunks and finds the most similar ones."""

from __future__ import annotations

from typing import Protocol

from bodeguero.domain.models import Domain, EmbeddedChunk, RetrievedChunk


class VectorStorePort(Protocol):
    def upsert(self, items: list[EmbeddedChunk]) -> int:
        """Insert chunks, skipping ones whose content_hash already exists.

        Returns how many NEW chunks were written. Must be idempotent:
        calling it twice with the same items writes them only once.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        domain: Domain | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks, best first.

        `domain` is a metadata filter: when routing is confident, we only
        search inside that domain (e.g. only "ventas").
        """
        ...

    def count(self) -> int:
        ...
