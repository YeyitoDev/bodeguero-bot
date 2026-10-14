"""Embeddings port: turns text into vectors.

Rule to remember (and say in interviews): the SAME embedding model must be
used at ingestion time and at query time, otherwise the vectors live in
different spaces and similarity search is meaningless.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingsPort(Protocol):
    @property
    def dimension(self) -> int:
        """Vector size. The vector store column must match it."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts (ingestion path). Adapters should batch requests."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query (online path)."""
        ...
