"""OpenAI embeddings adapter.

TODO(yee): implement.
  - Use an embedding model (check the docs for current names and dimensions)
    and set EMBEDDING_DIM in .env to its dimension. The pgvector column is
    created with that size.
  - embed_documents: send texts in batches (the API accepts a list), keep order.
  - embed_query: same model, single text.
  - Changing the embedding model means RE-INDEXING everything: old vectors
    are not comparable with new ones. Drop the table (or use a new one) and
    re-run the worker.
"""

from __future__ import annotations


class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for EMBEDDINGS_PROVIDER=openai")
        self._model = model
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("TODO(yee): implement OpenAIEmbeddings.embed_documents")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("TODO(yee): implement OpenAIEmbeddings.embed_query")
