"""Ingestion use case (OFFLINE path): load -> chunk -> embed -> upsert.

Runs in a background worker (see worker.py), never inside a user request.
Ingesting thousands of documents is slow; the chat must stay fast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bodeguero.application.chunking import chunk_document
from bodeguero.domain.models import EmbeddedChunk
from bodeguero.ports.document_source import DocumentSourcePort
from bodeguero.ports.embeddings import EmbeddingsPort
from bodeguero.ports.vector_store import VectorStorePort

log = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    documents: int = 0
    chunks: int = 0
    written: int = 0  # new chunks; re-runs on unchanged data give 0

    @property
    def skipped(self) -> int:
        return self.chunks - self.written


class IngestDocuments:
    def __init__(
        self,
        source: DocumentSourcePort,
        embeddings: EmbeddingsPort,
        store: VectorStorePort,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        batch_size: int = 64,
    ) -> None:
        self._source = source
        self._embeddings = embeddings
        self._store = store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size

    def run(self) -> IngestionReport:
        report = IngestionReport()
        batch = []
        for doc in self._source.load():
            report.documents += 1
            chunks = chunk_document(doc, self._chunk_size, self._chunk_overlap)
            report.chunks += len(chunks)
            batch.extend(chunks)
            if len(batch) >= self._batch_size:
                report.written += self._flush(batch)
                batch = []
        if batch:
            report.written += self._flush(batch)

        log.info(
            "ingestion done: %s docs, %s chunks, %s new, %s skipped",
            report.documents, report.chunks, report.written, report.skipped,
        )
        return report

    def _flush(self, chunks: list) -> int:
        # TODO(yee): skip embedding chunks whose content_hash already exists
        # (ask the store first). Today we embed everything and let the store
        # dedupe, which is correct but pays for embeddings we throw away.
        vectors = self._embeddings.embed_documents([c.text for c in chunks])
        items = [EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vectors, strict=True)]
        return self._store.upsert(items)
