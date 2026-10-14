"""Integration test against a real Postgres + pgvector.

Skipped automatically unless the database is reachable. Run it with:
    docker compose up -d db
    export PGVECTOR_TEST_URL=postgresql://bodeguero:bodeguero@localhost:5432/bodeguero
    pytest -q tests/test_pgvector_integration.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from bodeguero.adapters.outbound.embeddings.hashing import HashingEmbeddings
from bodeguero.adapters.outbound.loaders.filesystem import FileSystemSource
from bodeguero.application.ingestion import IngestDocuments
from bodeguero.domain.models import Domain

from .conftest import DATA_DIR

URL = os.getenv("PGVECTOR_TEST_URL")
pytestmark = pytest.mark.skipif(not URL, reason="set PGVECTOR_TEST_URL to run pgvector integration tests")


@pytest.fixture
def store():
    psycopg = pytest.importorskip("psycopg")
    from bodeguero.adapters.outbound.vector_store.pgvector_store import PgVectorStore

    # isolate the test in its own schema so it never touches real data
    schema = f"test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"CREATE SCHEMA {schema}")
    s = PgVectorStore.create(f"{URL}?options=-csearch_path%3D{schema},public", dimension=1024)
    yield s
    s.close()
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


def test_pgvector_ingest_is_idempotent_and_searchable(store):
    emb = HashingEmbeddings(1024)
    ingest = IngestDocuments(FileSystemSource(DATA_DIR), emb, store)

    first = ingest.run()
    second = ingest.run()
    assert first.written > 0 and second.written == 0
    assert store.count() == first.written

    results = store.search(emb.embed_query("precio aceite Primor"), top_k=3)
    assert results[0].chunk.source == "catalogo/abarrotes.md"
    assert results[0].score >= results[-1].score

    only_sales = store.search(emb.embed_query("aceite"), top_k=3, domain=Domain.SALES)
    assert all(r.chunk.domain is Domain.SALES for r in only_sales)
