"""pgvector adapter: Postgres + the `vector` extension as our vector database.

Why pgvector instead of Pinecone/Qdrant? Vectors live next to the relational
data (products, sales) in one database: one backup, one connection, SQL joins
and transactions. Easy to defend in an interview for this scale.
"""

from __future__ import annotations

import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from bodeguero.domain.models import Chunk, Domain, EmbeddedChunk, RetrievedChunk

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    content_hash TEXT PRIMARY KEY,          -- idempotency key
    source       TEXT NOT NULL,
    domain       TEXT NOT NULL,             -- metadata used for filtering
    position     INT  NOT NULL,
    text         TEXT NOT NULL,
    embedding    vector({dim}) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- HNSW = approximate nearest neighbour index; vector_cosine_ops matches the <=> operator.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_domain_idx ON chunks (domain);
"""


class PgVectorStore:
    def __init__(self, dsn: str, dimension: int) -> None:
        self._dim = dimension
        self._pool = ConnectionPool(dsn, min_size=1, max_size=5, configure=self._configure, open=True)

    @staticmethod
    def _bootstrap_extension(dsn: str) -> None:
        # The extension must exist BEFORE register_vector() runs on a connection.
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    @classmethod
    def create(cls, dsn: str, dimension: int) -> PgVectorStore:
        cls._bootstrap_extension(dsn)
        store = cls(dsn, dimension)
        store.ensure_schema()
        return store

    @staticmethod
    def _configure(conn: psycopg.Connection) -> None:
        register_vector(conn)

    def ensure_schema(self) -> None:
        # TODO(yee): replace this with real migrations (Alembic) once the
        # schema starts changing. CREATE IF NOT EXISTS is fine for a skeleton.
        with self._pool.connection() as conn:
            conn.execute(_SCHEMA.format(dim=self._dim))

    def upsert(self, items: list[EmbeddedChunk]) -> int:
        if not items:
            return 0
        rows = [
            (i.chunk.content_hash, i.chunk.source, i.chunk.domain.value, i.chunk.position, i.chunk.text, i.embedding)
            for i in items
        ]
        written = 0
        with self._pool.connection() as conn, conn.cursor() as cur:
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO chunks (content_hash, source, domain, position, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (content_hash) DO NOTHING
                    """,
                    row,
                )
                written += cur.rowcount
        return written

    def search(self, query_embedding: list[float], top_k: int, domain: Domain | None = None) -> list[RetrievedChunk]:
        # `<=>` is cosine DISTANCE (0 = identical). Similarity = 1 - distance.
        sql = """
            SELECT source, domain, position, text, 1 - (embedding <=> %(q)s::vector) AS score
            FROM chunks
            {where}
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
        """
        params: dict = {"q": query_embedding, "k": top_k}
        where = ""
        if domain is not None:
            where = "WHERE domain = %(domain)s"
            params["domain"] = domain.value
        # TODO(yee): hybrid search. Add a `tsv tsvector` column (Spanish config),
        # run a full-text query in parallel, and merge both rankings with
        # Reciprocal Rank Fusion. It catches exact product names that pure
        # vector search can miss.
        with self._pool.connection() as conn:
            rows = conn.execute(sql.format(where=where), params).fetchall()
        return [
            RetrievedChunk(
                chunk=Chunk(source=r[0], domain=Domain(r[1]), position=r[2], text=r[3]),
                score=float(r[4]),
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._pool.connection() as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._pool.close()
