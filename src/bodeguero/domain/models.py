"""Domain models.

These are plain, framework-free data structures. Nothing in here imports
FastAPI, psycopg, OpenAI or Telegram. That is the point of the hexagon: the
core speaks only in these types, and adapters translate to/from the outside.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class Domain(str, Enum):
    """Knowledge domains of a bodega. Used for routing and metadata filtering."""

    INVENTORY = "inventario"  # catalogue, prices, stock
    SALES = "ventas"  # sales records, best sellers, cash flow
    ADVICE = "consejos"  # business tips and policies
    UNKNOWN = "desconocido"


class Intent(str, Enum):
    """What the bodeguero wants. Same names as the lector's labels (lector/generar/comun.py)."""

    PRICE = "precio"
    STOCK = "stock"
    SALES = "ventas"
    ADVICE = "consejo"
    GREETING = "saludo"
    THANKS = "cortesia"
    OTHER = "fuera"


class ModelTier(str, Enum):
    """Which model size to use. Model routing = cheap model for easy questions."""

    SMALL = "small"
    LARGE = "large"


@dataclass(frozen=True)
class Document:
    """A raw source document before chunking (e.g. one markdown file)."""

    source: str  # stable identifier, e.g. "catalogo/bebidas.md"
    text: str
    domain: Domain


@dataclass(frozen=True)
class Chunk:
    """A piece of a document, ready to be embedded and stored."""

    source: str
    domain: Domain
    position: int  # index of the chunk inside its document
    text: str

    @property
    def content_hash(self) -> str:
        """Deterministic id. Re-ingesting the same content produces the same hash,
        which is what makes ingestion idempotent (upsert, never duplicate)."""
        raw = f"{self.source}\n{self.position}\n{self.text}".encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a search, with its similarity score (higher = closer)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Route:
    """Result of routing a question: where to look and which model to use."""

    domain: Domain
    tier: ModelTier
    reason: str = ""
    intent: Intent | None = None  # None: the router can't tell, so the answer goes through RAG
    confidence: float = 1.0
    spans: dict[str, str] = field(default_factory=dict)  # e.g. {"PRODUCTO": "aceite", "MARCA": "Primor"}


@dataclass(frozen=True)
class Citation:
    marker: int  # the [n] number used in the answer
    source: str
    content_hash: str


@dataclass(frozen=True)
class Completion:
    """What an LLM adapter returns. Token counts feed cost tracking later."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    route: Route | None = None
    model: str = ""
    grounded: bool = True  # False when we answered "I don't know"
