"""Chunking: split a document into overlapping pieces.

Why chunk at all? Embeddings of a whole document blur many topics together,
and the LLM context window is limited. Small, focused chunks retrieve better.

Why overlap? So a sentence cut at a boundary still appears whole in one chunk.
"""

from __future__ import annotations

from bodeguero.domain.models import Chunk, Document


def chunk_document(doc: Document, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Split by paragraphs, packing them into chunks of ~chunk_size characters.

    Simple and predictable on purpose. It respects paragraph boundaries when it
    can and only hard-splits paragraphs that are longer than chunk_size.

    TODO(yee): improve the strategy and measure the effect with evals/:
      - split by markdown headings first (section-aware chunking), so each
        chunk carries its section title as context;
      - size chunks in TOKENS instead of characters (tiktoken) so they map
        to model limits and cost;
      - try 300 / 600 / 1000 and compare recall@k. Pick by data, not by guess.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            pieces.append(p)
        else:  # hard-split very long paragraphs
            step = chunk_size - overlap
            pieces.extend(p[i : i + chunk_size] for i in range(0, len(p), step))

    chunks_text: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks_text.append(current)
            # carry the tail of the previous chunk forward = overlap
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = piece
    if current:
        chunks_text.append(current)

    return [
        Chunk(source=doc.source, domain=doc.domain, position=i, text=t)
        for i, t in enumerate(chunks_text)
    ]
