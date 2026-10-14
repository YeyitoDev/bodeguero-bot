import pytest

from bodeguero.application.chunking import chunk_document
from bodeguero.domain.models import Document, Domain


def _doc(text: str) -> Document:
    return Document(source="x.md", text=text, domain=Domain.INVENTORY)


def test_short_document_is_one_chunk():
    chunks = chunk_document(_doc("Inca Kola 500 ml: S/ 2.50"))
    assert len(chunks) == 1
    assert chunks[0].position == 0


def test_long_document_is_split_with_overlap():
    paragraphs = [f"Producto {i}: " + "detalle " * 20 for i in range(10)]
    chunks = chunk_document(_doc("\n\n".join(paragraphs)), chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 + 50 + 2 for c in chunks)
    # the tail of each chunk reappears at the start of the next one
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.text[-50:] in nxt.text


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_document(_doc("hola"), chunk_size=100, overlap=100)


def test_content_hash_is_deterministic():
    a = chunk_document(_doc("mismo texto"))[0]
    b = chunk_document(_doc("mismo texto"))[0]
    assert a.content_hash == b.content_hash
