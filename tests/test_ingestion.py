from bodeguero.adapters.outbound.embeddings.hashing import HashingEmbeddings
from bodeguero.adapters.outbound.loaders.filesystem import FileSystemSource
from bodeguero.adapters.outbound.vector_store.in_memory import InMemoryVectorStore
from bodeguero.application.ingestion import IngestDocuments
from bodeguero.domain.models import Domain

from .conftest import DATA_DIR


def test_ingestion_is_idempotent():
    store = InMemoryVectorStore()
    ingest = IngestDocuments(FileSystemSource(DATA_DIR), HashingEmbeddings(), store)

    first = ingest.run()
    second = ingest.run()

    assert first.written > 0
    assert second.written == 0  # nothing duplicated on re-run
    assert second.skipped == second.chunks
    assert store.count() == first.written


def test_folder_name_becomes_domain_metadata():
    docs = {d.source: d.domain for d in FileSystemSource(DATA_DIR).load()}
    assert docs["catalogo/bebidas.md"] is Domain.INVENTORY
    assert docs["ventas/octubre-semana-1.md"] is Domain.SALES
    assert docs["consejos/fiado.md"] is Domain.ADVICE
