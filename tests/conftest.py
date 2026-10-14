from __future__ import annotations

from pathlib import Path

import pytest

from bodeguero.adapters.outbound.embeddings.hashing import HashingEmbeddings
from bodeguero.adapters.outbound.llm.fake import FakeLLM
from bodeguero.adapters.outbound.loaders.filesystem import FileSystemSource
from bodeguero.adapters.outbound.messaging.telegram_client import ConsoleMessenger
from bodeguero.adapters.outbound.vector_store.in_memory import InMemoryVectorStore
from bodeguero.application.answer_question import AnswerQuestion
from bodeguero.application.ingestion import IngestDocuments
from bodeguero.application.routing import KeywordRouter
from bodeguero.bootstrap import Container
from bodeguero.config import Settings

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, data_dir=str(DATA_DIR), telegram_webhook_secret="test-secret")


@pytest.fixture
def container(settings: Settings) -> Container:
    """Everything wired with in-memory fakes: no network, no database, no keys."""
    embeddings = HashingEmbeddings(settings.embedding_dim)
    store = InMemoryVectorStore()
    llm = FakeLLM()
    IngestDocuments(FileSystemSource(DATA_DIR), embeddings, store).run()
    return Container(
        settings=settings,
        llm=llm,
        embeddings=embeddings,
        store=store,
        messenger=ConsoleMessenger(),
        answer_question=AnswerQuestion(KeywordRouter(), embeddings, store, llm),
    )
