"""Composition root: the ONE place that knows about concrete adapters.

Everything else receives its dependencies through constructors (dependency
injection). To swap OpenAI for Anthropic, or memory for pgvector, you change
settings, and this file picks the adapter. No other file changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bodeguero.adapters.outbound.embeddings.hashing import HashingEmbeddings
from bodeguero.adapters.outbound.loaders.filesystem import FileSystemSource
from bodeguero.adapters.outbound.messaging.telegram_client import ConsoleMessenger, TelegramMessenger
from bodeguero.adapters.outbound.vector_store.in_memory import InMemoryVectorStore
from bodeguero.application.answer_question import AnswerQuestion
from bodeguero.application.ingestion import IngestDocuments
from bodeguero.application.routing import KeywordRouter
from bodeguero.config import Settings
from bodeguero.ports.embeddings import EmbeddingsPort
from bodeguero.ports.llm import LLMPort
from bodeguero.ports.messaging import MessengerPort
from bodeguero.ports.vector_store import VectorStorePort

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    llm: LLMPort
    embeddings: EmbeddingsPort
    store: VectorStorePort
    messenger: MessengerPort
    answer_question: AnswerQuestion

    def ingestion(self, data_dir: str | None = None) -> IngestDocuments:
        return IngestDocuments(
            source=FileSystemSource(data_dir or self.settings.data_dir),
            embeddings=self.embeddings,
            store=self.store,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )


def build_llm(s: Settings) -> LLMPort:
    if s.llm_provider == "openai":
        from bodeguero.adapters.outbound.llm.openai_llm import OpenAILLM

        return OpenAILLM(s.openai_api_key, s.llm_small_model, s.llm_large_model)
    if s.llm_provider == "anthropic":
        from bodeguero.adapters.outbound.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(s.anthropic_api_key, s.llm_small_model, s.llm_large_model)
    from bodeguero.adapters.outbound.llm.fake import FakeLLM

    return FakeLLM()


def build_embeddings(s: Settings) -> EmbeddingsPort:
    if s.embeddings_provider == "openai":
        from bodeguero.adapters.outbound.embeddings.openai_embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(s.openai_api_key, s.embedding_model, s.embedding_dim)
    return HashingEmbeddings(s.embedding_dim)


def build_store(s: Settings, dimension: int) -> VectorStorePort:
    if s.vector_store == "pgvector":
        # imported lazily so tests and memory mode don't need psycopg installed/running
        from bodeguero.adapters.outbound.vector_store.pgvector_store import PgVectorStore

        return PgVectorStore.create(s.database_url, dimension)
    return InMemoryVectorStore()


def build_messenger(s: Settings) -> MessengerPort:
    if s.telegram_bot_token:
        return TelegramMessenger(s.telegram_bot_token)
    log.warning("TELEGRAM_BOT_TOKEN is empty: replies will be logged, not sent")
    return ConsoleMessenger()


def build_container(s: Settings) -> Container:
    llm = build_llm(s)
    embeddings = build_embeddings(s)
    store = build_store(s, embeddings.dimension)
    answer = AnswerQuestion(
        router=KeywordRouter(),
        embeddings=embeddings,
        store=store,
        llm=llm,
        top_k=s.top_k,
        min_score=s.min_score,
    )
    return Container(
        settings=s,
        llm=llm,
        embeddings=embeddings,
        store=store,
        messenger=build_messenger(s),
        answer_question=answer,
    )
