"""Settings loaded from environment variables / .env (12-factor style)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- adapters selection (change these to swap implementations) ---
    llm_provider: Literal["fake", "openai", "anthropic"] = "fake"
    embeddings_provider: Literal["hashing", "openai"] = "hashing"
    vector_store: Literal["memory", "pgvector"] = "memory"

    # --- LLM ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_small_model: str = ""  # cheap/fast model for simple lookups
    llm_large_model: str = ""  # stronger model for reasoning questions

    # --- embeddings ---
    embedding_model: str = ""
    embedding_dim: int = 1024

    # --- routing (who reads the question) ---
    router: Literal["lector", "keywords"] = "lector"
    lector_dir: str = "modelos/lector"
    typesafe_api_key: str = ""  # Jev; empty -> no fallback when the lector isn't sure
    jev_min_confidence: float = 0.8  # below this, the lector asks Jev

    # --- retrieval ---
    top_k: int = 4
    min_score: float = 0.05
    chunk_size: int = 400
    chunk_overlap: int = 80

    # --- infrastructure ---
    database_url: str = "postgresql://bodeguero:bodeguero@localhost:5432/bodeguero"
    data_dir: str = "data"

    # --- telegram ---
    telegram_bot_token: str = ""  # empty -> replies are logged, not sent
    telegram_webhook_secret: str = "change-me"  # validated on every webhook call

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
