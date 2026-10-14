"""Request/response schemas for the inbound HTTP adapters.

Validation happens HERE, at the edge. If a request doesn't match the schema,
FastAPI rejects it with 422 before it reaches the core.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ---------- Telegram (only the fields we use; Telegram sends many more) ----------


class TelegramChat(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int


class TelegramMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message_id: int
    chat: TelegramChat
    text: str | None = None  # photos, stickers, voice notes have no text


class TelegramUpdate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    update_id: int
    message: TelegramMessage | None = None


# ---------- Local HTTP API (handy for testing without Telegram) ----------


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class CitationOut(BaseModel):
    marker: int
    source: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    domain: str | None
    model_tier: str | None
    model: str
    citations: list[CitationOut]
