"""Messaging port: sending replies back to the user (Telegram today, WhatsApp tomorrow)."""

from __future__ import annotations

from typing import Protocol


class MessengerPort(Protocol):
    def send_message(self, chat_id: int, text: str) -> None:
        ...
