"""Telegram adapter for the Messenger port (sendMessage via the Bot API)."""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4096


class TelegramMessenger:
    def __init__(self, bot_token: str, timeout: float = 10.0) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._client = httpx.Client(timeout=timeout)

    def send_message(self, chat_id: int, text: str) -> None:
        # TODO(yee): retry on 429 using the `retry_after` value Telegram returns.
        resp = self._client.post(self._url, json={"chat_id": chat_id, "text": text[:TELEGRAM_MAX_LEN]})
        if resp.status_code != 200:
            log.error("telegram sendMessage failed: %s %s", resp.status_code, resp.text[:200])


class ConsoleMessenger:
    """Used when TELEGRAM_BOT_TOKEN is empty: prints replies instead of sending them."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []  # handy in tests

    def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))
        log.info("[console messenger] chat_id=%s reply=%s", chat_id, text)
