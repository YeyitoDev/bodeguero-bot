"""Telegram webhook: input adapter.

Flow: validate secret header -> validate body schema -> ACK 200 immediately
-> answer in a background task -> send reply through the Messenger port.

Why answer in the background? Telegram retries a webhook that doesn't answer
quickly, and LLM calls can take seconds. Retries would mean duplicate replies.
"""

from __future__ import annotations

import hmac
import html
import logging
import re

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status

from bodeguero.adapters.inbound.schemas import TelegramUpdate
from bodeguero.adapters.outbound.messaging.telegram_client import TELEGRAM_MAX_LEN
from bodeguero.bootstrap import Container

log = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])

WELCOME = (
    "<b>¡Hola! Soy tu asistente de bodega.</b>\n"
    "Pregúntame precios, stock, ventas o pídeme tu inventario.\n"
    "Por ejemplo: <i>¿cuánto está la Inca Kola de 500 ml?</i>"
)
_MARKER_RE = re.compile(r"\s*\[\d+\]")


def to_html(text: str) -> str:
    """Telegram HTML: names in bold, "Section:" lines as headings, no [n] markers (sources mean nothing in a chat)."""
    out: list[str] = []
    size = 0
    for line in _MARKER_RE.sub("", text).splitlines():
        line = html.escape(line, quote=False)
        name, sep, rest = line.partition(": ")
        if line.endswith(":") and ": " not in line:
            line = f"\n<b>{line[:-1]}</b>"
        elif sep and len(name) <= 60:
            line = f"<b>{name}</b>: {rest}"
        # Cut on whole lines: a cut inside a tag makes Telegram reject the message.
        if size + len(line) + 1 > TELEGRAM_MAX_LEN - 20:
            out.append("…")
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out).strip()


def _handle_message(container: Container, chat_id: int, text: str) -> None:
    try:
        reply = to_html(container.answer_question.execute(text).text)
    except Exception:  # never let one bad message kill the worker silently
        log.exception("failed to answer chat_id=%s", chat_id)
        reply = "Uy, tuve un problema respondiendo. Intenta de nuevo en un momento."
    container.messenger.send_message(chat_id, reply)


@router.post("/webhook", status_code=status.HTTP_200_OK)
async def telegram_webhook(
    update: TelegramUpdate,
    request: Request,
    background: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict:
    container: Container = request.app.state.container

    # Telegram sends the secret you configured in setWebhook in this header.
    # Constant-time comparison avoids leaking the secret through timing.
    expected = container.settings.telegram_webhook_secret
    if not x_telegram_bot_api_secret_token or not hmac.compare_digest(x_telegram_bot_api_secret_token, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid secret token")

    msg = update.message
    if msg is None or not msg.text:
        return {"ok": True, "ignored": "no text"}

    if msg.text.strip().startswith("/start"):
        background.add_task(container.messenger.send_message, msg.chat.id, WELCOME)
        return {"ok": True}

    # TODO(yee): dedupe by update_id (Telegram can deliver the same update
    # twice) and add per-chat rate limiting to protect your LLM budget.
    background.add_task(_handle_message, container, msg.chat.id, msg.text)
    return {"ok": True}
