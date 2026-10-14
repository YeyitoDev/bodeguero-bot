from fastapi.testclient import TestClient

from bodeguero.main import create_app


def _client(settings, container) -> TestClient:
    return TestClient(create_app(settings=settings, container=container))


def test_health(settings, container):
    with _client(settings, container) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0


def test_ask_returns_answer_with_citations(settings, container):
    with _client(settings, container) as client:
        resp = client.post("/ask", json={"question": "¿Cuánto cuesta la Inca Kola de 500 ml?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["citations"][0]["source"] == "catalogo/bebidas.md"


def test_ask_validates_input(settings, container):
    with _client(settings, container) as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422


SECRET = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def _update(text: str) -> dict:
    return {"update_id": 1, "message": {"message_id": 1, "chat": {"id": 42}, "text": text}}


def test_webhook_rejects_wrong_secret(settings, container):
    with _client(settings, container) as client:
        resp = client.post("/telegram/webhook", json=_update("hola"), headers={"X-Telegram-Bot-Api-Secret-Token": "x"})
    assert resp.status_code == 401


def test_webhook_answers_through_messenger_port(settings, container):
    with _client(settings, container) as client:
        resp = client.post(
            "/telegram/webhook",
            json=_update("¿Cuánto cuesta el arroz Costeño?"),
            headers=SECRET,
        )
    assert resp.status_code == 200
    chat_id, reply = container.messenger.sent[-1]
    assert chat_id == 42
    assert "4.80" in reply and "Fuentes:" in reply


def test_webhook_ignores_non_text_updates(settings, container):
    payload = {"update_id": 2, "message": {"message_id": 2, "chat": {"id": 42}}}
    with _client(settings, container) as client:
        resp = client.post("/telegram/webhook", json=payload, headers=SECRET)
    assert resp.json()["ignored"] == "no text"
