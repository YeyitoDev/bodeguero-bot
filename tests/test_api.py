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
    assert reply == "<b>Arroz Costeño 1 kg</b>: S/ 4.80"


def test_webhook_ignores_non_text_updates(settings, container):
    payload = {"update_id": 2, "message": {"message_id": 2, "chat": {"id": 42}}}
    with _client(settings, container) as client:
        resp = client.post("/telegram/webhook", json=payload, headers=SECRET)
    assert resp.json()["ignored"] == "no text"


def test_to_html_bolds_names_and_headings_escapes_and_drops_markers():
    from bodeguero.adapters.inbound.telegram_webhook import to_html

    text = "Gaseosas:\nInca Kola 500 ml: S/ 2.50 · 36 unidades [1]\nUsa <script> & no [2]"
    assert to_html(text) == (
        "<b>Gaseosas</b>\n<b>Inca Kola 500 ml</b>: S/ 2.50 · 36 unidades\nUsa &lt;script&gt; &amp; no"
    )


def test_to_html_cuts_long_answers_on_whole_lines():
    from bodeguero.adapters.inbound.telegram_webhook import to_html

    reply = to_html("\n".join(f"Producto {i}: S/ 1.00" for i in range(500)))
    assert len(reply) <= 4096 and reply.endswith("…") and reply.count("<b>") == reply.count("</b>")
