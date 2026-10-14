"""Plain HTTP API: /health and /ask. Lets you test the RAG pipeline with curl,
no Telegram needed. A second input adapter driving the SAME use case shows
the hexagon at work: the core doesn't care who is calling it."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bodeguero.adapters.inbound.schemas import AskRequest, AskResponse, CitationOut
from bodeguero.bootstrap import Container

router = APIRouter(tags=["api"])


@router.get("/health")
def health(request: Request) -> dict:
    container: Container = request.app.state.container
    return {"status": "ok", "chunks_indexed": container.store.count()}


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    container: Container = request.app.state.container
    answer = container.answer_question.execute(body.question)
    return AskResponse(
        answer=answer.text,
        grounded=answer.grounded,
        domain=answer.route.domain.value if answer.route else None,
        model_tier=answer.route.tier.value if answer.route else None,
        model=answer.model,
        citations=[CitationOut(marker=c.marker, source=c.source) for c in answer.citations],
    )
