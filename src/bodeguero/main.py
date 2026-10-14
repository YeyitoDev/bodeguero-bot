"""FastAPI app factory. Run with:  uvicorn bodeguero.main:app --reload"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from bodeguero.adapters.inbound import http_api, telegram_webhook
from bodeguero.bootstrap import Container, build_container
from bodeguero.config import Settings, get_settings
from bodeguero.logging_setup import configure_logging

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        app.state.container = container or build_container(settings)
        store = app.state.container.store
        if settings.vector_store == "memory" and store.count() == 0:
            # Memory mode has no separate worker process to share data with,
            # so we ingest the sample data at startup. With pgvector, the
            # worker does this offline (see worker.py / docker-compose).
            report = app.state.container.ingestion().run()
            log.info("memory mode: ingested sample data", extra={"chunks": report.chunks})
        yield
        close = getattr(store, "close", None)
        if close:
            close()

    app = FastAPI(title="Bodeguero Bot", version="0.1.0", lifespan=lifespan)
    app.include_router(http_api.router)
    app.include_router(telegram_webhook.router)
    return app


app = create_app()
