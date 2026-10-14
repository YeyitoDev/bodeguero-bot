"""Background worker for the OFFLINE ingestion path.

    python -m bodeguero.worker ingest            # ingest DATA_DIR
    python -m bodeguero.worker ingest ./otros    # ingest another folder

Runs as its own process (its own container in docker-compose), so heavy
ingestion never slows down the chat API.

TODO(yee): turn this one-shot command into a real queue consumer (e.g. a
Postgres-backed queue, Redis + RQ, or Celery) so uploads trigger ingestion
jobs automatically, with retries.
"""

from __future__ import annotations

import argparse
import logging

from bodeguero.bootstrap import build_container
from bodeguero.config import get_settings
from bodeguero.logging_setup import configure_logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(prog="bodeguero.worker")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="chunk, embed and upsert documents")
    ingest.add_argument("data_dir", nargs="?", default=None)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    container = build_container(settings)

    if args.command == "ingest":
        report = container.ingestion(args.data_dir).run()
        print(
            f"Ingested {report.documents} documents -> {report.chunks} chunks "
            f"({report.written} new, {report.skipped} already indexed). "
            f"Total in store: {container.store.count()}"
        )


if __name__ == "__main__":
    main()
