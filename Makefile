.PHONY: install up down ingest run test lint evals

# psycopg-binary 3.2.3 ships no wheels for Python 3.14
PYTHON ?= python3.13

install:            ## local venv setup
	$(PYTHON) -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e .

up:                 ## start Postgres + worker (ingestion) + API
	docker compose up --build

down:
	docker compose down

ingest:             ## re-run ingestion against the running database
	docker compose run --rm worker

run:                ## run the API locally (memory mode if VECTOR_STORE=memory)
	.venv/bin/uvicorn bodeguero.main:app --reload

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests evals

evals:
	.venv/bin/python -m evals.run_evals
