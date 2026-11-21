.PHONY: install up down ingest run test lint evals lector

# psycopg-binary 3.2.3 ships no wheels for Python 3.14
PYTHON ?= python3.13

install:            ## local venv setup
	$(PYTHON) -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pip install -e .

LECTOR_URL ?= https://github.com/YeyitoDev/bodeguero-bot/releases/download/lector-v1/lector.onnx

lector: modelos/lector/lector.onnx   ## download the lector weights (GitHub release)

modelos/lector/lector.onnx:
	curl -fL -o $@.tmp $(LECTOR_URL)
	cd modelos/lector && mv lector.onnx.tmp lector.onnx && shasum -a 256 -c lector.onnx.sha256 || { rm -f lector.onnx; exit 1; }

up: modelos/lector/lector.onnx   ## start Postgres + worker (ingestion) + API
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
