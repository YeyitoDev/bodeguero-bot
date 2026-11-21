# Bodeguero Bot

A Telegram assistant for Peruvian *bodegueros* (small shop owners). It answers questions about their catalogue, stock, sales and business rules using **Retrieval-Augmented Generation (RAG)**, and every answer cites its source.

> "¿Cuánto cuesta el aceite Primor?" → "Aceite Primor 900 ml: precio de venta S/ 9.90. Stock actual: 10 botellas. [1]" — *Fuentes: [1] catalogo/abarrotes.md*

Built with **hexagonal architecture (ports & adapters)**: the LLM provider, the embedding model, the vector store and the chat channel are all swappable without touching the core logic.

**Stack:** Python 3.11+ · FastAPI · Postgres + pgvector · Telegram Bot API · Docker Compose · pytest

---

## Architecture

```mermaid
flowchart LR
    subgraph IN["Inbound adapters"]
        TG["Telegram webhook<br/>(secret + schema validation)"]
        API["HTTP API<br/>/ask, /health"]
        WK["Worker CLI<br/>(ingestion)"]
    end

    subgraph CORE["Application core (no frameworks)"]
        AQ["AnswerQuestion<br/>orchestrator"]
        RT["Router<br/>domain + model tier"]
        PR["Grounded prompt<br/>+ citation check"]
        IG["IngestDocuments<br/>chunk → embed → upsert"]
    end

    subgraph PORTS["Ports (interfaces)"]
        LP(["LLMPort"])
        EP(["EmbeddingsPort"])
        VP(["VectorStorePort"])
        MP(["MessengerPort"])
        DP(["DocumentSourcePort"])
    end

    subgraph OUT["Outbound adapters"]
        LLM["Fake · OpenAI · Anthropic"]
        EMB["Hashing · OpenAI"]
        VS["pgvector · in-memory"]
        MSG["Telegram · console"]
        FS["File system loader"]
    end

    TG --> AQ
    API --> AQ
    WK --> IG
    AQ --> RT
    AQ --> PR
    AQ --> LP & EP & VP
    TG --> MP
    IG --> DP & EP & VP
    LP -.-> LLM
    EP -.-> EMB
    VP -.-> VS
    MP -.-> MSG
    DP -.-> FS
```

Dependencies point **inward**. The core only knows the ports. `bootstrap.py` is the single place that chooses the concrete adapters from settings.

### Online path (a user asks a question)

| # | Step | Where |
|---|------|-------|
| 1 | Telegram calls the webhook. Secret header and JSON schema are validated. Returns 200 immediately. | `adapters/inbound/telegram_webhook.py` |
| 2 | **Route**: pick a domain (`inventario` / `ventas` / `consejos`) and a model tier (small / large). | `application/routing.py` |
| 3 | **Embed** the question with the same model used at ingestion. | `EmbeddingsPort` |
| 4 | **Retrieve** top-k chunks, filtered by domain metadata. If nothing relevant is found, retry across all domains. | `VectorStorePort` |
| 5 | No relevant context → answer "No tengo esa información" **without calling the LLM**. | `application/answer_question.py` |
| 6 | **Build a grounded prompt**: numbered chunks, "answer only from context, cite [n]". | `application/prompts.py` |
| 7 | **Generate** through the LLM port. | `LLMPort` |
| 8 | **Check the output**: keep only citations that point to real chunks. | `application/answer_question.py` |
| 9 | Reply through the messenger port, with sources. | `MessengerPort` |

Steps 3–5 are the "R" in RAG.

### Offline path (ingestion)

`load → chunk (with overlap) → attach metadata (domain, source) → embed → upsert`

It runs in a separate **worker** process, so heavy ingestion never slows the chat. Each chunk's id is a **SHA-256 content hash**, and the store does `ON CONFLICT DO NOTHING`, so re-running ingestion is **idempotent**.

---

## Quick start

### Option A: no Docker, no API keys (in-memory)

```bash
make install                      # creates .venv, installs deps
make test                         # 21 tests, all offline
cp .env.example .env              # then set VECTOR_STORE=memory in .env
make run                          # http://localhost:8000/docs

curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "¿Cuánto cuesta la Inca Kola de 500 ml?"}'
```

### Option B: full stack with pgvector

```bash
cp .env.example .env              # VECTOR_STORE=pgvector (default)
make up                           # db → worker (ingests data/) → api
curl localhost:8000/health        # {"status":"ok","chunks_indexed":8}
make ingest                       # re-run ingestion: 0 new chunks (idempotent)

# integration tests against the real database
export PGVECTOR_TEST_URL=postgresql://bodeguero:bodeguero@localhost:5432/bodeguero
.venv/bin/pytest -q tests/test_pgvector_integration.py
```

### Connect Telegram

1. Create a bot with **@BotFather** and put the token in `TELEGRAM_BOT_TOKEN`.
2. Expose the API publicly (e.g. `cloudflared tunnel --url http://localhost:8000`).
3. Register the webhook with your secret:
   ```bash
   curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
     -d url=https://YOUR-TUNNEL/telegram/webhook \
     -d secret_token=$TELEGRAM_WEBHOOK_SECRET
   ```

Without a token, replies are written to the logs, so you can develop without Telegram.

---

## Evaluation

```bash
make evals
```

The pipeline is measured against `evals/golden.jsonl` (real bodeguero questions with their expected source and key fact). Retrieval and generation are measured **separately**:

| Metric | What it tells you |
|---|---|
| **recall@k** | Is the right document among the top-k retrieved chunks? |
| **MRR** | How high is it ranked? |
| **answer has fact** | Does the answer contain the key fact (price, quantity…)? |
| **grounded** | Does the answer cite at least one real chunk? |

Fake LLM + hashing embeddings + in-memory store, 10 questions:

| Router | recall@4 | MRR | answer has fact | grounded | intent accuracy |
|---|---|---|---|---|---|
| `keywords` (baseline) | 100% | 1.00 | 80% | 100% | 50% |
| `lector` (21/11/2026) | 100% | 1.00 | **100%** | 100% | **100%** |

With the lector, price and stock questions skip the LLM: the catalogue tool answers from the matched line, and
advice questions are filtered to the right domain (they used to be routed to `ventas`).

### The lector (question classifier)

A copy of restaurant-system's lector (DistilBERT in Spanish, int8 ONNX, ~4 ms per question on CPU), retrained
on synthetic bodega questions. It returns the intent (`precio`, `stock`, `ventas`, `consejo`, `saludo`,
`cortesia`, `fuera`) and the spans (`PRODUCTO`, `MARCA`, `MEDIDA`, `CANTIDAD`, `PERIODO`). On its held-out test
set (brands and templates it never saw): intent 96%, span F1 0.96. When its confidence is below
`JEV_MIN_CONFIDENCE`, Jev (TypeSafe) picks the intent instead, if `TYPESAFE_API_KEY` is set.

```bash
python3 lector/generar/sintetica.py                          # data -> lector/datos
LECTOR_TMP=/private/tmp/lector-bodega lector/entrenar/reentrenar.sh   # ~7 min -> modelos/lector
```

**Record every change here, with before/after numbers.**

---

## Design decisions

- **Hexagonal architecture.** The provider landscape changes every few months. Isolating OpenAI/Anthropic/pgvector behind ports keeps the business logic stable and makes the core testable with fakes: the whole test suite runs offline in under a second.
- **pgvector over a dedicated vector DB.** Vectors live next to relational data (products, sales) in one Postgres: one backup, transactions, SQL joins. An **HNSW** index keeps search fast as data grows. A dedicated vector DB is a later optimisation, not a starting point.
- **Content-hash ids → idempotent ingestion.** Re-running ingestion is always safe, which matters for retries and scheduled re-indexing.
- **Domain metadata + routing.** Filtering by domain improves precision and cuts noise. Because routers make mistakes, retrieval falls back to all domains when the filtered search finds nothing relevant (covered by a regression test).
- **Model routing (small vs large).** Most bodeguero questions are lookups ("¿cuánto cuesta…?"). Sending those to a small model and reserving the large one for reasoning questions is the biggest cost lever.
- **No context → no LLM call.** It's cheaper, and the model can't hallucinate an answer it was never asked to write.
- **Citations are verified, not trusted.** Citations pointing to chunks that don't exist are dropped, and the answer is marked ungrounded.
- **Webhook answers asynchronously.** Telegram retries slow webhooks. ACK first, answer in a background task.
- **Retrieved text is data, not instructions.** The system prompt says so explicitly (prompt-injection defence).

---

## Project structure

```
src/bodeguero/
├── domain/models.py              # plain dataclasses: Chunk, Route, Answer, Citation…
├── ports/                        # interfaces the core depends on
│   ├── llm.py  embeddings.py  vector_store.py  messaging.py  document_source.py
├── application/                  # use cases: the core logic
│   ├── answer_question.py        # online path orchestrator
│   ├── ingestion.py              # offline path
│   ├── chunking.py  routing.py  prompts.py
├── adapters/
│   ├── inbound/                  # who calls the core
│   │   ├── telegram_webhook.py  http_api.py  schemas.py
│   └── outbound/                 # what the core calls
│       ├── llm/                  # fake.py, openai_llm.py, anthropic_llm.py
│       ├── embeddings/           # hashing.py, openai_embeddings.py
│       ├── vector_store/         # pgvector_store.py, in_memory.py
│       ├── messaging/            # telegram_client.py
│       └── loaders/              # filesystem.py
├── bootstrap.py                  # composition root: wires adapters from settings
├── config.py                     # settings from env / .env
├── main.py                       # FastAPI app
└── worker.py                     # ingestion worker
evals/                            # golden dataset + evaluation runner
tests/                            # unit, API and pgvector integration tests
data/                             # sample bodega documents (folder = domain)
```

### Suggested reading order

1. `domain/models.py`: the vocabulary of the system.
2. `ports/`: the five contracts.
3. `application/answer_question.py`: the online path, top to bottom.
4. `application/ingestion.py` + `chunking.py`: the offline path.
5. `bootstrap.py`: how adapters get plugged into ports.
6. `adapters/`: the outside world.
7. `tests/`: each test documents one behaviour.

---

## Roadmap (`grep -rn "TODO(yee)" src evals`)

**Week 3–4: make it real**
- [ ] `OpenAILLM` / `AnthropicLLM` adapters (low temperature, token usage, retries with backoff)
- [ ] `OpenAIEmbeddings` adapter (batching; set `EMBEDDING_DIM`, re-index)
- [ ] Section-aware, token-based chunking, compared on the evals
- [ ] PDF / CSV loaders

**Week 5–6: senior-level**
- [ ] `LLMRouter` with structured output, compared to `KeywordRouter`
- [ ] Hybrid search (Postgres full-text + vectors, Reciprocal Rank Fusion) + reranking
- [ ] Query rewriting + short-term conversation memory
- [ ] Output guardrail: numbers without citations are never sent
- [ ] Tracing (Langfuse / OpenTelemetry): route, chunks, tokens, latency, cost per request
- [ ] LLM-as-a-judge evals: faithfulness, answer relevance
- [ ] Webhook dedupe by `update_id` + per-chat rate limiting

**Week 7–8: ship**
- [ ] Deploy (Fly.io / Railway) + Alembic migrations
- [ ] Queue-based ingestion worker
- [ ] 2-minute demo video and final eval numbers in this README
