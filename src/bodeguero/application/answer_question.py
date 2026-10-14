"""AnswerQuestion use case (ONLINE path). This is the orchestrator.

    question
      -> route (domain + model tier)
      -> embed query            \
      -> retrieve top-k         |  the "R" in RAG. Never skip these.
      -> (rerank)               /
      -> build grounded prompt
      -> LLM
      -> output checks (citations)
      -> Answer

The orchestrator depends only on ports. It has no idea whether the LLM is
OpenAI, Anthropic or a fake, or whether vectors live in pgvector or in memory.
"""

from __future__ import annotations

import logging
import re
import time

from bodeguero.application.prompts import NO_ANSWER, PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from bodeguero.application.routing import Router
from bodeguero.domain.models import Answer, Citation, Domain, RetrievedChunk
from bodeguero.ports.embeddings import EmbeddingsPort
from bodeguero.ports.llm import LLMPort
from bodeguero.ports.vector_store import VectorStorePort

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


class AnswerQuestion:
    def __init__(
        self,
        router: Router,
        embeddings: EmbeddingsPort,
        store: VectorStorePort,
        llm: LLMPort,
        top_k: int = 4,
        min_score: float = 0.05,
    ) -> None:
        self._router = router
        self._embeddings = embeddings
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._min_score = min_score

    def execute(self, question: str) -> Answer:
        started = time.perf_counter()

        # 1. Route
        route = self._router.route(question)

        # TODO(yee): query rewriting. Turn follow-ups like "¿y la de 1 litro?"
        # into a standalone question using chat history, before embedding.

        # 2. Retrieve (the R in RAG)
        chunks = self._retrieve(question, route.domain)

        # 3. No context -> don't call the LLM at all. Cheaper and can't hallucinate.
        if not chunks:
            log.info("no relevant context; answering NO_ANSWER", extra={"route": route.domain.value})
            return Answer(text=NO_ANSWER, route=route, grounded=False)

        # 4. Generate
        completion = self._llm.generate(SYSTEM_PROMPT, build_user_prompt(question, chunks), route.tier)

        # 5. Output check: keep only citations that point to real chunks
        citations = self._extract_citations(completion.text, chunks)
        grounded = bool(citations) and NO_ANSWER not in completion.text

        # TODO(yee): output guardrail. If the answer has numbers (prices,
        # quantities) but no valid citation, don't send it: retry once with
        # a stricter instruction, or fall back to NO_ANSWER.

        # TODO(yee): tracing. Send one trace per request (question, route,
        # chunk hashes + scores, prompt version, model, tokens, latency, cost)
        # to Langfuse or OpenTelemetry. Today it is just a log line.
        log.info(
            "answered",
            extra={
                "domain": route.domain.value,
                "tier": route.tier.value,
                "model": completion.model,
                "prompt_version": PROMPT_VERSION,
                "chunks": [round(c.score, 3) for c in chunks],
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        return Answer(
            text=completion.text,
            citations=citations,
            route=route,
            model=completion.model,
            grounded=grounded,
        )

    def _retrieve(self, question: str, domain: Domain) -> list[RetrievedChunk]:
        query_vec = self._embeddings.embed_query(question)
        domain_filter = None if domain is Domain.UNKNOWN else domain

        results = self._relevant(self._store.search(query_vec, self._top_k, domain=domain_filter))
        if not results and domain_filter is not None:
            # The router can be wrong ("¿a cuánto VENDO el aceite?" looks like
            # sales but is a price question). If the filtered search finds
            # nothing relevant, fall back to searching every domain.
            results = self._relevant(self._store.search(query_vec, self._top_k, domain=None))

        # TODO(yee): hybrid search + reranking.
        #   - add a keyword (BM25 / Postgres full-text) search and merge it with
        #     the vector results using Reciprocal Rank Fusion;
        #   - retrieve top-20, then rerank with a cross-encoder down to top_k.
        return results

    def _relevant(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [r for r in results if r.score >= self._min_score]

    @staticmethod
    def _extract_citations(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
        seen: set[int] = set()
        citations: list[Citation] = []
        for match in _CITATION_RE.finditer(text):
            n = int(match.group(1))
            if 1 <= n <= len(chunks) and n not in seen:
                seen.add(n)
                c = chunks[n - 1].chunk
                citations.append(Citation(marker=n, source=c.source, content_hash=c.content_hash))
        return citations
