"""AnswerQuestion use case (ONLINE path). This is the orchestrator, built as a LangGraph.

    question -> route (intent + spans)
      precio / stock / inventario -> catalog: scan the catalogue, pick the product lines, template answer
      ventas / consejo / unsure -> rag:     retrieve -> grounded prompt -> LLM -> citation check
      saludo / cortesia / fuera -> fixed:   canned answer, no retrieval

The orchestrator depends only on ports. It has no idea whether the router is
the lector, Jev or keywords, or whether vectors live in pgvector or in memory.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from bodeguero.application.catalog import ProductLine, find_products, products, reply
from bodeguero.application.prompts import NO_ANSWER, PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from bodeguero.application.routing import Router
from bodeguero.domain.models import Answer, Citation, Domain, Intent, RetrievedChunk, Route
from bodeguero.ports.embeddings import EmbeddingsPort
from bodeguero.ports.llm import LLMPort
from bodeguero.ports.vector_store import VectorStorePort

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_CATALOG = (Intent.PRICE, Intent.STOCK, Intent.INVENTORY)
_PRODUCT_LABELS = ("PRODUCTO", "MARCA", "MEDIDA")
# shortcut: the catalog node reads up to this many catalogue chunks per question, fine for one bodega;
# past that, filter product lines in SQL instead of scanning them.
_CATALOG_SCAN = 500

_FIXED = {
    Intent.GREETING: "¡Hola! Pregúntame precios, stock, ventas o consejos para tu bodega.",
    Intent.THANKS: "¡De nada! Aquí estoy para lo que necesites.",
    Intent.OTHER: "Solo sé de tu bodega: precios, stock, ventas y consejos.",
}


class _State(TypedDict, total=False):
    question: str
    route: Route
    answer: Answer


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

        graph = StateGraph(_State)
        graph.add_node("route", lambda s: {"route": self._router.route(s["question"])})
        graph.add_node("catalog", self._catalog)
        graph.add_node("rag", self._rag)
        graph.add_node("fixed", self._fixed)
        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self._branch, ["catalog", "rag", "fixed"])
        for node in ("catalog", "rag", "fixed"):
            graph.add_edge(node, END)
        self._graph = graph.compile()

    def execute(self, question: str) -> Answer:
        answer = self._graph.invoke({"question": question})["answer"]
        r = answer.route
        # DEBUG because it carries the user's text; turn it on only to review real questions.
        log.debug(
            "routed",
            extra={"question": question, "intent": r.intent, "why": r.reason, "spans": r.spans, "answer": answer.text},
        )
        return answer

    @staticmethod
    def _branch(state: _State) -> str:
        intent = state["route"].intent
        if intent in _CATALOG:
            return "catalog"
        return "fixed" if intent in _FIXED else "rag"

    def _fixed(self, state: _State) -> _State:
        route = state["route"]
        return {"answer": Answer(text=_FIXED[route.intent], route=route, grounded=route.intent is not Intent.OTHER)}

    def _catalog(self, state: _State) -> _State:
        question, route = state["question"], state["route"]
        vec = self._embeddings.embed_query(question)
        chunks = self._store.search(vec, _CATALOG_SCAN, domain=Domain.INVENTORY)
        named = [s for s in route.spans if s.keys() & set(_PRODUCT_LABELS)]
        # Keywords give no spans: search with the whole question. Inventory with nothing named lists everything.
        searches = named or ([] if route.intent is Intent.INVENTORY else [{"PRODUCTO": question}])

        found: list[ProductLine] = [] if searches else products(chunks)
        missing: list[str] = []
        for s in searches:
            query = " ".join(s[k] for k in _PRODUCT_LABELS if k in s)
            hits = find_products(query, chunks, brand=s.get("MARCA", ""), size=s.get("MEDIDA", ""))
            found += [h for h in hits if h.line not in {f.line for f in found}]
            if not hits:
                missing.append(query if named else "ese producto")

        markers: dict[str, int] = {}
        citations: list[Citation] = []
        lines = []
        for product in found:
            c = product.chunk.chunk
            if c.content_hash not in markers:
                markers[c.content_hash] = len(markers) + 1
                citations.append(Citation(marker=markers[c.content_hash], source=c.source, content_hash=c.content_hash))
            lines.append(f"{reply(route.intent, product)} [{markers[c.content_hash]}]")
        if missing:
            lines.append(f"No tienes {', '.join(missing)} en tu catálogo.")
        return {
            "answer": Answer(
                text="\n".join(lines) or "Tu catálogo está vacío.",
                citations=citations,
                route=route,
                model="catalog",
                grounded=bool(found),
            )
        }

    def _rag(self, state: _State) -> _State:
        started = time.perf_counter()
        question, route = state["question"], state["route"]

        # TODO(yee): query rewriting. Turn follow-ups like "¿y la de 1 litro?"
        # into a standalone question using chat history, before embedding.

        # 2. Retrieve (the R in RAG)
        chunks = self._retrieve(question, route.domain)

        # 3. No context -> don't call the LLM at all. Cheaper and can't hallucinate.
        if not chunks:
            log.info("no relevant context; answering NO_ANSWER", extra={"route": route.domain.value})
            return {"answer": Answer(text=NO_ANSWER, route=route, grounded=False)}

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
        return {
            "answer": Answer(
                text=completion.text,
                citations=citations,
                route=route,
                model=completion.model,
                grounded=grounded,
            )
        }

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
