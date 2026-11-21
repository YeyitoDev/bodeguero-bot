"""Routing: decide WHERE to search and WHICH model size to use.

Two decisions, one step:
  1. Intent/domain routing -> metadata filter for retrieval (inventario/ventas/consejos)
  2. Model routing -> small model for simple lookups, large model for reasoning

This is the "orchestrator sends the question to the right branch of knowledge"
idea from your interview answer, with its official name.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import replace
from typing import Protocol

from bodeguero.domain.models import Domain, Intent, ModelTier, Route

log = logging.getLogger(__name__)

INTENT_DOMAIN: dict[Intent, Domain] = {
    Intent.PRICE: Domain.INVENTORY,
    Intent.STOCK: Domain.INVENTORY,
    Intent.INVENTORY: Domain.INVENTORY,
    Intent.SALES: Domain.SALES,
    Intent.ADVICE: Domain.ADVICE,
}


class Router(Protocol):
    def route(self, question: str) -> Route:
        ...


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


_KEYWORDS: dict[Domain, tuple[str, ...]] = {
    Domain.INVENTORY: ("precio", "cuesta", "cuanto vale", "stock", "hay ", "queda", "producto", "marca", "inventario"),
    Domain.SALES: ("vend", "venta", "ingreso", "ganancia", "caja", "fiado", "mas salio", "semana", "mes pasado"),
    Domain.ADVICE: ("consejo", "como puedo", "deberia", "recomienda", "mejorar", "estrategia", "idea"),
}

def route_for(intent: Intent, confidence: float, reason: str, spans: list[dict[str, str]] | None = None) -> Route:
    """The Route a classifier gives once it knows the intent: domain filter and model size follow from it."""
    return Route(
        domain=INTENT_DOMAIN.get(intent, Domain.UNKNOWN),
        tier=ModelTier.LARGE if intent is Intent.ADVICE else ModelTier.SMALL,
        reason=reason,
        intent=intent,
        confidence=confidence,
        spans=spans or [],
    )


_PRICE_WORDS = ("precio", "cuesta", "vale", "cuanto esta", "a cuanto", "cobro")
_LIST_WORDS = ("inventario", "listado", "lista", "catalogo")

# Words that suggest the question needs reasoning, not just a lookup.
_COMPLEX_HINTS = ("por que", "deberia", "compar", "estrategia", "recomienda", "analiza", "conviene")


class KeywordRouter:
    """Baseline rule-based router. Fast, free, explainable, and good enough to start.

    TODO(yee): build an LLMRouter that asks the SMALL model to classify the
    question into a Domain and complexity using structured output (JSON),
    and fall back to KeywordRouter if the call fails. Then compare both on
    the golden dataset in evals/ and keep the one that routes better.
    """

    def route(self, question: str) -> Route:
        q = _normalize(question)
        scores = {d: sum(1 for kw in kws if kw in q) for d, kws in _KEYWORDS.items()}
        best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
        tied = sum(1 for s in scores.values() if s == best_score) > 1

        domain = best_domain if best_score > 0 and not tied else Domain.UNKNOWN
        is_complex = any(h in q for h in _COMPLEX_HINTS) or len(re.findall(r"\w+", q)) > 25
        tier = ModelTier.LARGE if is_complex else ModelTier.SMALL
        intent = {
            Domain.INVENTORY: (
                Intent.INVENTORY if any(w in q for w in _LIST_WORDS)
                else Intent.PRICE if any(w in q for w in _PRICE_WORDS)
                else Intent.STOCK
            ),
            Domain.SALES: Intent.SALES,
            Domain.ADVICE: Intent.ADVICE,
        }.get(domain)
        return Route(
            domain=domain,
            tier=tier,
            intent=intent,
            reason=f"keyword scores={ {d.value: s for d, s in scores.items()} }",
        )


class FallbackRouter:
    """Ask the primary router; when it isn't sure, ask the fallback and keep the surer answer.

    The fallback (Jev) only picks the intent, so the primary's spans are kept.
    If the fallback fails, the primary's answer stands: it is an improvement, not a dependency.
    """

    def __init__(self, primary: Router, fallback: Router, min_confidence: float) -> None:
        self._primary = primary
        self._fallback = fallback
        self._min_confidence = min_confidence

    def route(self, question: str) -> Route:
        first = self._primary.route(question)
        if first.confidence >= self._min_confidence:
            return first
        try:
            second = self._fallback.route(question)
        except Exception:  # noqa: BLE001 - any failure means "keep the primary"
            log.warning("fallback router failed; keeping the primary route", exc_info=True)
            return first
        return replace(second, spans=first.spans) if second.confidence > first.confidence else first
