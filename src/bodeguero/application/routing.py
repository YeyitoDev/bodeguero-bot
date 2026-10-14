"""Routing: decide WHERE to search and WHICH model size to use.

Two decisions, one step:
  1. Intent/domain routing -> metadata filter for retrieval (inventario/ventas/consejos)
  2. Model routing -> small model for simple lookups, large model for reasoning

This is the "orchestrator sends the question to the right branch of knowledge"
idea from your interview answer, with its official name.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from bodeguero.domain.models import Domain, ModelTier, Route


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
        return Route(domain=domain, tier=tier, reason=f"keyword scores={ {d.value: s for d, s in scores.items()} }")
