"""Fake LLM: deterministic, free, no API key. Used in tests and local dev.

It is "extractive": from the top-ranked chunk [1] it quotes the line that
shares the most words with the question, and cites it as [1]. That's enough
to exercise the whole pipeline end to end without paying for tokens.
"""

from __future__ import annotations

import re

from bodeguero.adapters.outbound.embeddings.hashing import terms
from bodeguero.application.prompts import NO_ANSWER
from bodeguero.domain.models import Completion, ModelTier

_FIRST_CHUNK_RE = re.compile(r"\[1\] \(fuente: [^)]+\)\n(.+?)(?:\n\n\[2\] \(fuente:|\n\nPREGUNTA:)", re.S)
_QUESTION_RE = re.compile(r"PREGUNTA: (.+)$", re.S)


class FakeLLM:
    def generate(self, system: str, user: str, tier: ModelTier) -> Completion:
        chunk = _FIRST_CHUNK_RE.search(user)
        question = _QUESTION_RE.search(user)
        if not chunk or not question:
            text = NO_ANSWER
        else:
            q_tokens = set(terms(question.group(1)))
            lines = [ln.strip() for ln in chunk.group(1).splitlines() if ln.strip() and not ln.startswith("#")]
            best = max(lines, key=lambda ln: len(q_tokens & set(terms(ln))), default="")
            text = f"Según tus documentos: {best} [1]" if best else NO_ANSWER
        return Completion(
            text=text,
            model=f"fake-{tier.value}",
            input_tokens=(len(system) + len(user)) // 4,  # rough: ~4 chars per token
            output_tokens=len(text) // 4,
        )
