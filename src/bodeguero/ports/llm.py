"""LLM port: the only way the core talks to a language model.

A port is an interface owned by the core. Adapters (OpenAI, Anthropic, a fake
for tests) implement it. Swapping providers = writing a new adapter and
changing one line in bootstrap.py. The core never changes.
"""

from __future__ import annotations

from typing import Protocol

from bodeguero.domain.models import Completion, ModelTier


class LLMPort(Protocol):
    def generate(self, system: str, user: str, tier: ModelTier) -> Completion:
        """Generate a completion for one system + user message pair.

        `tier` lets the core ask for a small or large model without knowing
        which concrete model names exist. The adapter maps tier -> model.
        """
        ...
