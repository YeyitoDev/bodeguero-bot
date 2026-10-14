"""Anthropic (Claude) adapter for the LLM port.

TODO(yee): implement this adapter, same contract as openai_llm.py.

Differences worth knowing (good interview material):
  - the system prompt is a top-level `system` parameter, not a message;
  - `max_tokens` is required;
  - usage comes back as usage.input_tokens / usage.output_tokens;
  - prompt caching can cut cost a lot when the system prompt is long and repeated.

Having two adapters for one port is the best live demo of why hexagonal
architecture matters: switch provider by changing LLM_PROVIDER in .env.
Check the current model names and SDK usage in the official docs.
"""

from __future__ import annotations

from bodeguero.domain.models import Completion, ModelTier


class AnthropicLLM:
    def __init__(self, api_key: str, small_model: str, large_model: str) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        self._models = {ModelTier.SMALL: small_model, ModelTier.LARGE: large_model}
        # TODO(yee): self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system: str, user: str, tier: ModelTier) -> Completion:
        raise NotImplementedError("TODO(yee): implement AnthropicLLM.generate (see module docstring)")
