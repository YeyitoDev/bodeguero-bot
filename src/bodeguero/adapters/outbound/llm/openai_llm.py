"""OpenAI adapter for the LLM port.

TODO(yee): implement this adapter. It's yours to write, on purpose: this is
where you learn the API details you'll be asked about in interviews.

Steps:
  1. `pip install openai` and add it to requirements.txt.
  2. In __init__, create `openai.OpenAI(api_key=...)` once and keep it
     (one client per process; it pools connections).
  3. Map tier -> model name using the settings (LLM_SMALL_MODEL / LLM_LARGE_MODEL).
  4. Call the Chat Completions (or Responses) API with the system + user
     messages and a LOW temperature (0-0.2): we want grounded facts, not creativity.
  5. Return a Completion with text, model and usage.prompt_tokens /
     usage.completion_tokens, so cost can be tracked.
  6. Handle failures: retry transient errors (429, 5xx, timeouts) with
     exponential backoff, and set a request timeout.

Check the current model names and SDK usage in the official docs before
you start. They change often.
"""

from __future__ import annotations

from bodeguero.domain.models import Completion, ModelTier


class OpenAILLM:
    def __init__(self, api_key: str, small_model: str, large_model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        self._models = {ModelTier.SMALL: small_model, ModelTier.LARGE: large_model}
        # TODO(yee): self._client = openai.OpenAI(api_key=api_key, timeout=30)

    def generate(self, system: str, user: str, tier: ModelTier) -> Completion:
        raise NotImplementedError("TODO(yee): implement OpenAILLM.generate (see module docstring)")
