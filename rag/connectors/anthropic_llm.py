"""Answer generation with Claude via the official `anthropic` SDK.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
"""
from __future__ import annotations

import os


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 2048):
        import anthropic  # imported lazily so the package stays optional

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
        self.max_tokens = max_tokens

    def generate(self, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return "(the model declined to answer)"
        return "".join(block.text for block in response.content if block.type == "text").strip()
