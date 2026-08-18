"""Connectors for any OpenAI-compatible HTTP API (OpenAI, Ollama, LM Studio,
vLLM, LiteLLM, ...).  Uses only the standard library.

Environment variables (all optional):
    OPENAI_BASE_URL   default https://api.openai.com/v1   (Ollama: http://localhost:11434/v1)
    OPENAI_API_KEY    default "none"  (local servers usually ignore it)
"""
from __future__ import annotations

import json
import os
import urllib.request


def _post(path: str, payload: dict, base_url: str | None, api_key: str | None) -> dict:
    base = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
    key = api_key or os.getenv("OPENAI_API_KEY", "none")
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


class OpenAIEmbedder:
    """POST /embeddings.  Default model works with OpenAI; for Ollama use e.g.
    `nomic-embed-text` (run `ollama pull nomic-embed-text` first)."""

    def __init__(self, model: str = "text-embedding-3-small", base_url: str | None = None,
                 api_key: str | None = None, batch_size: int = 64):
        self.model, self.base_url, self.api_key, self.batch_size = model, base_url, api_key, batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            data = _post("/embeddings", {"model": self.model, "input": batch}, self.base_url, self.api_key)
            # API may return items out of order -> sort by index.
            vectors.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        return vectors


class OpenAILLM:
    """POST /chat/completions."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None, api_key: str | None = None):
        self.model, self.base_url, self.api_key = model, base_url, api_key

    def generate(self, system: str, prompt: str) -> str:
        data = _post(
            "/chat/completions",
            {"model": self.model,
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
            self.base_url, self.api_key,
        )
        return data["choices"][0]["message"]["content"].strip()
