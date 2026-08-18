"""Pluggable embedding / LLM connectors.

`from_env()` picks connectors from environment variables so the CLI and demo
need no code changes to switch providers:

    RAG_EMBEDDER = hash (default) | openai | ollama | st
    RAG_LLM      = none (default) | anthropic | openai | ollama
    RAG_EMBED_MODEL / RAG_LLM_MODEL  override the model name
"""
from __future__ import annotations

import os

from .base import LLM, Embedder
from .local import ExtractiveLLM, HashEmbedder

__all__ = ["Embedder", "LLM", "HashEmbedder", "ExtractiveLLM", "from_env"]

OLLAMA_URL = "http://localhost:11434/v1"


def from_env() -> tuple[Embedder, LLM]:
    embedder_kind = os.getenv("RAG_EMBEDDER", "hash").lower()
    llm_kind = os.getenv("RAG_LLM", "none").lower()
    embed_model = os.getenv("RAG_EMBED_MODEL")
    llm_model = os.getenv("RAG_LLM_MODEL")

    if embedder_kind == "hash":
        embedder: Embedder = HashEmbedder()
    elif embedder_kind == "openai":
        from .openai_compat import OpenAIEmbedder
        embedder = OpenAIEmbedder(model=embed_model or "text-embedding-3-small")
    elif embedder_kind == "ollama":
        from .openai_compat import OpenAIEmbedder
        embedder = OpenAIEmbedder(model=embed_model or "nomic-embed-text", base_url=OLLAMA_URL)
    elif embedder_kind == "st":
        from .sentence_transformers import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(model=embed_model or "all-MiniLM-L6-v2")
    else:
        raise ValueError(f"unknown RAG_EMBEDDER={embedder_kind!r}")

    if llm_kind == "none":
        llm: LLM = ExtractiveLLM()
    elif llm_kind == "anthropic":
        from .anthropic_llm import AnthropicLLM
        llm = AnthropicLLM(model=llm_model)
    elif llm_kind == "openai":
        from .openai_compat import OpenAILLM
        llm = OpenAILLM(model=llm_model or "gpt-4o-mini")
    elif llm_kind == "ollama":
        from .openai_compat import OpenAILLM
        llm = OpenAILLM(model=llm_model or "llama3.2", base_url=OLLAMA_URL)
    else:
        raise ValueError(f"unknown RAG_LLM={llm_kind!r}")

    return embedder, llm
