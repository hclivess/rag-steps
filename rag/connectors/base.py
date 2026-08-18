"""Connector interfaces.  Anything that satisfies these two Protocols plugs in.

    Embedder : text -> vector           (used for both indexing and querying)
    LLM      : prompt -> answer text    (used only at answer time)

They are `typing.Protocol`s (structural typing), so you do not have to inherit
from anything — just implement the methods.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text.  All vectors must have equal length."""
        ...


@runtime_checkable
class LLM(Protocol):
    def generate(self, system: str, prompt: str) -> str:
        """Return the model's answer for a system instruction + user prompt."""
        ...
