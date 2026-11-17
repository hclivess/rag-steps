"""Dependency-free connectors so the demo runs offline with no API key.

They are intentionally simple — good enough to *see* retrieval work, not to
ship.  Swap them for real models via the other connectors in this package.
"""
from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")
# Very common words carry no topical signal; dropping them stops "is"/"the"
# from dominating the similarity score.
_STOPWORDS = frozenset("""
a an and are as at be but by do does for from has have how i if in into is it its
my of on or our so that the their there these they this to was we what when where
which who why will with you your
""".split())


class HashEmbedder:
    """Bag-of-words embedding via the "hashing trick".

    Each word (and each word bigram) is hashed to one of `dim` buckets; the
    vector counts hits per bucket and is L2-normalised.  It captures lexical
    overlap only — no semantics — but it needs zero dependencies and is
    deterministic, which makes it ideal for teaching and tests.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
        bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
        return words + bigrams

    def bucket(self, token: str) -> int:
        """The vector position a token lands in."""
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self._tokens(text):
                vec[self.bucket(tok)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class ExtractiveLLM:
    """A stand-in "LLM" that just returns the retrieved context.

    Useful to inspect exactly what a real model would have been given.
    """

    def generate(self, system: str, prompt: str) -> str:
        return "(no LLM configured — showing retrieved context)\n\n" + prompt
