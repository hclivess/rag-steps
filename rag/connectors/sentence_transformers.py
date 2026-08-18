"""Local, free, semantic embeddings via sentence-transformers.

    pip install sentence-transformers
"""
from __future__ import annotations


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import, optional dependency

        self.model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
