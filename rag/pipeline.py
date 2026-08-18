"""The RAG pipeline: ingest -> retrieve -> answer."""
from __future__ import annotations

import json
from pathlib import Path

from .chunking import chunk_text
from .connectors.base import LLM, Embedder
from .db import Database, Hit

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "Cite sources as [n] using the numbers in the context. "
    "If the context does not contain the answer, say you don't know."
)


class RAG:
    def __init__(self, db: Database, embedder: Embedder, llm: LLM,
                 chunk_size: int = 500, overlap: int = 80):
        self.db, self.embedder, self.llm = db, embedder, llm
        self.chunk_size, self.overlap = chunk_size, overlap

    # ---- indexing ---------------------------------------------------------
    def add_document(self, source: str, content: str, title: str | None = None,
                     metadata: dict | None = None) -> int:
        """Chunk + embed + store one document.  Returns number of chunks."""
        chunks = chunk_text(content, self.chunk_size, self.overlap)
        if not chunks:
            return 0
        embeddings = self.embedder.embed(chunks)
        doc_id = self.db.upsert_document(source, content, title,
                                         json.dumps(metadata) if metadata else None)
        self.db.insert_chunks(doc_id, chunks, embeddings)
        return len(chunks)

    def add_directory(self, folder: str | Path, patterns: tuple[str, ...] = ("*.md", "*.txt")) -> int:
        """Ingest every matching file under `folder` (recursively)."""
        folder = Path(folder)
        total = 0
        for pattern in patterns:
            for path in sorted(folder.rglob(pattern)):
                text = path.read_text(encoding="utf-8")
                n = self.add_document(str(path), text, title=path.stem.replace("_", " ").title())
                print(f"  indexed {path}  ({n} chunks)")
                total += n
        return total

    # ---- retrieval --------------------------------------------------------
    def retrieve(self, query: str, k: int = 4, mode: str = "hybrid") -> list[Hit]:
        """mode: 'vector' | 'keyword' | 'hybrid' (reciprocal-rank fusion of both)."""
        if mode == "vector":
            return self.db.vector_search(self.embedder.embed([query])[0], k)
        if mode == "keyword":
            return self.db.keyword_search(query, k)
        if mode == "hybrid":
            vec = self.db.vector_search(self.embedder.embed([query])[0], k * 2)
            kw = self.db.keyword_search(query, k * 2)
            return _reciprocal_rank_fusion([vec, kw])[:k]
        raise ValueError(f"unknown mode {mode!r}")

    # ---- generation -------------------------------------------------------
    def build_prompt(self, question: str, hits: list[Hit]) -> str:
        context = "\n\n".join(f"[{i}] ({h.source})\n{h.text}" for i, h in enumerate(hits, 1))
        return f"Context:\n{context}\n\nQuestion: {question}"

    def ask(self, question: str, k: int = 4, mode: str = "hybrid") -> tuple[str, list[Hit]]:
        hits = self.retrieve(question, k, mode)
        answer = self.llm.generate(SYSTEM_PROMPT, self.build_prompt(question, hits))
        return answer, hits


def _reciprocal_rank_fusion(rankings: list[list[Hit]], c: int = 60) -> list[Hit]:
    """Merge several ranked lists.  Each item scores sum(1 / (c + rank));
    items appearing high in *both* lists win.  Scores from different retrievers
    are not comparable, so RRF only uses ranks."""
    scores: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (c + rank)
            by_id.setdefault(hit.chunk_id, hit)
    fused = []
    for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        h = by_id[cid]
        fused.append(Hit(h.chunk_id, h.document_id, h.source, h.title, h.text, round(score, 4)))
    return fused
