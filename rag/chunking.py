"""Split long text into overlapping chunks.

Why chunk?  Embeddings represent one "idea" best when the text is short, and
LLM context windows are finite.  Overlap keeps sentences that straddle a
boundary from losing their context.
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Greedy paragraph-aware chunker.  Sizes are in characters, not tokens,
    to stay tokenizer-agnostic (roughly 4 chars ≈ 1 token)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # Paragraph fits into the current chunk -> append it.
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue
        # Flush what we have.
        if current:
            chunks.append(current)
        # Paragraph itself is too big -> hard-split it with overlap.
        if len(para) > chunk_size:
            start = 0
            while start < len(para):
                chunks.append(para[start : start + chunk_size])
                start += chunk_size - overlap
            current = ""
        else:
            # Start the next chunk with the tail of the previous one (overlap).
            tail = chunks[-1][-overlap:] if chunks and overlap else ""
            current = f"{tail}\n\n{para}" if tail else para

    if current:
        chunks.append(current)
    return chunks
