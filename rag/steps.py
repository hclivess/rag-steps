"""The two pipelines, one step at a time.

`ingest_events` and `ask_events` run the *same* building blocks as the CLI —
`chunk_text`, the embedder, `Database`, reciprocal-rank fusion — but yield a
small dict after every finished step, describing what the step just did.
The web interface (`rag/server.py`) streams these to the browser.

Standard library only, like the rest of the core.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from .chunking import chunk_text
from .connectors.base import LLM, Embedder
from .connectors.local import HashEmbedder
from .db import Database, _fts_query
from .pipeline import RAG, SYSTEM_PROMPT, _reciprocal_rank_fusion

PATTERNS = ("*.md", "*.txt")          # same file types as RAG.add_directory
MAX_TOKENS_SHOWN = 48


# ---- helpers that describe a step ---------------------------------------------
def connector_name(obj) -> str:
    model = getattr(obj, "model", None)
    return f"{type(obj).__name__} ({model})" if isinstance(model, str) else type(obj).__name__


def list_files(folder: str) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"not a folder: {folder}")
    return [p for pattern in PATTERNS for p in sorted(root.rglob(pattern))]


def title_of(path: Path) -> str:
    return path.stem.replace("_", " ").title()


def overlap_length(previous: str, current: str, limit: int) -> int:
    """How many leading characters of `current` repeat the end of `previous`."""
    for n in range(min(limit, len(previous), len(current)), 0, -1):
        if previous.endswith(current[:n]):
            return n
    return 0


def describe_chunks(text: str, chunk_size: int, overlap: int) -> list[dict]:
    chunks = chunk_text(text, chunk_size, overlap)
    out = []
    for i, chunk in enumerate(chunks):
        out.append({
            "position": i,
            "text": chunk,
            "chars": len(chunk),
            "overlap": overlap_length(chunks[i - 1], chunk, overlap) if i else 0,
        })
    return out


def describe_vector(embedder: Embedder, text: str, vector: list[float]) -> dict:
    info = {
        "dim": len(vector),
        "nonzero": sum(1 for v in vector if v),
        "vector": [round(v, 4) for v in vector],
    }
    if isinstance(embedder, HashEmbedder):
        # The hash embedder is simple enough to show completely: token -> bucket.
        tokens = embedder._tokens(text)
        info["token_count"] = len(tokens)
        info["tokens"] = [{"token": t, "bucket": embedder.bucket(t)} for t in tokens[:MAX_TOKENS_SHOWN]]
    return info


def read_settings(body: dict) -> tuple[int, int]:
    chunk_size, overlap = int(body.get("chunk_size", 500)), int(body.get("overlap", 80))
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    return chunk_size, overlap


def collect_documents(body: dict) -> list[dict]:
    """Files of the folder plus any text pasted into the page."""
    docs = []
    if body.get("folder"):
        for path in list_files(body["folder"]):
            docs.append({"source": str(path), "title": title_of(path), "path": path})
    for d in body.get("documents") or []:
        if str(d.get("text", "")).strip():
            source = str(d.get("source") or "pasted-text")
            docs.append({"source": source, "title": d.get("title") or source, "text": str(d["text"])})
    return docs


def state(db: Database, db_path: str, embedder: Embedder, llm: LLM) -> dict:
    rows = db.conn.execute(
        """
        SELECT d.id, d.source, d.title, LENGTH(d.content) AS chars, COUNT(c.id) AS chunks
        FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id ORDER BY d.id
        """
    ).fetchall()
    here = Path("demo_data")
    return {
        "db": db_path,
        "stats": db.stats(),
        "embedder": connector_name(embedder),
        "llm": connector_name(llm),
        "documents": [dict(r) for r in rows],
        "defaults": {
            "folder": str(here if here.is_dir() else Path(__file__).parent.parent / "demo_data"),
            "chunk_size": 500,
            "overlap": 80,
        },
    }


# ---- the two pipelines, one event per finished step ---------------------------
def ingest_events(db: Database, embedder: Embedder, docs: list[dict], chunk_size: int, overlap: int):
    started = time.perf_counter()
    batch = 1 if isinstance(embedder, HashEmbedder) else 16     # API embedders: fewer round trips
    yield {"step": "plan", "embedder": connector_name(embedder),
           "files": [{"source": d["source"], "title": d["title"]} for d in docs]}
    total = 0
    for i, doc in enumerate(docs):
        # 1. read
        text = doc["text"] if "text" in doc else doc["path"].read_text(encoding="utf-8")
        yield {"step": "read", "doc": i, "source": doc["source"], "title": doc["title"], "chars": len(text),
               "paragraphs": sum(1 for p in text.split("\n\n") if p.strip())}
        # 2. chunk
        chunks = describe_chunks(text, chunk_size, overlap)
        yield {"step": "chunk", "doc": i, "chunks": chunks}
        if not chunks:
            yield {"step": "store", "doc": i, "source": doc["source"], "rows": 0, "skipped": True}
            continue
        # 3. embed
        texts = [c["text"] for c in chunks]
        vectors: list[list[float]] = []
        for at in range(0, len(texts), batch):
            part = embedder.embed(texts[at:at + batch])
            for offset, vector in enumerate(part):
                position = at + offset
                yield {"step": "embed", "doc": i, "position": position, "of": len(texts),
                       **describe_vector(embedder, texts[position], vector)}
            vectors.extend(part)
        # 4. store
        doc_id = db.upsert_document(doc["source"], text, doc["title"])
        db.insert_chunks(doc_id, texts, vectors)
        total += len(texts)
        yield {"step": "store", "doc": i, "source": doc["source"], "document_id": doc_id, "rows": len(texts),
               "blob_bytes": len(vectors[0]) * 4, "stats": db.stats()}
    yield {"step": "done", "chunks": total, "documents": len(docs), "stats": db.stats(),
           "ms": round((time.perf_counter() - started) * 1000, 1)}


def ask_events(db: Database, embedder: Embedder, llm: LLM, question: str, k: int, mode: str):
    if mode not in ("vector", "keyword", "hybrid"):
        raise ValueError(f"unknown mode {mode!r}")
    rag = RAG(db, embedder, llm)
    limit = k * 2 if mode == "hybrid" else k        # hybrid over-fetches, then fuses down to k
    vec_hits, kw_hits = [], []

    if mode == "keyword":
        yield {"step": "embed_query", "skipped": True}
        yield {"step": "vector_search", "skipped": True}
    else:
        query_vector = embedder.embed([question])[0]
        yield {"step": "embed_query", **describe_vector(embedder, question, query_vector)}
        t = time.perf_counter()
        vec_hits = db.vector_search(query_vector, limit)
        yield {"step": "vector_search", "limit": limit, "scanned": db.stats()["chunks"],
               "ms": round((time.perf_counter() - t) * 1000, 1), "hits": [asdict(h) for h in vec_hits]}

    if mode == "vector":
        yield {"step": "keyword_search", "skipped": True}
    else:
        t = time.perf_counter()
        kw_hits = db.keyword_search(question, limit)
        yield {"step": "keyword_search", "limit": limit, "fts_query": _fts_query(question),
               "ms": round((time.perf_counter() - t) * 1000, 1), "hits": [asdict(h) for h in kw_hits]}

    if mode == "hybrid":
        fused = _reciprocal_rank_fusion([vec_hits, kw_hits])
        vec_rank = {h.chunk_id: r for r, h in enumerate(vec_hits)}
        kw_rank = {h.chunk_id: r for r, h in enumerate(kw_hits)}
        hits = fused[:k]
        yield {"step": "fusion", "c": 60, "k": k, "rows": [
            {"chunk_id": h.chunk_id, "source": h.source, "score": h.score, "kept": i < k,
             "vector_rank": vec_rank.get(h.chunk_id), "keyword_rank": kw_rank.get(h.chunk_id)}
            for i, h in enumerate(fused)]}
    else:
        hits = vec_hits or kw_hits
        yield {"step": "fusion", "skipped": True}

    prompt = rag.build_prompt(question, hits)
    yield {"step": "prompt", "system": SYSTEM_PROMPT, "prompt": prompt, "hits": [asdict(h) for h in hits]}
    t = time.perf_counter()
    answer = llm.generate(SYSTEM_PROMPT, prompt)
    yield {"step": "answer", "text": answer, "llm": connector_name(llm),
           "ms": round((time.perf_counter() - t) * 1000, 1)}

