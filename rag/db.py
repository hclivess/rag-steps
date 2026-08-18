"""SQLite as both the document store and the vector store.

Vectors are stored as packed float32 BLOBs.  Similarity search is done by
registering a Python `cosine_similarity` function with SQLite so that the
whole nearest-neighbour query is a single SQL statement:

    SELECT ..., cosine_similarity(embedding, :query) AS score
    FROM chunks ORDER BY score DESC LIMIT :k

This is a linear scan (O(n)), which is perfectly fine for thousands of chunks
and makes the mechanics transparent.  For millions of vectors you would swap
in an index such as the sqlite-vec extension — the interface stays the same.

Keyword search uses SQLite's built-in FTS5 full-text index; combining it with
vector search gives "hybrid" retrieval.
"""
from __future__ import annotations

import array
import math
import sqlite3
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id        INTEGER PRIMARY KEY,
    source    TEXT UNIQUE NOT NULL,      -- file path, URL, ... anything unique
    title     TEXT,
    content   TEXT NOT NULL,
    metadata  TEXT                       -- free-form JSON
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,        -- order of the chunk inside its document
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL            -- float32 vector
);

-- Full-text index over chunk text (external-content table, kept in sync by triggers).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


def encode_vector(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def decode_vector(blob: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(blob)
    return a.tolist()


def cosine_similarity(a: bytes, b: bytes) -> float:
    va, vb = decode_vector(a), decode_vector(b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    source: str
    title: str | None
    text: str
    score: float


class Database:
    def __init__(self, path: str = "rag.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        # Expose the Python function to SQL.  `deterministic=True` lets SQLite optimise.
        self.conn.create_function("cosine_similarity", 2, cosine_similarity, deterministic=True)
        self.conn.executescript(SCHEMA)

    # ---- writes -----------------------------------------------------------
    def upsert_document(self, source: str, content: str, title: str | None = None,
                        metadata: str | None = None) -> int:
        """Insert or replace a document.  Replacing deletes its old chunks (CASCADE)."""
        with self.conn:
            self.conn.execute("DELETE FROM documents WHERE source = ?", (source,))
            cur = self.conn.execute(
                "INSERT INTO documents (source, title, content, metadata) VALUES (?, ?, ?, ?)",
                (source, title, content, metadata),
            )
            return cur.lastrowid

    def insert_chunks(self, document_id: int, chunks: list[str], embeddings: list[list[float]]) -> None:
        rows = [(document_id, i, text, encode_vector(vec))
                for i, (text, vec) in enumerate(zip(chunks, embeddings))]
        with self.conn:
            self.conn.executemany(
                "INSERT INTO chunks (document_id, position, text, embedding) VALUES (?, ?, ?, ?)", rows)

    # ---- reads ------------------------------------------------------------
    def vector_search(self, query_vec: list[float], k: int = 5) -> list[Hit]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.document_id, d.source, d.title, c.text,
                   cosine_similarity(c.embedding, ?) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            ORDER BY score DESC LIMIT ?
            """,
            (encode_vector(query_vec), k),
        ).fetchall()
        return [Hit(*r) for r in rows]

    def keyword_search(self, query: str, k: int = 5) -> list[Hit]:
        # FTS5 `bm25()` returns lower-is-better; negate so higher-is-better like cosine.
        rows = self.conn.execute(
            """
            SELECT c.id, c.document_id, d.source, d.title, c.text,
                   -bm25(chunks_fts) AS score
            FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid
                            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY score DESC LIMIT ?
            """,
            (_fts_query(query), k),
        ).fetchall()
        return [Hit(*r) for r in rows]

    def stats(self) -> dict:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks}

    def close(self) -> None:
        self.conn.close()


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: quoted terms OR-ed together."""
    terms = [t for t in "".join(ch if ch.isalnum() else " " for ch in text).split() if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in terms) or '""'
