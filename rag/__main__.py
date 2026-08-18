"""Command-line interface.

    python -m rag ingest demo_data            # build rag.db from a folder
    python -m rag search "some question"      # show retrieved chunks
    python -m rag ask "some question"         # retrieve + generate an answer
    python -m rag stats
"""
from __future__ import annotations

import argparse
import textwrap

from .connectors import from_env
from .db import Database
from .pipeline import RAG


def main() -> None:
    p = argparse.ArgumentParser(prog="rag", description="Minimal RAG on SQLite")
    p.add_argument("--db", default="rag.db", help="SQLite file (default: rag.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="index all .md/.txt files in a folder")
    s.add_argument("folder")

    for name in ("search", "ask"):
        s = sub.add_parser(name)
        s.add_argument("query")
        s.add_argument("-k", type=int, default=4, help="number of chunks to retrieve")
        s.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")

    sub.add_parser("stats")
    args = p.parse_args()

    embedder, llm = from_env()
    rag = RAG(Database(args.db), embedder, llm)

    if args.cmd == "ingest":
        n = rag.add_directory(args.folder)
        print(f"done: {n} chunks -> {args.db}")
    elif args.cmd == "stats":
        print(rag.db.stats())
    elif args.cmd == "search":
        for i, h in enumerate(rag.retrieve(args.query, args.k, args.mode), 1):
            print(f"[{i}] score={h.score:.4f}  {h.source}")
            print(textwrap.indent(textwrap.shorten(h.text, 300), "    "))
    elif args.cmd == "ask":
        answer, hits = rag.ask(args.query, args.k, args.mode)
        print(answer)
        print("\nSources:")
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] {h.source} (score={h.score:.4f})")


if __name__ == "__main__":
    main()
