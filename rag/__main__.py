"""Command-line interface.

    python -m rag ingest demo_data            # build rag.db from a folder
    python -m rag search "some question"      # show retrieved chunks
    python -m rag ask "some question"         # retrieve + generate an answer
    python -m rag stats
    python -m rag serve                       # web interface on http://127.0.0.1:8765
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

    s = sub.add_parser("serve", help="local web interface that shows every step")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = p.parse_args()

    embedder, llm = from_env()
    if args.cmd == "serve":
        try:
            from .server import serve
        except ImportError as exc:
            raise SystemExit(f"the web interface needs FastAPI and uvicorn ({exc}):\n"
                             "    pip install fastapi uvicorn")
        serve(args.db, embedder, llm, args.host, args.port, open_browser=not args.no_open)
        return
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
