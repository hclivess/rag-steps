"""Local web interface:  python -m rag serve

A FastAPI app, started with uvicorn from inside this module (no separate
`uvicorn ...` command needed).  It serves the page in `rag/webui/` (plain
JavaScript, no build step) and a small JSON API.

Long operations (ingest, ask) answer with newline-delimited JSON: one JSON
object per line, sent as soon as a step is finished (see `rag/steps.py`).
The page reads the stream and lights up the matching step.

Needs:  pip install fastapi uvicorn      (the core stays dependency-free)
"""
from __future__ import annotations

import json
import queue
import threading
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .connectors.base import LLM, Embedder
from .db import Database
from .steps import (ask_events, collect_documents, describe_chunks, ingest_events, list_files,
                    read_settings, state, title_of)

WEBUI = Path(__file__).parent / "webui"
_END = object()


def _ndjson(db_path: str, make_events) -> StreamingResponse:
    """Stream one JSON line per event.

    The whole pipeline runs in one worker thread, because a SQLite connection
    must stay in the thread that opened it; a queue hands the events over to
    the response as they are produced.
    """
    events: queue.Queue = queue.Queue()

    def work() -> None:
        db = Database(db_path)
        try:
            for event in make_events(db):
                events.put(event)
        except Exception as exc:                # report it in the stream
            events.put({"step": "error", "message": str(exc)})
        finally:
            db.close()
            events.put(_END)

    def lines():
        threading.Thread(target=work, daemon=True).start()
        while (event := events.get()) is not _END:
            yield json.dumps(event) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store"})


def create_app(db_path: str, embedder: Embedder, llm: LLM) -> FastAPI:
    app = FastAPI(title="rag-sqlite", docs_url="/api/docs", redoc_url=None)

    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    @app.exception_handler(OSError)
    async def bad_request(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=400)

    # ---- the page ----
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEBUI / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/app.js", include_in_schema=False)
    def app_js():
        return FileResponse(WEBUI / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})

    @app.get("/style.css", include_in_schema=False)
    def style_css():
        return FileResponse(WEBUI / "style.css", headers={"Cache-Control": "no-store"})

    # ---- the API ----
    @app.get("/api/state")
    def get_state():
        db = Database(db_path)
        try:
            return state(db, db_path, embedder, llm)
        finally:
            db.close()

    @app.post("/api/files")
    def post_files(body: dict = Body(...)):
        """List the documents of a folder."""
        return {"files": [{"source": str(p), "title": title_of(p), "chars": p.stat().st_size}
                          for p in list_files(body.get("folder", ""))]}

    @app.post("/api/chunk")
    def post_chunk(body: dict = Body(...)):
        """Live preview: chunk one document with the given settings, store nothing."""
        chunk_size, overlap = read_settings(body)
        if "text" in body:
            text = str(body["text"])
        else:
            allowed = {str(p): p for p in list_files(body.get("folder", ""))}
            text = allowed[body["source"]].read_text(encoding="utf-8")
        return {"chars": len(text), "chunks": describe_chunks(text, chunk_size, overlap)}

    @app.post("/api/ingest")
    def post_ingest(body: dict = Body(...)):
        """Read, chunk, embed and store; streams one event per step."""
        chunk_size, overlap = read_settings(body)
        docs = collect_documents(body)
        return _ndjson(db_path, lambda db: ingest_events(db, embedder, docs, chunk_size, overlap))

    @app.post("/api/ask")
    def post_ask(body: dict = Body(...)):
        """Embed the question, search, fuse, build the prompt, generate; streams one event per step."""
        question = str(body.get("question", "")).strip()
        if not question:
            raise ValueError("the question is empty")
        k, mode = max(1, int(body.get("k", 4))), str(body.get("mode", "hybrid"))
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValueError(f"unknown mode {mode!r}")
        return _ndjson(db_path, lambda db: ask_events(db, embedder, llm, question, k, mode))

    @app.post("/api/reset")
    def post_reset():
        db = Database(db_path)
        try:
            with db.conn:
                db.conn.execute("DELETE FROM documents")    # chunks + FTS rows follow (CASCADE, trigger)
            return {"stats": db.stats()}
        finally:
            db.close()

    return app


def serve(db_path: str, embedder: Embedder, llm: LLM, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    """Start uvicorn in this process and (optionally) open the page."""
    url = f"http://{host}:{port}/"
    print(f"rag-sqlite web interface: {url}   (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(db_path, embedder, llm), host=host, port=port, log_level="warning",
                ws="none")                      # no websockets here: plain HTTP streaming
