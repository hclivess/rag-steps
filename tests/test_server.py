"""The web interface, tested over real HTTP against a uvicorn started in a thread."""
import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from rag.connectors import ExtractiveLLM, HashEmbedder
from rag.steps import describe_chunks, overlap_length

try:
    import uvicorn
    from rag.server import create_app
except ImportError:                     # fastapi / uvicorn are optional
    uvicorn = None

DEMO = str(Path(__file__).parent.parent / "demo_data")


class StepTests(unittest.TestCase):
    def test_overlap_is_found(self):
        self.assertEqual(overlap_length("hello world", "world again", 80), 5)
        self.assertEqual(overlap_length("hello", "there", 80), 0)

    def test_described_chunks_carry_their_overlap(self):
        text = "\n\n".join(f"paragraph {i} " * 12 for i in range(6))
        chunks = describe_chunks(text, 300, 50)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0]["overlap"], 0)
        self.assertTrue(any(c["overlap"] == 50 for c in chunks[1:]))
        for prev, cur in zip(chunks, chunks[1:]):
            self.assertTrue(prev["text"].endswith(cur["text"][:cur["overlap"]]))


@unittest.skipUnless(uvicorn, "fastapi and uvicorn are not installed")
class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        app = create_app(str(Path(cls.tmp.name) / "test.db"), HashEmbedder(), ExtractiveLLM())
        cls.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error",
                                                ws="none"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 10
        while not cls.server.started and time.time() < deadline:
            time.sleep(0.02)
        cls.base = f"http://127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.tmp.cleanup()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            return res.read().decode()

    def events(self, path, body):
        return [json.loads(line) for line in self.post(path, body).splitlines() if line.strip()]

    def test_page_is_served(self):
        with urllib.request.urlopen(self.base + "/") as res:
            self.assertIn("rag-sqlite", res.read().decode())
        with urllib.request.urlopen(self.base + "/app.js") as res:
            self.assertIn("/api/ingest", res.read().decode())

    def test_ingest_then_ask_reports_every_step(self):
        events = self.events("/api/ingest", {"folder": DEMO, "chunk_size": 500, "overlap": 80})
        steps = [e["step"] for e in events]
        self.assertEqual(steps[0], "plan")
        self.assertEqual(steps[-1], "done")
        self.assertEqual(steps.count("read"), 3)
        self.assertEqual(steps.count("store"), 3)
        done = events[-1]
        self.assertEqual(steps.count("embed"), done["chunks"])
        self.assertEqual(done["stats"], {"documents": 3, "chunks": done["chunks"]})
        embed = next(e for e in events if e["step"] == "embed")
        self.assertEqual(len(embed["vector"]), embed["dim"])
        self.assertTrue(embed["tokens"])

        events = self.events("/api/ask", {"question": "what water temperature for coffee", "k": 3,
                                          "mode": "hybrid"})
        self.assertEqual([e["step"] for e in events],
                         ["embed_query", "vector_search", "keyword_search", "fusion", "prompt", "answer"])
        by_step = {e["step"]: e for e in events}
        self.assertEqual(len(by_step["prompt"]["hits"]), 3)
        self.assertIn("coffee", by_step["prompt"]["hits"][0]["source"])
        self.assertEqual(sum(r["kept"] for r in by_step["fusion"]["rows"]), 3)
        self.assertIn("Question: what water temperature for coffee", by_step["answer"]["text"])

        events = self.events("/api/ask", {"question": "low light plants", "k": 2, "mode": "keyword"})
        self.assertTrue(events[0]["skipped"] and events[1]["skipped"] and events[3]["skipped"])

    def test_preview_stores_nothing_and_bad_input_is_refused(self):
        data = json.loads(self.post("/api/chunk", {"text": "one\n\n" + "two " * 100, "chunk_size": 120,
                                                   "overlap": 20}))
        self.assertGreater(len(data["chunks"]), 2)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/api/chunk", {"text": "abc", "chunk_size": 10, "overlap": 10})
        self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post("/api/files", {"folder": "no-such-folder-here"})
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
