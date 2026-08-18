import sqlite3
import unittest

from rag import RAG, Database
from rag.chunking import chunk_text
from rag.connectors import ExtractiveLLM, HashEmbedder
from rag.db import cosine_similarity, decode_vector, encode_vector


class ChunkingTests(unittest.TestCase):
    def test_paragraphs_grouped_and_split(self):
        text = "\n\n".join(["para one " * 10, "para two " * 10, "x" * 1200])
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        self.assertGreater(len(chunks), 3)

    def test_overlap_validation(self):
        with self.assertRaises(ValueError):
            chunk_text("abc", chunk_size=10, overlap=10)


class VectorTests(unittest.TestCase):
    def test_roundtrip_and_cosine(self):
        a, b = encode_vector([1.0, 0.0]), encode_vector([1.0, 0.0])
        self.assertEqual(decode_vector(a), [1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0, places=6)
        self.assertAlmostEqual(cosine_similarity(a, encode_vector([0.0, 1.0])), 0.0, places=6)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rag = RAG(Database(":memory:"), HashEmbedder(dim=256), ExtractiveLLM())
        self.rag.add_document("coffee", "Espresso needs a fine grind and hot water.\n\nStore beans airtight.")
        self.rag.add_document("bread", "Sourdough starter is fed with flour and water daily.")

    def test_vector_keyword_hybrid_agree_on_obvious_query(self):
        for mode in ("vector", "keyword", "hybrid"):
            hits = self.rag.retrieve("how do I feed a sourdough starter", k=1, mode=mode)
            self.assertEqual(hits[0].source, "bread", mode)

    def test_upsert_replaces_chunks(self):
        self.rag.add_document("coffee", "Completely new coffee text.")
        self.assertEqual(self.rag.db.stats(), {"documents": 2, "chunks": 2})

    def test_ask_returns_answer_and_sources(self):
        answer, hits = self.rag.ask("espresso grind?", k=1)
        self.assertIn("fine grind", answer)
        self.assertEqual(hits[0].source, "coffee")


if __name__ == "__main__":
    unittest.main()
