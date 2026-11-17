"""Using rag-steps as a library (no CLI)."""
from rag import RAG, Database
from rag.connectors import from_env

embedder, llm = from_env()                      # or build connectors yourself
rag = RAG(Database("rag.db"), embedder, llm)

rag.add_directory("demo_data")
rag.add_document("inline-note", "The office coffee machine is cleaned every Friday.")

answer, hits = rag.ask("How often is the coffee machine cleaned?", k=3)
print(answer)
for i, h in enumerate(hits, 1):
    print(f"[{i}] {h.source}  score={h.score:.3f}")
