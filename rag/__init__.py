"""Minimal RAG (Retrieval-Augmented Generation) on top of SQLite.

Pipeline:  documents -> chunks -> embeddings -> SQLite  ...  question -> retrieve -> LLM answer
"""
from .pipeline import RAG
from .db import Database

__all__ = ["RAG", "Database"]
