"""
atrium — Universal RAG ingestion engine.

A FastAPI-based document ingestion service that parses, chunks, embeds,
and stores PDFs, Office documents, HTML, Markdown, and more into a
vector database (Qdrant or pgvector). Retrieval is via a unified
helper that supports optional reranking, parent-child chunking, and
hybrid (vector + BM25) fusion.

Quickstart::

    from atrium.pipeline import run_ingestion
    from atrium.adapters import QdrantAdapter
    from atrium.embedder_config import build_embedder

    db = QdrantAdapter(host="localhost", port=6333)
    embedder = build_embedder()
    db.ensure_collection("documents", embedder.dimension)
    # ... pass db and embedder into run_ingestion(...) ...

The version constant is the single source of truth — packaged in
pyproject.toml and exposed here for runtime introspection.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Load `.env` into the process environment exactly once, here — before
# any submodule reads a setting. This is the fix for a real hole:
# `uvicorn atrium.main:app` used to ignore `.env` completely (there is
# no python-dotenv dependency), so an operator who set `API_KEY` in
# `.env` and booted the server with uvicorn directly got a server with
# no authentication on any route.
#
# Importing `atrium.config` here — rather than in each submodule —
# guarantees the load happens first no matter which module the
# operator imported.
from atrium.config import load_env_file as _load_env_file

_load_env_file()

__all__ = ["__version__"]