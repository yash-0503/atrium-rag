"""The local vector store: SQLite plus brute-force cosine.

A third implementation of the `VectorDBAdapter` protocol, alongside
Qdrant and pgvector. It exists so that the product works on a machine
with nothing running: no Docker, no database server, no service to
install first.

**Why SQLite and not an in-memory dict.** The API and the worker are
separate processes. A RAM store would mean the worker cannot see what
the API enqueued, and the corpus would disappear on restart. A file
backed by the standard library's `sqlite3` module survives both, needs
no dependency, and is readable by anything with a SQLite client.

**Why brute force.** The search is a cosine over every stored vector.
At the scale this product is aimed at — a personal corpus, a team's
documents, tens of thousands of chunks — that is milliseconds, and it
needs no index, no server and no tuning. Qdrant and pgvector are the
answers when the corpus outgrows it, and they stay supported.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["SqliteAdapter"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    ingest_run_id TEXT,
    text          TEXT,
    dim           INTEGER,
    vector        BLOB,
    payload       TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class SqliteAdapter:
    """A local, file-backed vector store.

    Implements the same protocol as `QdrantAdapter` and
    `PgVectorAdapter`, so every code path that talks to a vector store
    — ingest, search, the parent-child branch of `retrieve_and_rerank`,
    the retrieval lab — works against it unchanged.
    """

    name = "sqlite"

    def __init__(self, path: str = "atrium_local.db") -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # -- connection -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            db_path = Path(self.path).expanduser()
            if db_path.parent and str(db_path.parent) not in ("", "."):
                db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- collection -----------------------------------------------------

    def ensure_collection(self, name: str, dim: int) -> None:
        """Create the store, or check that it has the right dimension.

        A dimension disagreement is the same error the other adapters
        raise, and it names both numbers, because a silent mismatch
        shows up much later as nonsense search results.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'dim'"
        ).fetchone()
        if row is not None and row["value"]:
            existing = int(row["value"])
            if existing != int(dim):
                raise ValueError(
                    f"collection '{name}' exists with dim {existing}, but "
                    f"config requests {dim}"
                )
            return
        with self._lock:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('dim', ?)",
                (str(int(dim)),),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('collection', ?)",
                (name,),
            )
            conn.commit()

    def collection_schema_version(self) -> int:
        """The highest schema version among the stored chunks.

        Zero means a flat v0 corpus, which is what the flat chunkers
        produce — the same convention the other adapters use.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT payload FROM chunks LIMIT 200"
        ).fetchall()
        version = 0
        for item in row:
            try:
                payload = json.loads(item["payload"] or "{}")
            except (TypeError, ValueError):
                continue
            version = max(version, int(payload.get("schema_version", 0) or 0))
        return version

    # -- writing --------------------------------------------------------

    def upsert(self, chunks: Sequence[Any], batch_size: int = 100) -> None:
        import struct

        conn = self._connect()
        rows = []
        for chunk in chunks:
            vector = list(getattr(chunk, "embedding", None) or [])
            if not vector:
                continue
            payload = chunk.model_dump(exclude={"embedding"})
            rows.append((
                chunk.chunk_id,
                chunk.doc_id,
                getattr(chunk, "ingest_run_id", "") or "",
                chunk.text,
                len(vector),
                struct.pack(f"{len(vector)}f", *vector),
                json.dumps(payload, ensure_ascii=False, default=str),
            ))
        if not rows:
            return
        with self._lock:
            conn.executemany(
                "INSERT OR REPLACE INTO chunks"
                "(chunk_id, doc_id, ingest_run_id, text, dim, vector, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    def delete_stale_chunks(self, doc_id: str, current_run_id: str) -> None:
        """Drop this document's chunks that are not from the current run."""
        conn = self._connect()
        with self._lock:
            conn.execute(
                "DELETE FROM chunks WHERE doc_id = ? AND ingest_run_id != ?",
                (doc_id, current_run_id),
            )
            conn.commit()

    def replace_chunks(
        self,
        doc_id: str,
        chunks: Sequence[Any],
        current_run_id: str,
        batch_size: int = 100,
    ) -> None:
        """Write this version of the document, removing the previous one.

        Both statements run in one SQLite transaction, so the document
        is either the old version or the new one. Never a mix, and
        never two copies — which is what a crash between a separate
        upsert and a separate delete would leave behind.
        """
        import struct

        conn = self._connect()
        rows = []
        for chunk in chunks:
            vector = list(getattr(chunk, "embedding", None) or [])
            if not vector:
                continue
            payload = chunk.model_dump(exclude={"embedding"})
            rows.append((
                chunk.chunk_id,
                chunk.doc_id,
                getattr(chunk, "ingest_run_id", "") or "",
                chunk.text,
                len(vector),
                struct.pack(f"{len(vector)}f", *vector),
                json.dumps(payload, ensure_ascii=False, default=str),
            ))

        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "DELETE FROM chunks WHERE doc_id = ? AND ingest_run_id != ?",
                    (doc_id, current_run_id),
                )
                if rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO chunks"
                        "(chunk_id, doc_id, ingest_run_id, text, dim, vector, payload)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # -- reading --------------------------------------------------------

    def health(self) -> bool:
        try:
            conn = self._connect()
            conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone()
            return True
        except Exception:
            return False

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Flat v0 path: every chunk, scored, best first."""
        return self._search(query_vector, top_k, score_threshold, children_only=False)

    def search_children(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """v1 path: children only; parents are excluded."""
        return self._search(query_vector, top_k, score_threshold, children_only=True)

    def _search(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: Optional[float],
        children_only: bool,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        conn = self._connect()
        rows = conn.execute(
            "SELECT chunk_id, doc_id, text, dim, vector, payload FROM chunks"
        ).fetchall()
        if not rows:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.size == 0:
            return []

        # The vectors are stored as raw float32, so the whole corpus is
        # one buffer and the cosine is one matrix product. Doing this
        # in Python — unpacking and multiplying row by row — costs
        # 160 ms on a thousand chunks, which is the difference between
        # a search that feels instant and one that does not.
        keep = [
            index
            for index, row in enumerate(rows)
            if int(row["dim"] or 0) == query.size
        ]
        if not keep:
            return []
        buffer = b"".join(rows[index]["vector"] for index in keep)
        matrix = np.frombuffer(buffer, dtype=np.float32).reshape(len(keep), query.size)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query_norm = float(np.linalg.norm(query)) or 1.0
        scores = (matrix @ query).ravel() / (norms.ravel() * query_norm)

        scored: List[Dict[str, Any]] = []
        for position, index in enumerate(keep):
            row = rows[index]
            payload = json.loads(row["payload"] or "{}")
            if children_only and bool(payload.get("is_parent", False)):
                continue
            score = float(scores[position])
            if score_threshold is not None and score < float(score_threshold):
                continue
            scored.append({
                "id": row["chunk_id"],
                "score": float(score),
                "text": row["text"] or "",
                "doc_id": row["doc_id"] or "",
                "chunk_id": payload.get("chunk_id", row["chunk_id"]),
                "page_number": payload.get("page_number"),
                "section_path": payload.get("section_path") or [],
                "element_type": payload.get("element_type", ""),
                "is_parent": bool(payload.get("is_parent", False)),
                "parent_id": payload.get("parent_id"),
                "chunking_strategy": payload.get("chunking_strategy", ""),
            })

        scored.sort(key=lambda hit: hit["score"], reverse=True)
        return scored[: max(int(top_k), 0)]

    def retrieve_parents(
        self, parent_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """The parent chunks for the given ids, keyed by parent id."""
        if not parent_ids:
            return {}
        conn = self._connect()
        placeholders = ",".join("?" for _ in parent_ids)
        rows = conn.execute(
            f"SELECT chunk_id, payload FROM chunks WHERE chunk_id IN ({placeholders})",
            list(parent_ids),
        ).fetchall()
        parents: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            payload = json.loads(row["payload"] or "{}")
            parents[row["chunk_id"]] = {
                "id": row["chunk_id"],
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id", ""),
                "chunk_id": row["chunk_id"],
                "page_number": payload.get("page_number"),
                "section_path": payload.get("section_path") or [],
                "element_type": payload.get("element_type", ""),
                "is_parent": True,
                "parent_id": payload.get("parent_id"),
                "chunking_strategy": payload.get("chunking_strategy", ""),
            }
        return parents
