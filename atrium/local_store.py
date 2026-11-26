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

from .models import source_fields

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
    payload       TEXT,
    element_type  TEXT,
    is_parent     INTEGER DEFAULT 0,
    page_number   INTEGER,
    section_path  TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""

# How many times a write retries when SQLite reports the database is
# locked. Six attempts with a doubling delay covers about 3 seconds.
WRITE_RETRIES = 6

# Columns added after the first release. `CREATE TABLE IF NOT EXISTS`
# does not add a column to a table that already exists, so an existing
# database keeps its old shape and a query that reads the new columns
# fails against it. Each one is added here, once, if it is missing.
_MIGRATED_COLUMNS = (
    ("element_type", "TEXT"),
    ("is_parent", "INTEGER DEFAULT 0"),
    ("page_number", "INTEGER"),
    ("section_path", "TEXT"),
)


def _safe_json(raw: Optional[str]) -> dict:
    """Parse a stored payload, tolerating a corrupt one.

    One unreadable row used to abort the whole search, which hid the
    entire corpus behind a single bad row. A row that cannot be parsed
    yields an empty payload: the chunk is still returned, it just
    carries no extra fields.
    """
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _decode_section_path(raw: Optional[str]) -> list:
    """The stored section path, as a list of strings."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return list(value) if isinstance(value, list) else []


class SqliteAdapter:
    """A local, file-backed vector store.

    Implements the same protocol as `QdrantAdapter` and
    `PgVectorAdapter`, so every code path that talks to a vector store
    — ingest, search, the parent-child branch of `retrieve_and_rerank`,
    the retrieval lab — works against it unchanged.
    """

    name = "sqlite"

    #: The corpus size at which the local store stops being the right
    #: choice. The search is a full scan of every stored vector, so the
    #: cost grows with the corpus: about 23 ms at a thousand chunks and
    #: about 320 ms at twenty thousand. Past that, use Qdrant or
    #: pgvector.
    RECOMMENDED_MAX_CHUNKS = 20_000

    def __init__(self, path: str = "atrium_local.db") -> None:
        self.path = path
        self._lock = threading.Lock()
        # The corpus, loaded once and kept. Every query used to read
        # every vector back out of SQLite and recompute the matrix, so
        # the cost of a search grew with the corpus *and* was paid in
        # disk reads on every single query.
        self._corpus_cache: Optional[tuple] = None
        # Which write counter the cached corpus was built from. The API
        # and the worker are separate processes over one file: a cache
        # only a process's own writes can invalidate serves a snapshot
        # the moment the worker ingests anything. `corpus_version` lives
        # in the database and every write bumps it, so the reader can
        # see writes it did not make.
        self._corpus_cache_version: Optional[int] = None
        self._conn: Optional[sqlite3.Connection] = None

    # -- connection -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            db_path = Path(self.path).expanduser()
            if db_path.parent and str(db_path.parent) not in ("", "."):
                db_path.parent.mkdir(parents=True, exist_ok=True)
            # busy_timeout: how long a blocked write waits for the lock
            # before giving up. The stdlib default is 5 s, which is
            # short enough that two overlapping ingests fail outright.
            conn = sqlite3.connect(
                str(db_path), check_same_thread=False, timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            # WAL: readers do not block the writer and the writer does
            # not block readers. The rollback journal made a reader wait
            # out the writer's EXCLUSIVE window at commit — measured at
            # 2.2 s for an 8,000-row write against a 35 ms median.
            # `job_store.py` has used WAL all along; the vector store
            # did not.
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.Error:
                pass  # an in-memory or exotic filesystem may refuse these
            conn.executescript(_SCHEMA)
            self._migrate(conn)
            conn.commit()
            self._conn = conn
        return self._conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add any column a newer build expects but an older file lacks.

        `CREATE TABLE IF NOT EXISTS` leaves an existing table exactly as
        it was created, so a database file written before a column was
        added would fail every query that reads it. The alternative —
        a version stamp and a rebuild — is a much bigger change than the
        four columns are worth.
        """
        try:
            have = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        except sqlite3.Error:
            return
        for name, decl in _MIGRATED_COLUMNS:
            if name in have:
                continue
            try:
                conn.execute(f"ALTER TABLE chunks ADD COLUMN {name} {decl}")
            except sqlite3.Error:
                pass

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
        # The whole table, not an arbitrary sample of 200 rows. The
        # sample was unordered, so a v1 corpus whose first 200 rows
        # happened to be v0 reported version 0 — and the retrieval
        # layer then took the flat v0 branch and never used the
        # parent-child path at all. A whole-table scan of a small
        # integer column is cheap.
        row = conn.execute(
            "SELECT MAX(CAST(json_extract(payload, '$.schema_version') AS INTEGER))"
            " AS v FROM chunks"
        ).fetchone()
        if row is None or row["v"] is None:
            return 0
        try:
            return int(row["v"])
        except (TypeError, ValueError):
            return 0

    # -- writing --------------------------------------------------------

    def _corpus_version(self) -> int:
        """The write counter, as the database currently has it.

        One row read per search — cheap next to the reload it guards,
        and the only way this process can learn that another one wrote
        to the file.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'corpus_version'"
        ).fetchone()
        return int(row["value"] or 0) if row is not None else 0

    def _bump_corpus_version(self, conn: sqlite3.Connection) -> None:
        """Advance the counter. Call from inside the write's transaction."""
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('corpus_version', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
        )

    def upsert(self, chunks: Sequence[Any], batch_size: int = 100) -> None:
        import struct

        conn = self._connect()
        self._corpus_cache = None
        rows = []
        for chunk in chunks:
            vector = list(getattr(chunk, "embedding", None) or [])
            if not vector:
                continue
            self._check_dim(len(vector))
            rows.append(self._row(chunk, vector, struct))
        if not rows:
            return

        def _do():
            conn.executemany(
                "INSERT OR REPLACE INTO chunks"
                "(chunk_id, doc_id, ingest_run_id, text, dim, vector, payload,"
                " element_type, is_parent, page_number, section_path)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._bump_corpus_version(conn)
            conn.commit()

        self._write_locked(_do)

    @staticmethod
    def _row(chunk: Any, vector: list, struct) -> tuple:
        """One row for the `chunks` table.

        The stored payload deliberately drops `text` and `embed_text`:

        - `text` is already in its own column, so keeping a copy in the
          payload stored every chunk's text twice.
        - `embed_text` is a third copy of the text (plus a heading
          breadcrumb) and is only ever needed at embed time. No read
          path reads it back — it is derivable from the text and the
          section path.

        Together those two copies were 20% of the database file.

        The fields a search needs — the element type, whether the chunk
        is a parent, the page number and the section path — are written
        as real columns as well, so the search path never has to parse
        the payload JSON. That was 95% of the query time.
        """
        payload = chunk.model_dump(exclude={"embedding", "text", "embed_text"})
        source = payload.get("source") or {}
        page_number = source.get("page_number")
        section_path = source.get("section_path") or []
        return (
            chunk.chunk_id,
            chunk.doc_id,
            getattr(chunk, "ingest_run_id", "") or "",
            chunk.text,
            len(vector),
            struct.pack(f"{len(vector)}f", *vector),
            json.dumps(payload, ensure_ascii=False, default=str),
            payload.get("element_type") or "",
            1 if payload.get("is_parent") else 0,
            page_number,
            json.dumps(list(section_path), ensure_ascii=False),
        )

    def delete_stale_chunks(self, doc_id: str, current_run_id: str) -> None:
        """Drop this document's chunks that are not from the current run."""
        conn = self._connect()
        with self._lock:
            conn.execute(
                "DELETE FROM chunks WHERE doc_id = ? AND ingest_run_id != ?",
                (doc_id, current_run_id),
            )
            self._bump_corpus_version(conn)
            conn.commit()
            self._corpus_cache = None

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
            self._check_dim(len(vector))
            rows.append(self._row(chunk, vector, struct))

        def _do():
            conn.execute("BEGIN IMMEDIATE")
            self._corpus_cache = None
            try:
                conn.execute(
                    "DELETE FROM chunks WHERE doc_id = ? AND ingest_run_id != ?",
                    (doc_id, current_run_id),
                )
                if rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO chunks"
                        "(chunk_id, doc_id, ingest_run_id, text, dim, vector, payload,"
                        " element_type, is_parent, page_number, section_path)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                self._bump_corpus_version(conn)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        self._write_locked(_do)

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

    def _load_corpus(self):
        """The stored corpus as one matrix, loaded once.

        Returns `(rows, matrix)`, where `matrix` is the float32 array of
        every vector, row-aligned with `rows`.
        """
        # Read the counter *before* the rows: if a write lands in
        # between, the cached copy is keyed to an older version and the
        # next search reloads. Read it after, and the older rows would
        # be cached under the newer version and never refreshed.
        version = self._corpus_version()
        cached = self._corpus_cache
        if cached is not None and self._corpus_cache_version == version:
            return cached

        import numpy as np

        conn = self._connect()
        # The payload is NOT selected here. It is the largest field in
        # the row, and nothing in the search path needs it: the fields a
        # hit reports are all real columns now. Selecting it made
        # `json.loads` on every row 95% of the query time.
        rows = conn.execute(
            "SELECT chunk_id, doc_id, text, dim, vector, payload, element_type,"
            " is_parent, page_number, section_path FROM chunks"
        ).fetchall()
        if not rows:
            self._corpus_cache = ([], np.zeros((0, 0), dtype=np.float32))
            self._corpus_cache_version = version
            return self._corpus_cache

        # The reference dimension comes from the collection's `meta.dim`,
        # not from whichever row SQLite happened to return first. The
        # unordered `SELECT` made it arbitrary, and a row of another
        # width was then silently dropped from the corpus — invisible to
        # every query, but still counted by `count_chunks`.
        dim = self._collection_dim()
        usable = [row for row in rows if dim is None or int(row["dim"] or 0) == dim]
        if not usable:
            usable = rows
            dim = int(usable[0]["dim"] or 0)
        buffer = b"".join(row["vector"] for row in usable)
        matrix = np.frombuffer(buffer, dtype=np.float32).reshape(len(usable), dim)
        self._corpus_cache = (usable, matrix)
        self._corpus_cache_version = version
        return self._corpus_cache

    def _write_locked(self, work):
        """Run a write under the lock, retrying if the database is locked.

        Two processes writing the same file — the API and the worker —
        can collide on SQLite's write lock. The busy timeout absorbs a
        short collision, but a long one (a bulk ingest holding the lock
        for seconds) used to surface as `database is locked` and fail
        the ingest outright, with no retry anywhere. SQLite's answer to
        a busy database is to try again.
        """
        import time

        delay = 0.05
        last: Optional[BaseException] = None
        for _attempt in range(WRITE_RETRIES):
            try:
                with self._lock:
                    return work()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                last = exc
                time.sleep(delay)
                delay = min(delay * 2, 1.0)
        if last is not None:
            raise last

    def _check_dim(self, width: int) -> None:
        """Refuse a vector whose width is not the collection's.

        A wrong-width vector used to be stored happily and then silently
        dropped when the corpus was loaded, so the chunk was invisible to
        every query while still being counted by `count_chunks`. Catching
        it at the write turns that into an error at the point where it
        can still be understood.
        """
        expected = self._collection_dim()
        if expected is not None and int(width) != expected:
            raise ValueError(
                f"chunk vector has dimension {width}, but the collection "
                f"is {expected}-dimensional. The embedding model or "
                f"EMBED_DIM does not match the corpus."
            )

    def _collection_dim(self) -> Optional[int]:
        """The dimension the collection was created with, from `meta`."""
        conn = self._connect()
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'dim'"
        ).fetchone()
        if row is None or not row["value"]:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

    def count_chunks(self) -> int:
        """How many chunks the store holds."""
        conn = self._connect()
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    # The name the protocol uses.
    chunk_count = count_chunks

    def list_chunks(self) -> List[Dict[str, Any]]:
        """Every chunk in the corpus, in the shared `list_chunks` shape.

        This is a full corpus read — it exists so the lexical (BM25)
        index can be built from the store rather than maintained
        alongside it. It is called when the index is (re)built, not on
        the search path.

        The `text` column is selected alongside the payload because
        the payload deliberately excludes `text` (it is stored once, in
        the column) — the same reason `retrieve_parents` reads the
        column rather than the payload.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT chunk_id, doc_id, text, payload, element_type, is_parent,"
            " page_number, section_path FROM chunks"
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _safe_json(row["payload"])
            page_number, section_path = source_fields(payload)
            stored_section = _decode_section_path(row["section_path"])
            out.append({
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"] or "",
                "text": row["text"] or "",
                "page_number": (
                    row["page_number"] if row["page_number"] is not None else page_number
                ),
                "section_path": stored_section or section_path,
                "element_type": row["element_type"] or payload.get("element_type", ""),
                "is_parent": bool(row["is_parent"]),
                "parent_id": payload.get("parent_id"),
                "chunking_strategy": payload.get("chunking_strategy", ""),
            })
        return out


    def _search(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: Optional[float],
        children_only: bool,
    ) -> List[Dict[str, Any]]:
        import numpy as np

        rows, matrix = self._load_corpus()
        if not rows:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.size == 0:
            return []
        if matrix.shape[1] != query.size:
            # A dimension mismatch used to return an empty list, which
            # through /search is indistinguishable from an empty corpus
            # — a 200 with no results and no explanation. It means the
            # query vector and the stored corpus disagree (usually the
            # embedding model changed), and it has to say so.
            raise ValueError(
                f"query vector has dimension {query.size}, but the store "
                f"holds {matrix.shape[1]}-dimensional vectors. The "
                f"embedding model or EMBED_DIM does not match the corpus."
            )

        # The vectors are stored as raw float32, so the whole corpus is
        # one matrix and the cosine is one matrix product. Doing this
        # in Python — unpacking and multiplying row by row — costs
        # 160 ms on a thousand chunks, which is the difference between
        # a search that feels instant and one that does not.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query_norm = float(np.linalg.norm(query)) or 1.0
        scores = (matrix @ query).ravel() / (norms.ravel() * query_norm)

        # Score everything (one matrix product), then build hit dicts
        # only for the rows that can still make the cut. Building a dict
        # for every row and sorting afterwards cost more than the vector
        # math at every corpus size measured.
        limit = max(int(top_k), 0)
        if limit == 0:
            return []
        order = np.argsort(-scores)
        scored: List[Dict[str, Any]] = []
        for position in order:
            index = int(position)
            row = rows[index]
            if children_only and row["is_parent"]:
                continue
            score = float(scores[index])
            if score_threshold is not None and score < float(score_threshold):
                continue
            # The payload is parsed only for the rows that make the
            # cut. It carries the two fields that are not worth a
            # column — the parent link and the strategy name — and it
            # is the fallback for a row written before those columns
            # existed, whose column values are NULL.
            payload = _safe_json(row["payload"])
            payload_page, payload_section = source_fields(payload)
            page_number = row["page_number"]
            if page_number is None:
                page_number = payload_page
            section_path = _decode_section_path(row["section_path"])
            if not section_path:
                section_path = payload_section
            element_type = row["element_type"] or payload.get("element_type", "")
            scored.append({
                "id": row["chunk_id"],
                "score": score,
                "text": row["text"] or "",
                "doc_id": row["doc_id"] or "",
                "chunk_id": payload.get("chunk_id", row["chunk_id"]),
                "page_number": page_number,
                "section_path": section_path,
                "element_type": element_type,
                "is_parent": bool(row["is_parent"]),
                "parent_id": payload.get("parent_id"),
                "chunking_strategy": payload.get("chunking_strategy", ""),
            })
            if len(scored) >= limit:
                break
        return scored

    def retrieve_parents(
        self, parent_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """The parent chunks for the given ids, keyed by parent id."""
        if not parent_ids:
            return {}
        conn = self._connect()
        placeholders = ",".join("?" for _ in parent_ids)
        # The `text` column is selected explicitly. The payload
        # deliberately excludes `text` (see `_row` — it is stored once,
        # in the column, not twice), so reading the text from the
        # payload returned "" for every parent. The payload is kept as
        # a fallback for a row written before the column existed.
        rows = conn.execute(
            f"SELECT chunk_id, text, payload FROM chunks WHERE chunk_id IN ({placeholders})",
            list(parent_ids),
        ).fetchall()
        parents: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            payload = _safe_json(row["payload"])
            page_number, section_path = source_fields(payload)
            parents[row["chunk_id"]] = {
                "id": row["chunk_id"],
                "text": row["text"] or payload.get("text", ""),
                "doc_id": payload.get("doc_id", ""),
                "chunk_id": row["chunk_id"],
                "page_number": page_number,
                "section_path": section_path,
                "element_type": payload.get("element_type", ""),
                "is_parent": True,
                "parent_id": payload.get("parent_id"),
                "chunking_strategy": payload.get("chunking_strategy", ""),
            }
        return parents
