"""
job_store.py

SQLite-backed job state, replacing the in-memory dict that previously lived
in atrium/main.py. Persists across server restart, which is the difference
between "demo" and "deployable" — without it, every restart silently
loses the status of every in-flight or recent /ingest job.

Two kinds of job live in the same table
----------------------------------------
**Inline jobs** (``/ingest``) are created with
``create()`` and processed by the API process itself. They carry no
payload and are never handed to a worker.

**Queued jobs** (``POST /jobs``) are created with ``enqueue()``. They
carry a payload (the document bytes plus the per-request options) and
sit in ``pending`` until a worker claims one. This is the contract
``atrium.worker`` has always assumed — it calls ``claim_next()`` and
``release_stale_claims()``, which did not exist until this rewrite.
The four v0 contract tests in ``tests/test_v0_suite.py`` were the spec:
``POST /jobs`` returns 202 immediately with a job_id, rejects oversized
bodies with 413, and rejects a declared-type/detected-type mismatch
with 415.

Schema is intentionally minimal — one table, no migrations beyond
``CREATE TABLE IF NOT EXISTS`` — because the only consumers are
``/jobs``, ``/jobs/{id}`` and the worker, none of which needs
historical queries or analytics. If you ever need "all jobs for this
doc_id" or "failures in the last 24h", graduate this to Postgres and
add an index; don't try to scale SQLite beyond what its single-writer
model supports.

Concurrency
-----------
SQLite allows exactly one writer at a time. ``claim_next`` therefore
runs inside a ``BEGIN IMMEDIATE`` transaction so two workers cannot
claim the same row: the second one blocks on the write lock until the
first commits, then sees the row already claimed and moves on. The
connection is opened with ``isolation_level=None`` (autocommit) so
that this explicit transaction is the only one in play and the read
outside it can't be accidentally promoted into a write.

Thread-safety: ``check_same_thread=False`` plus a single connection
serialized through a lock is sufficient for FastAPI's threading model
(the API's inline path and the worker's claim loop both touch the
store, from different threads in the worker).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

# Job statuses. `pending` → `running` → `complete` | `failed`, with
# `pending` again when a stale claim is released back to the queue.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"


#: The columns `JobStore.update` may write. The names are
#: interpolated into the statement, so the list is a whitelist, not a
#: convention.
_UPDATABLE_COLUMNS = frozenset({
    "status",
    "doc_id",
    "chunks_written",
    "error",
    "ocr_failures",
    "attempts",
    "claimed_by",
    "claimed_at",
    "payload",
})


class JobStore:
    """Thin SQLite wrapper around a single `jobs` table."""

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        # isolation_level=None → autocommit. Explicit BEGIN IMMEDIATE in
        # claim_next is then the only transaction, which is what makes
        # the claim atomic across processes.
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL lets a reader and a writer work at the same time, and the
        # busy timeout makes a contended write *wait* instead of raising
        # "database is locked" — which used to kill the worker outright.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error:
            # A file system without WAL is not a reason to refuse to
            # run; the busy timeout alone still helps.
            pass
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id        TEXT PRIMARY KEY,
                doc_id        TEXT,
                status        TEXT NOT NULL,
                chunks_written INTEGER,
                error         TEXT,
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL,
                payload       TEXT,
                claimed_by    TEXT,
                claimed_at    REAL,
                attempts     INTEGER DEFAULT 0,
                ocr_failures INTEGER DEFAULT 0
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_doc_id ON jobs(doc_id)"
        )
        # Claim scan: pending jobs, oldest first.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_pending ON jobs(status, created_at)"
        )
        # Stale-claim scan: running jobs, oldest claim first. Without it
        # the worker's recovery sweep scans the whole table on every
        # poll.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_claimed ON jobs(status, claimed_at)"
        )
        self._migrate_legacy_columns()

    def _migrate_legacy_columns(self) -> None:
        """Add the queue columns to a pre-existing jobs.db.

        A database created by an earlier version has no payload /
        claimed_by / claimed_at columns, and `CREATE TABLE IF NOT
        EXISTS` does not add columns to an existing table. Without
        this the worker would fail on the first claim with
        "no such column: payload".
        """
        existing = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)")
        }
        for column, decl in (
            ("payload", "TEXT"),
            ("claimed_by", "TEXT"),
            ("claimed_at", "REAL"),
            ("attempts", "INTEGER"),
            ("ocr_failures", "INTEGER"),
        ):
            if column not in existing:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {decl}")

    # -- inline path ------------------------------------------------------

    def create(self, job_id: str, doc_id: Optional[str]) -> None:
        """Create a job row for the inline ingest path.

        No payload: the API process owns this job from the moment it is
        created, so no worker may claim it.
        """
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (job_id, doc_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, doc_id, STATUS_PENDING, now, now),
            )

    # -- queue path -------------------------------------------------------

    def enqueue(self, job_id: str, doc_id: Optional[str], payload: Dict[str, Any]) -> None:
        """Enqueue a job for a worker to claim.

        The payload is the work contract: it must contain
        ``file_bytes_b64``, ``filename`` and ``mime``, and may contain
        ``ocr_override``, ``chunking_mode`` and ``chunking_config``.
        """
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs "
                "(job_id, doc_id, status, payload, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, doc_id, STATUS_PENDING, json.dumps(payload), now, now),
            )

    def claim_next(self, worker_id: str) -> Optional[dict]:
        """Atomically claim the oldest pending queued job.

        Returns the job (with its payload decoded) or ``None`` when the
        queue is empty. The whole read-then-mark sequence runs in one
        ``BEGIN IMMEDIATE`` transaction, so two workers racing for the
        same row cannot both win it.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM jobs "
                    "WHERE status = ? AND payload IS NOT NULL "
                    "ORDER BY created_at LIMIT 1",
                    (STATUS_PENDING,),
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None
                now = time.time()
                self._conn.execute(
                    "UPDATE jobs SET status = ?, claimed_by = ?, claimed_at = ?, "
                    "updated_at = ?, attempts = COALESCE(attempts, 0) + 1 "
                    "WHERE job_id = ?",
                    (STATUS_RUNNING, worker_id, now, now, row["job_id"]),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        job = self._row_to_dict(row)
        raw = row["payload"] if "payload" in row.keys() else None
        job["payload"] = json.loads(raw) if raw else None
        # The row was read before the UPDATE, so it still carries the
        # pre-claim values. Report what we actually wrote.
        job["status"] = STATUS_RUNNING
        job["claimed_by"] = worker_id
        job["claimed_at"] = now
        job["attempts"] = int(row["attempts"] or 0) + 1
        return job

    def release_stale_claims(self, stale_seconds: float, max_attempts: int = 0) -> int:
        """Return jobs whose worker died mid-flight to the queue.

        A job is stale when it has been ``running`` for longer than
        ``stale_seconds``. The row goes back to ``pending`` so another
        worker picks it up; the claim fields are cleared.

        A job that has already been attempted ``max_attempts`` times is
        not released again: it is failed. Without that, a document that
        hangs the parser is retried every ``stale_seconds`` for ever,
        by every worker, and never leaves the queue.

        Returns the number of jobs released.
        """
        if stale_seconds <= 0:
            return 0
        cutoff = time.time() - stale_seconds
        with self._lock:
            if max_attempts:
                # The ones that have burned their attempts are done.
                self._conn.execute(
                    "UPDATE jobs SET status = ?, claimed_by = NULL, claimed_at = NULL, "
                    "updated_at = ?, error = ? WHERE status = ? "
                    "AND claimed_at IS NOT NULL AND claimed_at < ? "
                    "AND COALESCE(attempts, 0) >= ?",
                    (
                        STATUS_FAILED,
                        time.time(),
                        f"gave up after {max_attempts} attempts",
                        STATUS_RUNNING,
                        cutoff,
                        max_attempts,
                    ),
                )
            cur = self._conn.execute(
                "UPDATE jobs SET status = ?, claimed_by = NULL, claimed_at = NULL, "
                "updated_at = ? WHERE status = ? AND claimed_at IS NOT NULL "
                "AND claimed_at < ?"
                + (" AND COALESCE(attempts, 0) < ?" if max_attempts else ""),
                (STATUS_PENDING, time.time(), STATUS_RUNNING, cutoff)
                + ((max_attempts,) if max_attempts else ()),
            )
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def purge_terminal_jobs(self, older_than_seconds: float) -> int:
        """Delete finished jobs older than the retention window.

        The job rows are the record of what happened; the payloads in
        them are the user's documents. Those do not need to live
        forever, and a queue that only grows is a queue that eventually
        fills the disk.
        """
        if older_than_seconds <= 0:
            return 0
        cutoff = time.time() - older_than_seconds
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE status IN (?, ?) AND updated_at < ?",
                (STATUS_COMPLETE, STATUS_FAILED, cutoff),
            )
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    # -- shared -----------------------------------------------------------

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        # Only the columns below may be written. The field names are
        # interpolated into the SQL, so a name that came from anywhere
        # else would be an injection; a whitelist makes that impossible
        # rather than merely unlikely.
        unknown = set(fields) - _UPDATABLE_COLUMNS
        if unknown:
            raise ValueError(
                f"cannot update unknown job column(s): {sorted(unknown)}; "
                f"allowed: {sorted(_UPDATABLE_COLUMNS)}"
            )
        # JSON-encode list/dict values so they survive the TEXT column.
        encoded = {
            k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
            for k, v in fields.items()
        }
        set_clause = ", ".join(f"{k} = ?" for k in encoded)
        params = list(encoded.values()) + [time.time(), job_id]
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {set_clause}, updated_at = ? WHERE job_id = ?",
                params,
            )
            # A finished job keeps its record and loses its document. The
            # payload is the uploaded file, base64-encoded; keeping it
            # meant the job store grew by the size of every document
            # ever queued, forever, and every user's document stayed on
            # disk after the ingest was long done.
            if fields.get("status") in (STATUS_COMPLETE, STATUS_FAILED):
                self._conn.execute(
                    "UPDATE jobs SET payload = NULL WHERE job_id = ?", (job_id,)
                )

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_jobs(self, limit: int = 50) -> List[dict]:
        """Most recent jobs first. Used by `atrium doctor` / the CLI."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "job_id": row["job_id"],
            "doc_id": row["doc_id"],
            "status": row["status"],
            "chunks_written": row["chunks_written"],
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "claimed_by": row["claimed_by"] if "claimed_by" in row.keys() else None,
            "attempts": (row["attempts"] if "attempts" in row.keys() else 0) or 0,
            "ocr_failures": (
                row["ocr_failures"] if "ocr_failures" in row.keys() else 0
            ) or 0,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
