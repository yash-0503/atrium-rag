"""
tests/test_resilience_p1.py

Regression tests for the P1 findings of the resilience audit
(2026-11-01), in the order they were scheduled.

    R-18  the API binds loopback unless told otherwise
    R-06  chunking parameters are range-validated
    R-11  a document that would cost too much is refused
    R-13  a request may not hold every file in memory at once
    R-09  a job that hangs is failed, not retried for ever
    R-17  a locked job store does not kill the worker
    R-07  an OCR failure is reported, not swallowed
    R-08  a finished job does not keep the user's document
    R-10  the parser's per-document state is per-thread
    R-12  the API and the worker resolve the same paths
    R-15  the local store does not re-read the corpus per query
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from atrium.config import DEFAULTS, data_path, env_str
from atrium.local_store import SqliteAdapter
from atrium.models import ChunkRecord, SourceMetadata
from atrium.pipeline import ingest_document
from atrium.universal_parser import ParsedElement, UniversalParser


def _chunk(chunk_id: str, doc_id: str = "doc-1", text: str = "text") -> ChunkRecord:
    return ChunkRecord(
        doc_id=doc_id,
        chunk_id=chunk_id,
        ingest_run_id="run-1",
        text=text,
        embedding=[0.1] * 8,
        element_type="paragraph",
        source=SourceMetadata(file_name="a.pdf", page_number=1),
        char_span=(0, 4),
        content_hash=f"hash-{chunk_id}",
    )


@pytest.fixture
def app_with_mocks(monkeypatch):
    from atrium.job_store import JobStore
    from atrium.main import app

    parser = MagicMock()
    parser.parse.return_value = [
        ParsedElement(type="paragraph", text="A paragraph of text.", page=1),
    ]
    parser.last_parse_ocr_failures = 0

    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 8]
    embedder.dimension = 8
    embedder.health.return_value = True

    db = MagicMock()
    db.health.return_value = True
    db.ensure_collection.return_value = None
    db.upsert.return_value = None
    db.replace_chunks.return_value = None

    monkeypatch.setattr("atrium.main.parser", parser)
    monkeypatch.setattr("atrium.main.embedder", embedder)
    monkeypatch.setattr("atrium.main.db", db)
    monkeypatch.setattr("atrium.main.job_store", JobStore(db_path=":memory:"))
    return app, parser, embedder, db


# ---------------------------------------------------------------------------
# R-18 — the API binds loopback unless told otherwise
# ---------------------------------------------------------------------------


def test_the_server_binds_loopback_by_default():
    from atrium.cli import _is_loopback

    assert DEFAULTS["ATRIUM_BIND"] == "127.0.0.1"
    assert env_str("ATRIUM_BIND") == "127.0.0.1"
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0")


def test_the_container_still_binds_everywhere():
    """The image runs uvicorn directly with --host 0.0.0.0, so the
    CLI default does not change what the container does."""
    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text()
    assert '"--host", "0.0.0.0"' in dockerfile


# ---------------------------------------------------------------------------
# R-06 — chunking parameters are validated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, field",
    [
        ({"target_chars": 0}, "target_chars"),
        ({"target_chars": -5}, "target_chars"),
        ({"target_chars": 10**9}, "target_chars"),
        ({"target_chars": 100, "overlap_chars": 500}, "overlap_chars"),
        (
            {"target_chars": 400, "overlap_chars": 50, "min_chunk_chars": 500},
            "min_chunk_chars",
        ),
        (
            {"target_chars": 400, "overlap_chars": 50, "parent_target_chars": 50},
            "parent_target_chars",
        ),
        (
            {"target_chars": 400, "overlap_chars": 50, "parent_overlap_chars": 5000},
            "parent_overlap_chars",
        ),
        ({"target_chars": "700"}, "target_chars"),
    ],
)
def test_an_invalid_chunking_config_is_rejected(config, field):
    from atrium.smart_chunker import resolve_chunk_config

    with pytest.raises(ValueError) as excinfo:
        resolve_chunk_config("custom", config)
    assert field in str(excinfo.value)


def test_a_valid_chunking_config_is_accepted():
    from atrium.smart_chunker import resolve_chunk_config

    resolved = resolve_chunk_config("custom", {"target_chars": 900, "overlap_chars": 100})
    assert resolved.target_chars == 900


def test_an_invalid_chunking_config_is_a_422(app_with_mocks):
    app, _, _, _ = app_with_mocks
    client = TestClient(app)
    res = client.post(
        "/ingest",
        files=[("files", ("a.pdf", b"%PDF-1.4", "application/pdf"))],
        data={"chunking_mode": "custom", "chunking_config": '{"target_chars": -5}'},
    )
    assert res.status_code == 422
    assert "target_chars" in res.json()["detail"]


# ---------------------------------------------------------------------------
# R-11 — a document that would cost too much is refused
# ---------------------------------------------------------------------------


def test_a_document_with_too_many_chunks_is_refused(monkeypatch):
    monkeypatch.setenv("MAX_CHUNKS_PER_DOCUMENT", "1")
    parser = MagicMock()
    parser.parse.return_value = [
        ParsedElement(type="paragraph", text="one " * 600, page=1),
        ParsedElement(type="paragraph", text="two " * 600, page=1),
    ]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    embedder.dimension = 8
    store = SqliteAdapter(path=":memory:")
    store.ensure_collection("documents", 8)

    with pytest.raises(ValueError) as excinfo:
        ingest_document(
            doc_id="doc-1",
            file_bytes=b"%PDF-1.4",
            filename="a.pdf",
            mime_type="application/pdf",
            parser=parser,
            embedder=embedder,
            db=store,
        )
    assert "MAX_CHUNKS_PER_DOCUMENT" in str(excinfo.value)
    assert store.chunk_count() == 0


def test_a_document_with_too_many_pages_is_refused(monkeypatch):
    monkeypatch.setenv("MAX_PAGES_PER_DOCUMENT", "2")
    parser = MagicMock()
    parser.parse.return_value = [
        ParsedElement(type="paragraph", text="one", page=9),
    ]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    embedder.dimension = 8
    store = SqliteAdapter(path=":memory:")
    store.ensure_collection("documents", 8)

    with pytest.raises(ValueError) as excinfo:
        ingest_document(
            doc_id="doc-1",
            file_bytes=b"%PDF-1.4",
            filename="a.pdf",
            mime_type="application/pdf",
            parser=parser,
            embedder=embedder,
            db=store,
        )
    assert "MAX_PAGES_PER_DOCUMENT" in str(excinfo.value)


# ---------------------------------------------------------------------------
# R-13 — a request may not hold every file in memory at once
# ---------------------------------------------------------------------------


def test_a_request_with_too_many_files_is_rejected(app_with_mocks, monkeypatch):
    monkeypatch.setenv("MAX_FILES_PER_REQUEST", "2")
    app, _, _, _ = app_with_mocks
    client = TestClient(app)
    res = client.post(
        "/ingest",
        files=[
            ("files", ("a.pdf", b"%PDF-1.4", "application/pdf")),
            ("files", ("b.pdf", b"%PDF-1.4", "application/pdf")),
            ("files", ("c.pdf", b"%PDF-1.4", "application/pdf")),
        ],
    )
    assert res.status_code == 422
    assert "MAX_FILES_PER_REQUEST" in res.json()["detail"]


def test_an_oversize_file_is_rejected_before_it_is_read(app_with_mocks, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "16")
    app, _, _, _ = app_with_mocks
    client = TestClient(app)
    res = client.post(
        "/ingest",
        files=[("files", ("a.pdf", b"x" * 4096, "application/pdf"))],
    )
    assert res.status_code == 413


# ---------------------------------------------------------------------------
# R-09 — a job that hangs is failed, not retried for ever
# ---------------------------------------------------------------------------


def test_a_job_that_keeps_failing_is_given_up_on(tmp_path):
    from atrium.job_store import JobStore

    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    store.enqueue("j1", "doc-1", {"file_bytes_b64": "AAA", "filename": "a.pdf", "mime": "application/pdf"})
    claimed = store.claim_next("worker-1")
    assert claimed["attempts"] == 1
    # The worker died mid-job: the row is still 'running'. Wait past the
    # staleness window so the claim counts as stale.
    time.sleep(0.02)
    released = store.release_stale_claims(stale_seconds=0.001, max_attempts=3)
    assert released == 1
    assert store.get("j1")["attempts"] == 1

    claimed = store.claim_next("worker-1")
    assert claimed["attempts"] == 2
    time.sleep(0.02)
    store.release_stale_claims(stale_seconds=0.001, max_attempts=2)
    row = store.get("j1")
    assert row["status"] == "failed", "a job that keeps failing must not be released for ever"
    assert "gave up" in (row["error"] or "")


def test_the_per_job_timeout_is_armed(monkeypatch):
    from atrium import worker as worker_module

    monkeypatch.setattr(worker_module, "JOB_TIMEOUT_SECONDS", 1.0)
    worker_module._arm_job_timeout()
    try:
        with pytest.raises(TimeoutError):
            time.sleep(3)
    finally:
        worker_module._disarm_job_timeout()


# ---------------------------------------------------------------------------
# R-17 — a locked job store does not kill the worker
# ---------------------------------------------------------------------------


def test_the_job_store_uses_wal_and_a_busy_timeout(tmp_path):
    from atrium.job_store import JobStore

    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    assert store._conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
    store.close()


# ---------------------------------------------------------------------------
# R-07 — an OCR failure is reported
# ---------------------------------------------------------------------------


def test_the_ocr_adapter_counts_its_failures():
    from atrium.ocr_adapter import RapidOCRAdapter

    adapter = RapidOCRAdapter(lang="eng")

    def boom(_bytes):
        raise RuntimeError("the engine fell over")

    adapter._engine = boom
    assert adapter.transcribe(b"not an image") == ""
    assert adapter.failures == 1
    assert "fell over" in adapter.last_error
    adapter.reset_failures()
    assert adapter.failures == 0


def test_the_ingest_reports_the_ocr_failures():

    parser = MagicMock()
    parser.parse.return_value = [
        ParsedElement(type="paragraph", text="A paragraph.", page=1),
    ]
    parser.last_parse_ocr_failures = 3
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    embedder.dimension = 8
    store = SqliteAdapter(path=":memory:")
    store.ensure_collection("documents", 8)

    result = ingest_document(
        doc_id="doc-1",
        file_bytes=b"%PDF-1.4",
        filename="a.pdf",
        mime_type="application/pdf",
        parser=parser,
        embedder=embedder,
        db=store,
    )
    assert result.ocr_failures == 3


# ---------------------------------------------------------------------------
# R-08 — a finished job does not keep the user's document
# ---------------------------------------------------------------------------


def test_a_finished_job_drops_its_payload(tmp_path):
    from atrium.job_store import JobStore

    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    store.enqueue("j1", "doc-1", {"file_bytes_b64": "QUJD", "filename": "a.pdf", "mime": "application/pdf"})
    row = store._conn.execute("SELECT payload FROM jobs WHERE job_id = 'j1'").fetchone()
    assert row["payload"], "the payload is there while the job is pending"

    store.update("j1", status="complete", chunks_written=3)
    row = store._conn.execute("SELECT payload FROM jobs WHERE job_id = 'j1'").fetchone()
    assert row["payload"] is None, "a finished job kept the document"


def test_old_finished_jobs_are_purged(tmp_path):
    from atrium.job_store import JobStore

    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    store.create("j1", "doc-1")
    store.update("j1", status="complete")
    store._conn.execute("UPDATE jobs SET updated_at = 0 WHERE job_id = 'j1'")
    store._conn.commit()
    assert store.purge_terminal_jobs(60) == 1
    assert store.get("j1") is None


# ---------------------------------------------------------------------------
# R-10 — the parser's per-document state is per-thread
# ---------------------------------------------------------------------------


def test_the_parser_state_is_per_thread():
    parser = UniversalParser()
    mine = parser._parse_state()
    mine["pending"].append("mine")

    theirs: dict = {}

    def other():
        theirs["state"] = parser._parse_state()

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()

    assert theirs["state"] is not mine
    assert theirs["state"]["pending"] == []
    assert mine["pending"] == ["mine"]


# ---------------------------------------------------------------------------
# R-12 — the API and the worker resolve the same paths
# ---------------------------------------------------------------------------


def test_the_local_paths_are_absolute(monkeypatch, tmp_path):
    monkeypatch.setenv("ATRIUM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_DB_PATH", "jobs.db")
    job = data_path("JOB_DB_PATH", "jobs.db")
    assert job.is_absolute()
    assert job.parent == tmp_path.resolve()


def test_the_api_and_the_worker_resolve_the_same_job_store(monkeypatch, tmp_path):
    from atrium import worker as worker_module

    monkeypatch.setenv("ATRIUM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_DB_PATH", "jobs.db")
    api_path = data_path("JOB_DB_PATH", "jobs.db")
    worker_path = Path(worker_module._build_job_store().db_path)
    assert worker_path == api_path


# ---------------------------------------------------------------------------
# R-15 — the local store does not re-read the corpus per query
# ---------------------------------------------------------------------------


def test_the_local_store_caches_its_corpus(tmp_path):
    store = SqliteAdapter(path=str(tmp_path / "a.db"))
    store.ensure_collection("documents", 8)
    store.upsert([_chunk("c1"), _chunk("c2")])

    assert store._corpus_cache is None
    store.search([0.1] * 8, top_k=2)
    assert store._corpus_cache is not None

    # A write invalidates it, so the next search sees the new corpus.
    store.upsert([_chunk("c3")])
    assert store._corpus_cache is None
    hits = store.search([0.1] * 8, top_k=10)
    assert len(hits) == 3


def test_the_local_store_states_its_ceiling():
    assert SqliteAdapter.RECOMMENDED_MAX_CHUNKS > 0
