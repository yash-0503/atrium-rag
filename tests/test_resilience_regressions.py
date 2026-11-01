"""
tests/test_resilience_regressions.py

Regression tests for the five P0 findings of the resilience audit
(2026-11-01). Each test names the finding it exists to prevent from
coming back.

    R-01  the worker and the API must resolve the same store
    R-02  a parser crash must not become an indexed chunk
    R-03  a failed ingest must not answer 200
    R-04  a failed preview must not answer 200
    R-05  a re-ingest must replace, not duplicate

The tests do not need a live Ollama, Qdrant or Postgres.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from atrium.adapters import build_store
from atrium.models import ChunkRecord, SourceMetadata
from atrium.pipeline import ingest_document, run_ingestion
from atrium.universal_parser import ParsedElement, UniversalParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _element(text: str = "A paragraph of text.") -> ParsedElement:
    return ParsedElement(type="paragraph", text=text, page=1)


@pytest.fixture
def app_with_mocks(monkeypatch):
    """The API with the parser, embedder and store replaced by stubs."""
    from atrium.job_store import JobStore
    from atrium.main import app

    parser = MagicMock()
    parser.parse.return_value = [_element()]

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


@pytest.fixture
def local_store(tmp_path):
    """A real local store, on a real file."""
    from atrium.local_store import SqliteAdapter

    store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
    store.ensure_collection("documents", 8)
    return store


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


# ---------------------------------------------------------------------------
# R-01 — one store, everywhere
# ---------------------------------------------------------------------------


def test_the_api_and_the_worker_resolve_the_same_store(monkeypatch):
    """The worker used to build a Postgres adapter for the default
    SQLite configuration, so the two halves of the system wrote to
    different databases."""
    from atrium import main as main_module, worker as worker_module

    for kind, expected in (
        ("sqlite", "SqliteAdapter"),
        ("qdrant", "QdrantAdapter"),
        ("pgvector", "PgVectorAdapter"),
    ):
        monkeypatch.setenv("VECTOR_DB", kind)
        # The API builds its adapter once per process and caches it, so
        # the cache is cleared between the checks.
        monkeypatch.setattr(main_module, "_db", None)
        assert type(build_store()).__name__ == expected
        assert type(main_module._build_db()).__name__ == expected
        assert type(worker_module._build_db()).__name__ == expected


def test_an_unknown_store_name_is_rejected_with_the_supported_values(monkeypatch):
    monkeypatch.setenv("VECTOR_DB", "postgres")
    with pytest.raises(ValueError) as excinfo:
        build_store()
    message = str(excinfo.value)
    assert "sqlite" in message and "qdrant" in message and "pgvector" in message


def test_the_cli_resolves_the_same_store(monkeypatch):
    from atrium import cli as cli_module

    monkeypatch.setenv("VECTOR_DB", "sqlite")
    monkeypatch.setenv("LOCAL_DB_PATH", "/tmp/atrium_cli_store_test.db")
    _, _, db = cli_module._build_runtime()
    assert type(db).__name__ == "SqliteAdapter"


# ---------------------------------------------------------------------------
# R-02 — a crash is not content
# ---------------------------------------------------------------------------


def test_a_parser_crash_is_not_turned_into_a_placeholder_chunk():
    """The parser used to return a placeholder paragraph when a
    format parser crashed, which was then chunked, embedded and stored
    — a failure reported as content."""
    parser = UniversalParser()

    def boom(*args, **kwargs):
        raise KeyError("the parser tripped over something")

    with pytest.raises(KeyError):
        parser._safe_parse("PDF", boom)


def test_a_crashed_parser_indexes_nothing(tmp_path):
    """The whole pipeline: a parser that crashes must leave the store
    empty and the job marked failed."""
    from atrium.job_store import JobStore

    parser = MagicMock()
    parser.parse.side_effect = RuntimeError("the parser tripped over something")
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 8]
    embedder.dimension = 8
    store = SqliteAdapterForTest(tmp_path)
    job_store = JobStore(db_path=str(tmp_path / "jobs.db"))

    job_store.create("job-1", "doc-1")
    with pytest.raises(RuntimeError):
        run_ingestion(
            job_id="job-1",
            doc_id="doc-1",
            file_bytes=b"%PDF-1.4",
            filename="a.pdf",
            mime_type="application/pdf",
            parser=parser,
            embedder=embedder,
            db=store,
            job_store=job_store,
        )

    assert store.chunk_count() == 0
    job = job_store.get("job-1")
    assert job is not None and job["status"] == "failed", job


def SqliteAdapterForTest(tmp_path):
    from atrium.local_store import SqliteAdapter

    store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
    store.ensure_collection("documents", 8)
    store.chunk_count = lambda: len(store.search([0.0] * 8, top_k=1000))
    return store


# ---------------------------------------------------------------------------
# R-03 — a failed ingest is not a 200
# ---------------------------------------------------------------------------


def test_ingest_returns_422_when_the_document_cannot_be_parsed(app_with_mocks):
    app, parser, _, _ = app_with_mocks
    parser.parse.side_effect = ValueError("PDF is corrupt or unreadable")
    client = TestClient(app)
    res = client.post(
        "/ingest_single",
        files={"file": ("a.pdf", b"%PDF-1.4 broken", "application/pdf")},
    )
    assert res.status_code == 422
    assert "corrupt" in res.json()["detail"]


def test_ingest_returns_503_when_the_embedder_is_unavailable(app_with_mocks):
    app, _, embedder, _ = app_with_mocks
    from atrium.errors import AtriumSetupError

    embedder.embed.side_effect = AtriumSetupError(
        "The embedding endpoint at http://localhost:11434/v1 is not answering.",
        hints=("Start it, or pick another endpoint: `atrium init`",),
    )
    client = TestClient(app)
    res = client.post(
        "/ingest_single",
        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert res.status_code == 503
    assert "not answering" in res.json()["detail"]


def test_a_batch_where_one_file_fails_answers_207(app_with_mocks):
    """One bad file must not hide the good ones, and must not be
    reported as a whole successful batch."""
    app, parser, _, _ = app_with_mocks
    calls = {"n": 0}

    def parse(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("PDF is corrupt or unreadable")
        return [_element()]

    parser.parse.side_effect = parse
    client = TestClient(app)
    res = client.post(
        "/ingest",
        files=[
            ("files", ("good.pdf", b"%PDF-1.4 good", "application/pdf")),
            ("files", ("bad.pdf", b"%PDF-1.4 bad", "application/pdf")),
        ],
    )
    assert res.status_code == 207
    body = res.json()
    assert len(body["jobs"]) == 2
    assert any(job.get("error") for job in body["jobs"])
    assert any(job.get("job_id") for job in body["jobs"])


# ---------------------------------------------------------------------------
# R-04 — a failed preview is not a 200
# ---------------------------------------------------------------------------


def test_preview_returns_422_on_a_corrupt_document(app_with_mocks):
    app, parser, _, _ = app_with_mocks
    parser.parse.side_effect = ValueError("PDF is corrupt or unreadable")
    client = TestClient(app)
    res = client.post(
        "/preview",
        files={"file": ("a.pdf", b"%PDF-1.4 broken", "application/pdf")},
    )
    assert res.status_code == 422
    assert "corrupt" in res.json()["detail"]


def test_preview_keeps_the_status_of_a_rejected_upload(app_with_mocks):
    """A MIME rejection must stay a 415, not become a 200 with an
    error field."""
    app, _, _, _ = app_with_mocks
    monkey = pytest.MonkeyPatch()
    monkey.setenv("ALLOWED_MIME_TYPES", "text/markdown")
    try:
        client = TestClient(app)
        res = client.post(
            "/preview",
            files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert res.status_code == 415
    finally:
        monkey.undo()


# ---------------------------------------------------------------------------
# R-05 — a re-ingest replaces, it does not duplicate
# ---------------------------------------------------------------------------


def test_chunk_ids_are_stable_for_the_same_document_and_strategy():
    """A random id meant that ingesting the same document twice left
    two copies of every chunk in the store."""
    from atrium.smart_chunker import chunk_with_mode

    elements = [_element("First paragraph."), _element("Second paragraph.")]
    first = chunk_with_mode(elements, doc_id="d1", file_name="f.md", mode="precise")
    second = chunk_with_mode(elements, doc_id="d1", file_name="f.md", mode="precise")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    other = chunk_with_mode(elements, doc_id="d1", file_name="f.md", mode="balanced")
    assert {c.chunk_id for c in first} != {c.chunk_id for c in other}


def test_reingesting_the_same_document_does_not_duplicate_chunks(local_store):
    """Two ingests of one document must leave one version of it."""
    from atrium.job_store import JobStore

    parser = MagicMock()
    parser.parse.return_value = [_element("A paragraph of text.")]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    embedder.dimension = 8
    JobStore(db_path=":memory:")

    for _ in range(2):
        ingest_document(
            doc_id="doc-1",
            file_bytes=b"%PDF-1.4",
            filename="a.pdf",
            mime_type="application/pdf",
            parser=parser,
            embedder=embedder,
            db=local_store,
        )

    stored = local_store.search([0.1] * 8, top_k=100)
    assert len(stored) == len({hit["chunk_id"] for hit in stored}), "duplicate chunks"
    assert len(stored) <= 2


def test_concurrent_reingest_of_one_document_leaves_one_version(local_store):
    """Two ingests of the same document at the same time used to delete
    each other's chunks, leaving a mix of two versions."""
    from atrium.job_store import JobStore

    parser = MagicMock()
    parser.parse.return_value = [_element("A paragraph of text.")]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 8]
    embedder.dimension = 8
    JobStore(db_path=":memory:")

    errors: list = []

    def ingest():
        try:
            ingest_document(
                doc_id="doc-1",
                file_bytes=b"%PDF-1.4",
                filename="a.pdf",
                mime_type="application/pdf",
                parser=parser,
                embedder=embedder,
                db=local_store,
            )
        except Exception as exc:  # pragma: no cover - the failure path
            errors.append(exc)

    threads = [threading.Thread(target=ingest) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    stored = local_store.search([0.1] * 8, top_k=100)
    run_ids = {hit.get("ingest_run_id") for hit in stored}
    # Every surviving chunk belongs to one ingest, not to two.
    assert len(run_ids) == 1, f"the corpus holds two versions: {run_ids}"
