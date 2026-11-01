"""
tests/test_resilience_p2.py

Regression tests for the eleven P2 findings of the resilience audit
(2026-10-31).

    R-19  the vector store is closed when the service stops
    R-20  the unused path-traversal guard is gone
    R-21  the job store only writes known columns
    R-22  the stale-claim scan has an index
    R-23  an unparseable setting says so
    R-24  the retrieval lab's memory is bounded
    R-25  the schema needs the key when there is one
    R-26  /preview's response is capped
    R-27  a search says how big the corpus is
    R-28  the local model's dimension is pinned
    R-29  the CLI's file walk does not follow symlinked directories
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from atrium.job_store import JobStore
from atrium.universal_parser import ParsedElement


@pytest.fixture
def app_with_mocks(monkeypatch):
    from atrium.job_store import JobStore as Store
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
    db.count_chunks.return_value = 42

    monkeypatch.setattr("atrium.main.parser", parser)
    monkeypatch.setattr("atrium.main.embedder", embedder)
    monkeypatch.setattr("atrium.main.db", db)
    monkeypatch.setattr("atrium.main.job_store", Store(db_path=":memory:"))
    return app, parser, embedder, db


# ---------------------------------------------------------------------------
# R-19 — the store is closed on shutdown
# ---------------------------------------------------------------------------


def test_the_store_is_closed_when_the_service_stops(app_with_mocks):
    app, _, _, db = app_with_mocks
    with TestClient(app):
        pass
    assert db.close.called, "the vector store was not closed on shutdown"


def test_every_store_can_be_closed():
    from atrium.adapters import PgVectorAdapter, QdrantAdapter
    from atrium.local_store import SqliteAdapter

    for cls in (SqliteAdapter, QdrantAdapter, PgVectorAdapter):
        assert callable(getattr(cls, "close", None)), f"{cls.__name__} cannot be closed"


def test_closing_the_local_store_closes_its_connection(tmp_path):
    from atrium.local_store import SqliteAdapter

    store = SqliteAdapter(path=str(tmp_path / "a.db"))
    store.ensure_collection("documents", 8)
    store.close()
    assert store._conn is None, "the connection was left open"




# ---------------------------------------------------------------------------
# R-20 — the unused guard is gone
# ---------------------------------------------------------------------------


def test_the_unused_path_traversal_guard_is_gone():
    import atrium.security as security

    assert not hasattr(security, "safe_resolve"), (
        "safe_resolve is still there, still unused, and still claims to "
        "protect something"
    )


# ---------------------------------------------------------------------------
# R-21 — only known columns may be written
# ---------------------------------------------------------------------------


def test_an_unknown_job_column_is_refused(tmp_path):
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    store.create("j1", "doc-1")
    with pytest.raises(ValueError) as excinfo:
        store.update("j1", not_a_column="x")
    assert "not_a_column" in str(excinfo.value)


def test_a_known_job_column_is_written(tmp_path):
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    store.create("j1", "doc-1")
    store.update("j1", status="complete", chunks_written=7)
    row = store.get("j1")
    assert row["status"] == "complete"
    assert row["chunks_written"] == 7


# ---------------------------------------------------------------------------
# R-22 — the stale-claim scan has an index
# ---------------------------------------------------------------------------


def test_the_stale_claim_scan_has_an_index(tmp_path):
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    names = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        )
    }
    assert "idx_jobs_claimed" in names, sorted(names)


# ---------------------------------------------------------------------------
# R-23 — an unparseable setting says so
# ---------------------------------------------------------------------------


def test_an_unparseable_setting_warns_once_and_falls_back(monkeypatch, caplog):
    from atrium import config

    monkeypatch.setenv("ATRIUM_TEST_INT", "12x")
    monkeypatch.setattr(config, "_WARNED_UNPARSEABLE", set())
    with caplog.at_level(logging.WARNING, logger="ingest.config"):
        assert config.env_int("ATRIUM_TEST_INT", 5) == 5
        assert config.env_int("ATRIUM_TEST_INT", 5) == 5
    warnings = [
        record for record in caplog.records
        if "ATRIUM_TEST_INT" in record.getMessage()
    ]
    assert len(warnings) == 1, "the warning should be issued once, not per read"
    assert "12x" in warnings[0].getMessage()


def test_an_unparseable_float_warns(monkeypatch, caplog):
    from atrium import config

    monkeypatch.setenv("ATRIUM_TEST_FLOAT", "1.5x")
    monkeypatch.setattr(config, "_WARNED_UNPARSEABLE", set())
    with caplog.at_level(logging.WARNING, logger="ingest.config"):
        assert config.env_float("ATRIUM_TEST_FLOAT", 2.5) == 2.5
    assert any("ATRIUM_TEST_FLOAT" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# R-24 — the retrieval lab's memory is bounded
# ---------------------------------------------------------------------------


def test_the_lab_caps_the_chunks_of_one_session(app_with_mocks, monkeypatch):
    import atrium.main as main_module

    monkeypatch.setenv("MAX_LAB_CHUNKS_PER_SESSION", "2")
    main_module._retrieval_lab_sessions.clear()
    app, parser, embedder, _ = app_with_mocks
    # Long enough paragraphs that the chunker cannot merge them.
    parser.parse.return_value = [
        ParsedElement(type="paragraph", text=f"paragraph {n}. " * 400, page=1)
        for n in range(6)
    ]
    embedder.embed.side_effect = lambda texts: [[0.1] * 8 for _ in texts]

    client = TestClient(app)
    res = client.post(
        "/retrieval_lab/prepare",
        files={"file": ("a.md", b"# Title\n\nbody", "text/markdown")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["truncated"] is True
    assert len(body["chunks"]) == 2


def test_the_lab_evicts_sessions_over_the_total_cap(monkeypatch):
    import atrium.main as main_module

    monkeypatch.setenv("MAX_LAB_TOTAL_CHUNKS", "10")
    main_module._retrieval_lab_sessions.clear()

    from atrium.main import _retrieval_lab_put, _RetrievalLabSession

    def make(session_id: str, chunks: int):
        return _RetrievalLabSession(
            session_id=session_id,
            filename="a.md",
            doc_id="lab",
            chunks=[{"chunk_id": str(i), "text": "x"} for i in range(chunks)],
            kept_chunks=[],
            embeddings=None,
            bm25_index=None,
        )

    _retrieval_lab_put("s1", make("s1", 6))
    _retrieval_lab_put("s2", make("s2", 6))
    assert len(main_module._retrieval_lab_sessions) == 1, (
        "the lab kept more sessions than its total chunk cap allows"
    )


# ---------------------------------------------------------------------------
# R-25 — the schema needs the key when there is one
# ---------------------------------------------------------------------------


def test_the_schema_is_protected_when_a_key_is_set(app_with_mocks, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    app, _, _, _ = app_with_mocks
    client = TestClient(app)
    assert client.get("/health").status_code == 200, "/health stays open"
    assert client.get("/openapi.json").status_code == 401
    assert client.get("/docs").status_code == 401


def test_the_schema_is_open_when_there_is_no_key(app_with_mocks, monkeypatch):
    monkeypatch.setenv("API_KEY", "")
    app, _, _, _ = app_with_mocks
    client = TestClient(app)
    assert client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------------------
# R-26 — /preview's response is capped
# ---------------------------------------------------------------------------


def test_preview_caps_its_response(app_with_mocks, monkeypatch):
    monkeypatch.setenv("MAX_PREVIEW_CHUNKS", "3")
    app, parser, _, _ = app_with_mocks
    parser.parse.return_value = [
        ParsedElement(type="paragraph", text=f"paragraph {n} " * 400, page=1)
        for n in range(10)
    ]
    client = TestClient(app)
    res = client.post(
        "/preview",
        files={"file": ("a.md", b"# Title\n\nbody", "text/markdown")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["truncated"] is True
    assert body["total_chunks"] > 3
    assert len(body["chunks"]) == 3


# ---------------------------------------------------------------------------
# R-27 — a search says how big the corpus is
# ---------------------------------------------------------------------------


def test_a_search_reports_the_corpus_size(app_with_mocks):
    app, _, _, db = app_with_mocks
    db.count_chunks.return_value = 1234
    client = TestClient(app)
    res = client.post("/search", json={"query": "anything"})
    assert res.status_code == 200, res.text
    assert res.json()["corpus_chunks"] == 1234


# ---------------------------------------------------------------------------
# R-28 — the local model's dimension is pinned
# ---------------------------------------------------------------------------


def test_the_local_model_dimension_is_pinned():
    from atrium.local_embedder import DEFAULT_LOCAL_MODEL, SUGGESTED_MODELS

    assert DEFAULT_LOCAL_MODEL in SUGGESTED_MODELS
    assert SUGGESTED_MODELS[DEFAULT_LOCAL_MODEL] == 384


def test_a_local_model_of_the_wrong_width_is_refused():
    from atrium.errors import AtriumSetupError
    from atrium.local_embedder import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5", dim=384)
    with pytest.raises(AtriumSetupError) as excinfo:
        embedder._check_dimension([[0.1] * 768])
    assert "384" in str(excinfo.value)


def test_a_local_model_of_the_right_width_is_accepted():
    from atrium.local_embedder import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5", dim=384)
    embedder._check_dimension([[0.1] * 384])


# ---------------------------------------------------------------------------
# R-29 — the walk does not follow symlinked directories
# ---------------------------------------------------------------------------


def test_the_walk_does_not_follow_a_symlinked_directory(tmp_path):
    from atrium.cli import _walk_files

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.pdf").write_bytes(b"%PDF-1.4")

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "real.pdf").write_bytes(b"%PDF-1.4")
    (tree / "link").symlink_to(outside, target_is_directory=True)

    found = {p.name for p in _walk_files(tree)}
    assert "real.pdf" in found
    assert "secret.pdf" not in found, "the walk followed a symlinked directory"


def test_the_walk_still_reads_a_symlinked_file(tmp_path):
    from atrium.cli import _walk_files

    target = tmp_path / "elsewhere.pdf"
    target.write_bytes(b"%PDF-1.4")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "link.pdf").symlink_to(target)

    found = {p.name for p in _walk_files(tree)}
    assert "link.pdf" in found, "a file the operator named is still theirs to ingest"


def test_the_walk_skips_the_hidden_and_the_ignored(tmp_path):
    from atrium.cli import _walk_files

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "good.pdf").write_bytes(b"%PDF-1.4")
    (tree / ".hidden.pdf").write_bytes(b"%PDF-1.4")
    (tree / ".git").mkdir()
    (tree / ".git" / "config.pdf").write_bytes(b"%PDF-1.4")
    (tree / "notes.bin").write_bytes(b"not ingestable")

    found = {p.name for p in _walk_files(tree)}
    assert found == {"good.pdf"}
