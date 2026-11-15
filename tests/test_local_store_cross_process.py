"""tests/test_local_store_cross_process.py

The local store exists because the API and the worker are separate
processes sharing one SQLite file. That only works if a write made by
one process is visible to a query made by the other.

The corpus is cached in memory after the first search, so the cache
itself has to notice writes it did not make. These cases pin that: two
adapters over the same file stand in for the two processes.
"""

from __future__ import annotations

from atrium.local_store import SqliteAdapter
from atrium.models import ChunkRecord, SourceMetadata


def _record(chunk_id, embedding, doc_id="doc1", run_id="run1", text="hello"):
    return ChunkRecord(
        doc_id=doc_id,
        chunk_id=chunk_id,
        ingest_run_id=run_id,
        text=text,
        embedding=embedding,
        element_type="paragraph",
        source=SourceMetadata(
            file_name="a.pdf",
            page_number=1,
            section_path=["Introduction"],
        ),
        char_span=(0, 5),
        content_hash=f"hash-{chunk_id}",
    )


def _store(path):
    return SqliteAdapter(path=path)


def _ids(hits):
    return [hit["id"] for hit in hits]


def test_a_search_sees_a_chunk_written_by_another_process(tmp_path):
    path = str(tmp_path / "atrium_local.db")
    api = _store(path)
    api.ensure_collection("documents", 3)
    api.upsert([_record("c1", [1.0, 0.0, 0.0])])
    # The first search is what fills the API's cache.
    assert _ids(api.search([1.0, 0.0, 0.0], top_k=5)) == ["c1"]

    worker = _store(path)
    worker.replace_chunks(
        "doc2",
        [_record("c2", [0.0, 1.0, 0.0], doc_id="doc2", run_id="run2")],
        current_run_id="run2",
    )

    assert _ids(api.search([0.0, 1.0, 0.0], top_k=1)) == ["c2"]


def test_a_search_stops_seeing_a_chunk_another_process_deleted(tmp_path):
    path = str(tmp_path / "atrium_local.db")
    api = _store(path)
    api.ensure_collection("documents", 3)
    api.upsert([_record("c1", [1.0, 0.0, 0.0])])
    assert _ids(api.search([1.0, 0.0, 0.0], top_k=5)) == ["c1"]

    worker = _store(path)
    # A re-ingest of doc1 that produces no chunks is how a document
    # disappears from the corpus.
    worker.replace_chunks("doc1", [], current_run_id="run2")

    assert _ids(api.search([1.0, 0.0, 0.0], top_k=5)) == []


def test_an_unchanged_corpus_is_still_served_from_memory(tmp_path):
    path = str(tmp_path / "atrium_local.db")
    api = _store(path)
    api.ensure_collection("documents", 3)
    api.upsert([_record("c1", [1.0, 0.0, 0.0])])

    api.search([1.0, 0.0, 0.0], top_k=5)
    cached = api._corpus_cache
    api.search([1.0, 0.0, 0.0], top_k=5)

    # Same object: the check for a foreign write is one row read, not a
    # reload of every vector.
    assert api._corpus_cache is cached
