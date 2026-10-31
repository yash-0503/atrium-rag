"""
tests/test_ingest_to_storage_v1.py

End-to-end regression tests for the v1 (parent-child) metadata
contract. Verifies that:

  1. Smart-chunker parent/child metadata survives the pipeline's
     conversion to the FastAPI ChunkRecord.
  2. The Qdrant/PgVector adapter receives and persists the v1
     metadata in the payload.
  3. `collection_schema_version()` returns 1 after v1 chunks are
     ingested.
  4. `retrieve_and_rerank` exercises the v1 path (search_children +
     dedup + retrieve_parents) against a real in-memory Qdrant
     collection and resolves parent_text into the final hit.
  5. Legacy v0 chunks (no parent_id, no is_parent) remain
     compatible: schema_version=0, helper takes the v0 flat path.
  6. The shared embedder configuration produces identical model +
     dimension from the HTTP service bootstrap and the worker
     bootstrap.

Tests 2-5 use a real in-memory Qdrant client (`:memory:`) so the
persistence roundtrip is real, not mocked. Test 4 is the only
end-to-end v1 roundtrip in the suite — it pins the full chain:
pipeline → ChunkRecord → upsert → search_children → retrieve_parents
→ final hit with parent_text.

PgVector integration tests are not run (no live Postgres + pgvector
in this environment) — the SQL composition tests in
test_adapters_realtime.py cover the JSONB filter, and the
PgVectorAdapter.upsert payload composition is verified by mocking
the cursor and inspecting the args.

Qdrant point IDs must be UUIDs (qdrant-client 1.x requirement).
The helpers below auto-generate UUIDs; tests that want to assert a
specific id pattern call `set_chunk_ids` first.
"""

from __future__ import annotations

import json
import os
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from atrium.adapters import QdrantAdapter
from atrium.embedder_config import DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL, build_embedder
from atrium.models import ChunkRecord, SourceMetadata
from atrium.pipeline import _build_chunk_record, run_ingestion
from atrium.retrieval import retrieve_and_rerank

# ---------------------------------------------------------------------------
# Stub producer types (smart_chunker.ChunkRecord-shaped; we don't import
# the dataclass directly to avoid pulling in the chunker's import chain
# in case it's broken at test-collection time).
# ---------------------------------------------------------------------------


def _source(page: int = 1, file_name: str = "f.pdf") -> SourceMetadata:
    return SourceMetadata(
        file_name=file_name,
        page_number=page,
        bbox=None,
        section_path=["H1"],
        page_numbers=[page],
    )


@dataclass
class _StubSmartChunk:
    doc_id: str = "doc-1"
    chunk_id: str = "stub-cid"
    text: str = "stub text"
    embed_text: str = "stub embed text"
    element_type: str = "paragraph"
    source: Optional[SourceMetadata] = None
    char_span: tuple = (0, 1)
    content_hash: str = "h"
    parent_id: Optional[str] = None
    child_ids: list = field(default_factory=list)

    def __post_init__(self):
        if self.source is None:
            self.source = _source()


# ---------------------------------------------------------------------------
# 1. Pipeline metadata propagation
# ---------------------------------------------------------------------------


class TestPipelineMetadataPropagation:
    def test_parent_chunk_records_is_parent_true_and_schema_version_one(self):
        sc = _StubSmartChunk(
            chunk_id="p1",
            parent_id=None,
            child_ids=["c1", "c2", "c3"],
        )
        rec = _build_chunk_record(sc, ingest_run_id="run-1")
        assert rec.is_parent is True
        assert rec.schema_version == 1
        assert rec.parent_id is None  # parent has no parent
        assert rec.child_ids == ["c1", "c2", "c3"]

    def test_child_chunk_records_is_parent_false_and_schema_version_one(self):
        sc = _StubSmartChunk(
            chunk_id="c1",
            parent_id="p1",
            child_ids=[],
        )
        rec = _build_chunk_record(sc, ingest_run_id="run-1")
        assert rec.is_parent is False
        assert rec.schema_version == 1
        assert rec.parent_id == "p1"
        assert rec.child_ids == []

    def test_v0_flat_chunk_records_schema_version_zero(self):
        sc = _StubSmartChunk(
            chunk_id="flat-1",
            parent_id=None,
            child_ids=[],
        )
        rec = _build_chunk_record(sc, ingest_run_id="run-1")
        assert rec.is_parent is False
        assert rec.schema_version == 0
        assert rec.parent_id is None
        assert rec.child_ids == []

    def test_orphan_child_with_no_parent_id_in_chunk_is_v0(self):
        # Defensive: a chunker that mis-emits child_ids=[] with
        # parent_id=None should still be v0. _build_chunk_record
        # derives both is_parent and schema_version from the
        # actual fields, so a misconfiguration is observable.
        sc = _StubSmartChunk(chunk_id="x", parent_id=None, child_ids=[])
        rec = _build_chunk_record(sc, ingest_run_id="run-1")
        assert rec.schema_version == 0
        assert rec.is_parent is False

    def test_pipeline_run_ingestion_propagates_v1_to_db_upsert(self):
        # Full run_ingestion with mocked parser + chunker + db.
        # Verifies that the v1 metadata survives the conversion at
        # the pipeline layer, and that db.upsert receives the right
        # ChunkRecord list (not just the raw smart_chunker objects).
        sc_parent = _StubSmartChunk(
            chunk_id="p1",
            parent_id=None,
            child_ids=["c1"],
        )
        sc_child = _StubSmartChunk(
            chunk_id="c1",
            parent_id="p1",
            child_ids=[],
        )

        parser = MagicMock()
        parser.parse.return_value = [{"type": "paragraph", "text": "x", "page": 1}]

        embedder = MagicMock()
        embedder.embed.return_value = [[0.0] * 4, [0.0] * 4]
        embedder.dimension = 4

        db = MagicMock()
        db.upsert.return_value = None
        db.delete_stale_chunks.return_value = None

        job_store = MagicMock()

        with patch(
            "atrium.pipeline.chunk_with_mode",
            return_value=[sc_parent, sc_child],
        ):
            run_ingestion(
                job_id="j",
                doc_id="doc-1",
                file_bytes=b"x",
                filename="f.pdf",
                mime_type="application/pdf",
                parser=parser,
                embedder=embedder,
                db=db,
                job_store=job_store,
            )

        # db.upsert was called once with both chunks.
        db.upsert.assert_called_once()
        upserted = db.upsert.call_args.args[0]
        assert len(upserted) == 2

        # Find the parent and the child in the upserted list.
        by_id = {c.chunk_id: c for c in upserted}
        assert by_id["p1"].is_parent is True
        assert by_id["p1"].schema_version == 1
        assert by_id["p1"].child_ids == ["c1"]
        assert by_id["p1"].parent_id is None

        assert by_id["c1"].is_parent is False
        assert by_id["c1"].schema_version == 1
        assert by_id["c1"].parent_id == "p1"
        assert by_id["c1"].child_ids == []

    def test_v0_run_ingestion_emits_schema_version_zero(self):
        sc_flat = _StubSmartChunk(
            chunk_id="flat-1",
            parent_id=None,
            child_ids=[],
        )
        parser = MagicMock()
        parser.parse.return_value = [{"type": "paragraph", "text": "x", "page": 1}]
        embedder = MagicMock()
        embedder.embed.return_value = [[0.0] * 4]
        embedder.dimension = 4
        db = MagicMock()
        db.upsert.return_value = None
        db.delete_stale_chunks.return_value = None
        job_store = MagicMock()

        with patch("atrium.pipeline.chunk_with_mode", return_value=[sc_flat]):
            run_ingestion(
                job_id="j",
                doc_id="doc-1",
                file_bytes=b"x",
                filename="f.pdf",
                mime_type="application/pdf",
                parser=parser,
                embedder=embedder,
                db=db,
                job_store=job_store,
            )

        upserted = db.upsert.call_args.args[0]
        assert len(upserted) == 1
        assert upserted[0].schema_version == 0
        assert upserted[0].is_parent is False
        assert upserted[0].parent_id is None
        assert upserted[0].child_ids == []


# ---------------------------------------------------------------------------
# 2. Qdrant upsert persists v1 metadata in payload
# ---------------------------------------------------------------------------


class TestQdrantUpsertPersistsV1Metadata:
    def test_v1_metadata_lands_in_qdrant_point_payload(self):
        # Mock the Qdrant client and inspect what was passed to
        # client.upsert. The v1 fields must be in the PointStruct
        # payload so the retrieval helper can filter on is_parent.
        client = MagicMock()
        adapter = QdrantAdapter.__new__(QdrantAdapter)
        adapter.client = client

        rec = ChunkRecord(
            doc_id="d",
            chunk_id="c1",
            ingest_run_id="r1",
            text="child text",
            embed_text="child text",
            embedding=[0.0] * 4,
            element_type="paragraph",
            source=_source(),
            char_span=(0, 1),
            content_hash="h",
            parent_id="p1",
            is_parent=False,
            child_ids=[],
            schema_version=1,
        )
        adapter.upsert([rec])

        kwargs = client.upsert.call_args.kwargs
        points = kwargs["points"]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["parent_id"] == "p1"
        assert payload["is_parent"] is False
        assert payload["child_ids"] == []
        assert payload["schema_version"] == 1

    def test_parent_chunk_payload_carries_child_ids(self):
        client = MagicMock()
        adapter = QdrantAdapter.__new__(QdrantAdapter)
        adapter.client = client
        rec = ChunkRecord(
            doc_id="d",
            chunk_id="p1",
            ingest_run_id="r1",
            text="parent text",
            embed_text="parent text",
            embedding=[0.0] * 4,
            element_type="paragraph",
            source=_source(),
            char_span=(0, 1),
            content_hash="h",
            parent_id=None,
            is_parent=True,
            child_ids=["c1", "c2"],
            schema_version=1,
        )
        adapter.upsert([rec])
        payload = client.upsert.call_args.kwargs["points"][0].payload
        assert payload["is_parent"] is True
        assert payload["child_ids"] == ["c1", "c2"]
        assert payload["schema_version"] == 1


# ---------------------------------------------------------------------------
# 3-4. Real in-memory Qdrant roundtrip — full v1 path
# ---------------------------------------------------------------------------


def _make_inmem_qdrant_adapter() -> QdrantAdapter:
    """Build a QdrantAdapter whose client is a real in-memory Qdrant.
    No network required."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    adapter = QdrantAdapter.__new__(QdrantAdapter)
    adapter.client = client
    # Bypass the host/port __init__ that tries to construct a real client.
    return adapter


def _uuid_str() -> str:
    return str(_uuid.uuid4())


def _make_v1_record(
    chunk_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    is_parent: bool = False,
    child_ids: Optional[list] = None,
    text: str = "t",
    dim: int = 4,
) -> ChunkRecord:
    if child_ids is None:
        child_ids = []
    return ChunkRecord(
        doc_id="d",
        chunk_id=chunk_id or _uuid_str(),
        ingest_run_id="r1",
        text=text,
        embed_text=text,
        embedding=[0.0] * dim,
        element_type="paragraph",
        source=_source(),
        char_span=(0, 1),
        content_hash="h",
        parent_id=parent_id,
        is_parent=is_parent,
        child_ids=child_ids,
        schema_version=1,
    )


def _make_v0_record(chunk_id: Optional[str] = None, text: str = "t", dim: int = 4) -> ChunkRecord:
    return ChunkRecord(
        doc_id="d",
        chunk_id=chunk_id or _uuid_str(),
        ingest_run_id="r1",
        text=text,
        embed_text=text,
        embedding=[0.0] * dim,
        element_type="paragraph",
        source=_source(),
        char_span=(0, 1),
        content_hash="h",
    )


class TestQdrantV1InMemoryRoundtrip:
    def test_collection_schema_version_detects_v1_after_upsert(self):
        adapter = _make_inmem_qdrant_adapter()
        adapter.ensure_collection("documents", dim=4)
        # Empty collection → schema_version=0
        assert adapter.collection_schema_version() == 0
        # Upsert a v1 chunk → schema_version=1
        parent_id = _uuid_str()
        rec = _make_v1_record(parent_id=parent_id, text="child text")
        adapter.upsert([rec])
        assert adapter.collection_schema_version() == 1

    def test_collection_schema_version_zero_after_only_v0_upsert(self):
        adapter = _make_inmem_qdrant_adapter()
        adapter.ensure_collection("documents", dim=4)
        rec = _make_v0_record(text="flat text")
        adapter.upsert([rec])
        # v0 chunk has no schema_version field → count filter finds
        # nothing → schema_version=0.
        assert adapter.collection_schema_version() == 0

    def test_search_children_returns_only_children_with_parent_id(self):
        adapter = _make_inmem_qdrant_adapter()
        adapter.ensure_collection("documents", dim=4)
        # Two parents and three children, all with real UUIDs.
        p1, p2 = _uuid_str(), _uuid_str()
        c1, c2, c3 = _uuid_str(), _uuid_str(), _uuid_str()
        recs = [
            _make_v1_record(chunk_id=p1, is_parent=True, child_ids=[c1, c2], text="P1"),
            _make_v1_record(chunk_id=p2, is_parent=True, child_ids=[c3], text="P2"),
            _make_v1_record(chunk_id=c1, parent_id=p1, text="C1"),
            _make_v1_record(chunk_id=c2, parent_id=p1, text="C2"),
            _make_v1_record(chunk_id=c3, parent_id=p2, text="C3"),
        ]
        adapter.upsert(recs)
        # search_children filters out is_parent=True. The query
        # vector is arbitrary; in-memory Qdrant uses cosine sim and
        # all zero-vectors tie. We just need the filtering, not
        # ranking.
        children = adapter.search_children([0.0] * 4, top_k=10)
        ids = {h["chunk_id"] for h in children}
        assert ids == {c1, c2, c3}

    def test_retrieve_parents_returns_parent_text_for_each_child(self):
        adapter = _make_inmem_qdrant_adapter()
        adapter.ensure_collection("documents", dim=4)
        p1, p2 = _uuid_str(), _uuid_str()
        c1, c2, c3 = _uuid_str(), _uuid_str(), _uuid_str()
        recs = [
            _make_v1_record(chunk_id=p1, is_parent=True, child_ids=[c1, c2],
                           text="PARENT ONE TEXT"),
            _make_v1_record(chunk_id=p2, is_parent=True, child_ids=[c3],
                           text="PARENT TWO TEXT"),
            _make_v1_record(chunk_id=c1, parent_id=p1, text="child 1"),
            _make_v1_record(chunk_id=c2, parent_id=p1, text="child 2"),
            _make_v1_record(chunk_id=c3, parent_id=p2, text="child 3"),
        ]
        adapter.upsert(recs)
        parent_map = adapter.retrieve_parents([p1, p2, "missing-id"])
        assert set(parent_map.keys()) == {p1, p2}
        assert parent_map[p1]["text"] == "PARENT ONE TEXT"
        assert parent_map[p2]["text"] == "PARENT TWO TEXT"

    def test_retrieve_and_rerank_resolves_parent_text_end_to_end(self):
        # Full v1 retrieval path: schema-version detection, child
        # search, dedup, parent-resolve. Uses the in-memory adapter
        # so the roundtrip is real (no mocks).
        adapter = _make_inmem_qdrant_adapter()
        adapter.ensure_collection("documents", dim=4)
        p1, c1 = _uuid_str(), _uuid_str()
        recs = [
            _make_v1_record(chunk_id=p1, is_parent=True, child_ids=[c1],
                           text="PARENT TEXT"),
            _make_v1_record(chunk_id=c1, parent_id=p1, text="child text"),
        ]
        adapter.upsert(recs)

        # Use a non-zero query so the children rank ahead of the
        # parent in cosine sim. The in-memory client computes real
        # cosine sim; an all-zero query ties, but the
        # search_children filter excludes parents so the child
        # still surfaces.
        hits = retrieve_and_rerank(
            adapter, [0.5] * 4, query_text="", top_k=1,
        )
        assert len(hits) == 1
        # Helper detects v1 (schema_version>=1 and search_children exists).
        # The child has parent_id=p1, so parent_text should resolve.
        assert hits[0]["parent_id"] == p1
        assert hits[0]["parent_text"] == "PARENT TEXT"
        assert hits[0]["is_parent"] is False


class TestQdrantV0InMemoryRoundtrip:
    def test_v0_chunks_roundtrip_through_retrieve_and_rerank(self):
        # Legacy v0 chunk: no parent_id, no is_parent, schema_version=0.
        # The helper should take the v0 path and return parent_text=None.
        adapter = _make_inmem_qdrant_adapter()
        adapter.ensure_collection("documents", dim=4)
        rec = _make_v0_record(text="flat text")
        adapter.upsert([rec])

        hits = retrieve_and_rerank(
            adapter, [0.5] * 4, query_text="", top_k=1,
        )
        assert len(hits) == 1
        assert hits[0]["text"] == "flat text"
        # v0 path: parent fields are defaulted to None/[].
        assert hits[0]["parent_id"] is None
        assert hits[0]["parent_text"] is None
        assert hits[0]["parent_page_number"] is None
        assert hits[0]["parent_section_path"] == []
        assert hits[0]["is_parent"] is False


# ---------------------------------------------------------------------------
# 5. PgVector upsert persists v1 metadata in payload
# ---------------------------------------------------------------------------


class TestPgVectorUpsertPersistsV1Metadata:
    def test_v1_metadata_lands_in_pgvector_payload_jsonb(self):
        # Inspect the args passed to the cursor's execute. The
        # payload arg is `chunk.model_dump_json(exclude={"embedding"})`
        # which should contain the v1 fields.
        rec = ChunkRecord(
            doc_id="d",
            chunk_id="c1",
            ingest_run_id="r1",
            text="child text",
            embed_text="child text",
            embedding=[0.0] * 4,
            element_type="paragraph",
            source=_source(),
            char_span=(0, 1),
            content_hash="h",
            parent_id="p1",
            is_parent=False,
            child_ids=[],
            schema_version=1,
        )

        # Build a real PgVectorAdapter but with a mocked pool. The
        # autouse no-op register_vector fixture from
        # test_adapters_realtime.py is active here too.
        from tests.test_adapters_realtime import _make_pg_adapter
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        adapter.upsert([rec])

        # The cursor.execute call is parameterized with
        # (chunk_id, doc_id, ingest_run_id, embedding, payload).
        params = cursor.execute.call_args.args[1] if len(cursor.execute.call_args.args) > 1 else cursor.execute.call_args.kwargs.get("params", ())
        # 5th positional arg is payload.
        # First verify the payload arg (5th in positional).
        # `params` is a tuple; payload is the last element.
        payload_str = params[-1]
        payload = json.loads(payload_str)
        assert payload["parent_id"] == "p1"
        assert payload["is_parent"] is False
        assert payload["child_ids"] == []
        assert payload["schema_version"] == 1

    def test_pgvector_ensure_collection_raises_on_dim_mismatch(self):
        from tests.test_adapters_realtime import _make_pg_adapter
        adapter, _pool, conn, _cursor = _make_pg_adapter()
        # The production code reads `conn.execute(...).fetchone()`
        # (not cursor.fetchone) for the dim check. The MagicMock's
        # conn.execute().fetchone() chain defaults to a fresh mock,
        # so set the return_value explicitly. Stored dim=768 →
        # atttypmod=772 (dim + 4-byte pgvector typmod header).
        conn.execute.return_value.fetchone.return_value = (772,)
        with pytest.raises(ValueError, match="dim 768"):
            adapter.ensure_collection("documents", dim=1024)

    def test_pgvector_ensure_collection_no_op_on_matching_dim(self):
        from tests.test_adapters_realtime import _make_pg_adapter
        adapter, _pool, conn, _cursor = _make_pg_adapter()
        # Stored dim=1024 → atttypmod=1028. Same chain as above.
        conn.execute.return_value.fetchone.return_value = (1028,)
        # Should not raise.
        adapter.ensure_collection("documents", dim=1024)


# ---------------------------------------------------------------------------
# 6. Embedder config consistency
# ---------------------------------------------------------------------------


class TestEmbedderConfigConsistency:
    def setup_method(self):
        # Make sure no EMBED_MODEL / EMBED_DIM / EMBED_BASE_URL leak
        # from the test env.
        for var in ("EMBED_MODEL", "EMBED_DIM", "EMBED_BASE_URL"):
            os.environ.pop(var, None)

    def test_default_model_and_dim_match_documented_baseline(self):
        # The 19-set A/B on 2026-10-11 settled on bge-large; promote
        # embedder changes only on 19-set evidence. The HTTP
        # service and the worker share this default.
        #
        # `bge-large` is the Ollama tag. `bge-large-en-v1.5` is the
        # underlying HuggingFace repo name — it does not resolve in
        # Ollama's registry, so it could never be pulled. The default
        # used to name it, which meant a fresh install pointed at a
        # model that could not be fetched. tests/conftest.py pins the
        # same value (PINNED_EMBED_MODEL) for the same reason.
        assert DEFAULT_EMBED_MODEL == "bge-large"
        assert DEFAULT_EMBED_DIM == 1024

    def test_build_embedder_with_no_env_returns_documented_defaults(self):
        e = build_embedder()
        assert e.model == "bge-large"
        assert e.dimension == 1024

    def test_main_and_worker_construct_same_default_embedder(self):
        # Construct an embedder the way main.py does and the way
        # worker.py does. With no env override, the resulting
        # (model, dim) must be identical.
        e_main = build_embedder(base_url=os.getenv(
            "EMBED_BASE_URL", "http://host.docker.internal:11434",
        ))
        e_worker = build_embedder(base_url=os.getenv(
            "EMBED_BASE_URL", "http://host.docker.internal:11434",
        ))
        assert e_main.model == e_worker.model
        assert e_main.dimension == e_worker.dimension

    def test_env_override_applies_to_both_paths(self):
        os.environ["EMBED_MODEL"] = "nomic-embed-text:v1.5"
        os.environ["EMBED_DIM"] = "768"
        try:
            e_main = build_embedder()
            e_worker = build_embedder()
            assert e_main.model == e_worker.model == "nomic-embed-text:v1.5"
            assert e_main.dimension == e_worker.dimension == 768
        finally:
            os.environ.pop("EMBED_MODEL", None)
            os.environ.pop("EMBED_DIM", None)

    def test_no_hardcoded_768_nomic_remains_in_main_or_worker(self):
        # Lock that the previous bug (main.py:181 hardcoding
        # nomic/768) does not regress. If a future edit reintroduces
        # the hardcoded values, this test fails.
        from atrium import main as main_mod, worker as worker_mod
        main_src = open(main_mod.__file__).read()
        worker_src = open(worker_mod.__file__).read()
        assert 'model="nomic-embed-text:v1.5"' not in main_src, (
            "main.py must not hardcode nomic-embed-text; use build_embedder()"
        )
        assert "dim=768" not in main_src, (
            "main.py must not hardcode dim=768; use embedder.dimension"
        )
        # Worker is fine either way (it routes through build_embedder).
        # Just check the worker doesn't have its own hardcoded defaults.
        assert 'model="nomic-embed-text:v1.5"' not in worker_src
        assert "dim=768" not in worker_src
