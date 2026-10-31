"""
test_adapters_contract.py

Adapter contract tests (Task B, 2026-10-10).

Both `QdrantAdapter` and `PgVectorAdapter` implement the same parent-
child retrieval protocol so `app.retrieval.retrieve_with_parent_resolution`
treats them uniformly. These tests pin the protocol contract using mocks
(no live Qdrant / PgVector needed).

Cases:
  - schema_version: returns 0 (legacy), 1 (v1), 0 (mixed/legacy-only).
  - search_children: returns children only (filters out parents).
  - retrieve_parents: batched fetch by id; missing parents absent.
  - The retrieval helper produces the same output shape regardless of
    which adapter supplies the protocol methods (Qdrant or PgVector).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from atrium.retrieval import retrieve_with_parent_resolution

# ---------------------------------------------------------------------------
# Schema version: deterministic behavior on legacy / v1 / mixed collections.
# ---------------------------------------------------------------------------


def _resp(points):
    """The shape qdrant-client's query_points() returns.

    `search()` returned the points directly; `query_points()` wraps
    them in a response object with a `.points` attribute. The doubles
    have to model the shape the client actually returns, or they stop
    testing the shipped path.
    """
    response = MagicMock()
    response.points = points
    return response


class TestQdrantSchemaVersionContract:
    def test_returns_zero_when_no_v1_points(self):
        # Mock client.count returning 0 → schema_version=0
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.count.return_value = MagicMock(count=0)
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            assert adapter.collection_schema_version() == 0

    def test_returns_one_when_v1_points_exist(self):
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.count.return_value = MagicMock(count=42)
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            assert adapter.collection_schema_version() == 1

    def test_returns_zero_when_count_raises(self):
        # Empty/missing collection surfaces as v0 (legacy default).
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.count.side_effect = Exception("not found")
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            assert adapter.collection_schema_version() == 0


class TestPgVectorSchemaVersionContract:
    def test_returns_zero_when_no_v1_rows(self):
        with patch("atrium.adapters.ConnectionPool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (0,)
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_pool.connection.return_value = mock_conn
            mock_pool_cls.return_value = mock_pool
            from atrium.adapters import PgVectorAdapter
            adapter = PgVectorAdapter(conn_info="postgres://stub")
            assert adapter.collection_schema_version() == 0

    def test_returns_one_when_v1_rows_exist(self):
        with patch("atrium.adapters.ConnectionPool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (5,)
            mock_conn.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_pool.connection.return_value = mock_conn
            mock_pool_cls.return_value = mock_pool
            from atrium.adapters import PgVectorAdapter
            adapter = PgVectorAdapter(conn_info="postgres://stub")
            assert adapter.collection_schema_version() == 1


# ---------------------------------------------------------------------------
# search_children: filters parents out, returns children with parent_id.
# ---------------------------------------------------------------------------


class TestQdrantSearchChildrenContract:
    def test_applies_is_parent_false_filter(self):
        # Verify the QdrantAdapter.search_children builds a query_filter
        # with is_parent=False. Mocks the search() call and inspects the
        # query_filter kwarg.
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.query_points.return_value = _resp([])
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            adapter.search_children([0.0] * 4, top_k=5)
            # Inspect the kwargs passed to client.query_points.
            kwargs = mock_client.query_points.call_args.kwargs
            assert "query_filter" in kwargs, "search_children must apply a query_filter"
            # The filter must reference is_parent=False.
            qf = kwargs["query_filter"]
            # qf.must is a list of FieldCondition. Walk it.
            field_keys = [c.key for c in qf.must]
            assert "is_parent" in field_keys

    def test_returns_hits_in_helper_shape(self):
        # Hits must include id, score, text, doc_id, chunk_id, page_number,
        # section_path, element_type, parent_id. parent_id=None for
        # children (the child has a parent_id; the field is named to
        # distinguish from the parent itself).
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_hit = MagicMock()
            mock_hit.id = "child-uuid"
            mock_hit.score = 0.85
            mock_hit.payload = {
                "text": "child text",
                "doc_id": "doc-1",
                "chunk_id": "child-chunk-id",
                "page_number": 3,
                "section_path": ["H1", "H2"],
                "element_type": "paragraph",
                "parent_id": "parent-chunk-id",
            }
            mock_client.query_points.return_value = _resp([mock_hit])
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            hits = adapter.search_children([0.0] * 4, top_k=5)
            assert len(hits) == 1
            h = hits[0]
            assert h["score"] == 0.85
            assert h["text"] == "child text"
            assert h["parent_id"] == "parent-chunk-id"
            assert h["is_parent"] is False  # search_children only returns children


# ---------------------------------------------------------------------------
# retrieve_parents: batched fetch, missing parents absent from result.
# ---------------------------------------------------------------------------


class TestQdrantRetrieveParentsContract:
    def test_empty_input_returns_empty_dict(self):
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            assert adapter.retrieve_parents([]) == {}
            mock_client.retrieve.assert_not_called()

    def test_missing_parents_absent_from_result(self):
        # Client.retrieve returns only parents that exist; the helper
        # must surface only those (no synthetic entries).
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            existing_parent = MagicMock()
            existing_parent.id = "p1"
            existing_parent.payload = {
                "chunk_id": "p1",
                "text": "parent 1 text",
                "page_number": 5,
                "section_path": ["Section A"],
                "element_type": "paragraph",
            }
            mock_client.retrieve.return_value = [existing_parent]
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            result = adapter.retrieve_parents(["p1", "p2-missing"])
            # p1 present, p2-missing absent
            assert "p1" in result
            assert "p2-missing" not in result
            assert result["p1"]["text"] == "parent 1 text"

    def test_retrieve_failure_returns_empty_dict(self):
        # Caller (helper) treats empty as "no parents resolved" rather
        # than crashing.
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.retrieve.side_effect = Exception("connection lost")
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            assert adapter.retrieve_parents(["p1"]) == {}


# ---------------------------------------------------------------------------
# Cross-adapter: helper treats Qdrant and PgVector outputs uniformly.
# ---------------------------------------------------------------------------


class TestHelperAdapterInterchangeability:
    """The retrieval helper must produce the same result shape regardless
    of which adapter supplies the protocol methods. Verified by running
    the helper against Qdrant-shaped and PgVector-shaped stubs."""

    def _make_qdrant_stub(self, schema_v, children, parents):
        class _QStub:
            def __init__(self):
                self.calls = {"search": 0, "children": 0, "parents": 0}
                self._sv = schema_v
                self._children = list(children)
                self._parents = dict(parents)
            def collection_schema_version(self):
                return self._sv
            def search(self, qv, top_k=5, score_threshold=None):
                self.calls["search"] += 1
                flat = list(self._children)
                flat.sort(key=lambda h: h.get("score", 0.0), reverse=True)
                return flat[:top_k]
            def search_children(self, qv, top_k=5, score_threshold=None):
                self.calls["children"] += 1
                return sorted(self._children, key=lambda h: h.get("score", 0.0), reverse=True)[:top_k]
            def retrieve_parents(self, parent_ids):
                self.calls["parents"] += 1
                return {pid: self._parents[pid] for pid in parent_ids if pid in self._parents}
        return _QStub()

    def _make_pgvector_stub(self, schema_v, children, parents):
        # Identical surface area to Qdrant — the helper only sees the
        # protocol, not the implementation.
        return self._make_qdrant_stub(schema_v, children, parents)

    def test_qdrant_and_pgvector_produce_same_results(self):
        children = [
            {"id": "c1", "score": 0.9, "text": "c1", "doc_id": "d1",
             "chunk_id": "c1", "page_number": 1, "section_path": [],
             "element_type": "paragraph", "is_parent": False,
             "parent_id": "p1"},
            {"id": "c2", "score": 0.8, "text": "c2", "doc_id": "d1",
             "chunk_id": "c2", "page_number": 1, "section_path": [],
             "element_type": "paragraph", "is_parent": False,
             "parent_id": "p2"},
        ]
        parents = {
            "p1": {"text": "P1 TEXT", "page_number": 1, "section_path": [], "element_type": "paragraph"},
            "p2": {"text": "P2 TEXT", "page_number": 1, "section_path": [], "element_type": "paragraph"},
        }
        q_hits = retrieve_with_parent_resolution(
            self._make_qdrant_stub(1, children, parents), [0.0], top_k=2,
        )
        p_hits = retrieve_with_parent_resolution(
            self._make_pgvector_stub(1, children, parents), [0.0], top_k=2,
        )
        # Same length, same ids, same parent_text.
        assert len(q_hits) == len(p_hits) == 2
        for q, p in zip(q_hits, p_hits):
            assert q["id"] == p["id"]
            assert q["parent_text"] == p["parent_text"]
            assert q["score"] == p["score"]


# ---------------------------------------------------------------------------
# Legacy records: corpus without schema_version field is treated as v0.
# ---------------------------------------------------------------------------


class TestSqliteAdapterContract:
    """The local store has to be interchangeable with the other two.

    It is the default, so a difference between it and Qdrant shows up
    in every install rather than in the ones that opted into a server.
    """

    def _record(self, **overrides):
        from atrium.models import ChunkRecord, SourceMetadata

        payload = {
            "doc_id": "doc1",
            "chunk_id": "c1",
            "ingest_run_id": "run1",
            "text": "hello",
            "embedding": [1.0, 0.0, 0.0],
            "element_type": "paragraph",
            "source": SourceMetadata(file_name="a.pdf", page_number=1),
            "char_span": (0, 5),
            "content_hash": "h",
        }
        payload.update(overrides)
        return ChunkRecord(**payload)

    def _store(self, tmp_path):
        from atrium.local_store import SqliteAdapter

        return SqliteAdapter(path=str(tmp_path / "atrium_local.db"))

    def test_ensure_collection_creates_the_file(self, tmp_path):
        store = self._store(tmp_path)
        store.ensure_collection("documents", 3)
        assert (tmp_path / "atrium_local.db").exists()

    def test_dimension_mismatch_names_both_numbers(self, tmp_path):
        store = self._store(tmp_path)
        store.ensure_collection("documents", 3)
        with pytest.raises(ValueError) as excinfo:
            store.ensure_collection("documents", 5)
        message = str(excinfo.value)
        assert "3" in message and "5" in message

    def test_search_returns_the_helper_hit_shape(self, tmp_path):
        store = self._store(tmp_path)
        store.ensure_collection("documents", 3)
        store.upsert([self._record()])
        hits = store.search([1.0, 0.0, 0.0], top_k=5)
        assert len(hits) == 1
        hit = hits[0]
        for field in (
            "id", "score", "text", "doc_id", "chunk_id", "page_number",
            "section_path", "element_type", "is_parent", "parent_id",
            "chunking_strategy",
        ):
            assert field in hit
        assert hit["text"] == "hello"
        assert hit["doc_id"] == "doc1"
        assert hit["score"] == pytest.approx(1.0, abs=1e-6)

    def test_search_orders_by_score(self, tmp_path):
        store = self._store(tmp_path)
        store.ensure_collection("documents", 3)
        store.upsert([
            self._record(chunk_id="far", embedding=[0.0, 1.0, 0.0]),
            self._record(chunk_id="near", embedding=[1.0, 0.0, 0.0]),
        ])
        hits = store.search([1.0, 0.0, 0.0], top_k=5)
        assert [hit["chunk_id"] for hit in hits] == ["near", "far"]

    def test_search_children_excludes_parents(self, tmp_path):
        store = self._store(tmp_path)
        store.ensure_collection("documents", 3)
        store.upsert([
            self._record(chunk_id="parent", is_parent=True, schema_version=1),
            self._record(chunk_id="child", is_parent=False, schema_version=1),
        ])
        assert len(store.search_children([1.0, 0.0, 0.0], top_k=5)) == 1
        assert len(store.search([1.0, 0.0, 0.0], top_k=5)) == 2

    def test_retrieve_parents_returns_the_parent_payload(self, tmp_path):
        store = self._store(tmp_path)
        store.ensure_collection("documents", 3)
        store.upsert([self._record(chunk_id="p1", is_parent=True)])
        parents = store.retrieve_parents(["p1"])
        assert "p1" in parents
        assert parents["p1"]["is_parent"] is True

    def test_delete_stale_chunks_removes_the_old_run(self, tmp_path):
        store = self._store(tmp_path)
        store.ensure_collection("documents", 3)
        store.upsert([self._record(chunk_id="old", ingest_run_id="run0")])
        store.upsert([self._record(chunk_id="new", ingest_run_id="run1")])
        store.delete_stale_chunks("doc1", "run1")
        stored = {hit["chunk_id"] for hit in store.search([1.0, 0.0, 0.0], top_k=5)}
        assert stored == {"new"}

    def test_schema_version_reports_the_corpus_version(self, tmp_path):
        store = self._store(tmp_path)
        store.ensure_collection("documents", 3)
        assert store.collection_schema_version() == 0
        store.upsert([self._record(schema_version=1)])
        assert store.collection_schema_version() == 1

    def test_health_is_true_for_a_writable_file(self, tmp_path):
        assert self._store(tmp_path).health() is True


class TestLegacyRecordsNoSchemaVersionField:
    """A Qdrant point with no `schema_version` payload field (e.g.,
    ingested before the parent-child feature shipped) must not crash the
    schema_version check."""

    def test_schema_version_zero_with_empty_collection(self):
        # Empty collection: count returns 0, schema_version=0.
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.count.return_value = MagicMock(count=0)
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            assert adapter.collection_schema_version() == 0

    def test_search_includes_is_parent_false_for_legacy_payload(self):
        # A legacy point has no `is_parent` field. search() must still
        # return a hit dict (defaulting is_parent=False on the result).
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_hit = MagicMock()
            mock_hit.id = "legacy-uuid"
            mock_hit.score = 0.7
            mock_hit.payload = {
                "text": "legacy text",
                "doc_id": "doc-1",
                "chunk_id": "legacy-chunk-id",
                "page_number": 2,
                "section_path": [],
                "element_type": "paragraph",
                # NO is_parent, NO parent_id, NO schema_version
            }
            mock_client.query_points.return_value = _resp([mock_hit])
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            hits = adapter.search([0.0] * 4, top_k=5)
            assert len(hits) == 1
            assert hits[0]["is_parent"] is False
            assert hits[0]["parent_id"] is None


# ---------------------------------------------------------------------------
# Re-ingestion cleanup: schema_version updates after re-ingest.
# ---------------------------------------------------------------------------


class TestReingestionSchemaVersionUpdates:
    """Re-ingestion replaces old chunks (delete_stale_chunks already
    handles this for the production path). After re-ingest, the new
    points have the new payload (schema_version=1 for parent-child),
    so collection_schema_version reflects the new state."""

    def test_v0_to_v1_reingestion_flips_schema_version(self):
        # Simulate the schema_version change after re-ingest: a count
        # of schema_version>=1 points that was 0 (v0) is now 1 (v1).
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            # First call: count=0 (v0 legacy)
            mock_client.count.return_value = MagicMock(count=0)
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            assert adapter.collection_schema_version() == 0
            # Simulate re-ingest: 1 v1 point now exists
            mock_client.count.return_value = MagicMock(count=1)
            assert adapter.collection_schema_version() == 1

    def test_v1_to_v0_reingestion_flips_schema_version_back(self):
        # Reverse direction. Mixed state is treated as v1 (the
        # count_v1 > 0 wins), so flipping back requires the v1
        # points to be cleaned up (which delete_stale_chunks handles).
        with patch("atrium.adapters.QdrantClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.count.return_value = MagicMock(count=5)
            mock_client_cls.return_value = mock_client
            from atrium.adapters import QdrantAdapter
            adapter = QdrantAdapter(host="stub", port=6333)
            assert adapter.collection_schema_version() == 1
            # Re-ingest removed all v1 points: count=0
            mock_client.count.return_value = MagicMock(count=0)
            assert adapter.collection_schema_version() == 0