"""
tests/test_adapters_realtime.py

Complements tests/test_adapters_contract.py with cases that exercise
the production adapters (QdrantAdapter + PgVectorAdapter) directly
rather than via helper stubs. Contract tests pin the protocol through
the retrieval helper's expected shape; this file pins what the
production adapters themselves return on the edges that contract
tests don't reach.

Coverage:
  - QdrantAdapter.search() (v0 path) — empty results, score_threshold,
    return-type assertions, legacy payload (no is_parent field).
  - QdrantAdapter.search_children() — score_threshold passed through,
    empty results.
  - QdrantAdapter.retrieve_parents() — keys by chunk_id, falls back
    to str(point id) when payload.chunk_id absent.
  - QdrantAdapter._hit_to_dict() — None payload, missing fields.
  - PgVectorAdapter.search() / search_children() — SQL composition,
    empty results, threshold parameterization.
  - PgVectorAdapter.retrieve_parents() — empty input, missing rows,
    exception path.
  - PgVectorAdapter.collection_schema_version() — exception path.

All tests use MagicMock — no live Qdrant / Postgres required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from atrium.adapters import PgVectorAdapter, QdrantAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
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


def _make_qdrant_adapter():
    """Build a QdrantAdapter with a MagicMock client (bypasses real socket)."""
    client = MagicMock()
    adapter = QdrantAdapter.__new__(QdrantAdapter)
    adapter.client = client
    return adapter, client


def _make_pg_adapter():
    """Build a PgVectorAdapter with a MagicMock pool (bypasses real socket)."""
    pool = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    # conn as context manager
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    # cursor as context manager
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    conn.cursor.return_value = cursor
    # pool.connection() yields conn
    cm = MagicMock()
    cm.__enter__.return_value = conn
    cm.__exit__.return_value = False
    pool.connection.return_value = cm
    adapter = PgVectorAdapter.__new__(PgVectorAdapter)
    adapter.pool = pool
    return adapter, pool, conn, cursor


# ---------------------------------------------------------------------------
# QdrantAdapter.search() — v0 flat path
# ---------------------------------------------------------------------------


class TestQdrantClientApiRename:
    """qdrant-client renamed `search()` to `query_points()` in 1.10 and
    removed the old name in 1.20. `pip install atrium-rag` resolves the
    dependency to the newest release, so the adapter has to speak the
    installed client's dialect — and a version that only has the old
    method still has to work.

    Found by the performance measurement, which installs the built
    wheel rather than running this working tree: every /search came
    back 503 with "'QdrantClient' object has no attribute 'search'".
    """

    def test_modern_client_is_searched_with_query_points(self):
        adapter, client = _make_qdrant_adapter()
        client.query_points.return_value = _resp([])
        adapter.search([0.0] * 4, top_k=5)
        assert client.query_points.called
        # `search()` took query_vector=; `query_points()` takes query=.
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["query"] == [0.0] * 4
        assert "query_vector" not in kwargs

    def test_legacy_client_falls_back_to_search(self):
        adapter, client = _make_qdrant_adapter()
        # A client that predates the rename: no query_points at all.
        del client.query_points
        client.search.return_value = []
        adapter.search([0.0] * 4, top_k=5)
        assert client.search.called
        assert client.search.call_args.kwargs["query_vector"] == [0.0] * 4


class TestQdrantSearchV0Path:
    def test_search_returns_empty_list_when_no_results(self):
        adapter, client = _make_qdrant_adapter()
        client.query_points.return_value = _resp([])
        hits = adapter.search([0.0] * 4, top_k=5)
        assert hits == []

    def test_search_passes_top_k_through(self):
        adapter, client = _make_qdrant_adapter()
        client.query_points.return_value = _resp([])
        adapter.search([0.0] * 4, top_k=12)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["limit"] == 12
        assert kwargs["collection_name"] == "documents"

    def test_search_omits_score_threshold_when_none(self):
        adapter, client = _make_qdrant_adapter()
        client.query_points.return_value = _resp([])
        adapter.search([0.0] * 4, top_k=5, score_threshold=None)
        kwargs = client.query_points.call_args.kwargs
        assert "score_threshold" not in kwargs

    def test_search_passes_score_threshold_through(self):
        adapter, client = _make_qdrant_adapter()
        client.query_points.return_value = _resp([])
        adapter.search([0.0] * 4, top_k=5, score_threshold=0.42)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["score_threshold"] == 0.42

    def test_search_returns_full_hit_shape(self):
        adapter, client = _make_qdrant_adapter()
        h = MagicMock()
        h.id = "uuid-1"
        h.score = 0.91
        h.payload = {
            "text": "alpha",
            "doc_id": "doc-1",
            "chunk_id": "chunk-1",
            "page_number": 7,
            "section_path": ["H1", "H2"],
            "element_type": "paragraph",
        }
        client.query_points.return_value = _resp([h])
        hits = adapter.search([0.0] * 4, top_k=5)
        assert hits == [
            {
                "id": "uuid-1",
                "score": 0.91,
                "text": "alpha",
                "doc_id": "doc-1",
                "chunk_id": "chunk-1",
                "page_number": 7,
                "section_path": ["H1", "H2"],
                "element_type": "paragraph",
                "is_parent": False,  # default for v0 path
                "parent_id": None,
                # Empty for corpora ingested before the field existed.
                "chunking_strategy": "",
            }
        ]

    def test_search_returns_is_parent_true_when_payload_says_so(self):
        adapter, client = _make_qdrant_adapter()
        h = MagicMock()
        h.id = "p1"
        h.score = 0.95
        h.payload = {"text": "parent", "is_parent": True}
        client.query_points.return_value = _resp([h])
        hits = adapter.search([0.0] * 4, top_k=5)
        assert hits[0]["is_parent"] is True
        assert hits[0]["parent_id"] is None


# ---------------------------------------------------------------------------
# QdrantAdapter.search_children() — v1 child-only path
# ---------------------------------------------------------------------------


class TestQdrantSearchChildrenV1Path:
    def test_search_children_returns_empty_when_no_results(self):
        adapter, client = _make_qdrant_adapter()
        client.query_points.return_value = _resp([])
        hits = adapter.search_children([0.0] * 4, top_k=5)
        assert hits == []

    def test_search_children_passes_score_threshold_through(self):
        adapter, client = _make_qdrant_adapter()
        client.query_points.return_value = _resp([])
        adapter.search_children([0.0] * 4, top_k=5, score_threshold=0.3)
        kwargs = client.query_points.call_args.kwargs
        assert kwargs["score_threshold"] == 0.3

    def test_search_children_filter_has_is_parent_false(self):
        adapter, client = _make_qdrant_adapter()
        client.query_points.return_value = _resp([])
        adapter.search_children([0.0] * 4, top_k=5)
        kwargs = client.query_points.call_args.kwargs
        qf = kwargs["query_filter"]
        # Filter schema: must=[FieldCondition(key="is_parent", match=MatchValue(value=False))]
        keys = [c.key for c in qf.must]
        assert "is_parent" in keys
        # Confirm value is False on the match
        is_parent_cond = next(c for c in qf.must if c.key == "is_parent")
        assert is_parent_cond.match.value is False


# ---------------------------------------------------------------------------
# QdrantAdapter.retrieve_parents() — keying edge cases
# ---------------------------------------------------------------------------


class TestQdrantRetrieveParentsEdges:
    def test_retrieve_parents_keys_by_payload_chunk_id(self):
        adapter, client = _make_qdrant_adapter()
        p = MagicMock()
        p.id = "uuid-p1"  # point id differs from chunk_id
        p.payload = {
            "chunk_id": "logical-p1",
            "text": "parent text",
            "page_number": 2,
            "section_path": ["A"],
            "element_type": "paragraph",
        }
        client.retrieve.return_value = [p]
        result = adapter.retrieve_parents(["logical-p1"])
        # Keyed by chunk_id (not point id).
        assert "logical-p1" in result
        assert "uuid-p1" not in result
        assert result["logical-p1"]["text"] == "parent text"

    def test_retrieve_parents_falls_back_to_point_id_when_chunk_id_missing(self):
        adapter, client = _make_qdrant_adapter()
        p = MagicMock()
        p.id = "fallback-p2"
        p.payload = {"text": "p2 text"}  # no chunk_id in payload
        client.retrieve.return_value = [p]
        result = adapter.retrieve_parents(["fallback-p2"])
        assert "fallback-p2" in result
        assert result["fallback-p2"]["text"] == "p2 text"

    def test_retrieve_parents_handles_none_payload(self):
        adapter, client = _make_qdrant_adapter()
        p = MagicMock()
        p.id = "p3"
        p.payload = None
        client.retrieve.return_value = [p]
        result = adapter.retrieve_parents(["p3"])
        # All payload fields default cleanly.
        assert result["p3"]["text"] == ""
        assert result["p3"]["page_number"] is None
        assert result["p3"]["section_path"] == []
        assert result["p3"]["element_type"] == ""

    def test_retrieve_parents_only_returns_existing_parents(self):
        # Client.retrieve returns only the parents that exist; the helper
        # must surface only those — no synthetic entries for missing ids.
        adapter, client = _make_qdrant_adapter()
        p1 = MagicMock()
        p1.id = "p1"
        p1.payload = {"chunk_id": "p1", "text": "p1"}
        # Client was asked for p1+p2 but only p1 exists.
        client.retrieve.return_value = [p1]
        result = adapter.retrieve_parents(["p1", "p2-missing"])
        assert set(result.keys()) == {"p1"}

    def test_retrieve_parents_returns_empty_on_exception(self):
        adapter, client = _make_qdrant_adapter()
        client.retrieve.side_effect = RuntimeError("connection reset")
        assert adapter.retrieve_parents(["p1"]) == {}

    def test_retrieve_parents_passes_ids_to_client(self):
        adapter, client = _make_qdrant_adapter()
        client.retrieve.return_value = []
        adapter.retrieve_parents(["a", "b", "c"])
        kwargs = client.retrieve.call_args.kwargs
        assert kwargs["ids"] == ["a", "b", "c"]
        assert kwargs["collection_name"] == "documents"
        assert kwargs["with_payload"] is True
        assert kwargs["with_vectors"] is False


# ---------------------------------------------------------------------------
# ID roundtrip — what search_children emits must be exactly what
# retrieve_parents can resolve. The retrieval helper
# (app/retrieval.py:561-565) builds parent_ids from h["parent_id"]
# and looks them up in the dict retrieve_parents returns; the
# lookup must close. These tests pin that chain.
# ---------------------------------------------------------------------------


class TestQdrantParentIdRoundtrip:
    def test_search_children_parent_id_resolves_via_retrieve_parents(self):
        # Setup: parent point has chunk_id=parent-cid, child point has
        # parent_id=parent-cid. After search_children returns the
        # child, the helper passes child.parent_id to
        # retrieve_parents; the parent must be returned under that key.
        adapter, client = _make_qdrant_adapter()
        child = MagicMock()
        child.id = "child-uuid"
        child.score = 0.91
        child.payload = {
            "text": "child text",
            "doc_id": "doc-1",
            "chunk_id": "child-cid",
            "is_parent": False,
            "parent_id": "parent-cid",  # → emitted from search_children
        }
        parent = MagicMock()
        parent.id = "parent-uuid"  # point id; chunk_id is in payload
        parent.payload = {
            "chunk_id": "parent-cid",  # → key the dict is indexed by
            "text": "parent text",
            "page_number": 4,
            "section_path": ["S"],
            "element_type": "paragraph",
        }
        # search_children call → returns the child.
        client.query_points.return_value = _resp([child])
        # retrieve_parents call → returns the parent.
        client.retrieve.return_value = [parent]

        children = adapter.search_children([0.0] * 4, top_k=5)
        emitted_parent_id = children[0]["parent_id"]
        # The emitted ID flows directly into retrieve_parents.
        parent_map = adapter.retrieve_parents([emitted_parent_id])
        # The lookup closes.
        assert emitted_parent_id in parent_map
        assert parent_map[emitted_parent_id]["text"] == "parent text"

    def test_roundtrip_holds_when_parent_payload_lacks_chunk_id(self):
        # If the parent's payload.chunk_id is missing, retrieve_parents
        # falls back to str(point.id). The child.parent_id and the
        # parent's point id must agree for the lookup to close.
        adapter, client = _make_qdrant_adapter()
        child = MagicMock()
        child.id = "child-uuid"
        child.score = 0.5
        child.payload = {
            "chunk_id": "child-cid",
            "is_parent": False,
            "parent_id": "fallback-parent-id",  # = str(parent point id)
        }
        parent = MagicMock()
        parent.id = "fallback-parent-id"  # used as dict key when no chunk_id
        parent.payload = {
            "text": "fallback parent text",
            # No chunk_id field — adapter must fall back to str(p.id).
        }
        client.query_points.return_value = _resp([child])
        client.retrieve.return_value = [parent]

        children = adapter.search_children([0.0] * 4, top_k=5)
        parent_map = adapter.retrieve_parents([children[0]["parent_id"]])
        assert "fallback-parent-id" in parent_map
        assert parent_map["fallback-parent-id"]["text"] == "fallback parent text"

    def test_missing_parent_does_not_crash_helper(self):
        # If retrieve_parents returns nothing for the emitted parent_id,
        # the helper sees an empty dict and the child hit surfaces with
        # parent_text=None. Adapter must NOT raise.
        adapter, client = _make_qdrant_adapter()
        child = MagicMock()
        child.id = "child-uuid"
        child.score = 0.5
        child.payload = {
            "chunk_id": "child-cid",
            "is_parent": False,
            "parent_id": "missing-parent",
        }
        client.query_points.return_value = _resp([child])
        # retrieve returns nothing — parent not in collection.
        client.retrieve.return_value = []

        children = adapter.search_children([0.0] * 4, top_k=5)
        parent_map = adapter.retrieve_parents([children[0]["parent_id"]])
        assert parent_map == {}


# ---------------------------------------------------------------------------
# QdrantAdapter._hit_to_dict() — edge cases
# ---------------------------------------------------------------------------


class TestQdrantHitToDictEdges:
    def test_hit_to_dict_with_none_payload_uses_defaults(self):
        h = MagicMock()
        h.id = "x"
        h.score = 0.5
        h.payload = None
        d = QdrantAdapter._hit_to_dict(h)
        assert d["text"] == ""
        assert d["doc_id"] == ""
        assert d["chunk_id"] == "x"  # falls back to str(h.id)
        assert d["page_number"] is None
        assert d["section_path"] == []
        assert d["element_type"] == ""
        assert d["is_parent"] is False
        assert d["parent_id"] is None

    def test_hit_to_dict_normalizes_none_section_path_to_empty_list(self):
        h = MagicMock()
        h.id = "x"
        h.score = 0.5
        h.payload = {"section_path": None}
        d = QdrantAdapter._hit_to_dict(h)
        assert d["section_path"] == []

    def test_hit_to_dict_score_none_defaults_to_zero(self):
        h = MagicMock()
        h.id = "x"
        h.score = None
        h.payload = {}
        d = QdrantAdapter._hit_to_dict(h)
        assert d["score"] == 0.0


# ---------------------------------------------------------------------------
# QdrantAdapter.collection_schema_version() — edge cases
# ---------------------------------------------------------------------------


class TestQdrantSchemaVersionEdges:
    def test_schema_version_zero_on_empty_collection(self):
        adapter, client = _make_qdrant_adapter()
        client.count.return_value = MagicMock(count=0)
        assert adapter.collection_schema_version() == 0

    def test_schema_version_one_when_v1_points_present(self):
        adapter, client = _make_qdrant_adapter()
        client.count.return_value = MagicMock(count=1)
        assert adapter.collection_schema_version() == 1

    def test_schema_version_zero_when_count_raises(self):
        adapter, client = _make_qdrant_adapter()
        client.count.side_effect = RuntimeError("collection not found")
        assert adapter.collection_schema_version() == 0

    def test_schema_version_passes_range_filter_to_count(self):
        adapter, client = _make_qdrant_adapter()
        client.count.return_value = MagicMock(count=0)
        adapter.collection_schema_version()
        kwargs = client.count.call_args.kwargs
        assert kwargs["collection_name"] == "documents"
        # count_filter.must[0] should reference schema_version with Range(gte=1)
        cf = kwargs["count_filter"]
        keys = [c.key for c in cf.must]
        assert "schema_version" in keys


# ---------------------------------------------------------------------------
# PgVectorAdapter.search() / search_children() — SQL composition
# ---------------------------------------------------------------------------


class TestPgVectorSearchSQLComposition:
    def test_search_executes_flat_select_without_filter(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = []
        adapter.search([0.0] * 4, top_k=5)
        sql = cursor.execute.call_args.args[0]
        # No WHERE clause filtering is_parent; the helper builds SQL
        # dynamically and only adds WHERE when a filter applies.
        assert "WHERE COALESCE" not in sql
        assert "ORDER BY embedding" in sql
        assert "LIMIT" in sql

    def test_search_children_includes_is_parent_false_filter(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = []
        adapter.search_children([0.0] * 4, top_k=5)
        sql = cursor.execute.call_args.args[0]
        assert "COALESCE((payload->>'is_parent')::boolean, false) = false" in sql

    def test_search_with_score_threshold_adds_threshold_sql(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = []
        adapter.search([0.0] * 4, top_k=5, score_threshold=0.4)
        sql = cursor.execute.call_args.args[0]
        # Threshold SQL uses '1 - (embedding <=> %s::vector) >= %s'.
        assert "1 - (embedding <=> %s::vector) >= %s" in sql
        # Threshold params appear after the query vector in the params list.
        params = cursor.execute.call_args.args[1]
        assert 0.4 in params

    def test_search_no_threshold_omits_threshold_sql(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = []
        adapter.search([0.0] * 4, top_k=5, score_threshold=None)
        sql = cursor.execute.call_args.args[0]
        assert "1 - (embedding <=> %s::vector) >= %s" not in sql

    def test_search_returns_full_hit_shape(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = [
            (
                "uuid-1",
                "doc-1",
                {
                    "text": "alpha",
                    "page_number": 3,
                    "section_path": ["X"],
                    "element_type": "paragraph",
                },
                0.88,
            )
        ]
        hits = adapter.search([0.0] * 4, top_k=5)
        assert hits == [
            {
                "id": "uuid-1",
                "score": 0.88,
                "text": "alpha",
                "doc_id": "doc-1",
                "chunk_id": "uuid-1",
                "page_number": 3,
                "section_path": ["X"],
                "element_type": "paragraph",
                "is_parent": False,
                "parent_id": None,
                # Empty for corpora ingested before the field existed.
                "chunking_strategy": "",
            }
        ]

    def test_search_children_returns_is_parent_false(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = [
            ("uuid-c1", "doc-1", {"text": "c1", "is_parent": False, "parent_id": "p1"}, 0.7)
        ]
        hits = adapter.search_children([0.0] * 4, top_k=5)
        assert hits[0]["is_parent"] is False
        assert hits[0]["parent_id"] == "p1"

    def test_search_returns_empty_when_no_rows(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = []
        assert adapter.search([0.0] * 4, top_k=5) == []


# ---------------------------------------------------------------------------
# PgVectorAdapter.retrieve_parents() — edge cases
# ---------------------------------------------------------------------------


class TestPgVectorRetrieveParentsEdges:
    def test_retrieve_parents_empty_input_returns_empty_dict_without_db_call(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        result = adapter.retrieve_parents([])
        assert result == {}
        # No DB roundtrip on empty input.
        cursor.execute.assert_not_called()

    def test_retrieve_parents_keys_by_chunk_id_uuid(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = [
            ("uuid-p1", {"text": "p1", "page_number": 4, "section_path": ["S"], "element_type": "paragraph"}),
        ]
        result = adapter.retrieve_parents(["uuid-p1"])
        assert result["uuid-p1"]["text"] == "p1"
        assert result["uuid-p1"]["page_number"] == 4

    def test_retrieve_parents_missing_rows_omitted(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        # Asked for two parents, only one returned.
        cursor.fetchall.return_value = [
            ("uuid-p1", {"text": "p1"}),
        ]
        result = adapter.retrieve_parents(["uuid-p1", "uuid-p2-missing"])
        assert set(result.keys()) == {"uuid-p1"}

    def test_retrieve_parents_handles_none_payload(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.return_value = [("uuid-p3", None)]
        result = adapter.retrieve_parents(["uuid-p3"])
        # No crash on None payload; fields default cleanly.
        assert result["uuid-p3"]["text"] == ""
        assert result["uuid-p3"]["page_number"] is None
        assert result["uuid-p3"]["section_path"] == []

    def test_retrieve_parents_returns_empty_on_exception(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.execute.side_effect = RuntimeError("connection lost")
        assert adapter.retrieve_parents(["uuid-p1"]) == {}


# ---------------------------------------------------------------------------
# ID roundtrip — search_children's parent_id must resolve through
# retrieve_parents. Mirrors the Qdrant chain.
# ---------------------------------------------------------------------------


class TestPgVectorParentIdRoundtrip:
    def test_search_children_parent_id_resolves_via_retrieve_parents(self):
        # search_children → child with parent_id=<parent_uuid>.
        # retrieve_parents(<parent_uuid>) → parent row keyed by
        # str(chunk_id) = <parent_uuid>. Lookup closes.
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        # First fetchall: search_children returns the child.
        # Second fetchall: retrieve_parents returns the parent row.
        cursor.fetchall.side_effect = [
            [
                (
                    "child-uuid",
                    "doc-1",
                    {
                        "text": "child text",
                        "is_parent": False,
                        "parent_id": "parent-uuid",  # emitted
                    },
                    0.91,
                )
            ],
            [("parent-uuid", {"text": "parent text", "page_number": 4})],
        ]
        children = adapter.search_children([0.0] * 4, top_k=5)
        assert children[0]["parent_id"] == "parent-uuid"
        parent_map = adapter.retrieve_parents([children[0]["parent_id"]])
        assert "parent-uuid" in parent_map
        assert parent_map["parent-uuid"]["text"] == "parent text"

    def test_roundtrip_holds_when_parent_payload_is_null(self):
        # The chunk_id is the primary key (UUID), so the dict key
        # doesn't depend on payload.chunk_id. Even if payload IS NULL
        # the lookup closes — the row's chunk_id is the key.
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.side_effect = [
            [
                (
                    "child-uuid",
                    "doc-1",
                    {"is_parent": False, "parent_id": "parent-uuid"},
                    0.5,
                )
            ],
            [("parent-uuid", None)],  # payload is None
        ]
        children = adapter.search_children([0.0] * 4, top_k=5)
        parent_map = adapter.retrieve_parents([children[0]["parent_id"]])
        assert "parent-uuid" in parent_map
        # Defensive defaults from None payload.
        assert parent_map["parent-uuid"]["text"] == ""
        assert parent_map["parent-uuid"]["page_number"] is None

    def test_missing_parent_does_not_crash_helper(self):
        # search_children returns a child whose parent_id does not
        # correspond to any row. retrieve_parents returns an empty
        # result set. Lookup yields {}.
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchall.side_effect = [
            [
                (
                    "child-uuid",
                    "doc-1",
                    {"is_parent": False, "parent_id": "missing-parent"},
                    0.5,
                )
            ],
            [],  # retrieve_parents found no rows
        ]
        children = adapter.search_children([0.0] * 4, top_k=5)
        parent_map = adapter.retrieve_parents([children[0]["parent_id"]])
        assert parent_map == {}


# ---------------------------------------------------------------------------
# PgVectorAdapter.collection_schema_version() — edge cases
# ---------------------------------------------------------------------------


class TestPgVectorSchemaVersionEdges:
    def test_schema_version_zero_on_zero_count(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchone.return_value = (0,)
        assert adapter.collection_schema_version() == 0

    def test_schema_version_one_when_v1_rows_present(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchone.return_value = (3,)
        assert adapter.collection_schema_version() == 1

    def test_schema_version_zero_on_exception(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.execute.side_effect = RuntimeError("table missing")
        assert adapter.collection_schema_version() == 0

    def test_schema_version_sql_filters_by_schema_version_at_least_one(self):
        adapter, _pool, _conn, cursor = _make_pg_adapter()
        cursor.fetchone.return_value = (0,)
        adapter.collection_schema_version()
        sql = cursor.execute.call_args.args[0]
        assert "(payload->>'schema_version')::int >= 1" in sql
        assert "FROM chunks" in sql


# ---------------------------------------------------------------------------
# PgVectorAdapter._row_to_hit() — edge cases
# ---------------------------------------------------------------------------


class TestPgVectorRowToHitEdges:
    def test_row_to_hit_normalizes_none_payload(self):
        d = PgVectorAdapter._row_to_hit("uuid-x", "doc-1", None, 0.5)
        assert d["text"] == ""
        assert d["section_path"] == []
        assert d["element_type"] == ""
        assert d["is_parent"] is False
        assert d["parent_id"] is None

    def test_row_to_hit_normalizes_none_section_path_to_empty_list(self):
        d = PgVectorAdapter._row_to_hit("uuid-x", "doc-1", {"section_path": None}, 0.5)
        assert d["section_path"] == []

    def test_row_to_hit_score_default_zero_when_none(self):
        d = PgVectorAdapter._row_to_hit("uuid-x", "doc-1", {}, 0.0)
        assert d["score"] == 0.0
