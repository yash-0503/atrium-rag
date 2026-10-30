"""
test_retrieval.py

Unit tests for `app.retrieval.retrieve_with_parent_resolution` (Task B,
2026-10-10).

The helper is the single shared retrieval path used by both /search
(production) and the benchmark harness. These tests pin its contract
across the three corpus shapes:

  - Legacy (schema_version == 0): flat retrieval, no filter, no parent
    resolution. Hits carry None parent fields.
  - v1 (schema_version >= 1): `is_parent=False` filter on vector
    search, batched `retrieve_parents` for parent context, dedup by
    `parent_id` keeping the highest-scoring child per parent.
  - Mixed / no schema_version: same as legacy (deterministic — a
    collection with no v1 points is treated as legacy).

The tests use a stub adapter (no live Qdrant / PgVector) so the
helper's logic is verified without infrastructure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from atrium.retrieval import (
    DEDUP_FACTOR,
    MAX_CANDIDATE_FACTOR,
    collection_schema_version,
    retrieve_with_parent_resolution,
)


class _StubDB:
    """Minimal adapter stub that implements the parent-child protocol.

    Each instance is bound to a `schema_version` and a `_children` /
    `_parents` table. `search_children` returns up to `top_k` children
    sorted by score (descending). `retrieve_parents` returns the
    requested parents. No vector index — the stub filters and sorts
    in-process so the helper's logic is testable without Qdrant."""

    def __init__(
        self,
        schema_version: int = 0,
        children: Optional[List[Dict[str, Any]]] = None,
        parents: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._sv = schema_version
        self._children = list(children or [])
        self._parents = dict(parents or {})
        self.children_call_count = 0
        self.parents_call_count = 0
        self.search_call_count = 0

    def collection_schema_version(self) -> int:
        return self._sv

    def search(self, query_vector, top_k=5, score_threshold=None):
        self.search_call_count += 1
        # Return children + parents (no filter) — matches the legacy
        # /search shape that the helper recognizes.
        flat = list(self._children) + [
            {
                "id": f"parent-{pid}",
                "score": 0.9,
                "text": p.get("text", ""),
                "doc_id": p.get("doc_id", "doc-1"),
                "chunk_id": pid,
                "page_number": p.get("page_number"),
                "section_path": p.get("section_path") or [],
                "element_type": "paragraph",
                "is_parent": True,
                "parent_id": None,
            }
            for pid, p in self._parents.items()
        ]
        flat.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return flat[:top_k]

    def search_children(self, query_vector, top_k=5, score_threshold=None):
        self.children_call_count += 1
        sorted_children = sorted(
            self._children, key=lambda h: h.get("score", 0.0), reverse=True,
        )
        return sorted_children[:top_k]

    def retrieve_parents(self, parent_ids):
        self.parents_call_count += 1
        return {pid: self._parents[pid] for pid in parent_ids if pid in self._parents}


def _child(cid, score, parent_id, page=1, text="child text"):
    return {
        "id": cid,
        "score": score,
        "text": text,
        "doc_id": "doc-1",
        "chunk_id": cid,
        "page_number": page,
        "section_path": [],
        "element_type": "paragraph",
        "is_parent": False,
        "parent_id": parent_id,
    }


def _parent(pid, text, page=1):
    return {
        "text": text,
        "page_number": page,
        "section_path": [],
        "element_type": "paragraph",
    }


class TestSchemaVersionDetection:
    def test_legacy_corpus_returns_zero(self):
        db = _StubDB(schema_version=0)
        assert collection_schema_version(db) == 0

    def test_v1_corpus_returns_one(self):
        db = _StubDB(schema_version=1)
        assert collection_schema_version(db) == 1

    def test_adapter_without_method_returns_zero(self):
        class _NoMethodDB:
            pass
        assert collection_schema_version(_NoMethodDB()) == 0

    def test_adapter_method_raising_returns_zero(self):
        class _RaisingDB:
            def collection_schema_version(self):
                raise RuntimeError("boom")
        assert collection_schema_version(_RaisingDB()) == 0


class TestLegacyFlatPath:
    """schema_version == 0: no filter, no parent resolution, no dedup."""

    def test_legacy_returns_flat_hits(self):
        db = _StubDB(schema_version=0, children=[
            _child("c1", 0.9, None),  # no parent_id (flat chunk)
            _child("c2", 0.8, None),
        ])
        hits = retrieve_with_parent_resolution(db, [0.0, 0.0], top_k=2)
        assert len(hits) == 2
        # Hits sorted by score descending
        assert hits[0]["id"] == "c1"
        assert hits[1]["id"] == "c2"
        # No parent context (legacy has none)
        for h in hits:
            assert h["parent_text"] is None
            assert h["parent_id"] is None
            assert h["is_parent"] is False

    def test_legacy_calls_search_not_search_children(self):
        db = _StubDB(schema_version=0, children=[_child("c1", 0.9, None)])
        retrieve_with_parent_resolution(db, [0.0], top_k=1)
        assert db.search_call_count == 1
        assert db.children_call_count == 0
        assert db.parents_call_count == 0

    def test_legacy_top_k_respected(self):
        db = _StubDB(schema_version=0, children=[
            _child(f"c{i}", 1.0 - i * 0.1, None) for i in range(10)
        ])
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=3)
        assert len(hits) == 3


class TestV1ParentChildRetrieval:
    """schema_version >= 1: filter children, resolve parents, dedup."""

    def test_v1_returns_resolved_children(self):
        # 2 children, both pointing to parent "p1". Should dedup to 1
        # result with parent_text populated.
        db = _StubDB(
            schema_version=1,
            children=[
                _child("c1", 0.9, "p1"),
                _child("c2", 0.8, "p1"),
            ],
            parents={
                "p1": _parent("p1", "PARENT TEXT", page=2),
            },
        )
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=5)
        assert len(hits) == 1
        assert hits[0]["id"] == "c1"  # higher score wins
        assert hits[0]["parent_id"] == "p1"
        assert hits[0]["parent_text"] == "PARENT TEXT"
        assert hits[0]["parent_page_number"] == 2

    def test_v1_dedup_keeps_highest_scoring_child_per_parent(self):
        db = _StubDB(
            schema_version=1,
            children=[
                _child("c1", 0.7, "p1"),  # lower score
                _child("c2", 0.9, "p1"),  # higher score, must win
                _child("c3", 0.8, "p2"),
            ],
            parents={
                "p1": _parent("p1", "P1 TEXT"),
                "p2": _parent("p2", "P2 TEXT"),
            },
        )
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=5)
        assert len(hits) == 2
        # p1 → c2 (higher score), p2 → c3
        p1_hit = next(h for h in hits if h["parent_id"] == "p1")
        p2_hit = next(h for h in hits if h["parent_id"] == "p2")
        assert p1_hit["id"] == "c2"
        assert p2_hit["id"] == "c3"
        assert p1_hit["parent_text"] == "P1 TEXT"
        assert p2_hit["parent_text"] == "P2 TEXT"

    def test_v1_returns_top_k_distinct_parents(self):
        # 3 children, each pointing to a different parent. top_k=2
        # should return 2 results (one per parent), highest-scoring 2.
        db = _StubDB(
            schema_version=1,
            children=[
                _child("c1", 0.9, "p1"),
                _child("c2", 0.8, "p2"),
                _child("c3", 0.7, "p3"),
            ],
            parents={
                "p1": _parent("p1", "P1 TEXT"),
                "p2": _parent("p2", "P2 TEXT"),
                "p3": _parent("p3", "P3 TEXT"),
            },
        )
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=2)
        assert len(hits) == 2
        assert {h["parent_id"] for h in hits} == {"p1", "p2"}

    def test_v1_missing_parent_returns_none(self):
        # Child references parent "missing" — parent was deleted from
        # the collection (e.g., re-ingestion cleanup). Helper must not
        # crash; parent_text is None.
        db = _StubDB(
            schema_version=1,
            children=[_child("c1", 0.9, "missing-parent")],
            parents={},  # parent absent
        )
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=5)
        assert len(hits) == 1
        assert hits[0]["parent_id"] == "missing-parent"
        assert hits[0]["parent_text"] is None

    def test_v1_orphan_child_no_parent_id_treated_as_result(self):
        # Defensive — chunk with no parent_id in v1 corpus. Should
        # surface as a result with parent_text=None rather than be
        # silently dropped.
        db = _StubDB(
            schema_version=1,
            children=[
                _child("c1", 0.9, None),  # orphan
            ],
            parents={},
        )
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=5)
        assert len(hits) == 1
        assert hits[0]["parent_text"] is None
        assert hits[0]["parent_id"] is None

    def test_v1_retrieve_parents_failure_does_not_crash(self):
        class _FailingDB(_StubDB):
            def retrieve_parents(self, parent_ids):
                raise RuntimeError("Qdrant unavailable")

        db = _FailingDB(
            schema_version=1,
            children=[_child("c1", 0.9, "p1")],
            parents={"p1": _parent("p1", "P1 TEXT")},
        )
        # Helper must catch the exception and surface parent_text=None.
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=5)
        assert len(hits) == 1
        assert hits[0]["parent_id"] == "p1"
        assert hits[0]["parent_text"] is None

    def test_v1_overfetch_factor_constant(self):
        # Lock the dedup factor — bumping it changes over-fetch cost.
        assert DEDUP_FACTOR == 3
        assert MAX_CANDIDATE_FACTOR == 5

    def test_v1_overfetch_capped_at_max(self):
        # All children map to one parent (degenerate). top_k=5,
        # DEDUP_FACTOR=3 → over-fetch 15, dedup yields 1 result. We
        # should still cap at MAX_DEDUP_FACTOR (5) to bound work.
        db = _StubDB(
            schema_version=1,
            children=[
                _child(f"c{i}", 1.0 - i * 0.01, "p1") for i in range(50)
            ],
            parents={"p1": _parent("p1", "P1 TEXT")},
        )
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=5)
        # DEDUP_FACTOR=3 → 15 candidates fetched, but MAX_DEDUP_FACTOR=5
        # would bound candidate_k to 5*5=25. Either way, dedup yields 1.
        # The point: top_k=5 distinct parents is impossible here, but
        # we don't unbounded-fetch.
        assert len(hits) == 1
        assert db.children_call_count == 1  # exactly one over-fetch
        # Verify over-fetch wasn't unbounded (cap is 5x).
        # children passed in: 50, top_k=5 → candidate_k min(5*3, 5*5)=15
        # We can't directly assert the cap from outside, but the
        # constant pin above already does.

    def test_v1_sort_descending_by_score_after_dedup(self):
        # After dedup, results must be re-sorted by score descending.
        db = _StubDB(
            schema_version=1,
            children=[
                _child("c1", 0.7, "p1"),
                _child("c2", 0.9, "p2"),
                _child("c3", 0.8, "p3"),
            ],
            parents={
                "p1": _parent("p1", "P1"),
                "p2": _parent("p2", "P2"),
                "p3": _parent("p3", "P3"),
            },
        )
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=3)
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)


class TestV0AndV1ShapeCompatibility:
    """Both schema versions must produce the same response shape, so
    the caller (main.py or harness) doesn't need to branch."""

    def _hit_keys(self, hits):
        if not hits:
            return set()
        return set(hits[0].keys())

    def test_legacy_and_v1_have_same_keys(self):
        v0_db = _StubDB(schema_version=0, children=[_child("c1", 0.9, None)])
        v1_db = _StubDB(
            schema_version=1,
            children=[_child("c1", 0.9, "p1")],
            parents={"p1": _parent("p1", "P1")},
        )
        v0_hits = retrieve_with_parent_resolution(v0_db, [0.0], top_k=1)
        v1_hits = retrieve_with_parent_resolution(v1_db, [0.0], top_k=1)
        assert self._hit_keys(v0_hits) == self._hit_keys(v1_hits)


class TestLegacyVsV1DismissibleInput:
    """The helper must handle empty input the same in both modes."""

    @pytest.mark.parametrize("schema_v", [0, 1])
    def test_no_children_returns_empty_list(self, schema_v):
        db = _StubDB(schema_version=schema_v, children=[])
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=5)
        assert hits == []

    @pytest.mark.parametrize("schema_v", [0, 1])
    def test_top_k_zero_returns_empty(self, schema_v):
        db = _StubDB(schema_version=schema_v, children=[_child("c1", 0.9, None)])
        hits = retrieve_with_parent_resolution(db, [0.0], top_k=0)
        assert hits == []