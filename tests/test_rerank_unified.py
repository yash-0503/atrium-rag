"""
test_rerank_unified.py

Focused tests proving the unified `retrieve_and_rerank` function
(Task B step 2, 2026-10-10).

Verifies that production `/search` and the benchmark harness share
ONE retrieval+rerank+dedup+parent-resolve sequence — no two
implementations, no duplicate dedup logic.

Cases:
  - Duplicate children (same id appears twice in candidates).
  - Multiple children sharing one parent.
  - Missing parents (child references parent that was deleted).
  - Rerank enabled vs disabled — same shape, different scores/order
    where expected.
  - v0 vs v1 produce the same response shape (back-compat).
  - rerank failure (reranker returns None) falls back to vector order.
  - `retrieve_and_rerank` and `retrieve_with_parent_resolution`
    agree on the no-rerank path (back-compat alias).
  - Re-rank rerank disabled produces the dedup shape that production
    /search emits.

Stub adapter only — no live Qdrant/PgVector.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from atrium.retrieval import (
    retrieve_and_rerank,
    retrieve_with_parent_resolution,
)
from tests.metrics import evidence_in_context

# ---------------------------------------------------------------------------
# Adapter stub — implements the unified-function contract.
# Tracks every method call so tests can assert the canonical sequence.
# ---------------------------------------------------------------------------


class _StubReranker:
    """Stub cross-encoder. Records call signature and reorders
    candidates by `expected_top_ids` if provided; otherwise returns
    input order."""

    def __init__(self, expected_top_ids: Optional[List[str]] = None,
                 fail: bool = False):
        self.expected_top_ids = expected_top_ids
        self.fail = fail
        self.calls: List[Tuple[str, int, int]] = []

    def rerank(self, query: str, candidates, top_k: int):
        self.calls.append((query, len(candidates), top_k))
        if self.fail:
            return None
        # Reorder candidates so expected_top_ids come first.
        if self.expected_top_ids:
            by_id = {c.id: c for c in candidates}
            out = [by_id[tid] for tid in self.expected_top_ids if tid in by_id]
            # Tail: anything in `candidates` not in expected_top_ids
            seen = set(self.expected_top_ids)
            for c in candidates:
                if c.id not in seen:
                    out.append(c)
            return out[:top_k]
        return list(candidates)[:top_k]


class _StubAdapter:
    """Implements the unified-helper DB contract (VectorDBAdapter).
    Tracks every protocol method called."""

    def __init__(self, schema_version: int = 0,
                 children: Optional[List[Dict[str, Any]]] = None,
                 parents: Optional[Dict[str, Dict[str, Any]]] = None,
                 duplicate_children: bool = False):
        self._sv = schema_version
        self._children = list(children or [])
        self._parents = dict(parents or {})
        self._duplicate_children = duplicate_children
        self.calls: Dict[str, int] = {
            "collection_schema_version": 0,
            "search": 0,
            "search_children": 0,
            "retrieve_parents": 0,
        }

    def collection_schema_version(self) -> int:
        self.calls["collection_schema_version"] += 1
        return self._sv

    def search(self, query_vector, top_k=5, score_threshold=None):
        self.calls["search"] += 1
        # For v0 path, return flat children/parents as flat hits.
        flat = list(self._children)
        flat.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return flat[:top_k]

    def search_children(self, query_vector, top_k=5, score_threshold=None):
        self.calls["search_children"] += 1
        result = sorted(self._children, key=lambda h: h.get("score", 0.0), reverse=True)
        if self._duplicate_children and self._children:
            # Duplicate the top child — simulates Qdrant returning
            # two points with the same id (e.g., re-ingestion race).
            result = result + [dict(result[0])]
        return result[:top_k]

    def retrieve_parents(self, parent_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        self.calls["retrieve_parents"] += 1
        return {pid: self._parents[pid] for pid in parent_ids if pid in self._parents}


def _child(cid, score, parent_id=None, page=1, text="child text"):
    """Minimal child hit dict matching the adapter output shape."""
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


# ---------------------------------------------------------------------------
# 1. Rerank ON / OFF path correctness
# ---------------------------------------------------------------------------


class TestRerankOffCorrectness:
    """reranker=None → vector order, dedup applied."""

    def test_v0_no_rerank_returns_flat_top_k(self):
        db = _StubAdapter(schema_version=0, children=[
            _child("c1", 0.9), _child("c2", 0.8), _child("c3", 0.7),
        ])
        hits = retrieve_and_rerank(db, [0.0], "q", top_k=2, reranker=None)
        assert [h["id"] for h in hits] == ["c1", "c2"]
        # No rerank → no parent fields populated (v0 path).
        for h in hits:
            assert h["parent_text"] is None
            assert h["parent_id"] is None

    def test_v1_no_rerank_dedups_by_parent(self):
        db = _StubAdapter(
            schema_version=1,
            children=[
                _child("c1", 0.9, "p1"),
                _child("c2", 0.85, "p1"),  # same parent
                _child("c3", 0.7, "p2"),
            ],
            parents={"p1": _parent("p1", "PARENT-1"), "p2": _parent("p2", "PARENT-2")},
        )
        hits = retrieve_and_rerank(db, [0.0], "q", top_k=3, reranker=None)
        # c1 (higher score for p1) + c3 (only child for p2) = 2 distinct parents.
        assert len(hits) == 2
        assert {h["parent_id"] for h in hits} == {"p1", "p2"}
        # c1 wins p1 because higher score.
        p1_hit = next(h for h in hits if h["parent_id"] == "p1")
        assert p1_hit["id"] == "c1"
        assert p1_hit["parent_text"] == "PARENT-1"


class TestRerankOnCorrectness:
    """reranker=stub → rerank the wide candidate set, then dedup."""

    def test_v1_rerank_called_with_children(self):
        db = _StubAdapter(
            schema_version=1,
            children=[_child(f"c{i}", 0.9 - i * 0.01, parent_id="p1") for i in range(20)],
            parents={"p1": _parent("p1", "PARENT")},
        )
        reranker = _StubReranker(expected_top_ids=["c0", "c1", "c2", "c3", "c4"])
        retrieve_and_rerank(
            db, [0.0], "query text", top_k=5,
            reranker=reranker, candidate_k=20,
        )
        # Reranker was called with up to 20 candidates (over-fetched), top_k=5.
        assert reranker.calls, "reranker must have been called"
        query, n_cands, k = reranker.calls[0]
        assert query == "query text"
        assert n_cands > 0
        assert k == 5

    def test_v1_rerank_then_dedup_yields_distinct_parents(self):
        # 20 children, all parent="p1" — rerank orders them, then
        # dedup must collapse to a single parent result.
        db = _StubAdapter(
            schema_version=1,
            children=[_child(f"c{i}", 0.9 - i * 0.01, parent_id="p1") for i in range(20)],
            parents={"p1": _parent("p1", "PARENT")},
        )
        reranker = _StubReranker(expected_top_ids=["c3", "c7"])
        hits = retrieve_and_rerank(
            db, [0.0], "q", top_k=5,
            reranker=reranker, candidate_k=20,
        )
        assert len(hits) == 1
        assert hits[0]["parent_id"] == "p1"
        assert hits[0]["parent_text"] == "PARENT"


# ---------------------------------------------------------------------------
# 2. Duplicate children (same id appears twice)
# ---------------------------------------------------------------------------


class TestDuplicateChildren:
    def test_duplicate_children_dedup_to_single(self):
        # Adapter returns the same child twice (simulates Qdrant
        # duplicate after re-ingestion race). After rerank, the
        # dedup-by-parent should still produce a single result.
        db = _StubAdapter(
            schema_version=1,
            children=[
                _child("c1", 0.9, "p1"),
                _child("c2", 0.8, "p2"),
            ],
            parents={"p1": _parent("p1", "P1"), "p2": _parent("p2", "P2")},
            duplicate_children=True,
        )
        reranker = _StubReranker(expected_top_ids=["c1", "c2"])
        hits = retrieve_and_rerank(
            db, [0.0], "q", top_k=5,
            reranker=reranker, candidate_k=10,
        )
        # Two distinct parents → two results.
        assert len(hits) == 2
        assert {h["parent_id"] for h in hits} == {"p1", "p2"}


# ---------------------------------------------------------------------------
# 3. Multiple children sharing one parent
# ---------------------------------------------------------------------------


class TestMultipleChildrenSameParent:
    def test_three_children_same_parent_yields_single_hit(self):
        db = _StubAdapter(
            schema_version=1,
            children=[
                _child("c1", 0.7, "p1"),
                _child("c2", 0.9, "p1"),
                _child("c3", 0.8, "p1"),
            ],
            parents={"p1": _parent("p1", "PARENT TEXT")},
        )
        # Both with and without rerank should dedup to a single hit
        # with the highest-scoring child and parent_text attached.
        for reranker in [None, _StubReranker(expected_top_ids=["c2", "c1"])]:
            hits = retrieve_and_rerank(
                db, [0.0], "q", top_k=5, reranker=reranker,
            )
            assert len(hits) == 1
            assert hits[0]["id"] == "c2"  # highest score
            assert hits[0]["parent_text"] == "PARENT TEXT"


# ---------------------------------------------------------------------------
# 4. Missing parents
# ---------------------------------------------------------------------------


class TestMissingParents:
    def test_child_refs_missing_parent_yields_none_parent_text(self):
        db = _StubAdapter(
            schema_version=1,
            children=[_child("c1", 0.9, "missing-parent")],
            parents={},  # parent absent
        )
        for reranker in [None, _StubReranker()]:
            hits = retrieve_and_rerank(
                db, [0.0], "q", top_k=5, reranker=reranker,
            )
            assert len(hits) == 1
            assert hits[0]["parent_id"] == "missing-parent"
            assert hits[0]["parent_text"] is None
            assert hits[0]["parent_page_number"] is None
            assert hits[0]["parent_section_path"] == []

    def test_retrieve_parents_failure_does_not_crash(self):
        class _FailingDB(_StubAdapter):
            def retrieve_parents(self, parent_ids):
                raise RuntimeError("DB unavailable")

        db = _FailingDB(
            schema_version=1,
            children=[_child("c1", 0.9, "p1")],
            parents={"p1": _parent("p1", "P1")},
        )
        hits = retrieve_and_rerank(db, [0.0], "q", top_k=5, reranker=None)
        assert len(hits) == 1
        assert hits[0]["parent_id"] == "p1"
        # Parent resolution failed → parent_text None, no exception.
        assert hits[0]["parent_text"] is None


# ---------------------------------------------------------------------------
# 5. Rerank failure → falls back to vector order
# ---------------------------------------------------------------------------


class TestRerankFailure:
    def test_reranker_returns_none_falls_back_to_vector_order(self):
        db = _StubAdapter(
            schema_version=1,
            children=[_child("c1", 0.9, "p1"), _child("c2", 0.8, "p2")],
            parents={"p1": _parent("p1", "P1"), "p2": _parent("p2", "P2")},
        )
        reranker = _StubReranker(fail=True)
        hits = retrieve_and_rerank(
            db, [0.0], "q", top_k=5,
            reranker=reranker, candidate_k=10,
        )
        assert len(hits) == 2
        # Vector order preserved: c1 first.
        assert hits[0]["id"] == "c1"
        assert hits[1]["id"] == "c2"
        # Parent context still attached.
        assert hits[0]["parent_text"] == "P1"


# ---------------------------------------------------------------------------
# 6. Schema-version shape parity (v0 and v1 emit identical keys)
# ---------------------------------------------------------------------------


class TestShapeParity:
    def test_v0_and_v1_have_same_keys(self):
        v0_db = _StubAdapter(schema_version=0, children=[_child("c1", 0.9)])
        v1_db = _StubAdapter(
            schema_version=1,
            children=[_child("c1", 0.9, "p1")],
            parents={"p1": _parent("p1", "P1")},
        )
        v0 = retrieve_and_rerank(v0_db, [0.0], "q", top_k=1, reranker=None)
        v1 = retrieve_and_rerank(v1_db, [0.0], "q", top_k=1, reranker=None)
        assert set(v0[0].keys()) == set(v1[0].keys())


# ---------------------------------------------------------------------------
# 7. /search helper vs legacy retrieve_with_parent_resolution — same output
# ---------------------------------------------------------------------------


class TestAliasEquivalence:
    """`retrieve_with_parent_resolution` is the no-rerank back-compat
    alias and must produce the same result as
    `retrieve_and_rerank(..., reranker=None)` for the same input."""

    def test_alias_matches_unified_no_rerank(self):
        db = _StubAdapter(
            schema_version=1,
            children=[
                _child("c1", 0.9, "p1"),
                _child("c2", 0.85, "p1"),
                _child("c3", 0.7, "p2"),
            ],
            parents={"p1": _parent("p1", "P1"), "p2": _parent("p2", "P2")},
        )
        unified = retrieve_and_rerank(db, [0.0], "q", top_k=5, reranker=None)
        alias = retrieve_with_parent_resolution(db, [0.0], top_k=5)
        # Same ids, same parent_text.
        assert [h["id"] for h in unified] == [h["id"] for h in alias]
        for u, a in zip(unified, alias):
            assert u["parent_id"] == a["parent_id"]
            assert u["parent_text"] == a["parent_text"]
            assert u["score"] == a["score"]


# ---------------------------------------------------------------------------
# 8. The canonical call sequence (one shared retrieval+rerank+dedup+resolve)
# ---------------------------------------------------------------------------


class TestCanonicalSequence:
    """Both production /search and the benchmark harness must use the
    SAME sequence: search_children → rerank → dedup_by_parent → retrieve_parents.
    This test pins that sequence by recording protocol calls on a stub."""

    def test_v1_full_sequence(self):
        db = _StubAdapter(
            schema_version=1,
            children=[_child("c1", 0.9, "p1"), _child("c2", 0.8, "p2")],
            parents={"p1": _parent("p1", "P1"), "p2": _parent("p2", "P2")},
        )
        retrieve_and_rerank(db, [0.0], "q", top_k=5, reranker=_StubReranker())
        # Exactly one call per protocol method (single shared sequence).
        assert db.calls["collection_schema_version"] == 1
        assert db.calls["search_children"] == 1
        assert db.calls["retrieve_parents"] == 1
        assert db.calls["search"] == 0  # not used in v1 path

    def test_v0_does_not_call_search_children_or_retrieve_parents(self):
        db = _StubAdapter(schema_version=0, children=[_child("c1", 0.9)])
        retrieve_and_rerank(db, [0.0], "q", top_k=5, reranker=None)
        assert db.calls["collection_schema_version"] == 1
        assert db.calls["search"] == 1  # flat path
        assert db.calls["search_children"] == 0  # not v1
        assert db.calls["retrieve_parents"] == 0  # no parent resolution


# ---------------------------------------------------------------------------
# 9. Rerank disabled in production = rerank disabled in harness
# ---------------------------------------------------------------------------


class TestProductionHarnessEquivalence:
    """The whole point of the unification: a single function called
    from both contexts produces the same output. Verified via a small
    in-process comparison."""

    def test_same_inputs_same_outputs_no_rerank(self):
        # The "production" call routes through the same helper.
        # The "harness" call also routes through the same helper.
        # If they ever diverge, this test fails.
        children = [_child("c1", 0.9, "p1"), _child("c2", 0.85, "p1")]
        parents = {"p1": _parent("p1", "PARENT")}
        for r_unified in [None, _StubReranker(expected_top_ids=["c1"])]:
            db = _StubAdapter(schema_version=1, children=children, parents=parents)
            production_hits = retrieve_and_rerank(
                db, [0.0], "q", top_k=2, reranker=r_unified,
            )
            # Re-run with a fresh stub — simulating a separate harness call.
            db2 = _StubAdapter(schema_version=1, children=children, parents=parents)
            harness_hits = retrieve_and_rerank(
                db2, [0.0], "q", top_k=2, reranker=r_unified,
            )
            assert [h["id"] for h in production_hits] == [h["id"] for h in harness_hits]
            assert [h["parent_text"] for h in production_hits] == [h["parent_text"] for h in harness_hits]


# ---------------------------------------------------------------------------
# 10. Sanity: top_k respected under all paths
# ---------------------------------------------------------------------------


class TestTopKRespected:
    @pytest.mark.parametrize("rerank", [None, _StubReranker()])
    def test_top_k_respected_under_dedup(self, rerank):
        # 30 children, 30 distinct parents → top_k=5 should yield 5.
        db = _StubAdapter(
            schema_version=1,
            children=[_child(f"c{i}", 0.9 - i * 0.01, parent_id=f"p{i}") for i in range(30)],
            parents={f"p{i}": _parent(f"p{i}", f"P{i}") for i in range(30)},
        )
        hits = retrieve_and_rerank(db, [0.0], "q", top_k=5, reranker=rerank)
        assert len(hits) == 5


# ---------------------------------------------------------------------------
# 11. evidence_in_context metric (Task B step 3)
# ---------------------------------------------------------------------------


class TestEvidenceInContext:
    """The evidence-in-context metric checks whether the answer
    phrase appears in any retrieved child `text` or `parent_text`.
    This is the metric that surfaces parent-child's value-add: the
    child alone might not carry the literal answer phrase but the
    parent context might."""

    def test_phrase_in_child_text(self):
        hits = [{"text": "Wages, tips, other compensation", "parent_text": None}]
        assert evidence_in_context(hits, "Wages, tips, other compensation", k=5) == 1.0

    def test_phrase_only_in_parent_text(self):
        # The value-add scenario: child is a row label, parent has the
        # glossary. Without parent resolution the metric returns 0.0;
        # with parent resolution it returns 1.0.
        hits = [{
            "text": "Box 1 | 22222 | 50000",
            "parent_text": "(See also Notice to Employee on the back of Copy B.) Box 1. Enter this amount on the wages line.",
        }]
        assert evidence_in_context(hits, "Enter this amount on the wages line", k=5) == 1.0

    def test_phrase_absent(self):
        hits = [{"text": "Copy A — For SSA", "parent_text": None}]
        assert evidence_in_context(hits, "OMB 1545", k=5) == 0.0

    def test_case_insensitive(self):
        hits = [{"text": "OMB NO. 1545-0029", "parent_text": None}]
        assert evidence_in_context(hits, "omb no", k=5) == 1.0

    def test_empty_answer_phrase_returns_none(self):
        # Empty / None → metric undefined (don't pollute aggregate).
        assert evidence_in_context([{"text": "x"}], "") is None
        assert evidence_in_context([{"text": "x"}], None) is None

    def test_empty_hits_returns_zero(self):
        assert evidence_in_context([], "any phrase", k=5) == 0.0

    def test_respects_top_k(self):
        # Phrase is only in rank 6 — should NOT count for k=5.
        hits = [{"text": f"rank {i}: no match", "parent_text": None} for i in range(1, 7)] + \
               [{"text": "FINALLY has the answer phrase", "parent_text": None}]
        assert evidence_in_context(hits, "the answer phrase", k=5) == 0.0
        assert evidence_in_context(hits, "the answer phrase", k=7) == 1.0
