"""
test_retrieval_confidence_gate.py

Edge-case tests for the 2026-10-18 confidence-gate heuristic that
controls when `retrieve_and_rerank` consults BM25 on top of dense
retrieval. The gate is a heuristic, not a correctness guarantee —
these tests pin its observable behavior so a future tuning pass
can't silently change the routing policy.

Covered cases (each test asserts one observable property):

  * No `bm25_index` → gate never opens, even on uncertain dense.
  * Zero dense candidates + BM25 → gate opens (BM25 is the only
    retrieval path).
  * One dense candidate + BM25 → gate OPENS. A single result has
    no defined top-1/top-2 gap; the gate must NOT treat a singleton
    as confident (it would silently skip BM25 on real failures).
  * Two candidates, large gap, high top1 → gate stays closed.
  * Two candidates, tiny gap → gate opens.
  * Two candidates, top1 below DENSE_MIN_SCORE → gate opens even
    if the gap is wide.
  * Two candidates with equal scores (gap=0) → gate opens.
  * Tied scores at top-1 (gap undefined, not zero) → covered by
    the one-candidate case above; this one covers "two results
    that happen to have the same score" → gap=0 → open.
  * BM25 raises during consult → dense list is preserved, no
    exception propagates out of `retrieve_and_rerank`.
  * Raising both thresholds routes more queries to BM25.
  * Lowering both thresholds routes more queries to dense-only.

These tests use a stub DB and a stub BM25 index so no live
infrastructure is required.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytest

from atrium.retrieval import (
    DENSE_MIN_SCORE,
    _should_consult_bm25,
    retrieve_and_rerank,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubDB:
    """Minimal v0 adapter stub. Returns the children list verbatim."""

    def __init__(self, children: Optional[List[Dict[str, Any]]] = None) -> None:
        self._children = list(children or [])
        self.search_call_count = 0

    def collection_schema_version(self) -> int:
        return 0

    def search(self, query_vector, top_k=5, score_threshold=None):
        self.search_call_count += 1
        sorted_c = sorted(
            self._children, key=lambda h: h.get("score", 0.0), reverse=True,
        )
        return sorted_c[:top_k]


def _child(cid: str, score: float, text: str = "child text") -> Dict[str, Any]:
    return {
        "id": cid,
        "score": score,
        "text": text,
        "doc_id": "doc-1",
        "chunk_id": cid,
        "page_number": 1,
        "section_path": [],
        "element_type": "paragraph",
        "is_parent": False,
        "parent_id": None,
    }


class _StubBM25:
    """Records calls; `explode` flag raises on `search_with_meta`."""

    def __init__(self, hits: Optional[List[Dict[str, Any]]] = None, explode: bool = False):
        self._hits = list(hits or [])
        self.explode = explode
        self.search_with_meta_call_count = 0
        self.add_doc_call_count = 0

    def search_with_meta(self, query, top_k=5):
        self.search_with_meta_call_count += 1
        if self.explode:
            raise RuntimeError("BM25 exploded")
        return [(h["chunk_id"], h.get("score", 0.0), h) for h in self._hits[:top_k]]


# ---------------------------------------------------------------------------
# _should_consult_bm25 — pure decision tests
# ---------------------------------------------------------------------------


class TestShouldConsultBM25:
    """Pure decision tests for the gate. No I/O."""

    def test_no_bm25_index_never_consults(self):
        # Even on highly uncertain dense (empty list, gap 0), no
        # BM25 means no consult.
        assert _should_consult_bm25([], None) is False
        assert _should_consult_bm25([_child("c1", 0.9)], None) is False
        assert _should_consult_bm25(
            [_child("c1", 0.9), _child("c2", 0.9)], None,
        ) is False

    def test_zero_candidates_opens_gate(self):
        # BM25 is the only retrieval path.
        assert _should_consult_bm25([], _StubBM25()) is True

    def test_one_candidate_opens_gate(self):
        # CRITICAL: a single result has an undefined gap. The gate
        # must NOT assume confidence by assigning a missing
        # second score to zero. This is the trap the test is here
        # to prevent.
        assert _should_consult_bm25([_child("only", 0.99)], _StubBM25()) is True

    def test_high_gap_high_top1_keeps_gate_closed(self):
        # Clear winner: 0.95 - 0.50 = 0.45 gap (well above 0.05),
        # top1 0.95 well above 0.55 min. Gate stays closed.
        candidates = [_child("c1", 0.95), _child("c2", 0.50)]
        assert _should_consult_bm25(candidates, _StubBM25()) is False

    def test_tiny_gap_opens_gate(self):
        # 0.61 - 0.60 = 0.01 gap (below 0.05). Even with high
        # top1, the gate opens because dense is uncertain.
        candidates = [_child("c1", 0.61), _child("c2", 0.60)]
        assert _should_consult_bm25(candidates, _StubBM25()) is True

    def test_top1_below_min_score_opens_gate(self):
        # Wide gap, but top1 is below DENSE_MIN_SCORE. The score
        # itself is too low to trust.
        candidates = [_child("c1", 0.40), _child("c2", 0.10)]
        assert (
            DENSE_MIN_SCORE > 0.40
        )  # sanity: test is meaningful under current default
        assert _should_consult_bm25(candidates, _StubBM25()) is True

    def test_tied_scores_open_gate(self):
        # gap=0 → not confident → open. The trap the user
        # explicitly flagged: tied scores must not be treated as
        # "top2=0, gap=top1, confident".
        candidates = [_child("c1", 0.80), _child("c2", 0.80)]
        assert _should_consult_bm25(candidates, _StubBM25()) is True

    def test_missing_score_treated_as_zero(self):
        # Defensive: a malformed candidate with no `score` is
        # treated as 0.0. That alone opens the gate (top1=0 < min).
        candidates = [
            {"id": "c1", "chunk_id": "c1", "text": "no score"},
            _child("c2", 0.9),
        ]
        assert _should_consult_bm25(candidates, _StubBM25()) is True


# ---------------------------------------------------------------------------
# Threshold tuning — verify the user's stated direction
# ---------------------------------------------------------------------------


class TestThresholdDirection:
    """The gate's response to threshold changes must match the
    user-stated direction (2026-10-18): raising either threshold
    routes MORE queries to hybrid; lowering either routes MORE
    queries to dense-only."""

    def test_raising_gap_routes_more_to_hybrid(self, monkeypatch):
        # Boundary case: gap=0.05 exactly. At default gap=0.05,
        # the comparison is `gap >= gap` → True → confident. Raise
        # gap to 0.06 → `gap >= 0.06` → False → NOT confident →
        # consult. Same candidates, same top1, gate flips.
        candidates = [_child("c1", 0.80), _child("c2", 0.75)]
        # Default: gap=0.05, top1=0.80 (>= 0.55) → confident → no consult
        assert _should_consult_bm25(candidates, _StubBM25()) is False
        # Raise gap slightly: same candidates now NOT confident
        monkeypatch.setattr("atrium.retrieval.DENSE_CONFIDENT_GAP", 0.06)
        assert _should_consult_bm25(candidates, _StubBM25()) is True

    def test_raising_min_score_routes_more_to_hybrid(self, monkeypatch):
        # Boundary: top1=0.55 exactly. Default min=0.55 → confident.
        # Raise min to 0.56 → same top1 NOT confident.
        candidates = [_child("c1", 0.55), _child("c2", 0.20)]
        # Wide gap (0.35 ≥ 0.05), top1=0.55 (≥ 0.55) → confident.
        assert _should_consult_bm25(candidates, _StubBM25()) is False
        monkeypatch.setattr("atrium.retrieval.DENSE_MIN_SCORE", 0.56)
        # Now top1 < min → not confident → consult.
        assert _should_consult_bm25(candidates, _StubBM25()) is True

    def test_lowering_gap_routes_more_to_dense_only(self, monkeypatch):
        # Boundary: gap=0.01. At default gap=0.05, NOT confident
        # (gap < 0.05). Lower gap to 0.005 → 0.01 ≥ 0.005 → confident.
        candidates = [_child("c1", 0.80), _child("c2", 0.79)]
        assert _should_consult_bm25(candidates, _StubBM25()) is True
        monkeypatch.setattr("atrium.retrieval.DENSE_CONFIDENT_GAP", 0.005)
        assert _should_consult_bm25(candidates, _StubBM25()) is False


# ---------------------------------------------------------------------------
# retrieve_and_rerank integration — does the gate actually take effect?
# ---------------------------------------------------------------------------


class TestRetrieveAndRerankGateIntegration:
    """End-to-end through `retrieve_and_rerank` with a stub DB and
    stub BM25. Verifies that the gate's decision changes the
    observable behavior: BM25 is consulted or not, and BM25
    failures are handled gracefully."""

    def test_high_confidence_dense_skips_bm25(self):
        # Wide gap, high top1. Gate stays closed → BM25 is NOT
        # consulted. The dense list is returned as-is.
        db = _StubDB(children=[_child("c1", 0.95), _child("c2", 0.50)])
        bm25 = _StubBM25(hits=[_child("bm1", 0.5)])
        hits = retrieve_and_rerank(
            db, [0.0, 0.0], "any", top_k=2, bm25_index=bm25,
        )
        assert bm25.search_with_meta_call_count == 0
        # Dense-only: no RRF fusion, no BM25-only stub.
        assert [h["id"] for h in hits] == ["c1", "c2"]

    def test_low_confidence_dense_consults_bm25(self):
        # Tiny gap → gate opens → BM25 is consulted and fused.
        db = _StubDB(children=[_child("c1", 0.61), _child("c2", 0.60)])
        bm25 = _StubBM25(hits=[_child("bm1", 0.5)])
        hits = retrieve_and_rerank(
            db, [0.0, 0.0], "any", top_k=2, bm25_index=bm25,
        )
        assert bm25.search_with_meta_call_count == 1
        # Fused: the dense list is preserved, BM25 contributes RRF
        # ranks. The exact ordering depends on RRF math — we just
        # assert that BM25 was consulted.
        assert len(hits) >= 1

    def test_one_candidate_consults_bm25(self):
        # The trap test: a single dense result must consult BM25.
        db = _StubDB(children=[_child("only", 0.99)])
        bm25 = _StubBM25(hits=[_child("bm1", 0.5)])
        retrieve_and_rerank(
            db, [0.0, 0.0], "any", top_k=1, bm25_index=bm25,
        )
        assert bm25.search_with_meta_call_count == 1

    def test_zero_dense_consults_bm25(self):
        # No dense hits → BM25 is the only retrieval path.
        db = _StubDB(children=[])
        bm25 = _StubBM25(hits=[_child("bm1", 0.5)])
        hits = retrieve_and_rerank(
            db, [0.0, 0.0], "any", top_k=5, bm25_index=bm25,
        )
        assert bm25.search_with_meta_call_count == 1
        # The BM25 hit was promoted to a full hit shape.
        assert any(h.get("chunk_id") == "bm1" for h in hits)

    def test_no_bm25_index_skips_bm25_entirely(self):
        # When bm25_index is None, no consult ever happens, even
        # on the most uncertain dense result.
        db = _StubDB(children=[_child("c1", 0.61), _child("c2", 0.60)])
        hits = retrieve_and_rerank(
            db, [0.0, 0.0], "any", top_k=2, bm25_index=None,
        )
        # No exception, dense list returned as-is.
        assert [h["id"] for h in hits] == ["c1", "c2"]

    def test_bm25_raising_falls_back_to_dense(self, caplog):
        # If the gate opened and BM25 then raises, the dense
        # candidate list must be preserved. No exception escapes.
        db = _StubDB(children=[_child("c1", 0.61), _child("c2", 0.60)])
        bm25 = _StubBM25(explode=True)
        with caplog.at_level(logging.WARNING, logger="ingest.retrieval"):
            hits = retrieve_and_rerank(
                db, [0.0, 0.0], "any", top_k=2, bm25_index=bm25,
            )
        # The dense list survives the BM25 failure.
        assert [h["id"] for h in hits] == ["c1", "c2"]
        # And a warning was logged so an operator can see it.
        assert any("BM25 consult failed" in r.message for r in caplog.records)

    def test_routing_decision_logged(self, caplog):
        # The instrumented INFO log must fire with the decision,
        # top1, top2, and gap — so post-hoc analysis can tell
        # which queries the gate opened on.
        db = _StubDB(children=[_child("c1", 0.95), _child("c2", 0.50)])
        bm25 = _StubBM25(hits=[])
        with caplog.at_level(logging.INFO, logger="ingest.retrieval"):
            retrieve_and_rerank(
                db, [0.0, 0.0], "any", top_k=2, bm25_index=bm25,
            )
        routing_records = [
            r for r in caplog.records
            if "routing:" in r.getMessage()
        ]
        assert routing_records, "expected a routing log line"
        msg = routing_records[0].getMessage()
        assert "consult_bm25=False" in msg
        assert "top1=" in msg
        assert "top2=" in msg
        # gap is computed as float subtraction; the log value is
        # whatever Python's %s produces (e.g. 0.5 or
        # 0.44999999999999996). We only assert the key is present
        # and that the printed value rounds to the expected
        # 0.95 - 0.50 = 0.45.
        assert "gap=" in msg
        gap_token = [tok for tok in msg.split() if tok.startswith("gap=")][0]
        gap_value = float(gap_token.split("=")[1])
        assert gap_value == pytest.approx(0.45, abs=1e-6)
