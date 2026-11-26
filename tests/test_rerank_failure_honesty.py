"""
tests/test_rerank_failure_honesty.py

A reranker that fails must be REPORTED as having failed.

The adapters used to return their input order when inference failed.
That list is indistinguishable from a real ranking: it carries the
caller's original cosine scores, so `retrieve_and_rerank` set
`reranked=True` and labelled those scores `score_type="rerank"`. The
API then told the caller the cross-encoder had run when it had not —
and the numbers in the response were not the numbers it claimed to be.

The contract now: a failed rerank RAISES. `retrieve_and_rerank`
catches it, serves the vector ranking (the search still succeeds — the
failure is fail-open), and reports `reranked=False` with the dense
`score_type` intact.

These tests pin both halves of that: the retrieval layer reports the
truth, and the real adapters raise rather than returning a plausible
looking list.
"""

from __future__ import annotations

import pytest

from atrium.rerank_adapter import (
    CohereReranker,
    FlashRankReranker,
    LocalBgeReranker,
    RerankCandidate,
)
from atrium.retrieval import retrieve_and_rerank

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubDB:
    """A minimal v0 store: flat chunks, no parents."""

    def __init__(self, n: int = 3):
        self.hits = [
            {
                "id": f"c{i}", "chunk_id": f"c{i}", "text": f"text {i}",
                "score": 0.9 - i * 0.1, "doc_id": "d", "page_number": 1,
                "section_path": [], "element_type": "paragraph",
                "is_parent": False, "parent_id": None, "chunking_strategy": "",
            }
            for i in range(n)
        ]

    def collection_schema_version(self) -> int:
        return 0

    def search(self, v, top_k=5, score_threshold=None):
        return self.hits[:top_k]

    def search_children(self, v, top_k=5, score_threshold=None):
        return self.hits[:top_k]

    def retrieve_parents(self, ids):
        return {}


class _RaisingReranker:
    """A cross-encoder that failed — the shape a real adapter now has."""

    def rerank(self, query, candidates, top_k):
        raise RuntimeError("CUDA out of memory")


class _WorkingReranker:
    """A cross-encoder that worked."""

    def rerank(self, query, candidates, top_k):
        return list(reversed(candidates))


# ---------------------------------------------------------------------------
# 1. The retrieval layer reports the failure
# ---------------------------------------------------------------------------


class TestRetrievalReportsFailureHonestly:
    def test_a_raising_reranker_is_reported_as_not_reranked(self):
        db = _StubDB()
        stats: dict = {}
        hits = retrieve_and_rerank(
            db, [0.1] * 3, "q", top_k=3,
            reranker=_RaisingReranker(), stats=stats,
        )
        assert stats["reranked"] is False, (
            "a reranker that raised must not be reported as having reranked"
        )
        # Fail-open: the search still answers.
        assert len(hits) == 3

    def test_a_failed_rerank_keeps_the_dense_score_type(self):
        """The scores are still cosine values — say so."""
        db = _StubDB()
        hits = retrieve_and_rerank(
            db, [0.1] * 3, "q", top_k=3, reranker=_RaisingReranker(),
        )
        assert hits
        for hit in hits:
            assert hit["score_type"] == "cosine", (
                "a failed rerank must not label cosine scores as rerank scores"
            )

    def test_a_failed_rerank_keeps_the_original_cosine_scores(self):
        db = _StubDB()
        hits = retrieve_and_rerank(
            db, [0.1] * 3, "q", top_k=3, reranker=_RaisingReranker(),
        )
        assert [h["score"] for h in hits] == pytest.approx([0.9, 0.8, 0.7])

    def test_a_working_reranker_is_still_reported_as_reranked(self):
        """The control: the fix must not turn every rerank into a failure."""
        db = _StubDB()
        stats: dict = {}
        hits = retrieve_and_rerank(
            db, [0.1] * 3, "q", top_k=3, reranker=_WorkingReranker(), stats=stats,
        )
        assert stats["reranked"] is True
        assert all(h["score_type"] == "rerank" for h in hits)


# ---------------------------------------------------------------------------
# 2. The real adapters raise
# ---------------------------------------------------------------------------


def _candidates(n: int = 2):
    return [
        RerankCandidate(id=f"c{i}", text=f"text {i}", score=1.0 - i * 0.1)
        for i in range(n)
    ]


class TestLocalBgeRerankerRaises:
    def test_inference_failure_raises(self):
        adapter = LocalBgeReranker(model_name="bge-reranker-base")

        class _BrokenModel:
            def predict(self, pairs, **kwargs):
                raise RuntimeError("CUDA out of memory")

        # `_model` is set, so `_ensure_loaded` returns early and the
        # injected model is what `rerank()` calls.
        adapter._model = _BrokenModel()

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            adapter.rerank("q", _candidates(), 2)

    def test_it_does_not_return_the_input_order_instead(self):
        """The bug: a returned list of the input order looks like a ranking."""
        adapter = LocalBgeReranker(model_name="bge-reranker-base")

        class _BrokenModel:
            def predict(self, pairs, **kwargs):
                raise RuntimeError("boom")

        adapter._model = _BrokenModel()

        raised = False
        try:
            result = adapter.rerank("q", _candidates(), 2)
        except RuntimeError:
            raised = True
            result = None
        assert raised, "the adapter returned a list instead of raising"
        assert result is None


class TestFlashRankRerankerRaises:
    def test_model_load_failure_raises(self):
        adapter = FlashRankReranker(model_name="ms-marco-MiniLM-L-12-v2")
        # Simulate a model that could not be loaded. `_ensure_loaded`
        # re-raises this, and `rerank()` must not swallow it.
        adapter._load_error = RuntimeError("model download failed")

        with pytest.raises(RuntimeError, match="model download failed"):
            adapter.rerank("q", _candidates(), 2)

    def test_rerank_call_failure_raises(self):
        # The `flashrank` extra is optional, and `rerank()` imports
        # `RerankRequest` from it. Without the extra installed the
        # import fails before the call is ever made, so this test can
        # only reach the call path when the extra is present.
        pytest.importorskip(
            "flashrank", reason="the optional `flashrank` extra is not installed"
        )
        adapter = FlashRankReranker(model_name="ms-marco-MiniLM-L-12-v2")

        class _BrokenRanker:
            def rerank(self, request):
                raise RuntimeError("onnx session failed")

        adapter._ranker = _BrokenRanker()

        with pytest.raises(RuntimeError, match="onnx session failed"):
            adapter.rerank("q", _candidates(), 2)

    def test_a_missing_flashrank_extra_raises(self):
        """A deployment without the optional extra must degrade, not lie.

        `flashrank` is an optional extra, so "not installed" is a real
        deployment state. `rerank()` imports `RerankRequest` from it,
        so on such a deployment the import is the failure — and it has
        to raise for the same reason every other failure does.
        """
        adapter = FlashRankReranker(model_name="ms-marco-MiniLM-L-12-v2")
        # Whatever the local environment has installed, the failure the
        # adapter must report is the missing module.
        adapter._load_error = ModuleNotFoundError("No module named 'flashrank'")

        with pytest.raises(ModuleNotFoundError):
            adapter.rerank("q", _candidates(), 2)


class TestCohereRerankerRaises:
    def test_sdk_failure_raises(self):
        adapter = CohereReranker(api_key="test-key")

        class _BrokenClient:
            def rerank(self, **kwargs):
                raise RuntimeError("cohere 500")

        adapter._client = _BrokenClient()

        with pytest.raises(RuntimeError, match="cohere 500"):
            adapter.rerank("q", _candidates(), 2)

    def test_http_failure_raises(self, monkeypatch):
        adapter = CohereReranker(api_key="test-key")
        # The raw httpx path is used when the `cohere` SDK is absent.
        adapter._client = None

        def _boom(*args, **kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr("httpx.post", _boom)

        with pytest.raises(RuntimeError, match="connection refused"):
            adapter.rerank("q", _candidates(), 2)


# ---------------------------------------------------------------------------
# 3. The end-to-end contract
# ---------------------------------------------------------------------------


class TestFailureReachesTheApiShape:
    def test_the_api_level_flag_and_score_type_agree(self):
        """`rerank: false` in the response must match the hits' score_type.

        The API builds its `rerank` flag from `stats["reranked"]` and the
        hits' `score_type` from the same retrieval call. If they ever
        disagree, the caller is being told two different things about
        the same query.
        """
        db = _StubDB()
        stats: dict = {}
        hits = retrieve_and_rerank(
            db, [0.1] * 3, "q", top_k=3, reranker=_RaisingReranker(), stats=stats,
        )
        api_rerank_flag = bool(stats.get("reranked", False))
        hit_score_types = {h["score_type"] for h in hits}

        assert api_rerank_flag is False
        assert hit_score_types == {"cosine"}
        assert ("rerank" in hit_score_types) is False


# ---------------------------------------------------------------------------
# 4. The HTTP contract
# ---------------------------------------------------------------------------


class TestSearchEndpointReportsTheFailure:
    """`POST /search` must not claim a rerank that did not happen."""

    @pytest.fixture
    def client(self, monkeypatch):
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from atrium.job_store import JobStore
        from atrium.main import app

        embedder = MagicMock()
        embedder.embed.return_value = [[0.0] * 8]
        embedder.dimension = 8
        embedder.health.return_value = True

        db = MagicMock()
        db.health.return_value = True
        db.ensure_collection.return_value = None
        db.count_chunks.return_value = 3
        db.collection_schema_version.return_value = 0
        db.search.return_value = [
            {
                "id": "c0", "chunk_id": "c0", "text": "text 0", "score": 0.9,
                "doc_id": "d", "page_number": 1, "section_path": [],
                "element_type": "paragraph", "is_parent": False,
                "parent_id": None, "chunking_strategy": "",
            }
        ]
        db.search_children.return_value = db.search.return_value
        db.retrieve_parents.return_value = {}

        monkeypatch.setattr("atrium.main.embedder", embedder)
        monkeypatch.setattr("atrium.main.db", db)
        monkeypatch.setattr("atrium.main.job_store", JobStore(db_path=":memory:"))
        # The endpoint builds the reranker per request; make it return
        # the REAL adapter, with a model that fails. A stub that raises
        # would only test the retrieval layer (which already handled a
        # raising adapter); the real adapter is what used to swallow
        # the failure and report a successful rerank.
        def _broken_reranker(force: bool = False):
            adapter = LocalBgeReranker(model_name="bge-reranker-base")

            class _BrokenModel:
                def predict(self, pairs, **kwargs):
                    raise RuntimeError("CUDA out of memory")

            adapter._model = _BrokenModel()
            return adapter

        monkeypatch.setattr(
            "atrium.rerank_adapter.build_reranker", _broken_reranker
        )
        return TestClient(app)

    def test_search_reports_rerank_false(self, client):
        res = client.post("/search", json={"query": "retention policy", "rerank": True})
        assert res.status_code == 200
        body = res.json()

        assert body["rerank"] is False, (
            "the API reported a rerank that never ran"
        )
        # A reranker was configured, so the API can still say one was
        # available — the two flags are different facts.
        assert body["rerank_available"] is True
        assert body["results"], "the search returned nothing"
        for hit in body["results"]:
            # The cross-encoder did not run, so no score may be
            # labelled as a cross-encoder score. The score type may
            # legitimately be "cosine" or "rrf" — depending on whether
            # the hybrid fusion ran — but it must never be "rerank".
            assert hit["score_type"] != "rerank", (
                "a failed rerank must not label scores as rerank scores"
            )
