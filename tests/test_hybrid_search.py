"""
tests/test_hybrid_search.py

The hybrid (dense + BM25/RRF) retrieval path.

The sparse half of the hybrid path was written and measured but was
never wired into a product surface: `/search` called
`retrieve_and_rerank` without a `bm25_index`, so the RRF fusion, the
confidence gate, `DENSE_CONFIDENT_GAP` and `DENSE_MIN_SCORE` were all
unreachable in production.

These tests pin the wiring:

  * the corpus lexical index is built from the store and kept fresh;
  * the confidence gate decides when to consult it;
  * a query written in a script the sparse tokenizer cannot segment
    (Han / Kana / Hangul) does NOT consult it — the sparse ranker is
    blind to those documents, and fusing a blind signal into a good
    dense ranking only dilutes it;
  * the API says which of the two it did, in `hybrid` and in
    `score_type`.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from unittest.mock import MagicMock

import pytest

from atrium.bm25 import BM25Index, contains_cjk, tokenize
from atrium.corpus_index import CorpusBm25Index, build_bm25_index
from atrium.retrieval import _should_consult_bm25, retrieve_and_rerank

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str, doc_id: str = "d1", page: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "page_number": page,
        "section_path": ["Chapter 1"],
        "element_type": "paragraph",
        "is_parent": False,
        "parent_id": None,
        "chunking_strategy": "structure_aware",
    }


class _Store:
    """A store with a corpus, good enough to build an index from."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def count_chunks(self) -> int:
        return len(self._chunks)

    def list_chunks(self):
        return list(self._chunks)


class _Hit(dict):
    pass


def _dense_hits(scores):
    return [
        {
            "id": f"c{i}", "chunk_id": f"c{i}", "text": f"t{i}", "score": s,
            "doc_id": "d1", "page_number": 1, "section_path": [],
            "element_type": "paragraph", "is_parent": False,
            "parent_id": None, "chunking_strategy": "",
        }
        for i, s in enumerate(scores)
    ]


# ---------------------------------------------------------------------------
# 1. CJK detection
# ---------------------------------------------------------------------------


class TestCjkDetection:
    @pytest.mark.parametrize("text", [
        "日本語のテキスト",
        "中文文本",
        "한국어 텍스트",
        "ひらがな",
        "カタカナ",
        "an english line 日本語 mixed",
    ])
    def test_scripts_the_sparse_tokenizer_cannot_read_are_detected(self, text):
        assert contains_cjk(text) is True

    @pytest.mark.parametrize("text", [
        "retention policy",
        "naïve café résumé",
        "Ελληνικά κείμενο",
        "Русский текст",
        "policy for records",
        "",
    ])
    def test_latin_and_other_scripts_are_not_flagged(self, text):
        assert contains_cjk(text) is False


# ---------------------------------------------------------------------------
# 2. The confidence gate
# ---------------------------------------------------------------------------


class TestConfidenceGate:
    def test_no_index_means_no_consultation(self):
        assert _should_consult_bm25(_dense_hits([0.9, 0.8]), None, "retention") is False

    def test_a_confident_dense_result_skips_the_sparse_ranker(self):
        # A big gap between top-1 and top-2, and a high top-1: dense
        # is confident, so BM25 is not consulted.
        assert _should_consult_bm25(
            _dense_hits([0.95, 0.60]), object(), "retention policy"
        ) is False

    def test_an_uncertain_dense_result_consults_the_sparse_ranker(self):
        # A small gap: dense is not confident, so BM25 is consulted.
        assert _should_consult_bm25(
            _dense_hits([0.62, 0.60]), object(), "retention policy"
        ) is True

    def test_a_low_scoring_dense_result_consults_the_sparse_ranker(self):
        # top-1 below DENSE_MIN_SCORE: nothing here is a good match.
        assert _should_consult_bm25(
            _dense_hits([0.40, 0.30]), object(), "retention policy"
        ) is True

    def test_a_cjk_query_never_consults_the_sparse_ranker(self):
        r"""The sparse tokenizer cannot segment these scripts.

        `\w+` treats a run of Han characters as one token, so the
        sparse ranker is blind to the documents it would be asked
        about. Consulting it would dilute a good dense ranking with a
        signal that cannot see the corpus — so the gate declines,
        whatever the dense confidence looks like.
        """
        # Even with an uncertain dense result, a CJK query stays dense.
        assert _should_consult_bm25(
            _dense_hits([0.40, 0.30]), object(), "日本語 の テキスト"
        ) is False
        # And with no dense hits at all — where BM25 would otherwise be
        # the only retrieval path.
        assert _should_consult_bm25([], object(), "中文查询") is False

    def test_a_latin_query_still_consults_the_sparse_ranker(self):
        assert _should_consult_bm25(
            _dense_hits([0.40, 0.30]), object(), "retention policy"
        ) is True


# ---------------------------------------------------------------------------
# 3. The BM25 index itself
# ---------------------------------------------------------------------------


def _reference_search(docs, query, k1=1.5, b=0.75):
    """The original BM25 algorithm, transcribed.

    The index was rewritten to use an inverted postings map and an
    incrementally-maintained average document length. That is a
    performance change, not a scoring change, and this reference pins
    that: the rewritten index must return the same documents in the
    same order, with the same scores.
    """
    tf = {d: Counter(tokenize(t)) for d, t in docs.items()}
    dl = {d: len(tokenize(t)) for d, t in docs.items()}
    df: dict = {}
    for counter in tf.values():
        for term in counter:
            df[term] = df.get(term, 0) + 1
    n = len(docs)
    avgdl = sum(dl.values()) / max(1, n)
    scores: dict = {}
    for q in tokenize(query):
        d_f = df.get(q, 0)
        if d_f == 0:
            continue
        idf = math.log((n - d_f + 0.5) / (d_f + 0.5) + 1.0)
        for doc_id, counter in tf.items():
            f = counter.get(q, 0)
            if f == 0:
                continue
            num = f * (k1 + 1.0)
            den = f + k1 * (1.0 - b + b * dl[doc_id] / max(1e-9, avgdl))
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * (num / den)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class TestBm25Index:
    DOCS = {
        "d1": "The retention period for customer records is seven years after closure.",
        "d2": "Legal holds override the standard schedule and suspend all deletion timers.",
        "d3": "Audit logs are retained for three years and expire automatically.",
        "d4": "The quick brown fox jumps over the lazy dog retention schedule table.",
        "d5": "retention retention retention records records records are kept.",
    }

    @pytest.mark.parametrize("query", [
        "retention records", "legal holds suspend", "fox dog", "records",
        "nonexistent term here",
    ])
    def test_the_rewrite_scores_exactly_like_the_original(self, query):
        index = BM25Index()
        for doc_id, text in self.DOCS.items():
            index.add_doc(doc_id, text)
        mine = [(d, s) for d, s, _ in index.search_with_meta(query, top_k=50)]
        theirs = _reference_search(self.DOCS, query)
        assert [d for d, _ in mine] == [d for d, _ in theirs]
        assert all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(mine, theirs))

    def test_replacing_a_document_does_not_double_count_its_length(self):
        """The average document length must stay correct.

        The old implementation recomputed it from every document on
        every `add_doc`; the new one maintains a total. A replacement
        is where a wrong total shows up — the document is counted
        twice (or not at all), and every score shifts.
        """
        index = BM25Index()
        index.add_doc("d1", "one two three")
        index.add_doc("d2", "four five")
        index.add_doc("d1", "completely different words here now")
        assert index.n_docs == 2
        # 5 tokens ("completely different words here now") + 2 ("four five")
        assert index._avgdl == pytest.approx((5 + 2) / 2)

    def test_remove_doc_retracts_a_document(self):
        index = BM25Index()
        index.add_doc("d1", "retention records policy")
        index.add_doc("d2", "unrelated text about cooking")
        assert index.n_docs == 2

        assert index.remove_doc("d1") is True
        assert index.n_docs == 1
        assert index.remove_doc("d1") is False
        # The retracted document must not be findable any more.
        assert [d for d, _ in index.search("retention records", top_k=5)] == []
        # And it must not still be counted in the average length.
        assert index._avgdl == pytest.approx(len(tokenize("unrelated text about cooking")))

    def test_search_visits_only_the_documents_that_match(self):
        """A term nobody has must not cost a scan of the corpus.

        The postings map is what makes this possible; the old
        implementation walked every document for every query term.
        """
        index = BM25Index()
        for i in range(50):
            index.add_doc(f"d{i}", f"common filler uniquetoken{i}")
        # A term that exists in exactly one document.
        assert [d for d, _ in index.search("uniquetoken7", top_k=50)] == ["d7"]
        # A term that exists in none — nothing is scanned, nothing returned.
        assert index.search("absentterm", top_k=50) == []


# ---------------------------------------------------------------------------
# 4. The corpus index
# ---------------------------------------------------------------------------


class TestCorpusBm25Index:
    def test_it_is_built_from_the_store(self):
        store = _Store([
            _chunk("c1", "retention policy for records"),
            _chunk("c2", "legal holds suspend deletion"),
        ])
        index = CorpusBm25Index(store).get()
        assert index is not None
        assert index.n_docs == 2
        assert [d for d, _ in index.search("retention records", top_k=5)] == ["c1"]

    def test_the_index_carries_the_metadata_a_bm25_only_hit_needs(self):
        """A chunk the dense ranker missed must still come back whole."""
        store = _Store([_chunk("c1", "retention policy", doc_id="doc9", page=4)])
        index = CorpusBm25Index(store).get()
        doc_id, _score, meta = index.search_with_meta("retention", top_k=1)[0]
        assert doc_id == "c1"
        assert meta["doc_id"] == "doc9"
        assert meta["page_number"] == 4
        assert meta["section_path"] == ["Chapter 1"]
        assert meta["text"] == "retention policy"

    def test_it_is_cached_between_calls(self):
        store = _Store([_chunk("c1", "retention policy")])
        manager = CorpusBm25Index(store)
        first = manager.get()
        assert manager.get() is first

    def test_a_changed_chunk_count_rebuilds_it(self):
        """A document was ingested or removed — pick it up at once."""
        store = _Store([_chunk("c1", "retention policy")])
        manager = CorpusBm25Index(store)
        first = manager.get()

        store._chunks.append(_chunk("c2", "legal holds"))
        second = manager.get()
        assert second is not first
        assert second.n_docs == 2

    def test_an_explicit_invalidate_rebuilds_it(self):
        store = _Store([_chunk("c1", "retention policy")])
        manager = CorpusBm25Index(store)
        first = manager.get()
        manager.invalidate()
        assert manager.get() is not first

    def test_a_replacement_is_picked_up_when_the_ttl_expires(self):
        """A document replaced in place keeps the chunk count.

        The chunk count does not change, so the index is stale until
        the TTL expires. That is the documented limitation of the
        cache; the test pins it so the behaviour cannot change
        silently.
        """
        store = _Store([_chunk("c1", "the original text about retention")])
        manager = CorpusBm25Index(store, ttl_seconds=0.0)
        first = manager.get()

        store._chunks[0] = _chunk("c1", "a completely different replacement")
        second = manager.get()
        assert second is not first
        assert [d for d, _ in second.search("original", top_k=5)] == []
        assert [d for d, _ in second.search("replacement", top_k=5)] == ["c1"]
        assert first is not second

    def test_a_store_that_cannot_be_read_yields_no_index(self):
        """A broken store is a reason to serve the dense ranking."""
        store = MagicMock()
        store.count_chunks.side_effect = RuntimeError("database is down")
        assert CorpusBm25Index(store).get() is None

    def test_a_store_that_cannot_be_listed_yields_no_index(self):
        store = MagicMock()
        store.count_chunks.return_value = 3
        store.list_chunks.side_effect = RuntimeError("cannot read")
        assert CorpusBm25Index(store).get() is None

    def test_build_bm25_index_skips_chunks_with_no_id(self):
        index = build_bm25_index([
            {"chunk_id": "", "text": "orphan"},
            _chunk("c1", "kept"),
        ])
        assert index.n_docs == 1


# ---------------------------------------------------------------------------
# 5. The retrieval path
# ---------------------------------------------------------------------------


class TestRetrievalUsesTheIndex:
    class _DB:
        def __init__(self, hits):
            self._hits = hits

        def collection_schema_version(self):
            return 0

        def search(self, v, top_k=5, score_threshold=None):
            return self._hits[:top_k]

        def search_children(self, v, top_k=5, score_threshold=None):
            return self._hits[:top_k]

        def retrieve_parents(self, ids):
            return {}

    def test_a_fused_result_is_reported_as_hybrid(self):
        db = self._DB(_dense_hits([0.40, 0.30]))
        index = build_bm25_index([
            _chunk("c0", "a different document entirely"),
            _chunk("c1", "another one"),
        ])
        stats: dict = {}
        hits = retrieve_and_rerank(
            db, [0.1] * 4, "a different document", top_k=5,
            bm25_index=index, stats=stats,
        )
        assert stats["hybrid"] is True
        assert hits
        assert hits[0]["score_type"] == "rrf"

    def test_a_dense_only_result_is_reported_as_not_hybrid(self):
        db = self._DB(_dense_hits([0.40, 0.30]))
        stats: dict = {}
        retrieve_and_rerank(db, [0.1] * 4, "a different document", top_k=5, stats=stats)
        assert stats["hybrid"] is False

    def test_a_cjk_query_never_reaches_the_fusion(self):
        db = self._DB(_dense_hits([0.40, 0.30]))
        index = build_bm25_index([_chunk("c0", "another one")])
        stats: dict = {}
        hits = retrieve_and_rerank(
            db, [0.1] * 4, "日本語のテキスト", top_k=5,
            bm25_index=index, stats=stats,
        )
        assert stats["hybrid"] is False
        assert all(h["score_type"] == "cosine" for h in hits)


# ---------------------------------------------------------------------------
# 6. The HTTP contract
# ---------------------------------------------------------------------------


class TestSearchEndpointHybrid:
    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient

        from atrium.job_store import JobStore
        from atrium.local_store import SqliteAdapter
        from atrium.main import app

        embedder = MagicMock()
        embedder.embed.return_value = [[0.0] * 8]
        embedder.dimension = 8
        embedder.health.return_value = True

        # A real store, so the lexical index is built from real rows.
        store = SqliteAdapter(path=os.path.join(tempfile.mkdtemp(), "atrium_local.db"))
        store.ensure_collection("documents", 8)
        from atrium.models import ChunkRecord, SourceMetadata

        store.replace_chunks("d1", [
            ChunkRecord(
                doc_id="d1", chunk_id="c0", ingest_run_id="r1",
                text="the retention policy for customer records",
                embedding=[0.1] * 8, element_type="paragraph",
                source=SourceMetadata(file_name="f.pdf", page_number=1),
                char_span=(0, 1), content_hash="h",
            )
        ], "r1")

        monkeypatch.setattr("atrium.main.embedder", embedder)
        monkeypatch.setattr("atrium.main.db", store)
        monkeypatch.setattr("atrium.main.job_store", JobStore(db_path=":memory:"))
        return TestClient(app)

    def test_search_reports_whether_it_was_hybrid(self, client):
        res = client.post("/search", json={"query": "retention policy records"})
        assert res.status_code == 200
        body = res.json()
        # The flag exists and agrees with the hits' score type.
        assert "hybrid" in body
        if body["hybrid"]:
            assert body["results"][0]["score_type"] == "rrf"
        else:
            assert body["results"][0]["score_type"] == "cosine"

    def test_a_cjk_query_is_served_dense_only(self, client):
        res = client.post("/search", json={"query": "日本語のテキスト"})
        assert res.status_code == 200
        body = res.json()
        assert body["hybrid"] is False
        assert all(h["score_type"] != "rrf" for h in body["results"])

    def test_hybrid_can_be_switched_off(self, client, monkeypatch):
        monkeypatch.setenv("HYBRID_ENABLED", "0")
        res = client.post("/search", json={"query": "retention policy records"})
        assert res.status_code == 200
        body = res.json()
        assert body["hybrid"] is False
        assert all(h["score_type"] != "rrf" for h in body["results"])


# ---------------------------------------------------------------------------
# 7. The store contract
# ---------------------------------------------------------------------------


LIST_CHUNKS_FIELDS = (
    "chunk_id", "doc_id", "text", "page_number", "section_path",
    "element_type", "is_parent", "parent_id", "chunking_strategy",
)


class TestListChunks:
    """Every store must expose its corpus for the lexical index.

    The index is built from the store rather than maintained beside
    it, so `list_chunks` is the one method the hybrid path needs from
    every backend. The shape is pinned so the index builder does not
    have to special-case a backend.
    """

    def _assert_shape(self, chunks):
        assert chunks
        for chunk in chunks:
            for field in LIST_CHUNKS_FIELDS:
                assert field in chunk, f"missing {field}"
            assert chunk["text"], "a chunk with no text cannot be indexed"

    def test_sqlite_lists_its_corpus(self, tmp_path):
        from atrium.local_store import SqliteAdapter
        from atrium.models import ChunkRecord, SourceMetadata

        store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
        store.ensure_collection("documents", 4)
        store.replace_chunks("d1", [
            ChunkRecord(
                doc_id="d1", chunk_id="c0", ingest_run_id="r1",
                text="the retention policy text",
                embedding=[0.1] * 4, element_type="paragraph",
                source=SourceMetadata(
                    file_name="f.pdf", page_number=3, section_path=["S1"],
                ),
                char_span=(0, 1), content_hash="h",
            )
        ], "r1")

        chunks = store.list_chunks()
        self._assert_shape(chunks)
        assert chunks[0]["chunk_id"] == "c0"
        assert chunks[0]["doc_id"] == "d1"
        # The text lives in a column, not the payload — the same
        # reason `retrieve_parents` reads the column.
        assert chunks[0]["text"] == "the retention policy text"
        assert chunks[0]["page_number"] == 3
        assert chunks[0]["section_path"] == ["S1"]

    def test_qdrant_lists_its_corpus(self):
        import uuid

        from qdrant_client import QdrantClient

        from atrium.adapters import QdrantAdapter
        from atrium.models import ChunkRecord, SourceMetadata

        adapter = QdrantAdapter.__new__(QdrantAdapter)
        adapter.client = QdrantClient(":memory:")
        adapter.ensure_collection("documents", 4)
        adapter.upsert([
            ChunkRecord(
                doc_id="d1", chunk_id=str(uuid.uuid4()), ingest_run_id="r1",
                text="the retention policy text",
                embedding=[0.1] * 4, element_type="paragraph",
                source=SourceMetadata(
                    file_name="f.pdf", page_number=3, section_path=["S1"],
                ),
                char_span=(0, 1), content_hash="h",
            )
        ])

        chunks = adapter.list_chunks()
        self._assert_shape(chunks)
        assert chunks[0]["text"] == "the retention policy text"
        assert chunks[0]["doc_id"] == "d1"
        assert chunks[0]["page_number"] == 3
        assert chunks[0]["section_path"] == ["S1"]

    def test_pgvector_reads_the_corpus(self):
        from tests.test_adapters_realtime import _make_pg_adapter

        adapter, _pool, conn, _cursor = _make_pg_adapter()
        # The mocked connection returns an empty corpus.
        conn.execute.return_value.fetchall.return_value = []
        adapter.list_chunks()

        # The read must select the columns the index needs.
        sql = conn.execute.call_args.args[0]
        assert "chunk_id" in sql
        assert "payload" in sql
