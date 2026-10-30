"""
atrium/bm25.py

Pure-Python BM25 (Okapi) sparse retriever, used to compose hybrid
(dense + sparse) retrieval in the experiment harness. No native deps,
no rank_bm25 — keeps the requirements surface unchanged.

This is intentionally small: tokenize, index, search. No stemming, no
stopwords — the BM25 term-frequency signal is strong enough on its own
for the hybrid experiment, and adding linguistic pre-processing now
would conflate two changes (sparse-on + new analyzer).

Design notes
------------
* `BM25Index` is per-corpus: build once with `add_doc` per chunk,
  call `search(query, top_k)` to get ranked (doc_id, score) pairs.
* Tokenization: lowercase, split on Unicode word boundaries via
  `re.findall(r"\w+", text, flags=re.UNICODE)`. Catches Latin, CJK
  (each Han char is its own \w+ token), Arabic, accented Latin.
  The naive Han-character tokenization is fine for the BM25 hybrid
  experiment — it over-segments but doesn't lose signal.
* Score is Okapi BM25 with k1=1.5, b=0.75 (the textbook defaults
  used in rank_bm25). Not normalized to a 0..1 range; the fused
  ranker only cares about relative order (RRF operates on ranks).
* Doc ids are opaque strings. They are returned by `search` exactly
  as supplied, so the caller can key the result back to a chunk_id
  or content_hash without an extra mapping.
* Per-doc metadata is stored alongside the index (`add_doc(id, text,
  meta=...)`) and surfaced in `search_with_meta` for the hybrid
  fusion path — when a chunk is in the BM25 top-k but NOT in the
  dense top-k, the fusion function still needs `text`, `page_number`,
  etc. to emit a back-compat hit shape.

Public API
----------
* `BM25Index()` — empty index.
* `bm25.add_doc(doc_id, text, meta=None)` — register one document.
* `bm25.search(query, top_k=5)` — top-k (doc_id, score) sorted desc.
* `bm25.search_with_meta(query, top_k=5)` — same but each item is
  `(doc_id, score, meta)`; meta is the dict passed to `add_doc`
  (or `{}` if none).
* `bm25.n_docs` — number of indexed documents.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# Okapi BM25 defaults — same as the original BM25L paper and the
# rank_bm25 default. Holding these pinned so an experiment can
# reference them as constants if we tune later.
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> List[str]:
    """Lowercase + Unicode word-boundary tokenization.

    Han/CJK characters are individually split (each is its own
    `\\w+` token). The tokenization is intentionally simple —
    sophistication (jieba, MeCab, ICU) would change the sparse
    signal in ways orthogonal to the hybrid-retrieval experiment
    being tested here.
    """
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 Okapi index.

    Indexing complexity is O(N * L) where N=docs, L=avg doc length.
    Search is O(|Q| * postings-touch) — fine for fixture-scale
    corpora (a few hundred to a few thousand chunks). The 19-set
    benchmark builds ~12k chunks total, all rebuilt per fixture.
    """

    def __init__(self, k1: float = _K1, b: float = _B) -> None:
        self.k1 = k1
        self.b = b
        # Per-doc term frequency: {doc_id: Counter(term -> count)}
        self._tf: Dict[str, Counter] = {}
        # Per-doc length in tokens
        self._dl: Dict[str, int] = {}
        # Document frequency: {term: number of docs containing it}
        self._df: Dict[str, int] = {}
        # Per-doc metadata (the dict passed to add_doc(..., meta=...)).
        # Surfaces in search_with_meta so the hybrid fusion path can
        # promote a BM25-only hit to a full hit shape (with text,
        # page_number, etc.) without a second round-trip to the
        # vector store.
        self._meta: Dict[str, Dict[str, Any]] = {}
        # Average doc length across the corpus (recomputed on demand)
        self._avgdl: float = 0.0

    @property
    def n_docs(self) -> int:
        return len(self._tf)

    def add_doc(
        self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register one document. Idempotent on `doc_id` — re-adding
        replaces the prior entry, which matters when a chunk is
        re-chunked and the same content_hash collides."""
        # Capture the prior term set BEFORE we overwrite _tf, so the
        # doc-frequency table can be decremented for terms the old
        # version contributed to but the new one doesn't.
        prior_counter = self._tf.get(doc_id)
        prior_terms = set(prior_counter.keys()) if prior_counter else set()

        tokens = tokenize(text)
        self._tf[doc_id] = Counter(tokens)
        self._dl[doc_id] = len(tokens)
        self._meta[doc_id] = dict(meta) if meta else {}

        new_terms = set(self._tf[doc_id].keys())
        for t in prior_terms - new_terms:
            self._df[t] = self._df.get(t, 1) - 1
            if self._df[t] <= 0:
                self._df.pop(t, None)
        for t in new_terms - prior_terms:
            self._df[t] = self._df.get(t, 0) + 1
        self._avgdl = sum(self._dl.values()) / max(1, self.n_docs)

    def search(
        self, query: str, top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return up to `top_k` (doc_id, bm25_score) pairs sorted by
        score descending. Empty result if no query term matches any
        document (caller handles the empty case — RRF treats it as
        "this ranker contributed nothing for this query")."""
        return [(d, s) for d, s, _ in self.search_with_meta(query, top_k=top_k)]

    def search_with_meta(
        self, query: str, top_k: int = 5,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Same as `search` but each item also carries the per-doc
        metadata dict that was passed to `add_doc`. Used by the
        hybrid fusion path in `atrium/retrieval.py` to construct a
        full hit shape (text, page_number, section_path, etc.)
        for BM25-only hits without a second lookup round-trip."""
        q_tokens = tokenize(query)
        if not q_tokens or not self._tf:
            return []

        scores: Dict[str, float] = {}
        for q in q_tokens:
            df = self._df.get(q, 0)
            if df == 0:
                continue
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1) — the
            # "+ 1" inside the log is the Robertson–Sparck Jones
            # variant that keeps IDF non-negative for very common
            # terms (df > N/2). Without it, very common terms get
            # negative IDF and can rank spuriously low.
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf_counter in self._tf.items():
                f = tf_counter.get(q, 0)
                if f == 0:
                    continue
                dl = self._dl[doc_id]
                num = f * (self.k1 + 1.0)
                den = f + self.k1 * (1.0 - self.b + self.b * dl / max(1e-9, self._avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (num / den)

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (doc_id, score, self._meta.get(doc_id, {}))
            for doc_id, score in ranked[:top_k]
        ]
