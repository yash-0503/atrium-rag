r"""
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
  `re.findall(r"\w+", text, flags=re.UNICODE)`. Catches Latin, Arabic
  and accented Latin.
* **Han/CJK is NOT tokenized usefully.** `\w+` matches a run of Han
  characters as a SINGLE token, so a Chinese or Japanese document is
  one opaque token and cannot be matched by a query that does not
  contain a byte-identical run. This is a property of the regex, not
  a choice — see `contains_cjk`. The hybrid path therefore declines
  to consult the sparse ranker for a query written in those scripts,
  rather than fusing a signal that cannot see the corpus.
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
* `bm25.remove_doc(doc_id)` — retract a document (returns True if it
  was there).
* `bm25.search(query, top_k=5)` — top-k (doc_id, score) sorted desc.
* `bm25.search_with_meta(query, top_k=5)` — same but each item is
  `(doc_id, score, meta)`; meta is the dict passed to `add_doc`
  (or `{}` if none).
* `bm25.n_docs` — number of indexed documents.
* `contains_cjk(text)` — does this text contain Han/Kana/Hangul?
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# Scripts whose words are not separated by spaces. The `\w+` tokenizer
# below treats a whole run of Han characters as ONE token, which makes
# the sparse ranker useless for those languages: a Chinese or Japanese
# query is a single token that matches only a byte-identical run of
# characters.
#
# The hybrid path checks this and declines to consult the sparse
# ranker for such a query, rather than fusing a half-blind signal into
# a good dense ranking. See `contains_cjk`.
_CJK_RE = re.compile(
    "["
    "぀-ヿ"   # Hiragana + Katakana
    "㐀-䶿"   # CJK Extension A
    "一-鿿"   # CJK Unified Ideographs
    "豈-﫿"   # CJK Compatibility Ideographs
    "가-힯"   # Hangul syllables
    "]"
)

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


def contains_cjk(text: str) -> bool:
    """Does this text contain Han / Kana / Hangul characters?

    The sparse ranker's tokenizer cannot segment those scripts (see
    `_CJK_RE`), so a query written in them is one opaque token that
    matches almost nothing. Consulting the sparse ranker for such a
    query adds noise to the fusion rather than signal, so the hybrid
    path declines to consult it.

    This is a *query-side* check on purpose. A corpus may legitimately
    mix scripts — an English report quoting a Japanese term — and
    refusing to consult the sparse ranker for the whole corpus would
    cost recall for every query. The query is what decides.
    """
    if not text:
        return False
    return bool(_CJK_RE.search(text))


class BM25Index:
    """In-memory BM25 Okapi index.

    Indexing is O(L) per document — the average document length is
    maintained incrementally, and the document-frequency table is
    derived from the postings map rather than recomputed. Search is
    O(|Q| * postings-touch): only the documents that actually contain
    a query term are visited, not every document in the corpus.

    The postings map is what makes the index usable at corpus scale.
    The previous implementation kept a term-frequency counter per
    document and scanned every document for every query term, and
    recomputed the average document length on every `add_doc` — an
    O(N²) build and an O(|Q| * N) search. At twelve thousand chunks
    that is tens of millions of dict operations per query.
    """

    def __init__(self, k1: float = _K1, b: float = _B) -> None:
        self.k1 = k1
        self.b = b
        # Per-doc term frequency: {doc_id: Counter(term -> count)}
        self._tf: Dict[str, Counter] = {}
        # Per-doc length in tokens
        self._dl: Dict[str, int] = {}
        # Inverted index: {term: {doc_id: term_frequency}}. The
        # document frequency of a term is `len(_postings[term])`, so
        # there is no separate `_df` table to keep in sync.
        self._postings: Dict[str, Dict[str, int]] = {}
        # Per-doc metadata (the dict passed to add_doc(..., meta=...)).
        # Surfaces in search_with_meta so the hybrid fusion path can
        # promote a BM25-only hit to a full hit shape (with text,
        # page_number, etc.) without a second round-trip to the
        # vector store.
        self._meta: Dict[str, Dict[str, Any]] = {}
        # Total tokens across the corpus, maintained incrementally so
        # the average document length is O(1) to read.
        self._total_len: int = 0

    @property
    def n_docs(self) -> int:
        return len(self._tf)

    @property
    def _avgdl(self) -> float:
        """The average document length, in tokens."""
        return self._total_len / max(1, self.n_docs)

    def add_doc(
        self, doc_id: str, text: str, meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register one document. Idempotent on `doc_id` — re-adding
        replaces the prior entry, which matters when a chunk is
        re-chunked and the same content_hash collides."""
        tokens = tokenize(text)
        counter = Counter(tokens)

        # Replace: drop the old document's contribution first, so the
        # postings map and the length totals never double-count.
        if doc_id in self._tf:
            self._remove_doc_internal(doc_id)

        self._tf[doc_id] = counter
        self._dl[doc_id] = len(tokens)
        self._total_len += len(tokens)
        self._meta[doc_id] = dict(meta) if meta else {}

        for term, freq in counter.items():
            bucket = self._postings.setdefault(term, {})
            bucket[doc_id] = freq

    def remove_doc(self, doc_id: str) -> bool:
        """Drop a document from the index. Returns True if it was there.

        A corpus that is replaced — the same document re-ingested with
        new content — must be able to retract the old version, or the
        sparse ranker keeps scoring text that is no longer in the
        corpus.
        """
        if doc_id not in self._tf:
            return False
        self._remove_doc_internal(doc_id)
        return True

    def _remove_doc_internal(self, doc_id: str) -> None:
        """Remove a document's contribution to the postings and totals."""
        counter = self._tf.pop(doc_id, None)
        self._total_len -= self._dl.pop(doc_id, 0)
        self._meta.pop(doc_id, None)
        if not counter:
            return
        for term in counter:
            bucket = self._postings.get(term)
            if bucket is None:
                continue
            bucket.pop(doc_id, None)
            if not bucket:
                del self._postings[term]

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
        avgdl = self._avgdl
        n_docs = self.n_docs
        # Only the documents that contain the term are visited — the
        # postings map, not the whole corpus.
        for q in set(q_tokens):
            bucket = self._postings.get(q)
            if not bucket:
                continue
            df = len(bucket)
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1) — the
            # "+ 1" inside the log is the Robertson–Sparck Jones
            # variant that keeps IDF non-negative for very common
            # terms (df > N/2). Without it, very common terms get
            # negative IDF and can rank spuriously low.
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, f in bucket.items():
                dl = self._dl[doc_id]
                num = f * (self.k1 + 1.0)
                den = f + self.k1 * (1.0 - self.b + self.b * dl / max(1e-9, avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (num / den)

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            (doc_id, score, self._meta.get(doc_id, {}))
            for doc_id, score in ranked[:top_k]
        ]
