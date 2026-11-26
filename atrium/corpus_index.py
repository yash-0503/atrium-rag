"""
atrium/corpus_index.py

The lexical (BM25) index over the whole corpus, and the rules for
keeping it fresh enough to trust.

Why this exists
---------------
The hybrid retrieval path — dense candidates fused with a sparse BM25
ranking via RRF — was written and measured (it is the configuration
the published R@5 0.700 was measured on) but was never wired into a
product surface. `retrieve_and_rerank` only consults BM25 when it is
handed a `bm25_index`, and no product surface built one.

This module is the missing piece: it builds an index over the corpus
and hands it to the retrieval layer.

Freshness
---------
The index is a cache over the corpus, and the corpus changes under it.
Three things make it rebuild:

  * **The chunk count changed.** A document was ingested or removed.
    Detected immediately, from a cheap `count_chunks()`.
  * **The TTL expired.** A document was *replaced* — re-ingested with
    new content — which leaves the chunk count unchanged. The TTL
    bounds how long the index can hold text that is no longer in the
    corpus. `BM25_INDEX_TTL_SECONDS` (default 60).
  * **The process wrote to the corpus itself.** An ingest in *this*
    process invalidates the index immediately, so a developer who
    ingests and then searches sees their own document.

The honest limitation: a replacement that leaves the chunk count
unchanged is picked up within the TTL, not instantly. An exact
invalidation signal would need a corpus-version counter in every
store backend; the local store already has one (`corpus_version` in
its `meta` table) but the hosted backends do not, and a half-implemented
counter across three backends is worse than a documented TTL.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger("ingest.corpus_index")

__all__ = ["CorpusBm25Index", "build_bm25_index"]


def build_bm25_index(chunks: List[Dict[str, Any]]):
    """Build a `BM25Index` over a list of chunk dicts.

    The chunk dicts use the store's `list_chunks` shape. Each chunk is
    indexed under its `chunk_id`, with the metadata the hybrid fusion
    path needs to build a full hit for a BM25-only result — so a chunk
    the dense ranker missed can still be returned with its text, page
    and heading path, without a second round-trip to the store.
    """
    from atrium.bm25 import BM25Index

    index = BM25Index()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        index.add_doc(
            str(chunk_id),
            chunk.get("text") or "",
            meta={
                "id": str(chunk_id),
                "chunk_id": str(chunk_id),
                "doc_id": chunk.get("doc_id", ""),
                "text": chunk.get("text") or "",
                "page_number": chunk.get("page_number"),
                "section_path": chunk.get("section_path") or [],
                "element_type": chunk.get("element_type", ""),
                "is_parent": bool(chunk.get("is_parent", False)),
                "parent_id": chunk.get("parent_id"),
                "chunking_strategy": chunk.get("chunking_strategy", ""),
            },
        )
    return index


class CorpusBm25Index:
    """A lazily-built BM25 index over the corpus, kept reasonably fresh.

    Thread-safe: the build happens outside the lock, so concurrent
    searches do not serialise behind one another's corpus read. Two
    concurrent builders produce the same index and the last one to
    finish wins, which is harmless.
    """

    def __init__(self, db: Any, ttl_seconds: Optional[float] = None) -> None:
        self._db = db
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        # (index, chunk_count_at_build, built_at)
        self._cache: Optional[tuple] = None

    # -- freshness ------------------------------------------------------

    def invalidate(self) -> None:
        """Drop the cached index. The next `get()` rebuilds it."""
        with self._lock:
            self._cache = None

    def _ttl_seconds(self) -> float:
        if self._ttl is not None:
            return float(self._ttl)
        from atrium.config import env_float

        return env_float("BM25_INDEX_TTL_SECONDS", 60.0)

    def _is_stale(self, count: int) -> bool:
        if self._cache is None:
            return True
        _index, cached_count, built_at = self._cache
        if cached_count != count:
            # A document was added or removed.
            return True
        if (time.monotonic() - built_at) > self._ttl_seconds():
            # A document may have been replaced in place.
            return True
        return False

    # -- the index ------------------------------------------------------

    def get(self):
        """The index over the corpus, or `None` if it cannot be built.

        Returns `None` rather than raising: a lexical index that
        cannot be built is a reason to serve the dense ranking, not a
        reason to fail the search.
        """
        try:
            count = self._db.count_chunks()
        except Exception as exc:
            log.warning("corpus count failed (%s: %s); no lexical index",
                        type(exc).__name__, exc)
            return None

        with self._lock:
            if not self._is_stale(count):
                return self._cache[0]

        # Build outside the lock: a build is a full corpus read, and
        # holding the lock through it would serialise every search
        # behind it. Two concurrent builders produce the same index;
        # the last one to finish wins, which is harmless.
        try:
            chunks = self._db.list_chunks()
        except Exception as exc:
            log.warning("corpus read failed (%s: %s); no lexical index",
                        type(exc).__name__, exc)
            return None

        try:
            index = build_bm25_index(chunks)
        except Exception as exc:
            log.warning("lexical index build failed (%s: %s)",
                        type(exc).__name__, exc)
            return None

        with self._lock:
            self._cache = (index, count, time.monotonic())
        log.info("lexical index rebuilt: %d chunks", len(chunks))
        return index
