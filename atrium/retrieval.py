"""
atrium/retrieval.py

Shared retrieval + rerank + parent-child resolution. The single
sequence used by the production path and the benchmark harness so the
two stay in lockstep.

Public API:
  - retrieve_and_rerank(db, query_vector, query_text, top_k, ...) —
    one entry point. Internally: schema-version detection, wide
    candidate retrieval, optional BM25 RRF hybrid (opt-in via
    `bm25_index`), optional cross-encoder rerank, parent dedup,
    parent resolution, top_k truncation.
  - collection_schema_version(db) — reads
    `db.collection_schema_version()` defensively.
  - DEDUP_FACTOR, MAX_CANDIDATE_FACTOR, RRF_K, DENSE_CONFIDENT_GAP,
    DENSE_MIN_SCORE — internal constants. See each definition for
    the reasoning.

Schema-version contract:
  - schema_version == 0 (legacy): flat retrieval, no parent dedup or
    resolution.
  - schema_version >= 1 (parent-child): vector search filters out
    parents, dedupes by parent_id, attaches parent text via batched
    retrieve_parents.
  - Missing / failing adapter → treated as v0 (legacy default).

Backward-compat response shape:
    id, score, text, doc_id, chunk_id, page_number, section_path,
    element_type, is_parent, parent_id, parent_text,
    parent_page_number, parent_section_path, chunking_strategy
v0 hits carry parent_* as None / empty.
`chunking_strategy` names the chunking strategy that produced the
hit ("structure_aware", "parent_child", …). It is empty for corpora
ingested before the field existed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from atrium.config import env_float

log = logging.getLogger("ingest.retrieval")


# Over-fetch factor for parent dedup. When no reranker is supplied,
# we fetch top_k * DEDUP_FACTOR children so that, after dedup-by-
# parent, we have a good chance of returning top_k distinct parents.
DEDUP_FACTOR = 3

# Hard ceiling on over-fetch. A collection where most children map
# to the same parent (e.g., a single-page doc with one parent)
# shouldn't trigger an unbounded candidate pool.
MAX_CANDIDATE_FACTOR = 5

# RRF smoothing constant (Cormack et al., 2009). The original
# reciprocal-rank-fusion paper and rank_bm25's `BM25Plus` use 60.
# Pinning this avoids the "tune k to make the experiment look good"
# trap: the dense-vs-hybrid comparison isolates the ranker change
# from any fusion-tuning noise.
RRF_K = 60

# Confidence-gate thresholds (2026-10-18). The gate decides when
# retrieve_and_rerank consults BM25 on top of the dense result.
#
# Raising EITHER threshold makes the confidence bar STRICTER:
#   - top1 - top2 ≥ DENSE_CONFIDENT_GAP AND
#   - top1      ≥ DENSE_MIN_SCORE
# A stricter bar means fewer queries clear it → more queries get
# BM25 consulted.
#
# Lowering either threshold makes the bar LOOSER → more queries
# stay dense-only.
#
# Both are re-read from the environment at the start of every
# `retrieve_and_rerank` call, so an operator can tune the gate without
# a server restart. The values below are the 2026-10-18 starting
# point — not validated, just the first knob settings.
DENSE_CONFIDENT_GAP = env_float("DENSE_CONFIDENT_GAP")
DENSE_MIN_SCORE = env_float("DENSE_MIN_SCORE")


def _refresh_gate_thresholds() -> None:
    """Re-read the confidence-gate thresholds from the environment.

    The module-level values are the import-time snapshot; this refresh
    is what makes the documented "tune without a restart" behaviour
    real. It writes to the module globals on purpose: the gate reads
    those names, and the tests monkeypatch them.
    """
    global DENSE_CONFIDENT_GAP, DENSE_MIN_SCORE
    DENSE_CONFIDENT_GAP = env_float("DENSE_CONFIDENT_GAP")
    DENSE_MIN_SCORE = env_float("DENSE_MIN_SCORE")


def collection_schema_version(db) -> int:
    """Return the schema version of the underlying collection.

    Adapters that don't implement `collection_schema_version` (legacy
    callers) are treated as v0 — flat retrieval, no parent resolution.
    """
    fn = getattr(db, "collection_schema_version", None)
    if fn is None:
        return 0
    try:
        v = fn()
        return int(v)
    except Exception:
        return 0


def _should_consult_bm25(
    candidates: List[Dict[str, Any]],
    bm25_index: Optional[Any],
) -> bool:
    """Confidence-gate heuristic for the 2026-10-18 hybrid opt-in.

    Returns True if BM25 should be consulted on top of the dense
    candidate list. Treat this as a heuristic, not a correctness
    measure — it cannot tell whether BM25's evidence is right, only
    that dense retrieval is uncertain.

    Cases:
      - No `bm25_index` supplied → False (dense-only; cannot consult
        what isn't there).
      - Zero candidates → True (no dense answer, BM25 is the only
        hope; `_hybrid_fuse` handles the empty-dense case).
      - One candidate → True. A single result has no defined
        top-1/top-2 gap; we deliberately do NOT treat it as
        highly confident by assigning the missing second score to
        zero. That's a trap that would route every "singleton"
        dense result to dense-only and skip the BM25 safety net.
      - Two or more candidates → True iff the dense result is
        NOT confident, i.e. EITHER:
          * top_1 - top_2 < DENSE_CONFIDENT_GAP, or
          * top_1 < DENSE_MIN_SCORE.

    Raises are never propagated from this function — it's a pure
    decision, no I/O.
    """
    if bm25_index is None:
        return False
    if not candidates:
        return True
    if len(candidates) == 1:
        # One result: undefined gap, do not assume confidence.
        return True
    # 2+ candidates: apply the gap + min-score rule.
    top1 = float(candidates[0].get("score", 0.0))
    top2 = float(candidates[1].get("score", 0.0))
    gap = top1 - top2
    confident = (gap >= DENSE_CONFIDENT_GAP) and (top1 >= DENSE_MIN_SCORE)
    return not confident


def retrieve_and_rerank(
    db,
    query_vector: List[float],
    query_text: str,
    top_k: int,
    *,
    reranker=None,
    candidate_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    candidate_multiplier: Optional[int] = None,
    bm25_index: Optional[Any] = None,
    bm25_candidate_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Unified retrieval + rerank + parent-child resolution.

    Args:
        db: a `VectorDBAdapter`-compatible instance — must implement
            `search` for v0 and `search_children` + `retrieve_parents`
            for v1.
        query_vector: the embedded query.
        query_text: the raw query string — passed to the reranker for
            cross-encoder scoring on child text. Ignored when
            `reranker` is None. Also tokenized for the BM25 sparse
            ranker when `bm25_index` is supplied.
        top_k: result count to return.
        reranker: optional cross-encoder (`atrium.rerank_adapter`).
            When provided, the candidate set is widened to
            `max(top_k, candidate_k)` and rerank truncates to top_k.
            When None, vector order is used (no rerank).
        candidate_k: candidate-set width when `reranker` is set.
            Callers control this so /search and the harness can use
            their own default (30 and 50 respectively). Ignored when
            reranker is None.
        score_threshold: optional floor on vector similarity.
        candidate_multiplier: over-fetch factor for parent dedup when
            no reranker is supplied. Defaults to `DEDUP_FACTOR=3`.
            Cap is `MAX_CANDIDATE_FACTOR=5`.
        bm25_index: optional `atrium.bm25.BM25Index` instance. When
            supplied, the function consults BM25 only when the
            confidence-gate heuristic decides dense retrieval is
            uncertain (see `_should_consult_bm25`). When the gate
            says "confident", BM25 is skipped — this is the 2026-
            10-18 opt-in. When `None` (the default and the
            production path), only the dense ranker is used —
            backward compatible with all callers that existed
            before the hybrid experiment. Note: callers that want
            to force fusion regardless of confidence (e.g. for
            A/B comparison) should pass `bm25_index` AND set
            `DENSE_CONFIDENT_GAP=2.0` + `DENSE_MIN_SCORE=2.0` so
            the gate never trips "confident".
        bm25_candidate_k: BM25 candidate-set width. Defaults to
            `candidate_k` when that's set, else to `top_k *
            candidate_multiplier` (capped by `MAX_CANDIDATE_FACTOR`)
            for the rerank-off path. Mirrors the dense side so the
            two rankers see the same budget.

    Returns:
        list of hit dicts, each carrying the full back-compat schema
        (see module docstring). v0 hits have parent_text=None etc.
    """
    # Pick up any operator change to the gate thresholds made since
    # the last call.
    _refresh_gate_thresholds()

    schema_v = collection_schema_version(db)
    is_v1 = schema_v >= 1 and hasattr(db, "search_children")

    # Step 0 — resolve candidate set width.
    # - Reranker on: caller-supplied candidate_k (or default 30).
    #   No cap from this function — rerank adapters expect a
    #   caller-controlled candidate budget, and the harness uses 50
    #   while /search uses 30.
    # - Reranker off: over-fetch by DEDUP_FACTOR to absorb parent
    #   dedup loss in v1, capped by MAX_CANDIDATE_FACTOR so a
    #   degenerate corpus (mostly same parent) doesn't unbounded-
    #   fetch. v0 needs no over-fetch (no dedup).
    if reranker is not None:
        cand_k = candidate_k if candidate_k is not None else 30
        cand_k = max(top_k, cand_k)
    else:
        mult = candidate_multiplier if candidate_multiplier is not None else DEDUP_FACTOR
        mult = min(max(1, mult), MAX_CANDIDATE_FACTOR)
        cand_k = top_k * mult if is_v1 else top_k

    # Step 1 — wide candidate fetch.
    if is_v1:
        raw_candidates = db.search_children(
            query_vector, top_k=cand_k, score_threshold=score_threshold,
        )
    else:
        raw_candidates = db.search(
            query_vector, top_k=cand_k, score_threshold=score_threshold,
        )
    if not raw_candidates and bm25_index is None:
        return []

    # Step 1b — optional hybrid fusion with BM25 (2026-10-11
    # experiment). When bm25_index is supplied, do a BM25 search
    # over the same corpus, look up the original dense candidates
    # by chunk_id, and RRF-fuse the two ranked lists. The fused
    # list is then re-sorted by RRF score and passed to the rest
    # of the pipeline (rerank, dedup, parent-resolve). The dense
    # list may be empty in the degenerate case (vector DB down);
    # we still try BM25 and fuse the two halves, falling back to
    # whichever side returns hits.
    #
    # 2026-10-18 — confidence gate: only consult BM25 when dense
    # retrieval is uncertain (see `_should_consult_bm25`). High-
    # confidence dense results skip BM25 entirely. The gate is a
    # heuristic; a misfire routes to the wrong ranker but does not
    # drop hits. Logging below records the decision for
    # post-hoc analysis.
    if bm25_index is not None:
        should_consult = _should_consult_bm25(raw_candidates, bm25_index)
        if raw_candidates:
            top1 = float(raw_candidates[0].get("score", 0.0))
            top2 = (
                float(raw_candidates[1].get("score", 0.0))
                if len(raw_candidates) >= 2 else None
            )
            gap = (top1 - top2) if top2 is not None else None
            log.info(
                "routing: n_candidates=%d consult_bm25=%s "
                "top1=%.4f top2=%s gap=%s "
                "threshold_gap=%.4f threshold_min=%.4f",
                len(raw_candidates), should_consult, top1, top2, gap,
                DENSE_CONFIDENT_GAP, DENSE_MIN_SCORE,
            )
        else:
            log.info(
                "routing: n_candidates=0 consult_bm25=True (no dense hits, "
                "BM25 is the only retrieval path)"
            )
        if should_consult:
            bm25_top = bm25_candidate_k or cand_k
            try:
                raw_candidates = _hybrid_fuse(
                    raw_candidates, bm25_index, query_text, top_k=cand_k,
                    sparse_top_k=bm25_top,
                )
            except Exception as exc:
                # Graceful fallback: BM25 raised (tokenization error,
                # corpus mismatch, etc.) — continue with the dense
                # candidate list. The gate's promise is "consult when
                # uncertain", not "BM25 always works". A failure here
                # is logged and the dense path is preserved.
                log.warning(
                    "BM25 consult failed (%s: %s); falling back to dense "
                    "candidate list (n=%d)",
                    type(exc).__name__, exc, len(raw_candidates),
                )

    # Step 2 — optional rerank. Cross-encoders score child text (not
    # parent) — they're optimized for short passages. The reranker
    # returns up to top_k RerankCandidate entries with new scores.
    hits = _apply_rerank(
        query_text, raw_candidates, top_k, reranker,
    ) if reranker is not None else raw_candidates

    # Step 3 — dedup by parent_id (v1 only).
    if is_v1:
        hits = _dedup_by_parent(hits, top_k)
    else:
        # v0 hits are all flat — but we still emit back-compat shape
        # with parent fields defaulted to None / [].
        hits = [_with_v0_parent_defaults(h) for h in hits]

    # Step 4 — parent resolution (v1 only). Batched single call.
    if is_v1:
        hits = _resolve_parents(db, hits)

    return hits


def _hybrid_fuse(
    dense_hits: List[Dict[str, Any]],
    bm25_index: Any,
    query_text: str,
    top_k: int,
    sparse_top_k: int,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of dense + BM25 candidate lists.

    Algorithm (Cormack et al. 2009): for each ranker, assign each
    hit a rank (1-based). For every unique chunk_id, the fused
    score is `sum_over_rankers(1 / (RRF_K + rank))`. Hits present
    in both rankers get a higher score; hits in only one get the
    contribution of that one ranker.

    Args:
        dense_hits: candidate list from the vector DB. Each hit
            must have an `id` (Qdrant point id) AND a `chunk_id`
            (application-level id) — the BM25 index is keyed on
            `chunk_id` so we look up sparse hits in the dense list
            by chunk_id, not point id. Hits without `chunk_id` are
            skipped from the dense half (logged once, not per-call).
        bm25_index: a `BM25Index` already populated with the
            corpus. The same `chunk_id` is used as the BM25 doc_id.
        query_text: the user's query string; tokenized for BM25.
        top_k: how many candidates the fused list should hold.
            This is the same budget the dense side had — fusion
            cannot add candidates that neither ranker surfaced.
        sparse_top_k: how many BM25 hits to pull. Defaults to
            `top_k` when the caller doesn't override.

    Returns:
        The fused candidate list, sorted by RRF score descending,
        truncated to `top_k`. Each hit carries the original dense
        payload (text, page_number, section_path, is_parent, etc.)
        — the BM25 side contributes the score, not new fields.

    Defensive behavior:
        - Empty dense, non-empty sparse: dense list is augmented
          with stub records for the BM25 hits that aren't already
          in the dense list. We use `bm25_index.search_with_meta`
          to also pull each hit's stored per-doc metadata (text,
          page_number, etc.) so the final hit shape is complete
          even when the dense ranker missed the chunk.
        - Non-empty dense, empty sparse: dense list is returned
          unchanged (RRF contributes nothing from the missing
          ranker — identical to dense-only).
        - Both empty: returns [] (caller short-circuits the
          per-query scoring loop with zero hits).
    """
    sparse_results = (
        bm25_index.search_with_meta(query_text, top_k=sparse_top_k)
        if hasattr(bm25_index, "search_with_meta")
        else [(d, s, {}) for d, s in (bm25_index.search(query_text, top_k=sparse_top_k) or [])]
    ) or []

    # Map chunk_id -> dense hit (for payload lookup). We index by
    # chunk_id (not Qdrant point id) because that's the same key
    # the BM25 index was populated with.
    dense_by_chunk: Dict[str, Dict[str, Any]] = {}
    for h in dense_hits:
        cid = h.get("chunk_id") or h.get("id")
        if cid:
            dense_by_chunk[str(cid)] = h

    # RRF score accumulator. We use a dict of dicts so we can
    # carry the original dense payload alongside the running score.
    fused: Dict[str, Dict[str, Any]] = {}

    for rank, h in enumerate(dense_hits, start=1):
        cid = str(h.get("chunk_id") or h.get("id") or "")
        if not cid:
            continue
        fused[cid] = {
            **h,
            "_rrf_score": 1.0 / (RRF_K + rank),
            "_rrf_rank_dense": rank,
        }

    for rank, (cid, _bm25_score, bm25_meta) in enumerate(sparse_results, start=1):
        rrf_contrib = 1.0 / (RRF_K + rank)
        if cid in fused:
            fused[cid]["_rrf_score"] = fused[cid].get("_rrf_score", 0.0) + rrf_contrib
            fused[cid]["_rrf_rank_sparse"] = rank
        else:
            # BM25-only hit — promote to a candidate. We prefer the
            # dense hit's payload (text, page_number, etc.) when
            # available, but if the dense side never surfaced this
            # chunk we fall back to the per-doc metadata that was
            # passed to `bm25.add_doc(..., meta=...)`. The caller
            # is responsible for populating that meta with the
            # fields `retrieve_and_rerank` emits (text, page_number,
            # section_path, element_type, is_parent, parent_id,
            # doc_id, chunk_id).
            base = dict(dense_by_chunk.get(cid, {}))
            if not base and bm25_meta:
                # No dense hit — construct a complete hit from the
                # BM25 metadata. `id` falls back to chunk_id; the
                # downstream dedup / parent-resolve code is tolerant
                # of bare ids and None parent_id.
                base = {
                    "id": bm25_meta.get("id") or cid,
                    "chunk_id": cid,
                    "doc_id": bm25_meta.get("doc_id", ""),
                    "text": bm25_meta.get("text", ""),
                    "page_number": bm25_meta.get("page_number"),
                    "section_path": bm25_meta.get("section_path") or [],
                    "element_type": bm25_meta.get("element_type", ""),
                    "is_parent": bm25_meta.get("is_parent", False),
                    "parent_id": bm25_meta.get("parent_id"),
                    "chunking_strategy": bm25_meta.get("chunking_strategy", ""),
                }
            fused[cid] = {
                **base,
                "id": base.get("id", cid),
                "chunk_id": cid,
                "score": float(_bm25_score),
                "_rrf_score": rrf_contrib,
                "_rrf_rank_sparse": rank,
            }

    if not fused:
        return []

    ranked = sorted(
        fused.values(),
        key=lambda h: h.get("_rrf_score", 0.0),
        reverse=True,
    )
    # Promote the RRF score to the surface `score` field. Downstream
    # callers (e.g., PerQueryResult.top_5_similarities) read `score`
    # for diagnostics — keeping the RRF value there is more honest
    # than a stale cosine sim that no longer reflects the ranker's
    # view of relevance. Strip the private _rrf_* bookkeeping keys.
    out: List[Dict[str, Any]] = []
    for h in ranked[:top_k]:
        h2 = {k: v for k, v in h.items() if not k.startswith("_rrf_")}
        h2["score"] = float(h.get("_rrf_score", 0.0))
        out.append(h2)
    return out


def _apply_rerank(query_text, candidates, top_k, reranker):
    """Optional cross-encoder rerank. On success returns the reranked
    list (preserving original child payload via id lookup); on failure
    returns the input list sorted by vector score."""
    # Lazy import: rerank_adapter is heavy (it pulls sentence-transformers
    # for the local model); the /search path without rerank shouldn't
    # pay that import cost.
    from atrium.rerank_adapter import RerankCandidate

    try:
        rq = [
            RerankCandidate(
                id=h.get("id"),
                text=h.get("text", ""),
                score=float(h.get("score", 0.0)),
                payload=h,  # full dict so we can re-attach parent_id etc.
            )
            for h in candidates
        ]
        reranked = reranker.rerank(query_text, rq, top_k)
    except Exception:
        reranked = None

    if reranked is None:
        # Rerank failed (or no reranker input) — fall back to vector order.
        out = list(candidates)
        out.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return out

    # Map reranked ids back to original children for parent_id and
    # other fields the cross-encoder doesn't see.
    by_id = {h.get("id"): h for h in candidates}
    out = []
    for c in reranked:
        orig = by_id.get(c.id, {"id": c.id, "text": c.text, "score": 0.0})
        out.append({
            "id": c.id,
            "score": float(c.score),
            "text": c.text,
            "doc_id": orig.get("doc_id", ""),
            "chunk_id": orig.get("chunk_id", c.id),
            "page_number": orig.get("page_number"),
            "section_path": orig.get("section_path") or [],
            "element_type": orig.get("element_type", ""),
            "is_parent": orig.get("is_parent", False),
            "parent_id": orig.get("parent_id"),
            "chunking_strategy": orig.get("chunking_strategy", ""),
        })
    out.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    return out


def _dedup_by_parent(hits, top_k):
    """Dedupe by parent_id, keeping highest-scoring child per parent.
    Top up with orphans (children with parent_id=None) if dedup left
    us under top_k. Returns the combined list sorted by score desc,
    truncated to top_k."""
    by_parent: Dict[str, Dict[str, Any]] = {}
    orphans: List[Dict[str, Any]] = []
    for h in hits:
        pid = h.get("parent_id")
        if not pid:
            orphans.append(h)
            continue
        existing = by_parent.get(pid)
        if existing is None or float(h.get("score", 0.0)) > float(existing.get("score", 0.0)):
            by_parent[pid] = h

    combined = list(by_parent.values()) + orphans
    combined.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    return combined[:top_k]


def _resolve_parents(db, hits):
    """Attach parent_text / parent_page_number / parent_section_path
    to each hit via a single batched retrieve_parents call. Hits
    without parent_id (or whose parent was deleted) carry None."""
    parent_ids = [h.get("parent_id") for h in hits if h.get("parent_id")]
    parent_map: Dict[str, Dict[str, Any]] = {}
    if parent_ids and hasattr(db, "retrieve_parents"):
        try:
            parent_map = db.retrieve_parents(parent_ids) or {}
        except Exception:
            # Parent retrieval failure doesn't drop children — they
            # remain valid results with parent_text=None.
            parent_map = {}

    out = []
    for h in hits:
        pid = h.get("parent_id")
        p = (parent_map.get(pid) if pid else None) or {}
        out.append({
            "id": h.get("id"),
            "score": float(h.get("score", 0.0)),
            "text": h.get("text", ""),
            "doc_id": h.get("doc_id", ""),
            "chunk_id": h.get("chunk_id", h.get("id")),
            "page_number": h.get("page_number"),
            "section_path": h.get("section_path") or [],
            "element_type": h.get("element_type", ""),
            "is_parent": h.get("is_parent", False),
            "parent_id": pid,
            "parent_text": p.get("text"),
            "parent_page_number": p.get("page_number"),
            "parent_section_path": p.get("section_path") or [],
            "chunking_strategy": h.get("chunking_strategy", ""),
        })
    return out


def _with_v0_parent_defaults(h):
    """Add parent fields with None / [] defaults for v0 (legacy) hits."""
    return {
        "id": h.get("id"),
        "score": float(h.get("score", 0.0)),
        "text": h.get("text", ""),
        "doc_id": h.get("doc_id", ""),
        "chunk_id": h.get("chunk_id", h.get("id")),
        "page_number": h.get("page_number"),
        "section_path": h.get("section_path") or [],
        "element_type": h.get("element_type", ""),
        "is_parent": False,
        "parent_id": None,
        "parent_text": None,
        "parent_page_number": None,
        "parent_section_path": [],
        "chunking_strategy": h.get("chunking_strategy", ""),
    }


# ---------------------------------------------------------------------------
# Back-compat alias (Task B): callers that still reference
# `retrieve_with_parent_resolution` (legacy test imports, etc.) get
# the unified function with reranker disabled. Internal callers were
# migrated; this alias keeps external test modules working.
# ---------------------------------------------------------------------------
def retrieve_with_parent_resolution(
    db,
    query_vector: List[float],
    top_k: int,
    score_threshold: Optional[float] = None,
    candidate_multiplier: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Back-compat wrapper: vector-only retrieve (no rerank) through
    the unified function. `query_text` is unused in vector-only mode
    but required by the unified signature — pass a placeholder."""
    return retrieve_and_rerank(
        db,
        query_vector,
        query_text="",  # unused when reranker=None
        top_k=top_k,
        reranker=None,
        candidate_k=None,
        score_threshold=score_threshold,
        candidate_multiplier=candidate_multiplier,
    )
