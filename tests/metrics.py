"""
metrics.py

Pure-Python retrieval metrics for the benchmark harness. No numpy/pandas
deliberately — keeps the harness importable everywhere (CI, dev, prod)
without forcing a numeric-stack install just to score a number.

All metrics operate on iterables of "ids" where the id is whatever you
want to score against (page number, content_hash, section_path). For
page-level recall@k the "ids" are integers; for chunk-level they're strings.
Mixing types in a single query is fine — the metric just does `in` checks.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Mapping, Optional, Sequence


def recall_at_k(
    retrieved: Sequence[Hashable],
    relevant: Iterable[Hashable],
    k: int = 5,
) -> float:
    """Fraction of relevant items that appear in the top-k retrieved.

    Returns 0.0 if `relevant` is empty (avoids divide-by-zero; also signals
    'nothing to be right about' which is the right null behavior for an
    unused query).

    Duplicates in `retrieved` are deduplicated before measuring. Without
    this, the same chunk appearing twice in top-k (it can — Qdrant can
    return the same point twice across shards) would inflate the hit
    count past |relevant|, producing recall@k > 1.0 which is a bug.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    top_k = list(retrieved[:k])
    hits = len({x for x in top_k if x in rel})
    return hits / len(rel)


def mrr(
    retrieved: Sequence[Hashable],
    relevant: Iterable[Hashable],
) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant hit, else 0.0.
    Works on raw retrieved sequence (not deduped) since rank is the
    natural notion here."""
    rel = set(relevant)
    for rank, x in enumerate(retrieved, start=1):
        if x in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(
    retrieved: Sequence[Hashable],
    graded_relevance: Mapping[Hashable, float],
    k: int = 5,
) -> float:
    """Discounted Cumulative Gain with standard log2(rank+1) discount.
    Operates on the raw retrieved sequence (not deduped) — the standard
    formulation, where gain decays with rank position."""
    score = 0.0
    for rank, x in enumerate(retrieved[:k], start=1):
        gain = graded_relevance.get(x, 0.0)
        score += gain / math.log2(rank + 1)
    return score


def ndcg_at_k(
    retrieved: Sequence[Hashable],
    graded_relevance: Mapping[Hashable, float],
    k: int = 5,
) -> float:
    """Normalized DCG. Returns 0.0 if ideal DCG is 0 (no relevant items
    exist at all — same null behavior as recall@k)."""
    actual = dcg_at_k(retrieved, graded_relevance, k)
    ideal_gains = sorted(graded_relevance.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    if ideal == 0:
        return 0.0
    return actual / ideal


def evidence_in_context(
    hits: Sequence[Mapping],
    answer_phrase: str,
    k: int = 5,
) -> Optional[float]:
    """Evidence-in-context metric (Task B step 3, 2026-10-10).

    For each hit in top-k, check whether `answer_phrase` (case-
    insensitive substring) appears in either the child `text` or the
    `parent_text`. The metric is 1.0 if any hit in top-k contains the
    phrase, 0.0 otherwise. `None` is returned if the check is
    undefined (empty hits or empty answer_phrase).

    This is the metric that surfaces parent-child's value-add: a flat
    child chunk might not contain the literal answer phrase, but the
    parent context might. The classic example is "What does box 1
    report?" — the child (the Box 1 row) is just "1 | 22222 | 50000
    | ..." with no glossary, but the parent instructions page has
    "Box 1. Enter this amount on the wages line..."

    Limitations:
      - Substring match is a coarse approximation of "answer
        present." It will false-positive on partial matches
        ("Wages" matching in unrelated contexts) and false-negative
        on paraphrased answers. The user-facing report (Task B
        step 5) supplements with manual inspection of the JSON
        dump.
      - This metric requires `answer_phrase` to be supplied per
        query in the ground truth. Fixtures without it report
        `None` and are not aggregated into a fixture-level mean.
      - Case-insensitive by default; pass a lowercase `answer_phrase`
        to get case-sensitive matching (the comparison does
        `phrase.lower() in text.lower()`).

    Args:
        hits: top-k hit dicts from `retrieve_and_rerank`. Each hit
            may carry `text` (child) and `parent_text` (parent).
        answer_phrase: literal phrase expected in the answer. Empty
            string → return None (undefined).
        k: how many top hits to inspect. Default 5.

    Returns:
        1.0 if the phrase appears in any of the top-k hits' child
        text or parent_text, 0.0 otherwise, None if undefined.
    """
    if not answer_phrase:
        return None
    if not hits:
        return 0.0
    needle = answer_phrase.lower()
    for h in hits[:k]:
        for field in ("text", "parent_text"):
            v = h.get(field) or ""
            if needle in v.lower():
                return 1.0
    return 0.0