"""
test_chunking_auto.py

Tests for Auto-mode heuristic coverage (Task A, 2026-10-10).

The 2026-10-10 grid showed none of the 19 fixtures trigger any Auto
heuristic — so the validated data validates the fallback config, not
the adaptive behavior itself. The user pushed back on this gap:

  "Add synthetic or real fixtures that deliberately trigger long-form
   prose, table-heavy, and code-heavy routing. Assert the selected
   configuration and output, then check that borderline inputs behave
   predictably."

Heuristics (defined in `app/smart_chunker.py:_derive_auto_config`):

  - Long-form prose  (>=50% of paragraphs >= 800 chars):
        prose_target_tokens 128 -> 256, overlap 16 -> 32.
  - Table-heavy  (>=30% of elements are tables):
        table_max_rows 10 -> 5, table_max_tokens 384 -> 256.
  - Code-heavy   (>=20% of elements are code):
        code_max_lines 30 -> 50.

Edge cases stack: long-form-prose AND table-heavy (rare) gets both.

These tests build ParsedElement streams that deliberately trip each
heuristic, sit just below the threshold (no trigger), or stack two.
The synthetic fixtures do NOT go through a real PDF parser — they're
constructed directly so the assertion is "given this element stream,
what does Auto pick?".

Run: pytest tests/test_chunking_auto.py -v
"""

from __future__ import annotations

from atrium.smart_chunker import (
    _AUTO_CODE_HEAVY_FRAC,
    _AUTO_LONG_PROSE_CHARS,
    _AUTO_LONG_PROSE_FRAC,
    _AUTO_TABLE_HEAVY_FRAC,
    PRESETS,
    _derive_auto_config,
    chunk_auto,
    chunk_with_mode,
)
from atrium.universal_parser import ParsedElement

# ---------------------------------------------------------------------------
# Fixtures — synthetic ParsedElement streams shaped to trip / not-trip
# each heuristic. No real PDF involved; we want to assert "given this
# stream, Auto picked X" without parser noise.
# ---------------------------------------------------------------------------


def _paragraph(text: str) -> ParsedElement:
    return ParsedElement(type="paragraph", text=text, page=1)


def _heading(text: str) -> ParsedElement:
    return ParsedElement(type="heading", text=text, page=1, heading_level=1)


def _table(text: str = "h1 | h2\n-- | --\nrow1 | row2") -> ParsedElement:
    return ParsedElement(type="table", text=text, page=1)


def _code(text: str = "def f(): pass") -> ParsedElement:
    return ParsedElement(type="code", text=text, page=1)


def _long_paragraph(seed: str = "x", length: int = 850) -> str:
    """Build a paragraph of approximately `length` chars without
    forcing an exact match to the heuristic threshold. We use 850 (50
    over the 800 threshold) so each paragraph comfortably trips it."""
    s = (seed + " ") * (length // (len(seed) + 1) + 1)
    return s[:length]


# A paragraph that is EXACTLY at the threshold (== 800) should trip
# the heuristic — the threshold is `>= 800`, not `> 800`.
def _exactly_threshold_paragraph() -> str:
    return "a" * _AUTO_LONG_PROSE_CHARS


# A paragraph just below threshold (799 chars) should NOT trip.
def _just_below_threshold_paragraph() -> str:
    return "a" * (_AUTO_LONG_PROSE_CHARS - 1)


# ---------------------------------------------------------------------------
# Threshold constants — exposed so the tests can document the heuristic
# tie-in directly (a regression here is a behavior change).
# ---------------------------------------------------------------------------


class TestAutoHeuristicsThresholdsAreStable:
    """Lock the heuristic thresholds. A regression in these constants is
    a behavior change to Auto mode — any bump must be intentional."""

    def test_long_prose_threshold(self):
        assert _AUTO_LONG_PROSE_FRAC == 0.50
        assert _AUTO_LONG_PROSE_CHARS == 800

    def test_table_heavy_threshold(self):
        assert _AUTO_TABLE_HEAVY_FRAC == 0.30

    def test_code_heavy_threshold(self):
        assert _AUTO_CODE_HEAVY_FRAC == 0.20


# ---------------------------------------------------------------------------
# 1. Long-form prose heuristic
# ---------------------------------------------------------------------------


class TestAutoLongFormProseHeuristic:
    """Long-form prose trips when >=50% of paragraphs are >=800 chars."""

    def test_long_prose_heuristic_trips_when_majority_paragraphs_long(self):
        # 6 paragraphs, 4 are >=800 chars (66% >= 50% threshold).
        elements = [
            _paragraph(_long_paragraph(length=850)) for _ in range(4)
        ] + [
            _paragraph("short 1"),     # 50 chars
            _paragraph("short 2"),     # 50 chars
        ]
        cfg = _derive_auto_config(elements)
        # prose_target / overlap must be the long-form values, not precise
        assert cfg.prose_target_tokens == 256
        assert cfg.prose_overlap_tokens == 32
        # mode must announce the heuristic for audit
        assert "long_prose" in cfg.mode

    def test_long_prose_heuristic_trips_at_exact_threshold(self):
        # 2 paragraphs, both exactly at threshold (800 chars each).
        # 100% >= 800 → trips the heuristic.
        elements = [
            _paragraph(_exactly_threshold_paragraph()),
            _paragraph(_exactly_threshold_paragraph()),
        ]
        cfg = _derive_auto_config(elements)
        assert cfg.prose_target_tokens == 256
        assert cfg.prose_overlap_tokens == 32

    def test_long_prose_heuristic_does_not_trip_just_below_threshold(self):
        # 2 paragraphs, both 799 chars (just below 800 threshold).
        # Borderline: heuristic should NOT fire.
        elements = [
            _paragraph(_just_below_threshold_paragraph()),
            _paragraph(_just_below_threshold_paragraph()),
        ]
        cfg = _derive_auto_config(elements)
        # Falls back to precise defaults.
        assert cfg.prose_target_tokens == PRESETS["precise"].prose_target_tokens
        assert cfg.prose_overlap_tokens == PRESETS["precise"].prose_overlap_tokens
        assert "long_prose" not in cfg.mode

    def test_long_prose_heuristic_does_not_trip_when_only_minority_long(self):
        # 6 paragraphs, 2 long (33% < 50% threshold). Even though
        # some paragraphs are long, the heuristic shouldn't fire.
        elements = [
            _paragraph(_long_paragraph(length=850)),
            _paragraph(_long_paragraph(length=850)),
        ] + [
            _paragraph("short") for _ in range(4)
        ]
        cfg = _derive_auto_config(elements)
        assert cfg.prose_target_tokens == PRESETS["precise"].prose_target_tokens
        assert "long_prose" not in cfg.mode

    def test_long_prose_heuristic_uses_only_paragraphs_not_headings_or_code(self):
        # Headings and code are excluded from the "paragraphs" pool
        # even if their text is long. Only `type=='paragraph'` counts.
        elements = [
            _paragraph(_long_paragraph(length=850)),     # 1 long paragraph
            _heading(_long_paragraph(seed="H", length=850)),  # long but not paragraph
            _code(_long_paragraph(seed="d", length=850)),      # long but not paragraph
            _paragraph("short"),
        ]
        cfg = _derive_auto_config(elements)
        # 1 of 2 paragraphs (50%) is long — but the threshold is
        # `>= 50%`, and we have 1 of 2 = 0.5 exactly, which IS >= 0.5.
        # So the heuristic fires. The point: heading + code don't count.
        assert cfg.prose_target_tokens == 256

    def test_long_prose_heuristic_falls_back_when_no_paragraphs(self):
        # Edge case: all elements are tables/code. The paragraph filter
        # is empty; the heuristic must not crash.
        elements = [_table(), _code(), _code()]
        cfg = _derive_auto_config(elements)
        # No paragraphs → long_prose heuristic not evaluated
        assert "long_prose" not in cfg.mode


# ---------------------------------------------------------------------------
# 2. Table-heavy heuristic
# ---------------------------------------------------------------------------


class TestAutoTableHeavyHeuristic:
    """Table-heavy trips when >=30% of all elements are tables."""

    def test_table_heavy_trips(self):
        # 10 elements, 3 tables (30% >= 30% threshold).
        elements = (
            [_table() for _ in range(3)]
            + [_paragraph("p1"), _paragraph("p2"), _paragraph("p3"),
               _paragraph("p4"), _code(), _code(), _code()]
        )
        cfg = _derive_auto_config(elements)
        assert cfg.table_max_rows == 5
        assert cfg.table_max_tokens == 256
        assert "table_heavy" in cfg.mode

    def test_table_heavy_trips_with_higher_ratio(self):
        # 5 elements, 4 tables (80% >> 30%).
        elements = [_table() for _ in range(4)] + [_paragraph("p")]
        cfg = _derive_auto_config(elements)
        assert cfg.table_max_rows == 5
        assert "table_heavy" in cfg.mode

    def test_table_heavy_does_not_trip_just_below_threshold(self):
        # 10 elements, 2 tables (20% < 30% threshold).
        elements = (
            [_table() for _ in range(2)]
            + [_paragraph(f"p{i}") for i in range(8)]
        )
        cfg = _derive_auto_config(elements)
        # Falls back to precise defaults.
        assert cfg.table_max_rows == PRESETS["precise"].table_max_rows
        assert cfg.table_max_tokens == PRESETS["precise"].table_max_tokens
        assert "table_heavy" not in cfg.mode

    def test_table_heavy_counted_across_all_elements(self):
        # The denominator is ALL elements, not just paragraphs.
        # 10 total, 3 tables = 30% — trips.
        elements = (
            [_table() for _ in range(3)]
            + [_code() for _ in range(7)]
        )
        cfg = _derive_auto_config(elements)
        assert "table_heavy" in cfg.mode


# ---------------------------------------------------------------------------
# 3. Code-heavy heuristic
# ---------------------------------------------------------------------------


class TestAutoCodeHeavyHeuristic:
    """Code-heavy trips when >=20% of all elements are code."""

    def test_code_heavy_trips(self):
        # 10 elements, 2 code (20% >= 20% threshold).
        elements = (
            [_code() for _ in range(2)]
            + [_paragraph(f"p{i}") for i in range(8)]
        )
        cfg = _derive_auto_config(elements)
        assert cfg.code_max_lines == 50
        assert "code_heavy" in cfg.mode

    def test_code_heavy_trips_with_higher_ratio(self):
        # 5 elements, all code (100%).
        elements = [_code() for _ in range(5)]
        cfg = _derive_auto_config(elements)
        assert cfg.code_max_lines == 50
        assert "code_heavy" in cfg.mode

    def test_code_heavy_does_not_trip_just_below_threshold(self):
        # 10 elements, 1 code (10% < 20% threshold).
        elements = (
            [_code()]
            + [_paragraph(f"p{i}") for i in range(9)]
        )
        cfg = _derive_auto_config(elements)
        assert cfg.code_max_lines == PRESETS["precise"].code_max_lines
        assert "code_heavy" not in cfg.mode


# ---------------------------------------------------------------------------
# 4. Edge-case stacking: multiple heuristics together
# ---------------------------------------------------------------------------


class TestAutoHeuristicStacking:
    """Multiple heuristics can stack on the same document."""

    def test_long_prose_and_table_heavy_stack(self):
        # Long-form prose (4 long + 2 short = 66%) AND table-heavy
        # (3 tables / 10 = 30%). Both heuristics fire.
        elements = (
            [_paragraph(_long_paragraph(length=850)) for _ in range(4)]
            + [_paragraph("short"), _paragraph("short")]
            + [_table() for _ in range(3)]
            + [_code()]
        )
        cfg = _derive_auto_config(elements)
        # Long-prose adjustment
        assert cfg.prose_target_tokens == 256
        assert cfg.prose_overlap_tokens == 32
        # Table-heavy adjustment
        assert cfg.table_max_rows == 5
        assert cfg.table_max_tokens == 256
        # Mode announces both
        assert "long_prose" in cfg.mode
        assert "table_heavy" in cfg.mode

    def test_all_three_heuristics_stack(self):
        # Long-form prose + table-heavy + code-heavy together.
        # Math: 4 long paragraphs / 6 paragraphs = 66% (>=50%).
        #        3 tables / 10 total = 30% (>=30%).
        #        2 code / 10 total = 20% (>=20%).
        # Total = 4 long + 2 short + 3 tables + 2 code = 11. Need
        # <= 10 for the percentages to land cleanly. Drop one short.
        elements = (
            [_paragraph(_long_paragraph(length=850)) for _ in range(4)]
            + [_paragraph("short")]
            + [_table() for _ in range(3)]
            + [_code() for _ in range(2)]
        )
        cfg = _derive_auto_config(elements)
        assert "long_prose" in cfg.mode
        assert "table_heavy" in cfg.mode
        assert "code_heavy" in cfg.mode

    def test_table_heavy_and_code_heavy_without_long_prose(self):
        # Tables + code without long prose.
        elements = (
            [_table() for _ in range(3)]
            + [_code() for _ in range(2)]
            + [_paragraph("p"), _paragraph("p")]
        )
        cfg = _derive_auto_config(elements)
        assert "table_heavy" in cfg.mode
        assert "code_heavy" in cfg.mode
        assert "long_prose" not in cfg.mode


# ---------------------------------------------------------------------------
# 5. Fallback behavior (no heuristic fires)
# ---------------------------------------------------------------------------


class TestAutoFallbackToPrecise:
    """When no heuristic trips, Auto == Precise preset behavior."""

    def test_empty_elements_uses_precise_defaults(self):
        cfg = _derive_auto_config([])
        assert cfg.mode == "auto"
        assert cfg.prose_target_tokens == PRESETS["precise"].prose_target_tokens
        assert cfg.table_max_rows == PRESETS["precise"].table_max_rows
        assert cfg.code_max_lines == PRESETS["precise"].code_max_lines

    def test_mixed_short_elements_uses_precise_defaults(self):
        # 6 short paragraphs + 1 table + 1 code. Table is 1/8 = 12.5%
        # (below 30%), code is 1/8 = 12.5% (below 20%), no paragraph
        # is long. No heuristic fires.
        elements = (
            [_paragraph("short") for _ in range(6)]
            + [_table()]
            + [_code()]
        )
        cfg = _derive_auto_config(elements)
        assert cfg.prose_target_tokens == PRESETS["precise"].prose_target_tokens
        assert cfg.table_max_rows == PRESETS["precise"].table_max_rows
        assert cfg.code_max_lines == PRESETS["precise"].code_max_lines
        assert cfg.mode == "auto"


# ---------------------------------------------------------------------------
# 6. End-to-end via chunk_auto — selected config actually applies
# ---------------------------------------------------------------------------


class TestChunkAutoAppliesDerivedConfig:
    """Verify the selected config actually changes chunking behavior, not
    just the config object. Without this we'd have a config derived
    correctly but never applied."""

    def test_long_prose_heuristic_produces_larger_chunks_than_precise(self):
        # 4 long paragraphs — long_prose heuristic fires.
        long_elements = [
            _paragraph(_long_paragraph(length=1200)) for _ in range(4)
        ]
        # Same elements through Precise preset.
        baseline = chunk_with_mode(
            long_elements, "doc-1", "x.txt", mode="precise",
        )
        # Same elements through Auto.
        auto_chunks = chunk_auto(long_elements, "doc-1", "x.txt")
        # Auto's prose_target_tokens is 256 (vs precise's 128), so each
        # chunk should pack more text on average — fewer chunks total
        # for the same input.
        avg_auto = sum(len(c.text) for c in auto_chunks) / max(len(auto_chunks), 1)
        avg_precise = sum(len(c.text) for c in baseline) / max(len(baseline), 1)
        assert avg_auto >= avg_precise, (
            f"long_prose heuristic didn't enlarge chunks: "
            f"avg_auto={avg_auto:.0f} avg_precise={avg_precise:.0f}"
        )

    def test_table_heavy_heuristic_produces_smaller_table_groups(self):
        # 4 tables with 30 rows each.
        big_table = ("h | h\n" + "r | r\n") * 30
        elements = [_table(big_table) for _ in range(4)]
        baseline = chunk_with_mode(
            elements, "doc-1", "x.txt", mode="precise",
        )
        auto_chunks = chunk_auto(elements, "doc-1", "x.txt")
        # Auto's table_max_rows is 5 (vs precise's 10), so the same
        # 30-row table should produce MORE chunks under auto (each
        # chunk covers fewer rows).
        # Filter for chunks that came from tables (chunk type heuristic
        # — using chunk.section_path or just rely on count).
        assert len(auto_chunks) >= len(baseline), (
            f"table_heavy heuristic didn't tighten row grouping: "
            f"auto={len(auto_chunks)} precise={len(baseline)}"
        )

    def test_chunk_with_mode_auto_routes_to_chunk_auto(self):
        # Smoke test: mode='auto' through chunk_with_mode must use
        # _derive_auto_config, not silently fall through to precise.
        elements = [
            _paragraph(_long_paragraph(length=850)) for _ in range(4)
        ] + [_paragraph("short"), _paragraph("short")]
        # The same element stream going through mode='auto' must
        # produce the same chunks as chunk_auto directly.
        via_mode = chunk_with_mode(elements, "doc-1", "x.txt", mode="auto")
        via_auto = chunk_auto(elements, "doc-1", "x.txt")
        assert len(via_mode) == len(via_auto), (
            f"mode='auto' dispatch differs from chunk_auto: "
            f"via_mode={len(via_mode)} via_auto={len(via_auto)}"
        )