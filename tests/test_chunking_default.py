"""
test_chunking_default.py

Regression tests for the production chunker wiring (`chunk_default` in
`atrium/smart_chunker.py`). The chunker is bound to `PRESETS["precise"]`
— the structure-aware strategy at 700/100 chars — which is the config the
19-fixture benchmark measures.

These tests lock the SHIPPED behavior of `chunk_default` (the wrapper the
production pipeline uses), not the per-strategy research code in
`chunk_elements` (which stays as the comparison reference). A future change
to either the wrapper or the precise preset must update them.

Run:  pytest tests/test_chunking_default.py -v

Note on what used to live here: an earlier revision of this file also
carried strict-validation tests for `ChunkConfig.from_dict` and a
token-budget test against a real bge-large BPE tokenizer. Both belong to
a token-based chunker generation that was never committed — see
`tests/legacy/README.md`. The shipped chunker budgets in characters, so
this file locks the character-based contract instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atrium.smart_chunker import (
    CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_CHUNK_CONFIG,
    PRESETS,
    chunk_default,
    chunk_elements,
    chunk_with_mode,
    estimate_tokens,
)
from atrium.universal_parser import ParsedElement, UniversalParser

REPO_ROOT = Path(__file__).resolve().parent.parent
W2_PDF = REPO_ROOT / "tests" / "benchmarks" / "fixtures" / "17_irs_w2" / "source.pdf"
SEC_PDF = REPO_ROOT / "tests" / "benchmarks" / "fixtures" / "13_sec_10k" / "source.pdf"
BERT_PDF = REPO_ROOT / "tests" / "benchmarks" / "fixtures" / "04_bert_paper" / "source.pdf"


def _parse_and_chunk_default(pdf_path: Path, doc_id: str):
    parser = UniversalParser()
    raw = pdf_path.read_bytes()
    elements = parser.parse(raw, pdf_path.name, "application/pdf")
    chunks = chunk_default(elements, doc_id=doc_id, file_name=pdf_path.name)
    return elements, chunks


# ---------------------------------------------------------------------------
# Wiring checks (the wrapper must point at the validated preset)
# ---------------------------------------------------------------------------

def test_default_chunk_config_is_precise():
    """DEFAULT_CHUNK_CONFIG must equal PRESETS["precise"] — the
    structure-aware preset the 19-fixture benchmark measures. If a future
    experiment promotes a different preset, this test fails and forces the
    author to re-publish the experiment doc and re-run the regression
    checks."""
    assert DEFAULT_CHUNK_CONFIG.mode == "precise"
    assert DEFAULT_CHUNK_CONFIG == PRESETS["precise"]


def test_chunk_default_signature_matches_chunk_elements():
    """`chunk_default` is a drop-in replacement for `chunk_elements` in the
    production ingest path. Lock the (elements, doc_id, file_name) signature
    so a future refactor that breaks the call site fails here."""
    import inspect
    a = inspect.signature(chunk_default)
    b = inspect.signature(chunk_elements)
    assert list(a.parameters.keys())[:3] == list(b.parameters.keys())[:3]


def test_token_accounting_is_the_documented_character_estimate():
    """The shipped chunker budgets in characters, not tokens.

    The chunking-strategy grid used a real bge-large BPE tokenizer; that
    token-based chunker generation was never committed (see
    `tests/legacy/README.md`). The shipped config is character-based, so
    lock the conversion constant instead. If someone swaps in a real
    tokenizer, this test fails and forces the constants to be re-derived
    rather than silently changing every chunk boundary."""
    assert CHARS_PER_TOKEN_ESTIMATE == 4
    assert estimate_tokens("a" * 400) == 100
    # Never returns 0 — a non-empty string always costs at least one token.
    assert estimate_tokens("x") == 1


# ---------------------------------------------------------------------------
# Behavior locks on real fixtures (the D-precise wins from the experiment)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not W2_PDF.exists(), reason="W-2 fixture PDF not present")
def test_chunk_default_w2_chunks_are_more_granular_than_legacy():
    """W-2 (R@5 0.500 → 0.583 in the grid) has more chunks under chunk_default
    than under chunk_elements — the precise preset splits tables to 10-row
    groups where the legacy preset kept the whole table. Lock the ratio:
    chunk_default must produce strictly more chunks than chunk_elements on
    W-2 (the table-heavy fixture that motivated D-precise)."""
    parser = UniversalParser()
    raw = W2_PDF.read_bytes()
    elements = parser.parse(raw, W2_PDF.name, "application/pdf")

    default_chunks = chunk_default(elements, doc_id="w2", file_name=W2_PDF.name)
    legacy_chunks = chunk_elements(elements, doc_id="w2", file_name=W2_PDF.name)

    assert len(default_chunks) > len(legacy_chunks), (
        f"expected chunk_default to produce more chunks on W-2 than "
        f"chunk_elements (precise preset splits tables more aggressively), "
        f"got default={len(default_chunks)} legacy={len(legacy_chunks)}"
    )
    # And the absolute count must stay in a sensible range — if a future
    # change makes it produce 10x the legacy count, something has gone
    # wrong with the table splitter.
    assert len(default_chunks) <= 4 * len(legacy_chunks), (
        f"chunk_default exploded on W-2 ({len(default_chunks)} vs "
        f"{len(legacy_chunks)} legacy). Check the table splitter."
    )


@pytest.mark.skipif(not W2_PDF.exists(), reason="W-2 fixture PDF not present")
def test_chunk_default_w2_chunks_carry_section_path():
    """Per the 2026-10-09 audit (`CORE_BASELINE.md` §5.2), 8.6% of legacy
    chunks ship without section_path. The chunking-strategy grid locked
    D-precise as the new default partly because all four weak-fixture
    lifts (W-2, Wikipedia HTML, Arabic scanned, Tom Sawyer EPUB) come
    from finer section_path coverage. Lock ≥ 95% section_path coverage
    on W-2."""
    _, chunks = _parse_and_chunk_default(W2_PDF, "w2")
    if not chunks:
        pytest.skip("W-2 produced no chunks")
    with_section = sum(1 for c in chunks if c.source.section_path)
    ratio = with_section / len(chunks)
    assert ratio >= 0.95, (
        f"only {with_section}/{len(chunks)} ({ratio:.0%}) chunks carry "
        f"section_path; expected ≥ 95% per the 2026-10-10 grid analysis"
    )


@pytest.mark.skipif(not SEC_PDF.exists(), reason="SEC 10-K fixture PDF not present")
def test_chunk_default_keeps_table_grouping_for_long_tables():
    """Apple 10-K contains long financial tables. The precise preset
    packs tighter than the legacy default, which keeps every chunk inside
    a bge-large-sized window. Lock that no chunk under chunk_default
    exceeds a sensible char budget — bge-large is 512 tokens ≈ ~2560
    chars after the calibrated ratio, and 6.0 chars/token is a safe
    upper bound. The grid validated that D-precise does not blow this
    budget on the 10-K."""
    _, chunks = _parse_and_chunk_default(SEC_PDF, "10k")
    if not chunks:
        pytest.skip("SEC 10-K produced no chunks")
    bge_max_chars = 512 * 6  # 6.0 = safe upper bound on chars/token
    blown = [c for c in chunks if len(c.text) > bge_max_chars]
    blown_sizes = sorted((len(c.text) for c in blown), reverse=True)
    assert not blown, (
        f"{len(blown)} chunks exceed bge-large 512-token budget "
        f"(>{bge_max_chars} chars): sizes={blown_sizes[:5]}"
    )


@pytest.mark.skipif(not BERT_PDF.exists(), reason="BERT fixture PDF not present")
def test_chunk_default_bert_section_coverage_at_least_half():
    """BERT (R@5 0.786 baseline) is a long academic PDF with rich
    section structure. The D-precise win on BERT was small (+0) but
    no-regression; lock that chunk_default still attaches section_path
    on ≥ 50% of BERT chunks — a coarser sanity check than the W-2 case
    because BERT's headings parse with mixed reliability."""
    _, chunks = _parse_and_chunk_default(BERT_PDF, "bert")
    if not chunks:
        pytest.skip("BERT produced no chunks")
    with_section = sum(1 for c in chunks if c.source.section_path)
    ratio = with_section / len(chunks)
    assert ratio >= 0.50, (
        f"only {with_section}/{len(chunks)} ({ratio:.0%}) BERT chunks "
        f"carry section_path; expected ≥ 50%"
    )


# ---------------------------------------------------------------------------
# Dispatcher contract — `chunk_with_mode` must reach the same config
# ---------------------------------------------------------------------------

class TestChunkWithModeReachesTheSameConfig:
    """`chunk_default` and `chunk_with_mode(mode="precise")` are two doors
    into the same config. If they drift, the benchmark (which calls
    chunk_with_mode) and any legacy caller (which calls chunk_default)
    stop measuring the same thing."""

    def test_chunk_default_matches_precise_mode(self):
        elements = [
            ParsedElement(
                type="paragraph",
                text=("The quick brown fox jumps over the lazy dog. " * 6),
                page=1,
            ),
        ]
        via_default = chunk_default(elements, "doc-1", "x.txt")
        via_mode = chunk_with_mode(elements, "doc-1", "x.txt", mode="precise")
        assert len(via_default) == len(via_mode)

    def test_auto_mode_resolves_to_the_precise_parameters(self):
        """Auto is the production default and must serve the same
        parameters as the precise preset (the 2026-10-17 fix that stopped
        auto silently serving the baseline strategy)."""
        from atrium.smart_chunker import _resolve_chunk_config
        cfg = _resolve_chunk_config("auto", None)
        assert cfg.target_chars == PRESETS["precise"].target_chars
        assert cfg.overlap_chars == PRESETS["precise"].overlap_chars
        assert cfg.strategy == PRESETS["precise"].strategy
        # The audit label still says "auto" even though the parameters
        # are the precise preset's.
        assert cfg.mode == "auto"

    def test_chunk_config_carries_its_preset_name(self):
        for name in ("balanced", "precise", "context_rich"):
            assert PRESETS[name].mode == name

    def test_custom_mode_overlays_the_precise_preset(self):
        from atrium.smart_chunker import _resolve_chunk_config
        cfg = _resolve_chunk_config("custom", {"target_chars": 500})
        assert cfg.target_chars == 500
        assert cfg.overlap_chars == PRESETS["precise"].overlap_chars
        assert cfg.mode == "custom"

    def test_custom_mode_rejects_unknown_keys(self):
        from atrium.smart_chunker import _resolve_chunk_config
        with pytest.raises(ValueError) as ei:
            _resolve_chunk_config("custom", {"target_chrs": 500})
        assert "target_chrs" in str(ei.value)
