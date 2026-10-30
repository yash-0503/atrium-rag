"""
test_docx_page_derivation.py

Focused regression test for the `docx_page_by_heading` opt-in flag
(UniversalParser.__init__, app/universal_parser.py).

Why this exists (2026-10-18):
- The only DOCX fixture in the 19-set (10_gov_docx) declares "Pages
  derive from Heading 1 boundaries" in its ground truth, but the
  production parser's default bumps on Heading 1 + Heading 2
  (`bump_levels=(1, 2)`). That caused a 4-page drift on the 4
  ROUTING_ISSUE queries (`analyze_gate_diff.py`) where the gate was
  confident-but-wrong.
- The opt-in flag (`docx_page_by_heading=True`) switches to
  `bump_levels=(1,)` and brings the relevant chunks onto the ground
  truth's expected pages. Without this test, a future regression in
  the bump_levels wiring would silently re-introduce the 4-page
  drift.

Each test uses the 10_gov_docx fixture (the only DOCX in the 19-set)
and a few synthetic Heading-1-only DOCX bodies built in the test
itself, so we cover both real-fixture behaviour and the unit-level
boundary cases that motivated the flag (single H1, multiple H1s, no
H1 at all).
"""
from __future__ import annotations

import io
from pathlib import Path

import docx
import pytest

from atrium.universal_parser import UniversalParser

FIXTURES = Path(__file__).parent / "benchmarks" / "fixtures" / "10_gov_docx"


def _read_fixture() -> tuple[bytes, str, str]:
    """The 10_gov_docx fixture.

    It is a real government document and is not part of this
    repository, so the tests that need it skip rather than fail.
    """
    src = FIXTURES / "source.docx"
    if not src.exists():
        pytest.skip("10_gov_docx fixture not present")
    return src.read_bytes(), "source.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _build_docx_with_headings(heading_levels: list[int], heading_texts: list[str], bodies: list[str]) -> bytes:
    """Build a minimal DOCX in-memory with the given heading/body pairs.

    Each `heading_levels[i]` and `heading_texts[i]` produces one heading
    paragraph; the corresponding `bodies[i]` is a non-heading paragraph
    emitted right after. Returns raw DOCX bytes."""
    doc = docx.Document()
    for level, text, body in zip(heading_levels, heading_texts, bodies):
        doc.add_heading(text, level=level)
        if body:
            doc.add_paragraph(body)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_opt_in_aligns_dense_corpus_to_heading1_pages():
    """Under `docx_page_by_heading=True`, the 10_gov_docx fixture's
    Research Team, Timeline, and Budget table chunks land on the
    pages the ground truth expects. Before the fix these drifted by
    4 pages (PI on page 11, expected page 7)."""
    from atrium.smart_chunker import chunk_with_mode

    fb, fn, mime = _read_fixture()
    parser = UniversalParser(docx_page_by_heading=True)
    elements = parser.parse(fb, fn, mime)
    chunks = chunk_with_mode(
        elements, doc_id="test", file_name=fn, mode="auto", config=None,
    )

    # The Title page also contains "Dr. Jane Smith" (author line), so
    # we filter to chunks inside a section path (the body chunks, not
    # the title chunk which has section_path=[]).
    pi_chunks = [
        c for c in chunks
        if "Dr. Jane Smith" in (c.text or "")
        and c.source.section_path  # exclude title page (no section_path)
    ]
    timeline_chunks = [
        c for c in chunks
        if "Year 1" in (c.text or "") and "Months" in (c.text or "")
    ]
    budget_chunks = [
        c for c in chunks
        if "Personnel" in (c.text or "") and "$420K" in (c.text or "")
    ]

    assert pi_chunks, "Research Team chunk should contain 'Dr. Jane Smith' (excluding title page)"
    assert pi_chunks[0].source.page_number == 7, (
        f"Research Team should be on page 7 (Heading 1 boundary), got page {pi_chunks[0].source.page_number}"
    )

    assert timeline_chunks, "Timeline chunk should contain 'Year 1' / 'Months'"
    assert timeline_chunks[0].source.page_number == 8, (
        f"Timeline should be on page 8, got page {timeline_chunks[0].source.page_number}"
    )

    assert budget_chunks, "Budget table chunk should contain 'Personnel' and '$420K'"
    assert budget_chunks[0].source.page_number == 9, (
        f"Budget table should be on page 9, got page {budget_chunks[0].source.page_number}"
    )


def test_default_flag_keeps_heading1_plus_heading2_behavior():
    """Without the opt-in, the parser still bumps on Heading 1 + Heading 2
    (the production default — must not silently regress). On the
    10_gov_docx fixture this preserves the legacy page numbers, even
    though they don't match ground truth."""
    from atrium.smart_chunker import chunk_with_mode

    fb, fn, mime = _read_fixture()
    parser = UniversalParser()  # default: docx_page_by_heading=False
    elements = parser.parse(fb, fn, mime)
    chunks = chunk_with_mode(
        elements, doc_id="test", file_name=fn, mode="auto", config=None,
    )
    pi_chunks = [
        c for c in chunks
        if "Dr. Jane Smith" in (c.text or "")
        and c.source.section_path  # exclude title page
    ]
    assert pi_chunks, "Research Team chunk should still exist under default flag"
    # Under default, the H1+H2 bumps push Research Team to page 11
    # (verified against the 2026-10-17 baseline run).
    assert pi_chunks[0].source.page_number == 11, (
        f"Default flag should keep Research Team on page 11; got {pi_chunks[0].source.page_number}. "
        "If this regresses, the production default silently changed and other DOCX consumers may be affected."
    )


def test_heading1_only_bumps_one_page_per_h1():
    """Unit-level boundary: a 4-H1 doc with no Heading 2 should produce
    exactly 4 page bumps under the opt-in flag. Pre-heading content
    lands on page 1; each H1 bumps to the next page."""
    fb = _build_docx_with_headings(
        heading_levels=[1, 1, 1, 1],
        heading_texts=["Section A", "Section B", "Section C", "Section D"],
        bodies=["body A", "body B", "body C", "body D"],
    )
    parser = UniversalParser(docx_page_by_heading=True)
    elements = parser.parse(fb, "doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    pages = [e.page for e in elements]
    # Pre-heading-1 content (none here, since H1 is the first element) +
    # first H1 → page 1 (first H1 doesn't bump because out is empty)
    # second H1 → page 2, third → page 3, fourth → page 4
    assert pages == [1, 1, 2, 2, 3, 3, 4, 4], (
        f"Expected pages [1,1,2,2,3,3,4,4] for 4 H1s with bodies, got {pages}"
    )


def test_heading1_plus_heading2_default_behavior_kept():
    """Default flag bumps on both H1 and H2. 2 H1s + 2 H2s = 4 bumps
    instead of 2 — proves the default flag preserves the original
    behaviour and is non-trivially different from the opt-in."""
    fb = _build_docx_with_headings(
        heading_levels=[1, 2, 1, 2],
        heading_texts=["Section A", "Subsection A.1", "Section B", "Subsection B.1"],
        bodies=["body", "", "body", ""],
    )
    parser = UniversalParser()  # default: H1+H2
    elements = parser.parse(fb, "doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    pages = [e.page for e in elements]
    # Empty-body paragraphs are filtered out by the parser. With
    # H1+H2 bump_levels, the surviving 6 elements land on:
    #   H1 Section A → page 1 (out was empty, no bump)
    #   body "body" → page 1
    #   H2 Subsection A.1 → page 2
    #   H1 Section B → page 3
    #   body "body" → page 3
    #   H2 Subsection B.1 → page 4
    assert pages == [1, 1, 2, 3, 3, 4], (
        f"Expected [1,1,2,3,3,4] for H1+H2 default, got {pages}"
    )


def test_no_headings_at_all_stays_on_page_one():
    """A doc with zero headings should produce a single page-1 chunk
    under both flag settings — sanity check that the absence of
    headings doesn't blow up."""
    from docx import Document
    doc = Document()
    doc.add_paragraph("Just a plain paragraph, no headings anywhere.")
    doc.add_paragraph("Another plain paragraph.")
    buf = io.BytesIO()
    doc.save(buf)
    fb = buf.getvalue()
    for opt_in in (False, True):
        parser = UniversalParser(docx_page_by_heading=opt_in)
        elements = parser.parse(fb, "doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert all(e.page == 1 for e in elements), (
            f"opt_in={opt_in}: expected all page=1 for no-heading input, got {[e.page for e in elements]}"
        )


def test_opt_in_does_not_affect_non_docx_parsers():
    """The opt-in flag must be inert on PDF / HTML / ODT / MD / etc.
    Sanity-check by parsing an HTML fixture and asserting page
    distribution is unchanged regardless of the flag."""
    html_path = FIXTURES.parent / "07_wikipedia_html" / "source.html"
    if not html_path.exists():
        pytest.skip("07_wikipedia_html fixture not present")
    fb = html_path.read_bytes()
    p_default = UniversalParser()
    p_optin = UniversalParser(docx_page_by_heading=True)
    el_default = p_default.parse(fb, "source.html", "text/html")
    el_optin = p_optin.parse(fb, "source.html", "text/html")
    pages_default = [e.page for e in el_default]
    pages_optin = [e.page for e in el_optin]
    assert pages_default == pages_optin, (
        f"HTML page distribution must be identical regardless of docx_page_by_heading. "
        f"default={pages_default[:10]} optin={pages_optin[:10]}"
    )