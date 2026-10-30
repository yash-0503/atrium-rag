"""
test_phase1a.py

Batched tests for Phase 1A (1-day wins) and the phantom-table fix
discovered against a developer-supplied example PDF.

Run:  pytest tests/test_phase1a.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import fitz
import pytest

from atrium.job_store import JobStore
from atrium.smart_chunker import _split_sentences, chunk_elements
from atrium.universal_parser import ParsedElement, UniversalParser

# The document this test was written against. It is not in the repo
# (it is a local file), so the test skips unless the path is pointed at
# one — via $ATRIUM_DEV_EXAMPLE_PDF, or by dropping a file with this
# name next to the tests directory. The absolute path that used to be
# hardcoded here was the developer's home directory and leaked into
# the public history.
DEV_EXAMPLE_PDF = Path(
    os.environ.get("ATRIUM_DEV_EXAMPLE_PDF")
    or (Path(__file__).resolve().parent / "dev-example.pdf")
)


# ---------------------------------------------------------------------------
# Phantom-table fix — the critical bug from the user's screenshot
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DEV_EXAMPLE_PDF.exists(), reason="dev-example.pdf not present")
def test_dev_example_pdf_no_phantom_table_chunks():
    """The bug: page.find_tables() detected a phantom full-page 'table'
    on every page of the Canva-style dev-example.pdf, which then masked
    every real paragraph and shipped the whole page text as one fake
    table chunk. The fix filters phantom tables in _parse_pdf_page."""
    parser = UniversalParser()
    raw = DEV_EXAMPLE_PDF.read_bytes()
    elements = parser.parse(raw, "dev-example.pdf", "application/pdf")

    # The fake "phantom" table chunks would have a `table` type on every
    # page (pages 1-3, 5-6) where there is no real table. The real table
    # only exists on page 4.
    table_pages = sorted({e.page for e in elements if e.type == "table"})
    assert table_pages == [4], (
        f"expected only page 4 to have a table element, got {table_pages}"
    )


@pytest.mark.skipif(not DEV_EXAMPLE_PDF.exists(), reason="dev-example.pdf not present")
def test_dev_example_pdf_paragraphs_extracted():
    """After the fix, the actual paragraph content of each page must be
    reachable as paragraph (or heading) elements — not buried under a
    phantom table that swallows the whole page."""
    parser = UniversalParser()
    raw = DEV_EXAMPLE_PDF.read_bytes()
    elements = parser.parse(raw, "dev-example.pdf", "application/pdf")

    paragraph_pages = sorted({e.page for e in elements if e.type == "paragraph"})
    # Pages 1, 2, 3, 4, 5, 6 should all have at least one paragraph now.
    assert paragraph_pages == [1, 2, 3, 4, 5, 6], (
        f"expected paragraphs on every page, got {paragraph_pages}"
    )

    # Spot-check: the page 2 body text should be a paragraph, not buried.
    page2_paragraphs = [e.text for e in elements if e.type == "paragraph" and e.page == 2]
    assert any("designed to showcase" in t for t in page2_paragraphs), (
        "page 2 paragraph missing after phantom-table fix"
    )


@pytest.mark.skipif(not DEV_EXAMPLE_PDF.exists(), reason="dev-example.pdf not present")
def test_dev_example_pdf_real_table_survives():
    """The phantom filter must NOT drop the real ID/Name/Role table
    on page 4 — only the page-level phantom rectangle."""
    parser = UniversalParser()
    raw = DEV_EXAMPLE_PDF.read_bytes()
    elements = parser.parse(raw, "dev-example.pdf", "application/pdf")
    real_tables = [e for e in elements if e.type == "table" and e.page == 4]
    assert len(real_tables) == 1
    assert "ID" in real_tables[0].text
    assert "Name" in real_tables[0].text
    assert "001" in real_tables[0].text
    assert "John Doe" in real_tables[0].text


def test_is_phantom_table_unit():
    """Direct unit test of _is_phantom_table using a synthetic fitz
    TableFinder result is awkward (Tab is opaque), so instead we test
    the geometric signal directly by feeding the real dev-example.pdf
    and asserting that the whole-page rectangle is filtered while the
    real page-4 data table survives."""
    if not DEV_EXAMPLE_PDF.exists():
        pytest.skip("dev-example.pdf not present")
    parser = UniversalParser()
    doc = fitz.open(str(DEV_EXAMPLE_PDF))
    page = doc[3]  # page 4
    raw_tables = page.find_tables().tables
    assert len(raw_tables) == 2
    kept = [t for t in raw_tables if not parser._is_phantom_table(t, page.rect)]
    assert len(kept) == 1
    doc.close()


# ---------------------------------------------------------------------------
# CJK sentence splitter (1A.2)
# ---------------------------------------------------------------------------

def test_cjk_sentence_split_chinese_period():
    parts = _split_sentences("第一句。这是第二句。还有第三句。")
    assert parts == ["第一句。", "这是第二句。", "还有第三句。"]


def test_cjk_sentence_split_mixed_punctuation():
    """Mixed Latin/CJK without whitespace is inherently ambiguous — we
    can't tell where 'English first.' ends and '第二句话' begins because
    there's no boundary character. The CJK path fires (text contains
    CJK punctuation) and splits on CJK terminals, producing one chunk
    per CJK sentence."""
    parts = _split_sentences("English first. 第二句话！And English again? 第三句话。")
    # Two CJK sentences -> two chunks. The Latin sentence runs into the
    # first CJK chunk because there's no whitespace boundary.
    assert len(parts) == 2
    assert "第二句话！" in parts[0]
    assert "第三句话。" in parts[1]


def test_cjk_sentence_split_japanese_period():
    parts = _split_sentences("日本語の文章。これはテストです。最後。")
    assert len(parts) == 3


def test_cjk_sentence_split_no_split_inside_abbrev():
    """The Latin abbreviation handling must still work — 'e.g.' should
    not be split on the period."""
    parts = _split_sentences("We use Python e.g. for scripts. It is great.")
    assert len(parts) == 2
    assert "e.g." in parts[0]


# ---------------------------------------------------------------------------
# JobStore (1A.4)
# ---------------------------------------------------------------------------

def test_job_store_create_get_update(tmp_path):
    db_path = str(tmp_path / "jobs.db")
    store = JobStore(db_path=db_path)
    store.create("job-1", "doc-1")
    record = store.get("job-1")
    assert record is not None
    assert record["doc_id"] == "doc-1"
    assert record["status"] == "pending"
    store.update("job-1", status="running")
    record = store.get("job-1")
    assert record["status"] == "running"


def test_job_store_persists_across_reload(tmp_path):
    db_path = str(tmp_path / "jobs.db")
    store = JobStore(db_path=db_path)
    store.create("job-2", "doc-2")
    store.update("job-2", status="complete", chunks_written=42)
    del store

    store2 = JobStore(db_path=db_path)
    record = store2.get("job-2")
    assert record is not None
    assert record["status"] == "complete"
    assert record["chunks_written"] == 42


def test_job_store_get_missing_returns_none(tmp_path):
    store = JobStore(db_path=str(tmp_path / "jobs.db"))
    assert store.get("nope") is None


# ---------------------------------------------------------------------------
# Parser error handling (1A.1)
# ---------------------------------------------------------------------------

def test_encrypted_pdf_raises_value_error(tmp_path):
    """Encrypted PDFs (or any FileDataError input) must surface as a
    ValueError with a user-readable message, not an uncaught RuntimeError
    leaking through to a 500. Different PyMuPDF versions raise different
    exception classes for encrypted PDFs; we assert on the error MESSAGE,
    not the exception type, to stay portable."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "secret content")
    out = tmp_path / "encrypted.pdf"
    doc.save(str(out), encryption=fitz.PDF_ENCRYPT_AES_256,
              owner_pw="secret", user_pw="secret")
    doc.close()

    parser = UniversalParser()
    with pytest.raises(ValueError, match=r"(?i)password|encrypt|unreadable"):
        parser.parse(out.read_bytes(), "enc.pdf", "application/pdf")


def test_corrupt_pdf_raises_value_error():
    """Random non-PDF bytes must produce a clean ValueError, not a
    PyMuPDF RuntimeError leaking to a 500."""
    parser = UniversalParser()
    with pytest.raises(ValueError, match=r"(?i)corrupt|unreadable"):
        parser.parse(b"this is definitely not a pdf", "bad.pdf", "application/pdf")


def test_empty_pdf_raises_value_error():
    """Zero-byte input must be handled gracefully."""
    parser = UniversalParser()
    with pytest.raises(ValueError):
        parser.parse(b"", "empty.pdf", "application/pdf")


# ---------------------------------------------------------------------------
# Caption/note attachment (regression — the original user complaint)
# ---------------------------------------------------------------------------

def test_caption_attaches_to_table():
    elements = [
        ParsedElement(type="paragraph", text="Sample tables", page=1),
        ParsedElement(type="table", text="A | B\n--- | ---\n1 | 2", page=1),
    ]
    chunks = chunk_elements(elements, "d", "f")
    assert len(chunks) == 1
    assert chunks[0].element_type == "table"
    assert "Sample tables" in chunks[0].text
    assert "A | B" in chunks[0].text


def test_note_attaches_to_table():
    elements = [
        ParsedElement(type="table", text="A | B\n--- | ---\n1 | 2", page=1),
        ParsedElement(type="paragraph", text="see layout problems", page=1),
    ]
    chunks = chunk_elements(elements, "d", "f")
    assert len(chunks) == 1
    assert "see layout problems" in chunks[0].text


def test_caption_not_eaten_when_real_paragraph_between():
    """A caption-shaped paragraph followed by a real paragraph before a
    table must NOT attach — the real paragraph breaks adjacency."""
    elements = [
        ParsedElement(type="paragraph", text="Sample tables", page=1),
        ParsedElement(type="paragraph", text="Some real prose that ends with a period.", page=1),
        ParsedElement(type="table", text="A | B\n--- | ---\n1 | 2", page=1),
    ]
    chunks = chunk_elements(elements, "d", "f")
    # Expect: caption+real-paragraph packed into one paragraph chunk,
    # table as its own chunk. Caption NOT eaten by the table.
    table_chunks = [c for c in chunks if c.element_type == "table"]
    assert len(table_chunks) == 1
    assert "Sample tables" not in table_chunks[0].text
    assert "Some real prose" not in table_chunks[0].text
