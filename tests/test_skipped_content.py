"""
test_skipped_content.py

Content the parser could not read must not be indexed, and the reason
must be reported.

Two failures this pins down:

* a placeholder element ("[Scanned page — no vision or OCR adapter
  configured]") used to be embedded and stored like real content, so
  it came back in the top-k of every search that touched the document
  and crowded out the text that was actually there;
* a document whose every page was unreadable used to report a clean
  success — "ingested, 12 chunks" of placeholder — so an operator had
  no way to tell that their scan had not been read at all.

The skip is counted on the parse, carried on the ingest result, and
written to the job row, so it can be asked about after the fact.
"""

from __future__ import annotations

import pytest

from atrium.errors import NoReadableContentError
from atrium.pipeline import describe_skipped, format_skipped
from atrium.universal_parser import UniversalParser


# A one-pixel JPEG. Nothing can read it without an OCR or vision
# adapter, which is the point: the parse has to drop it and say so.
_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300"
    "08060607060508070707090909080a0c140d0c0b0b0c191213"
    "0f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30"
    "313434341f27393d38323c2e333432ffc0000b0800010001010111"
    "00ffc40014000100000000000000000000000000000000000009"
    "ffda0008010100003f00fbfe8a28a2803fffd9"
)


class TestParserDropsWhatItCannotRead:
    def test_a_raw_image_with_no_vision_or_ocr_yields_nothing(self):
        parser = UniversalParser()
        elements = parser.parse(_JPEG, "photo.jpg", "image/jpeg")
        assert elements == []

    def test_the_skip_is_counted_and_named(self):
        parser = UniversalParser()
        parser.parse(_JPEG, "photo.jpg", "image/jpeg")
        assert parser.last_parse_skipped == {"no_vision_or_ocr_adapter": 1}

    def test_no_placeholder_text_survives_the_parse(self):
        """The placeholder used to be the element's text.

        It was then chunked, embedded and stored like any other
        content, so a search for anything at all could return it.
        """
        parser = UniversalParser()
        elements = parser.parse(_JPEG, "photo.jpg", "image/jpeg")
        for element in elements:
            assert "no vision or OCR" not in (element.text or "")
            assert "pending vision" not in (element.text or "")

    def test_the_reason_is_a_page_level_one(self):
        """A page skip means the document is short of content it should
        have; an image skip means one picture is missing from an
        otherwise readable page."""
        from atrium.universal_parser import PAGE_LEVEL_SKIP_REASONS

        assert "no_vision_or_ocr_adapter" in PAGE_LEVEL_SKIP_REASONS
        assert "image_blank" not in PAGE_LEVEL_SKIP_REASONS


class TestTheReport:
    def test_the_reasons_are_one_comparable_string(self):
        assert format_skipped(
            {"image_blank": 3, "no_vision_or_ocr_adapter": 12}
        ) == "no_vision_or_ocr_adapter=12,image_blank=3"

    def test_an_empty_breakdown_is_an_empty_string(self):
        assert format_skipped({}) == ""
        assert format_skipped(None) == ""

    def test_the_sentence_separates_pages_from_elements(self):
        assert describe_skipped(
            {"no_vision_or_ocr_adapter": 12, "image_blank": 3}
        ) == "12 page(s) unreadable, 3 image(s) skipped"

    def test_the_sentence_omits_what_did_not_happen(self):
        assert describe_skipped({"no_vision_or_ocr_adapter": 12}) == (
            "12 page(s) unreadable"
        )
        assert describe_skipped({"image_blank": 3}) == "3 image(s) skipped"


class TestTheIngest:
    """A document with nothing readable in it must not look ingested."""

    def _ingest(self, db, parser, filename="photo.jpg", mime="image/jpeg"):
        from atrium.pipeline import ingest_document

        class _Embedder:
            dimension = 4

            def embed(self, texts):
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        return ingest_document(
            doc_id="doc-1",
            file_bytes=_JPEG,
            filename=filename,
            mime_type=mime,
            parser=parser,
            embedder=_Embedder(),
            db=db,
        )

    def test_an_unreadable_document_fails_the_ingest(self, tmp_path):
        from atrium.local_store import SqliteAdapter

        store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
        store.ensure_collection("documents", 4)
        with pytest.raises(NoReadableContentError) as excinfo:
            self._ingest(store, UniversalParser())
        message = str(excinfo.value)
        assert "No readable content" in message
        # The message says what to do about it.
        assert "OCR_ENABLED" in message and "VISION_ENABLED" in message

    def test_the_failure_carries_the_counts(self, tmp_path):
        from atrium.local_store import SqliteAdapter

        store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
        store.ensure_collection("documents", 4)
        with pytest.raises(NoReadableContentError) as excinfo:
            self._ingest(store, UniversalParser())
        assert excinfo.value.skipped_elements == 1
        assert excinfo.value.skipped_reasons == {"no_vision_or_ocr_adapter": 1}

    def test_it_is_a_value_error(self, tmp_path):
        """So the API answers 422 — the document is the problem — and not
        503, which would tell the client to retry."""
        from atrium.local_store import SqliteAdapter

        store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
        store.ensure_collection("documents", 4)
        with pytest.raises(ValueError):
            self._ingest(store, UniversalParser())

    def test_nothing_was_stored(self, tmp_path):
        from atrium.local_store import SqliteAdapter

        store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
        store.ensure_collection("documents", 4)
        with pytest.raises(NoReadableContentError):
            self._ingest(store, UniversalParser())
        assert store.count_chunks() == 0

    def test_a_mixed_document_completes_with_the_skip_reported(self, tmp_path):
        """40 readable pages and 3 unreadable ones is not a failure.

        Only a document with *nothing* readable fails. The check is on
        the elements, not on the skips — a PDF with three scanned pages
        and forty typed ones is a document that was ingested.
        """
        from atrium.local_store import SqliteAdapter
        from atrium.universal_parser import ParsedElement
        from atrium.pipeline import ingest_document

        class _MixedParser:
            """A document that is mostly readable, with some pages that
            are not."""

            last_parse_skipped = {"no_vision_or_ocr_adapter": 3}

            def parse(self, file_bytes, filename, mime_type):
                return [
                    ParsedElement(type="paragraph", text=f"Readable page {i}", page=i)
                    for i in range(40)
                ]

        class _Embedder:
            dimension = 4

            def embed(self, texts):
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
        store.ensure_collection("documents", 4)
        result = ingest_document(
            doc_id="doc-1",
            file_bytes=b"whatever",
            filename="mixed.pdf",
            mime_type="application/pdf",
            parser=_MixedParser(),
            embedder=_Embedder(),
            db=store,
        )
        assert result.elements_parsed == 40
        assert result.chunks_written > 0
        assert result.skipped_elements == 3
        assert result.skipped_reasons == {"no_vision_or_ocr_adapter": 3}
        assert store.count_chunks() > 0


class TestTheJobRow:
    """The skip is recorded on the job, so it can be asked about later."""

    def test_the_job_row_records_the_skip(self, tmp_path):
        from atrium.job_store import JobStore
        from atrium.local_store import SqliteAdapter
        from atrium.pipeline import run_ingestion

        store = SqliteAdapter(path=str(tmp_path / "atrium_local.db"))
        store.ensure_collection("documents", 4)
        jobs = JobStore(db_path=str(tmp_path / "jobs.db"))
        job_id = "job-1"
        jobs.create(job_id, "doc-1")

        class _Embedder:
            dimension = 4

            def embed(self, texts):
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        with pytest.raises(NoReadableContentError):
            run_ingestion(
                job_id, "doc-1", _JPEG, "photo.jpg", "image/jpeg",
                UniversalParser(), _Embedder(), store, jobs,
            )

        record = jobs.get(job_id)
        assert record["status"] == "failed"
        assert record["skipped_elements"] == 1
        assert record["skipped_reasons"] == "no_vision_or_ocr_adapter=1"
        assert "No readable content" in record["error"]
