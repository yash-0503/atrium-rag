"""A zip-container document may not expand without bound.

`MAX_UPLOAD_BYTES` caps the compressed upload. It says nothing about the
number that costs memory: deflate reaches ~1000:1 on repetitive data, so
a few kilobytes of archive can declare gigabytes of members. Every parser
for these formats reads a member into `bytes` in full, which turns a small
upload into an out-of-memory kill inside a worker, with no message that
points back at the file.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from atrium import universal_parser
from atrium.universal_parser import UniversalParser

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
EPUB = "application/epub+zip"


def _archive(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def _bomb(megabytes: int = 8) -> bytes:
    """A small archive that expands to `megabytes` — the shape in the issue.

    Deflate collapses the zeroes, so the compressed bytes stay in the
    kilobytes while the declared uncompressed size is megabytes.
    """
    return _archive({"word/document.xml": b"\0" * (megabytes * 1024 * 1024)})


def test_bomb_is_compressed_far_smaller_than_it_expands():
    """Guard the premise: this fixture is a bomb, not just a big file."""
    payload = _bomb(8)
    assert len(payload) < 8 * 1024 * 1024 // 100
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert sum(m.file_size for m in zf.infolist()) == 8 * 1024 * 1024


def test_oversized_archive_is_rejected_before_it_is_decompressed(monkeypatch):
    monkeypatch.setenv("MAX_DECOMPRESSED_BYTES", str(1024 * 1024))
    parser = UniversalParser()
    with pytest.raises(ValueError, match="MAX_DECOMPRESSED_BYTES"):
        parser.parse(_bomb(8), "bomb.docx", DOCX)


def test_rejection_names_the_mime_type_and_the_limit(monkeypatch):
    monkeypatch.setenv("MAX_DECOMPRESSED_BYTES", str(1024 * 1024))
    parser = UniversalParser()
    with pytest.raises(ValueError) as excinfo:
        parser.parse(_bomb(4), "bomb.docx", DOCX)
    message = str(excinfo.value)
    assert DOCX in message
    assert "1048576" in message


def test_epub_takes_the_same_path(monkeypatch):
    """The guard is keyed on the container, not on the Office formats.

    A regression that only covered docx/pptx/xlsx would leave EPUB — the
    format the issue names first — reading members unguarded.
    """
    monkeypatch.setenv("MAX_DECOMPRESSED_BYTES", str(1024 * 1024))
    parser = UniversalParser()
    with pytest.raises(ValueError, match="MAX_DECOMPRESSED_BYTES"):
        parser.parse(_bomb(8), "bomb.epub", EPUB)


def test_archive_within_the_limit_passes_the_guard():
    parser = UniversalParser()
    parser._check_archive_expansion(_bomb(1), DOCX)  # must not raise


def test_member_count_is_bounded_on_its_own(monkeypatch):
    """Many empty members cost memory without tripping the byte cap."""
    monkeypatch.setattr(universal_parser, "MAX_ARCHIVE_MEMBERS", 4)
    parser = UniversalParser()
    payload = _archive({f"part{i}.xml": b"" for i in range(5)})
    with pytest.raises(ValueError, match="members"):
        parser._check_archive_expansion(payload, DOCX)


def test_corrupt_archive_is_left_to_the_format_parser():
    """A non-zip must not be rejected here — the parser has the better message."""
    parser = UniversalParser()
    parser._check_archive_expansion(b"not a zip at all", DOCX)  # must not raise


def test_all_zip_container_formats_are_covered():
    assert DOCX in universal_parser.ZIP_CONTAINER_MIME_TYPES
    assert EPUB in universal_parser.ZIP_CONTAINER_MIME_TYPES
    for mime in (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
    ):
        assert mime in universal_parser.ZIP_CONTAINER_MIME_TYPES
