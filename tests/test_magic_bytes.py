"""
test_magic_bytes.py

M0-4 (2026-10-10): unit tests for the lightweight signature-based
MIME detection that supplements libmagic on environments without it
(CI, dev, slim Docker images).

The M0-4 contract: `/jobs` rejects uploads where the declared
content_type disagrees with the detected MIME (returns 415). When
libmagic is installed it does the detection; when it isn't, the
fallback `_match_signature` reads the first bytes and matches
against known file signatures.

These tests pin `_match_signature`'s behavior for the formats the
API advertises. False positives (e.g. a random binary blob matching
a prefix by accident) are accepted as long as the common cases
(PDF, ZIP, PNG, JPEG) classify correctly.
"""

from __future__ import annotations

import pytest

from atrium.main import _match_signature


@pytest.mark.parametrize("content,expected", [
    # PDF — the canonical signature for the 19-fixture grid.
    (b"%PDF-1.4\n%...", "application/pdf"),
    (b"%PDF-1.7", "application/pdf"),
    # ZIP — covers DOCX, XLSX, PPTX, EPUB, ODS, ODP, ODT.
    (b"PK\x03\x04some content", "application/zip"),
    # PNG / JPEG / GIF.
    (b"\x89PNG\r\n\x1a\n...", "image/png"),
    (b"\xff\xd8\xff\xe0...", "image/jpeg"),
    (b"GIF87a...", "image/gif"),
    (b"GIF89a...", "image/gif"),
    # Plain text / markdown / json / csv — detected via the printable
    # ASCII heuristic (no real signature).
    (b"# Hello world\n", "text/plain"),
    (b"key,value\nfoo,bar\n", "text/plain"),
    (b'{"foo": "bar"}', "text/plain"),
    # HTML — starts with `<` which is < 0x80, so it falls into the
    # text branch. The fixture is text/html; the API accepts the
    # upload and lets the parser figure it out.
    (b"<!DOCTYPE html><html>...</html>", "text/plain"),
    # RTF.
    (b'{\\rtf1\\ansi\\ansicpg1252...', "application/rtf"),
])
def test_match_signature_recognized(content: bytes, expected: str):
    """Common file signatures map to the right MIME. Covers the
    19-fixture grid plus the formats the API advertises."""
    assert _match_signature(content) == expected


def test_match_signature_empty_returns_none():
    """Empty content is the caller's responsibility to reject; the
    signature matcher returns None so the upload isn't classified
    as opaque octet-stream and accidentally 415'd by /jobs."""
    assert _match_signature(b"") is None


def test_match_signature_binary_returns_none():
    """Random binary content with no recognizable signature
    returns None — the caller treats the upload as opaque. This
    is the desired behavior: `_match_signature` is a supplement
    to libmagic, not a replacement, so a non-match doesn't fail
    the upload (only a content-type-mismatch does)."""
    # Random bytes that don't match any signature prefix.
    assert _match_signature(b"\x00\x01\x02\x03\x04\x05") is None
    # A binary blob that happens to start with PK (the ZIP magic)
    # is classified as ZIP — that prefix is too short to be
    # ambiguous in practice (PK is 2 bytes and 2 bytes of randomness
    # gives 1/65536 chance). The fixture-level test above already
    # pins the common cases.
    assert _match_signature(b"\x99\xab\xcd\xef") is None