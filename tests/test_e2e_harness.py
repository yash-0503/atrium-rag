"""
test_e2e_harness.py

End-to-end harness for fixture-driven tests. Each fixture under
tests/fixtures/<name>/ contains:
  - source.{pdf|docx|pptx|xlsx|html}  -- the document to ingest
  - meta.json                          -- {"element_types": [...], "min_chunks": N, ...}

The harness parses and chunks the document, then asserts that:
  - The chunk list is non-empty and reaches `min_chunks`.
  - The expected element types all appear at least once.
  - Every chunk has a content_hash and a non-empty text.

Embedders and DBs are mocked out — this is purely a parser/chunker
contract test. Tests that need a real embedder/DB live in
tests/test_phase2.py (hybrid search, search endpoint) and rely on
either a fixture docker-compose stack or stubbed adapters.

Fixtures themselves are NOT included in this commit — drop real PDFs
into tests/fixtures/<name>/source.pdf and a matching meta.json. The
test list itself acts as the contract: any fixture whose meta.json is
missing is silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atrium.smart_chunker import chunk_elements
from atrium.universal_parser import UniversalParser

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _discover_fixtures() -> list[Path]:
    if not FIXTURES_DIR.exists():
        return []
    return [p for p in FIXTURES_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists()]


def _fixture_source(fixture_dir: Path) -> tuple[bytes, str, str]:
    """Return (file_bytes, filename, mime_type) for the fixture's source file."""
    for ext, mime in [("pdf", "application/pdf"),
                       ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                       ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
                       ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                       ("html", "text/html")]:
        candidate = fixture_dir / f"source.{ext}"
        if candidate.exists():
            return candidate.read_bytes(), candidate.name, mime
    raise FileNotFoundError(f"No source file in {fixture_dir}")


@pytest.mark.parametrize("fixture_dir", _discover_fixtures(), ids=lambda p: p.name)
def test_fixture_parses_and_chunks(fixture_dir: Path):
    meta: dict[str, Any] = json.loads((fixture_dir / "meta.json").read_text())
    file_bytes, filename, mime = _fixture_source(fixture_dir)

    parser = UniversalParser()
    elements = parser.parse(file_bytes, filename, mime)
    assert elements, f"{fixture_dir.name}: parser produced zero elements"

    chunks = chunk_elements(elements, doc_id=f"fixture-{fixture_dir.name}", file_name=filename)
    assert chunks, f"{fixture_dir.name}: chunker produced zero chunks"
    min_chunks = meta.get("min_chunks", 1)
    assert len(chunks) >= min_chunks, (
        f"{fixture_dir.name}: expected >= {min_chunks} chunks, got {len(chunks)}"
    )

    expected_types = set(meta.get("element_types", []))
    if expected_types:
        seen = {c.element_type for c in chunks}
        missing = expected_types - seen
        assert not missing, f"{fixture_dir.name}: missing element types {missing}"

    for c in chunks:
        assert c.text.strip(), f"{fixture_dir.name}: empty chunk text"
        assert c.content_hash, f"{fixture_dir.name}: empty content_hash"
