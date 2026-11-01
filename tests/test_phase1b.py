"""
test_phase1b.py

Batched tests for Phase 1B (1-week wins): multi-file ingest concurrency,
Office/HTML format hardening, and the e2e harness scaffold.

Run:  pytest tests/test_phase1b.py -v

Most tests use mocked embedder / DB where appropriate. The /ingest
endpoint tests use FastAPI's TestClient and stub the parser/embedder/db.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from atrium.universal_parser import UniversalParser

# These tests are designed to run without a live Ollama/Qdrant. We mock
# the heavy lifting via monkeypatching the module-level singletons that
# app.main creates at import time.


@pytest.fixture
def app_with_mocks(monkeypatch):
    """Build a FastAPI app instance with parser/embedder/db replaced by
    stubs. Avoids importing app.main directly (which calls Ollama)."""
    from atrium.main import app

    fake_parser = MagicMock()
    # A parser that returns an element, so the ingest succeeds. A parser
    # that returns nothing is a failed ingest — and a failed ingest is a
    # 4xx, which is what the resilience tests assert.
    from atrium.universal_parser import ParsedElement

    fake_parser.parse.return_value = [
        ParsedElement(type="paragraph", text="A paragraph of text.", page=1),
    ]

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.0] * 8]
    fake_embedder.health.return_value = True

    fake_db = MagicMock()
    fake_db.health.return_value = True
    fake_db.ensure_collection.return_value = None
    fake_db.upsert.return_value = None

    monkeypatch.setattr("atrium.main.parser", fake_parser)
    monkeypatch.setattr("atrium.main.embedder", fake_embedder)
    monkeypatch.setattr("atrium.main.db", fake_db)

    # Replace JobStore with a fresh in-memory one so tests don't share state.
    from atrium.job_store import JobStore
    fresh_store = JobStore(db_path=":memory:")
    monkeypatch.setattr("atrium.main.job_store", fresh_store)

    return app, fake_parser, fake_embedder, fake_db


# ---------------------------------------------------------------------------
# 1A.3 — /health endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint_returns_ok_when_db_and_embedder_up(app_with_mocks):
    app, _, _, _ = app_with_mocks
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["db"] is True
    assert body["embedder"] is True


def test_health_endpoint_returns_503_when_embedder_down(app_with_mocks):
    app, _, embedder, _ = app_with_mocks
    embedder.health.return_value = False
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 503
    assert res.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# 1B.1 — Multi-file /ingest + concurrency
# ---------------------------------------------------------------------------

def test_multi_file_ingest_returns_list_of_jobs(app_with_mocks):
    app, parser, embedder, db = app_with_mocks
    client = TestClient(app)

    # Build two minimal valid PDF byte streams (PyMuPDF is happy with empty).
    import fitz
    def make_pdf(name):
        d = fitz.open()
        d.new_page()
        out = io.BytesIO()
        d.save(out)
        d.close()
        return (name, out.getvalue(), "application/pdf")

    f1 = ("a.pdf", b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f\ntrailer\n<<>>\nstartxref\n9\n%%EOF", "application/pdf")
    f2 = ("b.pdf", b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f\ntrailer\n<<>>\nstartxref\n9\n%%EOF", "application/pdf")

    res = client.post("/ingest", files=[("files", f1), ("files", f2)])
    assert res.status_code == 200
    body = res.json()
    assert "jobs" in body
    assert len(body["jobs"]) == 2
    for entry in body["jobs"]:
        assert "job_id" in entry
        assert "doc_id" in entry


def test_ingest_single_back_compat(app_with_mocks):
    app, _, _, _ = app_with_mocks
    client = TestClient(app)
    f = ("a.pdf", b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f\ntrailer\n<<>>\nstartxref\n9\n%%EOF", "application/pdf")
    res = client.post("/ingest_single", files={"file": f})
    assert res.status_code == 200
    body = res.json()
    assert "job_id" in body
    assert "doc_id" in body


def test_ingest_oversize_rejected(app_with_mocks, monkeypatch):
    app, _, _, _ = app_with_mocks
    client = TestClient(app)
    # The cap is lowered rather than sending 100 MB: the assertion is
    # about the status, not about the size.
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    big = b"x" * (51 * 1024 * 1024)
    f = ("big.pdf", big, "application/pdf")
    res = client.post("/ingest_single", files={"file": f})
    # Rejected synchronously with 413. It used to be allowed to answer
    # 200 with an error in the body, which is a failure reported as a
    # success.
    assert res.status_code == 413


# ---------------------------------------------------------------------------
# 1B.2 — Office / HTML hardening
# ---------------------------------------------------------------------------

def _make_pptx_with_dense_bullets() -> bytes:
    """Build a minimal PPTX where one slide has a text frame with many
    bullet paragraphs — the old code collapsed them into one; the new
    code emits one ParsedElement per bullet."""
    from pptx import Presentation
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]  # title + content
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    title.text = "Dense Bullets"
    body = slide.placeholders[1]
    tf = body.text_frame
    tf.text = "First bullet"
    p = tf.add_paragraph()
    p.text = "Second bullet"
    p = tf.add_paragraph()
    p.text = "Third bullet"
    out = io.BytesIO()
    prs.save(out)
    return out.getvalue()


def test_pptx_dense_slide_emits_multiple_paragraphs():
    parser = UniversalParser()
    elements = parser.parse(_make_pptx_with_dense_bullets(), "d.pptx",
                             "application/vnd.openxmlformats-officedocument.presentationml.presentation")
    texts = [e.text for e in elements if e.type in ("paragraph", "list_item")]
    # Old code produced ONE paragraph "First bullet\nSecond bullet\nThird bullet".
    # New code must produce at least three elements.
    assert any("First bullet" in t for t in texts)
    assert any("Second bullet" in t for t in texts)
    assert any("Third bullet" in t for t in texts)


def _make_xlsx_with_many_rows(n_rows: int = 100) -> bytes:
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["ID", "Name", "Value"])
    for i in range(n_rows):
        ws.append([i, f"name-{i}", i * 10])
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()


def test_xlsx_large_sheet_chunked_at_40_rows_with_header_repeat():
    parser = UniversalParser()
    elements = parser.parse(_make_xlsx_with_many_rows(100), "d.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    tables = [e for e in elements if e.type == "table"]
    # 100 rows / 40 = 3 table chunks; each must repeat the header.
    assert len(tables) >= 3
    for t in tables:
        assert "ID" in t.text
        assert "Name" in t.text


def test_xlsx_hidden_sheet_skipped():
    import openpyxl
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Visible"
    ws1.append(["a", "b"])
    ws2 = wb.create_sheet("Hidden")
    ws2.append(["x", "y"])
    ws2.sheet_state = "hidden"
    out = io.BytesIO()
    wb.save(out)

    parser = UniversalParser()
    elements = parser.parse(out.getvalue(), "h.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    headings = [e.text for e in elements if e.type == "heading"]
    assert "Visible" in headings
    assert "Hidden" not in headings


def test_html_strips_nav_footer_aside():
    html = b"""
    <html><body>
      <nav><a href="/x">Nav link</a><a href="/y">Another nav</a></nav>
      <main>
        <h1>Real Content</h1>
        <p>This is the actual content.</p>
      </main>
      <footer>Footer junk</footer>
      <aside>Sidebar junk</aside>
    </body></html>
    """
    parser = UniversalParser()
    elements = parser.parse(html, "p.html", "text/html")
    texts = " ".join(e.text for e in elements)
    assert "Real Content" in texts
    assert "actual content" in texts
    assert "Nav link" not in texts
    assert "Footer junk" not in texts
    assert "Sidebar junk" not in texts


def test_html_prefers_main_when_present():
    html = b"""
    <html><body>
      <div>Outside main</div>
      <main>
        <h1>Inside main</h1>
        <p>Body text.</p>
      </main>
    </body></html>
    """
    parser = UniversalParser()
    elements = parser.parse(html, "p.html", "text/html")
    texts = " ".join(e.text for e in elements)
    assert "Inside main" in texts
    assert "Outside main" not in texts
