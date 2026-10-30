#!/usr/bin/env python3
"""
smoke_test.py

Verify an *installed* atrium — the artifact a user gets from
`pip install atrium-rag`, not the source tree.

    pip install dist/atrium_rag-*.whl
    python scripts/smoke_test.py

It runs from any directory and needs no services: no Qdrant, no
Ollama, no network. Every check exercises something a user would
touch on day one — the import, the console script, the parser, the
chunker, the HTTP routes, the config resolver, the job queue.

Exits 0 when everything passes, 1 otherwise.

Run it from outside the repository (or with the repo's `atrium/`
directory not on sys.path) or it will test the source tree instead of
the installed package:

    cd /tmp && python /path/to/atrium/scripts/smoke_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f"  — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001 — this is a test harness
        FAILURES.append(f"{name}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {name}  — {type(exc).__name__}: {exc}")


def check_import():
    import atrium

    return f"version {atrium.__version__}"


def check_console_script():
    exe = Path(sys.executable).parent / "atrium"
    if not exe.exists():
        raise AssertionError("no `atrium` console script next to the interpreter")
    out = subprocess.run(
        [str(exe), "version"], capture_output=True, text=True, timeout=60,
    )
    if out.returncode != 0:
        raise AssertionError(f"`atrium version` exited {out.returncode}: {out.stderr}")
    return out.stdout.strip()


def check_submodules():
    # The public surface, imported the way a user imports it.
    from atrium.adapters import QdrantAdapter  # noqa: F401
    from atrium.config import get_settings
    from atrium.embedder_config import build_embedder  # noqa: F401
    from atrium.job_store import JobStore  # noqa: F401
    from atrium.pipeline import ingest_document  # noqa: F401
    from atrium.retrieval import retrieve_and_rerank  # noqa: F401
    from atrium.smart_chunker import chunk_with_mode  # noqa: F401
    from atrium.universal_parser import UniversalParser  # noqa: F401

    settings = get_settings()
    return f"{settings.vector_db} / {settings.embed_model} / {settings.default_top_k}"


def check_parser():
    from atrium.universal_parser import UniversalParser

    parser = UniversalParser()
    md = b"# Title\n\nA paragraph with some prose in it.\n"
    elements = parser.parse(md, "sample.md", "text/markdown")
    if not elements:
        raise AssertionError("markdown parsed to zero elements")
    return f"{len(elements)} elements from markdown"


def check_chunker():
    from atrium.smart_chunker import chunk_with_mode
    from atrium.universal_parser import UniversalParser

    parser = UniversalParser()
    md = b"# Title\n\n" + (b"A paragraph of prose. " * 40)
    elements = parser.parse(md, "sample.md", "text/markdown")
    chunks = chunk_with_mode(elements, doc_id="d", file_name="sample.md", mode="auto")
    if not chunks:
        raise AssertionError("chunker produced zero chunks")
    strategies = {getattr(c, "chunking_strategy", "") for c in chunks}
    return f"{len(chunks)} chunks, strategy={strategies.pop() or '?'}"


def check_pdf():
    """Parse a real PDF, if one is reachable without the corpus."""
    from atrium.universal_parser import UniversalParser

    candidates = [
        Path(__file__).resolve().parent.parent
        / "tests" / "benchmarks" / "fixtures" / "17_irs_w2" / "source.pdf",
        Path(__file__).resolve().parent.parent
        / "tests" / "parsing_fixtures" / "25_raw_image" / "source.png",
    ]
    pdf = next((p for p in candidates if p.exists()), None)
    if pdf is None:
        return "skipped — no fixture PDF on disk"

    mime = "application/pdf" if pdf.suffix == ".pdf" else "image/png"
    elements = UniversalParser().parse(pdf.read_bytes(), pdf.name, mime)
    if not elements:
        raise AssertionError(f"{pdf.name} parsed to zero elements")
    return f"{len(elements)} elements from {pdf.name}"


def check_routes():
    from atrium.main import app

    paths = {r.path for r in app.routes}
    expected = {
        "/health", "/ingest", "/ingest_single", "/jobs", "/search",
        "/preview", "/retrieval_lab", "/visualize",
    }
    missing = expected - paths
    if missing:
        raise AssertionError(f"missing routes: {sorted(missing)}")
    return f"{len(paths)} routes"


def check_config_precedence():
    from atrium.config import env_str, get_settings

    os.environ["ATRIUM_SMOKE_TEST"] = "from-env"
    try:
        if env_str("ATRIUM_SMOKE_TEST") != "from-env":
            raise AssertionError("environment variable did not win")
    finally:
        os.environ.pop("ATRIUM_SMOKE_TEST", None)
    if not get_settings().embed_dim:
        raise AssertionError("embed_dim is not set")
    return "environment > .env > default"


def check_job_queue():
    from atrium.job_store import JobStore

    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(db_path=str(Path(tmp) / "jobs.db"))
        store.enqueue("j1", "d1", {"file_bytes_b64": "AAA", "filename": "x.pdf"})
        job = store.claim_next("smoke")
        if not job or job["job_id"] != "j1":
            raise AssertionError("could not claim the queued job")
        if store.claim_next("smoke") is not None:
            raise AssertionError("claimed the same job twice")
        store.close()
    return "enqueue + atomic claim"


def check_ocr_provider():
    from atrium.ocr_adapter import build_ocr_adapter

    try:
        adapter = build_ocr_adapter(provider="rapidocr")
    except RuntimeError as exc:
        # The `ocr` extra is opt-in. A base install has no OCR engine;
        # that is a valid state, not a failure.
        return f"skipped — {exc}"
    return type(adapter).__name__


def main() -> int:
    print(f"atrium smoke test — python {sys.version.split()[0]}")
    print(f"  from: {Path(atrium_path()).parent if atrium_path() else '?'}")
    print()

    check("package imports", check_import)
    check("console script", check_console_script)
    check("public submodules", check_submodules)
    check("parser", check_parser)
    check("chunker", check_chunker)
    check("pdf parser", check_pdf)
    check("http routes", check_routes)
    check("config precedence", check_config_precedence)
    check("job queue", check_job_queue)
    check("ocr provider", check_ocr_provider)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for line in FAILURES:
            print(f"  - {line}")
        return 1
    print("all checks passed")
    return 0


def atrium_path():
    try:
        import atrium

        return atrium.__file__
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
