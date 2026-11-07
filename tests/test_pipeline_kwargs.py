"""
tests/test_pipeline_kwargs.py

run_ingestion must accept chunking_mode and chunking_config as keyword
arguments and forward them unchanged to chunk_with_mode.

Two callers rely on this signature:

  1. app/main.py:218-221 (asyncio.to_thread call from /ingest and
     /ingest). Invoked with positional args only — no
     chunking kwargs. The historical behavior (mode='auto',
     config=None) must be preserved.

  2. app/worker.py:212-216 (the standalone drain loop). Passes
     chunking_mode and chunking_config from the job payload. Both
     must arrive at chunk_with_mode unchanged.

Before the fix this test file errors with TypeError on every test
that supplies chunking kwargs; the default-only test passes (it
exercises the existing hardcoded 'auto' / None path).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from atrium.pipeline import run_ingestion


def _stub_parser() -> MagicMock:
    p = MagicMock()
    # parser.parse must return a non-empty list or run_ingestion
    # raises ValueError("No content parsed.") before reaching
    # chunk_with_mode.
    p.parse.return_value = [{"type": "paragraph", "text": "x", "page": 1}]
    return p


def _stub_embedder() -> MagicMock:
    e = MagicMock()
    e.embed.return_value = []  # empty → no per-chunk embedding work
    return e


def _stub_db() -> MagicMock:
    d = MagicMock()
    d.upsert.return_value = None
    d.delete_stale_chunks.return_value = None
    return d


def _stub_job_store() -> MagicMock:
    j = MagicMock()
    j.update.return_value = None
    return j


def _invoke(
    *,
    chunking_mode: Optional[str] = None,
    chunking_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Invoke run_ingestion with stubs; capture chunk_with_mode args.

    Returns the dict of forwarded kwargs (mode, config, doc_id,
    file_name). Raises TypeError if run_ingestion's signature
    rejects the supplied kwargs — which is the regression this test
    guards against.
    """
    captured: Dict[str, Any] = {}

    def fake_chunk_with_mode(elements, doc_id, file_name, mode, config):
        captured["doc_id"] = doc_id
        captured["file_name"] = file_name
        captured["mode"] = mode
        captured["config"] = config
        return []  # empty list → run_ingestion completes with 0 chunks

    with patch("atrium.pipeline.chunk_with_mode", side_effect=fake_chunk_with_mode):
        kwargs: Dict[str, Any] = {}
        if chunking_mode is not None:
            kwargs["chunking_mode"] = chunking_mode
        if chunking_config is not None:
            kwargs["chunking_config"] = chunking_config
        run_ingestion(
            job_id="j",
            doc_id="d",
            file_bytes=b"x",
            filename="f.pdf",
            mime_type="application/pdf",
            parser=_stub_parser(),
            embedder=_stub_embedder(),
            db=_stub_db(),
            job_store=_stub_job_store(),
            **kwargs,
        )
    return captured


# ---------------------------------------------------------------------------
# Forwarding behavior — the explicit path worker.py:212-216 takes.
# ---------------------------------------------------------------------------


def test_run_ingestion_forwards_explicit_chunking_settings():
    """Caller passes chunking_mode='precise' and a custom config dict;
    both must arrive at chunk_with_mode unchanged."""
    captured = _invoke(
        chunking_mode="precise",
        chunking_config={"target_chunk_chars": 500},
    )
    assert captured["mode"] == "precise"
    assert captured["config"] == {"target_chunk_chars": 500}


def test_run_ingestion_forwards_only_mode_defaulting_config():
    """Caller can pass chunking_mode without chunking_config. config
    must default to None, not be undefined / KeyError."""
    captured = _invoke(chunking_mode="balanced")
    assert captured["mode"] == "balanced"
    assert captured["config"] is None


def test_run_ingestion_forwards_only_config_defaulting_mode():
    """Caller can pass chunking_config without chunking_mode. mode
    must default to 'auto'."""
    captured = _invoke(chunking_config={"max_chunk_chars": 999})
    assert captured["mode"] == "auto"
    assert captured["config"] == {"max_chunk_chars": 999}


def test_run_ingestion_forwards_context_rich_explicit_settings():
    """context_rich mode (Strategy E, parent-child) is the only mode
    that materially differs from balanced/precise. Verify it's
    forwarded by name."""
    captured = _invoke(chunking_mode="context_rich")
    assert captured["mode"] == "context_rich"
    assert captured["config"] is None


# ---------------------------------------------------------------------------
# Default behavior — the legacy /ingest path (app/main.py:218-221) takes.
# ---------------------------------------------------------------------------


def test_run_ingestion_defaults_to_auto_and_none_when_no_kwargs():
    """Existing /ingest callers invoke run_ingestion
    without any chunking kwargs. run_ingestion must still use the
    historical defaults: mode='auto', config=None.

    This test passes both before and after the fix; it guards against
    an accidental default change in a future edit.
    """
    captured = _invoke()
    assert captured["mode"] == "auto"
    assert captured["config"] is None


def test_run_ingestion_forwards_doc_id_and_filename_alongside_settings():
    """Sanity check: the doc_id and file_name that chunk_with_mode
    receives must match the values run_ingestion was called with,
    even when chunking settings are explicitly overridden."""
    captured = _invoke(chunking_mode="precise", chunking_config={})
    assert captured["doc_id"] == "d"
    assert captured["file_name"] == "f.pdf"