"""
tests/conftest.py

M0-3 (EMBED_MODEL pinning, 2026-10-10): enforce a single pinned
embedder for every pytest run so a developer can't silently re-baseline
the benchmark by setting EMBED_MODEL in their shell.

The pinned value matches `atrium/embedder_config.py` — the production default
embedder that ships with the API. Any other value is treated as a
configuration error: pytest fails fast with a clear message rather
than running the suite against a different embedding model and
producing numbers that don't match the published baseline.

How to override (intentional, not accidental):

    EMBED_MODEL=bge-large-en-v1.5 pytest
    EMBED_MODEL_ALLOW_OVERRIDE=1 EMBED_MODEL=mixedbread-ai/mxbai-embed-large pytest
    # ^ second form sets EMBED_MODEL_ALLOW_OVERRIDE=1 to skip the pin

Without EMBED_MODEL set, the conftest sets it to the pinned value so
downstream tests that read `os.environ['EMBED_MODEL']` (or
`os.getenv('EMBED_MODEL')`) get the production default.

Hermetic environment
--------------------
`atrium` loads `.env` into the process environment when it is imported.
That is the fix that makes `uvicorn atrium.main:app` honour `API_KEY`
instead of serving an unauthenticated API — but it also means a
developer's local `.env` can reach the test suite.

That is a hazard, not a convenience: a local `.env` typically holds
Docker-only host names (`qdrant`, `host.docker.internal`) that resolve
nowhere on a laptop. Inheriting it makes the e2e tier skip itself
("stack not reachable") and makes the run irreproducible on another
machine.

So the suite opts out of `.env`: `RAG_INGEST_ENV` points at a path
that does not exist. Variables exported in the developer's *shell* still
apply — the shell sits at the top of the precedence chain, and a
deliberate override should beat a file.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Snapshot the shell environment before anything can import `atrium`.
# Keys that appear later and are not in this snapshot came from a `.env`
# file and are stripped in `pytest_configure`.
_SHELL_ENV = dict(os.environ)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Point the `.env` loader at a path that does not exist. This is the
# switch that stops a local `.env` from reaching the suite.
os.environ["RAG_INGEST_ENV"] = str(_REPO_ROOT / ".env.pytest-absent")


def _strip_dotenv_leakage() -> None:
    """Remove settings that arrived from a `.env` file.

    `atrium` merges `.env` into `os.environ` on import. Anything that
    was not in the shell snapshot came from a file, so it is removed
    here — the suite runs on the built-in defaults unless the operator
    exported something deliberately.
    """
    from atrium.config import DEFAULTS

    for key in list(DEFAULTS):
        if key not in _SHELL_ENV:
            os.environ.pop(key, None)


# M0-3 (2026-10-10): pinned production embedder. Must match
# `atrium/embedder_config.py` (EMBED_MODEL default) so the test suite and the
# running server use the same model — otherwise baseline numbers
# diverge silently between CI and the API a user deploys.
#
# The Ollama tag is `bge-large` (Ollama aliases the underlying
# bge-large-en-v1.5 model under that tag); prior benchmark runs
# (2026-10-08 baseline, 2026-10-09 embedder+rerank swap) used
# `bge-large` / `bge-large:latest`. `bge-large-en-v1.5` is the
# underlying HF model name and won't resolve in Ollama's registry
# unless someone explicitly tags it that way.
PINNED_EMBED_MODEL = "bge-large"


def _enforce_embed_model_pin() -> None:
    """Session-start hook: assert EMBED_MODEL is pinned.

    Raises pytest.UsageError with a clear actionable message if the
    pin is violated. Sets the env var to the pinned value if it's
    unset so downstream code that reads os.environ finds the same
    value the production server uses.
    """
    override_allowed = os.getenv("EMBED_MODEL_ALLOW_OVERRIDE", "") == "1"
    current = os.getenv("EMBED_MODEL")

    if current is None:
        # Default to the pinned value so downstream os.getenv() calls
        # get the production embedder without a shell export.
        os.environ["EMBED_MODEL"] = PINNED_EMBED_MODEL
        return

    if current == PINNED_EMBED_MODEL:
        return

    if override_allowed:
        # User explicitly opted out. Don't enforce.
        return

    import pytest

    raise pytest.UsageError(
        f"EMBED_MODEL is pinned to {PINNED_EMBED_MODEL!r} "
        f"(matches atrium/embedder_config.py) so the test suite and the "
        f"production server use the same embedder. Got: {current!r}.\n\n"
        f"To run with the pinned value, leave EMBED_MODEL unset or:\n"
        f"    EMBED_MODEL={PINNED_EMBED_MODEL} pytest\n\n"
        f"To intentionally override (you accept the baseline will "
        f"drift), set:\n"
        f"    EMBED_MODEL_ALLOW_OVERRIDE=1 EMBED_MODEL={current} pytest"
    )


# pytest_configure runs once per pytest session, before any test
# collection. This is the right place to enforce an env-var pin.
def pytest_configure(config):
    # Import atrium *after* the sentinel above is in place, so the
    # package's own `.env` load finds nothing.
    import atrium  # noqa: F401

    _strip_dotenv_leakage()
    _enforce_embed_model_pin()


# Autouse across the whole suite: PgVectorAdapter.upsert calls
# `register_vector(conn)` from pgvector.psycopg before using the
# cursor. With a MagicMock conn (the standard test setup in
# test_adapters_realtime / test_ingest_to_storage_v1),
# `register_vector` raises TypeError ("expected Connection or
# AsyncConnection, got MagicMock"), and tenacity's retry decorator
# on upsert retries 3× then bubbles RetryError.
#
# Patching `app.adapters.register_vector` for every test makes the
# mock pool/conn safe to use in any test file without each one
# having to remember to stub the adapter-specific call. Real
# integration tests (not currently in this suite) would skip this
# fixture via `@pytest.mark.no_register_vector_noop` if added.
@pytest.fixture(autouse=True)
def _no_op_register_vector():
    with patch("atrium.adapters.register_vector", lambda conn: None):
        yield