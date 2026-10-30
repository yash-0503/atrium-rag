"""
test_embed_model_pin.py

M0-3 (EMBED_MODEL pinning, 2026-10-10) — proves the conftest hook
`pytest_configure` enforces the embedder pin. We exercise the pin
logic by calling `_enforce_embed_model_pin` directly with various
env-var states, which keeps these tests independent of pytest's
collection order.

The pin guarantees a single embedder (`bge-large-en-v1.5`) is the
test-suite baseline. Without it, a developer setting
`EMBED_MODEL=...` in their shell could produce a green test run
against a different model and the published baseline numbers
(`benchmark/RESULTS.md`) would silently diverge from what CI sees.
"""

from __future__ import annotations

import importlib
import os

import pytest


def _reload_conftest():
    """Reload tests.conftest so the module-level PINNED_EMBED_MODEL
    and the `_enforce_embed_model_pin` function are re-evaluated
    against the current environment. Otherwise an earlier test that
    mutated `os.environ` would be cached at import time."""
    import tests.conftest as _cf
    return importlib.reload(_cf)


@pytest.fixture(autouse=True)
def _clean_embed_env(monkeypatch):
    """Snapshot EMBED_MODEL + EMBED_MODEL_ALLOW_OVERRIDE around each
    test so a mutation in one test doesn't leak into the next."""
    saved = {
        k: os.environ.get(k)
        for k in ("EMBED_MODEL", "EMBED_MODEL_ALLOW_OVERRIDE")
    }
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("EMBED_MODEL_ALLOW_OVERRIDE", raising=False)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_pin_default_is_bge_large():
    """The pinned value must match `app/main.py:321`."""
    import tests.conftest as _cf
    assert _cf.PINNED_EMBED_MODEL == "bge-large", (
        "PINNED_EMBED_MODEL drifted from the production default in "
        "app/main.py:321. Update both together or the test suite "
        "will diverge from the running server."
    )


def test_unset_env_sets_pinned_value():
    """When EMBED_MODEL is unset, the hook sets it to the pinned
    value so downstream code that reads `os.environ` finds the
    production embedder without an explicit shell export."""
    import os
    os.environ.pop("EMBED_MODEL", None)
    _cf = _reload_conftest()
    _cf._enforce_embed_model_pin()
    assert os.environ.get("EMBED_MODEL") == "bge-large"


def test_correct_pinned_value_passes():
    """Setting EMBED_MODEL to the pinned value is a no-op (no
    failure, no override of the env)."""
    import os
    os.environ["EMBED_MODEL"] = "bge-large"
    _cf = _reload_conftest()
    _cf._enforce_embed_model_pin()  # should not raise
    assert os.environ["EMBED_MODEL"] == "bge-large"


def test_wrong_embed_model_fails_fast():
    """The whole point of the pin: a different embedder must
    fail with a clear actionable error, not silently re-baseline."""
    import os
    os.environ["EMBED_MODEL"] = "nomic-embed-text:v1.5"
    _cf = _reload_conftest()
    with pytest.raises(pytest.UsageError) as excinfo:
        _cf._enforce_embed_model_pin()
    msg = str(excinfo.value)
    assert "EMBED_MODEL is pinned" in msg
    assert "bge-large" in msg
    assert "nomic-embed-text:v1.5" in msg


def test_override_allowed_with_explicit_opt_in():
    """`EMBED_MODEL_ALLOW_OVERRIDE=1` lets the user run with a
    different embedder. The hook must honor that explicitly — it's
    the escape hatch for one-off experiments (regression on a
    different model, etc.)."""
    import os
    os.environ["EMBED_MODEL"] = "mixedbread-ai/mxbai-embed-large"
    os.environ["EMBED_MODEL_ALLOW_OVERRIDE"] = "1"
    _cf = _reload_conftest()
    _cf._enforce_embed_model_pin()  # must not raise
    assert os.environ["EMBED_MODEL"] == "mixedbread-ai/mxbai-embed-large"