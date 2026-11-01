"""
tests/test_release_readiness.py

Regression tests for the defects the release-readiness pass found, in
the shipped wheel rather than the source tree.

    * `_err_console()` had lost its `return`, so every CLI error path
      raised `AttributeError: 'NoneType' object has no attribute
      'print'` instead of printing the message it was about to print.
    * The in-process embedder indexed fastembed's generator, so the
      `local` extra failed with "'generator' object is not subscriptable".
    * A hint containing `atrium-rag[local]` was read as console markup
      and lost its brackets.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from rich.console import Console

# ---------------------------------------------------------------------------
# The error console
# ---------------------------------------------------------------------------


def test_the_error_console_returns_a_console():
    from atrium.cli import _err_console

    console = _err_console()
    assert isinstance(console, Console), (
        "the error console returned "
        f"{type(console).__name__}, so every error path crashes"
    )


def test_a_setup_failure_prints_its_message_and_its_hints(tmp_path):
    """`atrium doctor` with no .env must say so, not raise."""
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    proc = subprocess.run(
        [sys.executable, "-m", "atrium.cli", "doctor", "--env", str(tmp_path / "nope.env")],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(tmp_path),
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "AttributeError" not in combined, combined
    assert "atrium init" in combined, combined


# ---------------------------------------------------------------------------
# The in-process embedder
# ---------------------------------------------------------------------------


def test_the_local_embedder_accepts_a_generator():
    """fastembed returns a generator of arrays; it cannot be indexed."""
    from atrium.local_embedder import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5", dim=4)

    class _Model:
        def embed(self, texts):
            return (list([0.1] * 4) for _ in texts)

    embedder._model = _Model()
    vectors = embedder.embed(["one", "two"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 4


def test_the_local_embedder_reports_a_wrong_width(monkeypatch):
    from atrium.errors import AtriumSetupError
    from atrium.local_embedder import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(model="BAAI/bge-small-en-v1.5", dim=4)

    class _Model:
        def embed(self, texts):
            return (list([0.1] * 8) for _ in texts)

    embedder._model = _Model()
    with pytest.raises(AtriumSetupError) as excinfo:
        embedder.embed(["one"])
    assert "384" in str(excinfo.value) or "4" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The hints survive the console
# ---------------------------------------------------------------------------


def test_a_hint_keeps_its_brackets(capsys):
    """`atrium-rag[local]` is markup to rich, and rich eats it."""
    from atrium.cli import _setup_failed
    from atrium.errors import AtriumSetupError

    error = AtriumSetupError(
        "The local embedder could not load the model.",
        hints=('Install the extra: `pip install "atrium-rag[local]"`',),
    )
    import typer

    with pytest.raises(typer.Exit):
        _setup_failed(error)
    captured = capsys.readouterr()
    assert "atrium-rag[local]" in captured.out + captured.err, (
        "the brackets were eaten by the console: "
        + repr(captured.out + captured.err)
    )
