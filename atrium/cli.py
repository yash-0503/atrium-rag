"""
atrium/cli.py

Configure and operate the Universal RAG ingestion server.

The CLI is the primary interface — HTTP is secondary. A developer
should be able to go from `pip install atrium` to "my documents are
searchable" without writing a single HTTP request.

Commands:
  init        First-run wizard. Writes `.env` with chosen settings.
  configure   Edit a single config key non-interactively. Idempotent.
  validate    Confirm `.env` is well-formed and consistent with the
              installed runtime (extras installed, defaults parse).
  doctor      Sanity-check the running stack. Probes vector DB and
              embedder reachability unless `--skip-remote` is set.
              Exits non-zero when something is actually broken.
  ingest      Parse, chunk, embed and store documents.
  search      Query the ingested corpus.
  reingest    Re-chunk a document with a different strategy.
  demo        Run a self-contained example end to end.
  serve       Boot uvicorn with the FastAPI service.
  version     Print the package version.

Every command reads `.env` fresh. `atrium.config` owns the precedence
chain (real environment > `.env` > built-in default) and the defaults
table, so the wizard, the runtime and `doctor` cannot disagree about
what a setting means.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import typer

from atrium import __version__
from atrium.config import (
    DEFAULTS,
    env_path,
    load_env_file,
    read_env_file,
)

app = typer.Typer(
    name="atrium",
    help="Configure and operate the Atrium ingestion server.",
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Presentation
#
# `rich` ships with typer, so it is always present; declaring it
# directly is honest about the dependency rather than relying on a
# transitive one.
# ---------------------------------------------------------------------------

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

_THEME = Theme(
    {
        "ok": "bold green",
        "warn": "bold yellow",
        "bad": "bold red",
        "key": "bold cyan",
        "dim": "dim",
        "accent": "bold magenta",
    }
)

console = Console(theme=_THEME)


def _err_console() -> Console:
    return Console(theme=_THEME, stderr=True)


def _quiet_logs(verbose: bool = False) -> None:
    """Keep CLI output readable.

    The atrium library logs at INFO — useful on a server, noise in a
    terminal. The CLI shows its own progress instead; `--verbose`
    turns the library logs back on.
    """
    if verbose:
        return
    import logging

    for name in ("httpx", "httpcore", "ingest", "atrium"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Schema -- one entry per env key the wizard manages.
#
# There is deliberately no "default" key here. Defaults live in
# `atrium.config.DEFAULTS`, which is the single source of truth shared
# with the runtime and `doctor`. A second copy of a default is how
# `EMBED_BASE_URL` ended up with four different values.
#
# Type drives parsing + validation. secret=True masks the value in
# `configure --show`.
# ---------------------------------------------------------------------------

CONFIG_SCHEMA: List[Dict[str, Any]] = [
    {
        "key": "VECTOR_DB",
        "type": "choice",
        "choices": ["sqlite", "qdrant", "pgvector"],
        "prompt": "Vector database backend",
        "help": "sqlite (a local file, nothing to run) or qdrant or pgvector",
        "secret": False,
    },
    {
        "key": "QDRANT_HOST",
        "type": "str",
        "prompt": "Qdrant host",
        "help": "Hostname for the Qdrant service",
        "secret": False,
        "depends_on": ("VECTOR_DB", "qdrant"),
    },
    {
        "key": "QDRANT_PORT",
        "type": "int",
        "prompt": "Qdrant port",
        "help": "Qdrant HTTP port",
        "secret": False,
        "depends_on": ("VECTOR_DB", "qdrant"),
    },
    {
        "key": "LOCAL_DB_PATH",
        "type": "str",
        "prompt": "Where the local store keeps its file",
        "help": "the SQLite file used when VECTOR_DB=sqlite",
        "secret": False,
        "depends_on": ("VECTOR_DB", "sqlite"),
    },
    {
        "key": "PG_CONN",
        "type": "str",
        "prompt": "Postgres connection string",
        "help": "DSN of the pgvector database",
        "secret": True,  # contains password
        "depends_on": ("VECTOR_DB", "pgvector"),
    },
    {
        "key": "EMBED_BASE_URL",
        "type": "str",
        "prompt": "Embedding provider URL (OpenAI-compatible)",
        "help": "Base URL for the /v1/embeddings endpoint, including /v1",
        "secret": False,
    },
    {
        "key": "EMBED_API_KEY",
        "type": "str",
        "prompt": "Embedding provider API key",
        "help": "blank for local servers; required by hosted providers",
        "secret": True,
    },
    {
        "key": "EMBED_PROVIDER",
        "type": "choice",
        "choices": ["http", "fastembed"],
        "prompt": "Embedding provider",
        "help": "http (any OpenAI-compatible endpoint) or fastembed (in-process)",
        "secret": False,
    },
    {
        "key": "EMBED_MODEL",
        "type": "str",
        "prompt": "Embedding model name",
        "help": "Model tag (Ollama) or model id (OpenAI)",
        "secret": False,
    },
    {
        "key": "EMBED_DIM",
        "type": "int",
        "prompt": "Embedding vector dimension",
        "help": "Must match the model's actual output dim",
        "secret": False,
    },
    {
        "key": "OCR_ENABLED",
        "type": "bool",
        "prompt": "Enable OCR fallback?",
        "help": "Set to 1 to OCR scanned pages and image-only docs",
        "secret": False,
    },
    {
        "key": "OCR_PROVIDER",
        "type": "choice",
        "choices": ["tesseract", "rapidocr"],
        "prompt": "OCR provider (only relevant if OCR_ENABLED=1)",
        "help": "tesseract (needs binary) or rapidocr (pip-only)",
        "secret": False,
    },
    {
        "key": "RERANK_ENABLED",
        "type": "bool",
        "prompt": "Enable reranking?",
        "help": "Set to 1 to add a cross-encoder rerank stage",
        "secret": False,
    },
    {
        "key": "RERANK_PROVIDER",
        "type": "choice",
        "choices": ["bge", "flashrank", "cohere"],
        "prompt": "Rerank provider",
        "help": "bge (default), flashrank (lightweight), or cohere (hosted)",
        "secret": False,
    },
    {
        "key": "DEFAULT_TOP_K",
        "type": "int",
        "prompt": "Default top-k for search",
        "help": "Higher = more candidates but slower",
        "secret": False,
    },
    {
        "key": "MAX_UPLOAD_BYTES",
        "type": "int",
        "prompt": "Max upload size in bytes (default 100 MB)",
        "help": "Uploads larger than this are rejected with 413",
        "secret": False,
    },
    {
        "key": "API_KEY",
        "type": "str",
        "prompt": "API key for bearer auth (blank = no auth, dev only)",
        "help": "Require Authorization: Bearer <key> on all routes except /health",
        "secret": True,
    },
]

CONFIG_KEYS: Dict[str, Dict[str, Any]] = {entry["key"]: entry for entry in CONFIG_SCHEMA}

# Env keys the schema doesn't manage but the runtime expects. Kept
# out of the interactive wizard to avoid asking about them; the
# schema validator only warns (doesn't fail) when these are absent.
#
# The value is the type, not a bare set: `atrium configure` coerces
# whatever it is given, so it needs to know whether a key is an int,
# a bool or a string.
_MANAGED_BY_RUNTIME: Dict[str, str] = {
    "JOB_DB_PATH": "str",
    "JOB_STALE_SECONDS": "int",
    "INGEST_CONCURRENCY": "int",
    "WORKER_POLL_BASE": "float",
    "WORKER_POLL_MAX": "float",
    "WORKER_JOB_TIMEOUT": "float",
    "OCR_LANG": "str",
    "TESSERACT_CMD": "str",
    "RERANK_MODEL": "str",
    "RERANK_DEVICE": "str",
    "RERANK_BACKEND": "str",
    "RERANK_MAX_LENGTH": "int",
    "FLASHRANK_CACHE_DIR": "str",
    "COHERE_API_KEY": "str",
    "LOG_LEVEL": "str",
    "CORS_ALLOW_ORIGINS": "str",
    "ALLOWED_MIME_TYPES": "str",
    "ALLOW_CHUNKED_UPLOADS": "bool",
    "VISION_ENABLED": "bool",
    "VISION_MODEL": "str",
    "VISION_TIMEOUT_SECONDS": "float",
    "DENSE_CONFIDENT_GAP": "float",
    "DENSE_MIN_SCORE": "float",
    "RETRIEVAL_LAB_MAX_SESSIONS": "int",
}

# Keys whose values must never be echoed back to the terminal.
_SECRET_RUNTIME_KEYS = {"COHERE_API_KEY"}


def _runtime_entries() -> Dict[str, Dict[str, Any]]:
    """Schema-shaped entries for the runtime-managed keys.

    `atrium init` writes these too — with their current or default
    value — so a freshly initialised `.env` shows every knob an
    operator can turn. Hiding 24 of them behind "see the docs" is how
    a config surface feels like a trap.
    """
    return {
        key: {
            "key": key,
            "type": kind,
            "secret": key in _SECRET_RUNTIME_KEYS,
        }
        for key, kind in _MANAGED_BY_RUNTIME.items()
    }


# Every key `atrium init` and `atrium configure` know about.
_ALL_KEYS: Dict[str, Dict[str, Any]] = {
    **CONFIG_KEYS,
    **_runtime_entries(),
}


def _default_for(key: str) -> str:
    """The built-in default for a config key."""
    return DEFAULTS.get(key, "")


# ---------------------------------------------------------------------------
# .env writing
# ---------------------------------------------------------------------------


def _split_value_comment(raw: str) -> Tuple[str, str]:
    """Split a raw .env value into (value, trailing comment).

    A `#` only starts a comment when it is preceded by whitespace and
    is not inside quotes. Returns (value, " # comment") or (value, "").
    """
    in_single = in_double = False
    for i, ch in enumerate(raw):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if i > 0 and raw[i - 1] in " \t":
                return raw[:i].rstrip(), "  " + raw[i:].strip()
    return raw, ""


def _format_value(entry: Dict[str, Any], value: str) -> str:
    """Render a value the way it should appear in the file."""
    if entry.get("secret") and value:
        # Keep double-quoted so re-parsing handles embedded
        # whitespace; not a hardening measure, just a round-trip
        # guarantee for tokens that contain spaces.
        return f'"{value}"'
    return value


def _write_env(path: Path, values: Dict[str, str]) -> None:
    """Write a `.env` file, preserving comments and unmanaged keys.

    The previous implementation rebuilt the file from scratch, which
    threw away every comment the operator had written and moved
    runtime-managed keys to the bottom. This version edits the file in
    place: comment lines and blank lines survive, managed keys are
    rewritten in place, and new keys are appended in one block.
    """
    existing_lines: List[str] = []
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    managed = set(_ALL_KEYS)
    written: set = set()
    out: List[str] = []

    for raw in existing_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            out.append(raw)
            continue
        if "=" not in raw:
            out.append(raw)
            continue
        key, _, raw_value = raw.partition("=")
        key = key.strip()
        if key not in managed:
            out.append(raw)
            continue
        if key not in values:
            # Managed key the caller did not supply — leave the
            # operator's line exactly as it is.
            out.append(raw)
            written.add(key)
            continue
        entry = _ALL_KEYS[key]
        current, comment = _split_value_comment(raw_value.strip())
        del current
        out.append(f"{key}={_format_value(entry, values[key])}{comment}")
        written.add(key)

    # Only keys the caller actually supplied get appended. `atrium init`
    # supplies every key (so the file comes out complete); `atrium
    # configure` supplies one (so it doesn't pad the file with blanks).
    missing = [k for k in _ALL_KEYS if k not in written and k in values]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# --- added by atrium ---")
        for key in missing:
            entry = _ALL_KEYS[key]
            out.append(f"{key}={_format_value(entry, values[key])}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _coerce(entry: Dict[str, Any], raw: str) -> Tuple[bool, Any, str]:
    """Coerce a raw string to the schema-typed value. Returns
    (ok, parsed, error_message)."""
    t = entry["type"]
    try:
        if t == "int":
            return True, int(raw), ""
        if t == "float":
            value = float(raw)
            # "240" reads better in a .env than "240.0".
            return True, (int(value) if value.is_integer() else value), ""
        if t == "bool":
            if raw.lower() in ("1", "true", "yes", "on"):
                return True, "1", ""
            if raw.lower() in ("0", "false", "no", "off", ""):
                return True, "0", ""
            return False, raw, f"expected 0/1, true/false; got {raw!r}"
        if t == "choice":
            if raw in entry["choices"]:
                return True, raw, ""
            return False, raw, f"expected one of {entry['choices']}; got {raw!r}"
        # Default: str
        return True, raw, ""
    except ValueError:
        return False, raw, f"expected {t}; got {raw!r}"


def _effective_values(values: Dict[str, str]) -> Dict[str, str]:
    """The configuration the runtime will actually use.

    The `.env` file's values on top of the built-in defaults, with the
    same precedence the runtime applies: an *empty* value in the file
    counts as unset (see `atrium.config.env_str`), so it does not
    shadow the default.

    Validating this merged view is what stops `atrium doctor` from
    reporting "missing key MAX_UPLOAD_BYTES" for a key that has a
    perfectly good default and simply isn't written down.
    """
    merged = {key: _default_for(key) for key in DEFAULTS}
    for key, value in values.items():
        if value != "":
            merged[key] = value
    return merged


def _validate(values: Dict[str, str]) -> List[str]:
    """Run schema validation + cross-key consistency checks. Returns
    a list of human-readable problems; empty list means OK."""
    problems: List[str] = []
    effective = _effective_values(values)
    for entry in CONFIG_SCHEMA:
        k = entry["key"]
        ok, _, err = _coerce(entry, effective.get(k, ""))
        if not ok:
            problems.append(f"{k}: {err}")

    # Cross-key: EMBED_DIM must be positive.
    if "EMBED_DIM" in effective:
        try:
            dim = int(effective["EMBED_DIM"])
            if dim <= 0:
                problems.append("EMBED_DIM must be > 0")
        except ValueError:
            pass  # already flagged above

    # Cross-key: OCR_PROVIDER=rapidocr requires rapidocr extra.
    if effective.get("OCR_ENABLED") == "1" and effective.get("OCR_PROVIDER") == "rapidocr":
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError:
            problems.append(
                "OCR_PROVIDER=rapidocr requires `pip install atrium-rag[ocr]`",
            )

    # Cross-key: RERANK_ENABLED=1 needs the matching extra.
    if effective.get("RERANK_ENABLED") == "1":
        provider = effective.get("RERANK_PROVIDER", "bge")
        if provider == "bge":
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                problems.append(
                    "RERANK_PROVIDER=bge requires `pip install atrium-rag[rerank]`",
                )
        elif provider == "flashrank":
            try:
                import flashrank  # noqa: F401
            except ImportError:
                problems.append(
                    "RERANK_PROVIDER=flashrank requires `pip install atrium-rag[flashrank]`",
                )
        elif provider == "cohere":
            if not effective.get("COHERE_API_KEY"):
                problems.append(
                    "RERANK_PROVIDER=cohere requires COHERE_API_KEY in the environment",
                )

    return problems


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def _probe_embedder(values: Dict[str, str]) -> Tuple[bool, str]:
    """Is the configured embedding provider there?

    Two providers: an OpenAI-compatible HTTP endpoint, or the
    in-process embedder from the `local` extra. Returns (ok, detail).
    """
    provider = (values.get("EMBED_PROVIDER", "http") or "http").strip().lower()
    if provider in ("fastembed", "local", "inprocess", "in-process"):
        try:
            from atrium.local_embedder import FastEmbedEmbedder

            ok, detail = FastEmbedEmbedder(
                model=values.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
            ).probe()
            return ok, detail
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    base = values.get("EMBED_BASE_URL", "")
    if not base:
        return False, "EMBED_BASE_URL is empty"
    url = base.rstrip("/").replace("/v1", "") + "/v1/models"
    headers = {}
    key = values.get("EMBED_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        import httpx

        r = httpx.get(url, timeout=3.0, headers=headers)
        if r.status_code == 200:
            return True, url
        return False, f"{url} -> HTTP {r.status_code}"
    except Exception as exc:
        return False, f"{url} -> {type(exc).__name__}: {exc}"


def _probe_vector_db(values: Dict[str, str]) -> Tuple[bool, str]:
    """Cheap probe for the configured vector DB backend. Returns
    (ok, detail)."""
    backend = values.get("VECTOR_DB", "sqlite")
    try:
        if backend in ("sqlite", "local", "file"):
            from atrium.local_store import SqliteAdapter

            path = values.get("LOCAL_DB_PATH", "atrium_local.db")
            SqliteAdapter(path=path).health()
            return True, f"local file {path}"
        if backend == "qdrant":
            from qdrant_client import QdrantClient

            client = QdrantClient(
                host=values.get("QDRANT_HOST", "localhost"),
                port=int(values.get("QDRANT_PORT", "6333")),
            )
            client.get_collections()
            return True, f"{values.get('QDRANT_HOST', 'localhost')}:{values.get('QDRANT_PORT', '6333')}"
        if backend == "pgvector":
            import psycopg

            with psycopg.connect(values.get("PG_CONN", ""), connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            return True, "postgres"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return False, f"unknown backend {backend!r}"


def _probe_embedder_model(values: Dict[str, str]) -> Tuple[bool, str]:
    """Check that the configured embedding model is actually available.

    A wrong model name is the single most common misconfiguration:
    `bge-large-en-v1.5` (the HuggingFace repo name) does not resolve
    in Ollama's registry, and the error only shows up on the first
    ingest, minutes later.
    """
    provider = (values.get("EMBED_PROVIDER", "http") or "http").strip().lower()
    model = values.get("EMBED_MODEL", "")
    if provider in ("fastembed", "local", "inprocess", "in-process"):
        from atrium.local_embedder import SUGGESTED_MODELS

        if not model:
            return False, "EMBED_MODEL is empty"
        if model in SUGGESTED_MODELS:
            return True, f"{model} ({SUGGESTED_MODELS[model]}-d)"
        try:
            from atrium.local_embedder import FastEmbedEmbedder

            ok, detail = FastEmbedEmbedder(model=model).probe()
            return ok, detail
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
    base = values.get("EMBED_BASE_URL", "")
    if not base or not model:
        return False, "EMBED_BASE_URL or EMBED_MODEL is empty"
    url = base.rstrip("/").replace("/v1", "") + "/v1/models"
    headers = {}
    key = values.get("EMBED_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        import httpx

        r = httpx.get(url, timeout=3.0, headers=headers)
        if r.status_code != 200:
            return False, f"{url} -> HTTP {r.status_code}"
        # The OpenAI-compatible shape uses `id`; Ollama's native
        # /api/tags uses `name`. Accept either.
        names: set = set()
        for entry in r.json().get("data", []):
            for field in ("id", "name"):
                value = entry.get(field)
                if value:
                    names.add(value)
        if not names:
            return False, f"{url} returned no model list"
        # Ollama reports "bge-large:latest"; accept a bare-tag match too.
        hit = any(
            n == model or n.split(":")[0] == model.split(":")[0] for n in names
        )
        if hit:
            return True, model
        return False, f"model {model!r} not pulled (available: {sorted(names)})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    env: Optional[Path] = typer.Option(
        None,
        "--env",
        help="Where to write the .env file (default: ./.env or $RAG_INGEST_ENV)",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Skip prompts: detect what is available and write the best "
             "configuration. Useful for CI.",
    ),
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        help="Answer yes to every setup action (pull a model, start a "
             "container, install an extra).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing .env without asking.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Print the result as JSON.",
    ),
) -> None:
    """Set Atrium up on this machine.

    Detects what is installed and running, proposes a vector store and
    an embedding provider, offers to fix what it can — each action
    behind its own yes/no — and shows you the result before anything is
    written.

    Existing settings are kept: re-running never silently drops what
    you configured.
    """
    from atrium.setup import run_wizard

    target = env_path(env)
    existing = read_env_file(target) if target.exists() else {}

    if target.exists() and non_interactive and not force:
        _err_console().print(
            f"[bad]refusing to overwrite {target}[/] (use --force)",
        )
        raise typer.Exit(code=2)
    if target.exists() and not force:
        if not typer.confirm(
            f"{target} already exists — review and update it?", default=True
        ):
            raise typer.Abort()

    code = run_wizard(
        target,
        defaults=DEFAULTS,
        existing=existing,
        non_interactive=non_interactive,
        assume_yes=assume_yes,
        force=force,
        as_json=as_json,
        console=console,
        err_console=_err_console(),
    )
    if code:
        raise typer.Exit(code=code)


@app.command()
def configure(
    key: str = typer.Option(..., "--key", "-k", help="Config key, e.g. EMBED_MODEL"),
    value: str = typer.Option(..., "--value", "-v", help="Value to set"),
    env: Optional[Path] = typer.Option(
        None, "--env", help="Path to .env (default ./.env)",
    ),
) -> None:
    """Set a single config key. Idempotent.

    Works for every key atrium reads — the ones the wizard asks about
    and the runtime-managed ones (CORS_ALLOW_ORIGINS, VISION_*,
    WORKER_*, …) alike.
    """
    target = env_path(env)
    if key not in _ALL_KEYS:
        _err_console().print(
            f"[bad]unknown key {key!r}[/]; valid keys: {', '.join(sorted(_ALL_KEYS))}",
        )
        raise typer.Exit(code=2)
    entry = _ALL_KEYS[key]
    ok, coerced, err = _coerce(entry, value)
    if not ok:
        _err_console().print(f"[bad]invalid value for {key}: {err}[/]")
        raise typer.Exit(code=2)
    values = read_env_file(target)
    values[key] = str(coerced)
    _write_env(target, values)
    shown = "***" if entry.get("secret") and coerced else coerced
    console.print(f"[ok]set[/] {key}={shown}")


@app.command()
def validate(
    env: Optional[Path] = typer.Option(
        None, "--env", help="Path to .env (default ./.env)",
    ),
) -> None:
    """Validate the .env file. Exits non-zero on problems."""
    target = env_path(env)
    if not target.exists():
        _err_console().print(f"[bad]{target} does not exist[/]; run `atrium init` first.")
        raise typer.Exit(code=1)
    values = read_env_file(target)
    problems = _validate(values)
    if problems:
        _err_console().print("[bad]Configuration is invalid:[/]")
        for p in problems:
            _err_console().print(f"  [bad]-[/] {p}")
        raise typer.Exit(code=1)
    console.print(f"[ok]{target} looks good.[/]")


@app.command()
def doctor(
    env: Optional[Path] = typer.Option(
        None, "--env", help="Path to .env (default ./.env)",
    ),
    skip_remote: bool = typer.Option(
        False,
        "--skip-remote",
        help="Don't probe the vector DB or embedder (faster, useful in CI).",
    ),
) -> None:
    """Sanity-check the installed stack.

    Reports vector DB health, embedder reachability, the configured
    model, and any permissive security posture. Exits non-zero when
    something is actually broken — a dead stack is not "fine".
    """
    target = env_path(env)
    if not target.exists():
        _err_console().print(
            f"[bad]no .env found at {target}[/]; run `atrium init` first.",
        )
        raise typer.Exit(code=1)

    # Load the file the operator pointed at so the effective settings
    # below match what the server would use. The shell still wins.
    load_env_file(target)
    values = read_env_file(target)
    effective = _effective_values(values)
    problems = _validate(values)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("k", style="key", no_wrap=True)
    table.add_column("v")

    table.add_row("atrium", f"v{__version__}")
    table.add_row("python", sys.version.split()[0])
    table.add_row(".env", str(target))
    table.add_row("vector db", effective.get("VECTOR_DB", "?"))
    table.add_row("embedder", effective.get("EMBED_BASE_URL", "?"))
    table.add_row("model", effective.get("EMBED_MODEL", "?"))
    api_key_state = "set" if effective.get("API_KEY") else "[warn]not set[/]"
    table.add_row("api key", api_key_state)
    console.print(Panel(table, title="[accent]atrium doctor[/]", border_style="cyan"))

    if problems:
        _err_console().print("[bad]Configuration problems:[/]")
        for p in problems:
            _err_console().print(f"  [bad]-[/] {p}")
        console.print()

    remote_ok = True
    if not skip_remote:
        console.print("[dim]Probing services…[/]")
        for label, probe in (
            ("embedder", _probe_embedder),
            ("model", _probe_embedder_model),
            ("vector db", _probe_vector_db),
        ):
            ok, detail = probe(effective)
            mark = "[ok]OK[/]" if ok else "[bad]unreachable[/]"
            console.print(f"  {label:<10} {mark}  [dim]{detail}[/]")
            if not ok:
                remote_ok = False

    # Security posture notes — flag a permissive config before the
    # operator ships it. Warnings, not failures.
    try:
        from atrium.security import diagnostics as security_diagnostics

        notes = security_diagnostics()
        if notes:
            console.print()
            console.print("[warn]Security notes:[/]")
            for n in notes:
                console.print(f"  [warn]![/] {n}")
    except Exception as exc:
        console.print(f"  [warn]security diagnostics unavailable: {exc}[/]")

    if problems or not remote_ok:
        console.print()
        console.print("[bad]doctor found problems.[/]")
        raise typer.Exit(code=1)
    console.print()
    console.print("[ok]Everything checks out.[/]")


# ---------------------------------------------------------------------------
# Ingest / search
# ---------------------------------------------------------------------------

# Extensions the CLI will pick up when walking a directory. Anything
# else is skipped with a note rather than silently ingested.
_INGESTABLE_SUFFIXES = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".md", ".markdown", ".txt", ".csv", ".json",
    ".epub", ".rtf", ".odt", ".ods", ".odp",
}

_SKIP_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache"}


def _collect_files(paths: Sequence[str]) -> List[Path]:
    """Expand the given paths into a list of ingestable files."""
    out: List[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if not child.is_file():
                    continue
                if any(part in _SKIP_DIR_NAMES for part in child.parts):
                    continue
                if child.name.startswith("."):
                    continue
                if child.suffix.lower() in _INGESTABLE_SUFFIXES:
                    out.append(child)
        elif p.is_file():
            out.append(p)
        else:
            _err_console().print(f"[warn]skipping[/] {p} [dim](not found)[/]")
    # De-duplicate while preserving order.
    seen: set = set()
    unique: List[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def _setup_failed(exc) -> None:
    """Report a setup problem the way the user should see it.

    A missing service or an unset variable is not a crash: it is a
    sentence and a way out.
    """
    _err_console().print(f"[bad]{exc.message}[/]")
    for hint in getattr(exc, "hints", ()):  # pragma: no branch
        _err_console().print(f"  [dim]{hint}[/]")
    raise typer.Exit(code=1)


def _build_runtime():
    """Construct the parser, embedder and vector DB from the resolved
    configuration. Imported lazily so `atrium --help` stays fast.

    A setup problem — no embedding endpoint, a store that is not there
    — is reported as a sentence and a way out, never as a traceback.
    """
    from atrium.errors import AtriumSetupError

    try:
        return _build_runtime_inner()
    except AtriumSetupError as exc:
        _setup_failed(exc)


def _build_runtime_inner():
    """The construction itself; `_build_runtime` handles the failures."""
    from atrium.adapters import PgVectorAdapter, QdrantAdapter
    from atrium.config import env_bool, env_float, env_str
    from atrium.embedder_config import build_embedder
    from atrium.errors import AtriumSetupError
    from atrium.universal_parser import UniversalParser
    from atrium.vision_adapters import OllamaVisionAdapter

    try:
        embedder = build_embedder()
    except AtriumSetupError as exc:
        _setup_failed(exc)

    vision = None
    if env_bool("VISION_ENABLED"):
        base = env_str("EMBED_BASE_URL").replace("/v1", "").rstrip("/")
        vision = OllamaVisionAdapter(
            base_url=base,
            model=env_str("VISION_MODEL"),
            timeout_seconds=env_float("VISION_TIMEOUT_SECONDS"),
        )

    ocr = None
    if env_bool("OCR_ENABLED"):
        try:
            from atrium.ocr_adapter import build_ocr_adapter

            ocr = build_ocr_adapter(
                lang=env_str("OCR_LANG"),
                tesseract_cmd=env_str("TESSERACT_CMD") or None,
            )
        except Exception as exc:
            _err_console().print(f"[warn]OCR unavailable: {exc}[/]")

    parser = UniversalParser(vision_adapter=vision, ocr_adapter=ocr)

    backend = env_str("VECTOR_DB") or "sqlite"
    if backend == "pgvector":
        db = PgVectorAdapter(conn_info=env_str("PG_CONN"))
    elif backend in ("sqlite", "local", "file"):
        from atrium.local_store import SqliteAdapter

        db = SqliteAdapter(path=env_str("LOCAL_DB_PATH") or "atrium_local.db")
    else:
        db = QdrantAdapter(
            host=env_str("QDRANT_HOST"),
            port=int(env_str("QDRANT_PORT")),
        )

    # A store that is not there is a setup problem, and this is the
    # moment to say so — before a document has been parsed and embedded.
    if not db.health():
        from atrium.errors import AtriumSetupError

        if backend == "qdrant":
            raise AtriumSetupError(
                f"Qdrant is not reachable at {env_str('QDRANT_HOST')}:"
                f"{env_str('QDRANT_PORT')}.",
                hints=(
                    "Start it: `docker compose --profile qdrant up -d qdrant`",
                    "Or use the local store, which needs no server: "
                    "set VECTOR_DB=sqlite in .env",
                ),
            )
        if backend == "pgvector":
            raise AtriumSetupError(
                "The Postgres server is not reachable.",
                hints=(
                    "Start Postgres, or point PG_CONN at it",
                    "Or use the local store: set VECTOR_DB=sqlite in .env",
                ),
            )
    return parser, embedder, db


def _detect_mime(file_bytes: bytes, filename: str) -> str:
    """Detect the MIME type. Uses the same four-tier fallback as the
    HTTP service so the CLI and the server agree on what a file is.

    Imported from `atrium.mime` rather than `atrium.main`: the CLI
    never serves a request, and importing FastAPI to sniff a file
    would add ~220 ms to every CLI invocation."""
    from atrium.mime import detect_mime

    return detect_mime(file_bytes, filename)


@app.command()
def ingest(
    paths: List[str] = typer.Argument(..., help="Files or directories to ingest."),
    doc_id: Optional[str] = typer.Option(
        None, "--doc-id",
        help="Document id for a single file (default: the file stem).",
    ),
    chunking_mode: str = typer.Option(
        "auto", "--chunking-mode",
        help="Chunking preset: auto, balanced, precise, context_rich, custom.",
    ),
    chunking_config: Optional[str] = typer.Option(
        None, "--chunking-config",
        help="JSON object of chunking overrides, e.g. '{\"target_chars\": 900}'.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show the library's own log output.",
    ),
) -> None:
    """Parse, chunk, embed and store documents in the vector store."""
    from atrium.pipeline import ingest_document

    _quiet_logs(verbose)

    files = _collect_files(paths)
    if not files:
        _err_console().print("[bad]No ingestable files found.[/]")
        raise typer.Exit(code=1)

    overrides: Optional[dict] = None
    if chunking_config:
        import json

        try:
            overrides = json.loads(chunking_config)
        except json.JSONDecodeError as exc:
            _err_console().print(f"[bad]--chunking-config is not valid JSON: {exc}[/]")
            raise typer.Exit(code=2) from None
        chunking_mode = "custom"

    parser, embedder, db = _build_runtime()
    try:
        db.ensure_collection("documents", embedder.dimension)
    except Exception as exc:
        _err_console().print(f"[bad]Could not prepare the vector store: {exc}[/]")
        raise typer.Exit(code=1) from None

    results: List[Tuple[str, int, int]] = []
    failures: List[Tuple[str, str]] = []
    started = time.monotonic()

    with console.status("[accent]Ingesting…[/]") as status:
        for path in files:
            the_doc_id = doc_id if (doc_id and len(files) == 1) else path.stem
            status.update(f"[accent]Ingesting[/] {path.name}")
            try:
                raw = path.read_bytes()
                mime = _detect_mime(raw, path.name)
                result = ingest_document(
                    doc_id=the_doc_id,
                    file_bytes=raw,
                    filename=path.name,
                    mime_type=mime,
                    parser=parser,
                    embedder=embedder,
                    db=db,
                    chunking_mode=chunking_mode,
                    chunking_config=overrides,
                )
                results.append((path.name, result.elements_parsed, result.chunks_written))
            except Exception as exc:
                failures.append((path.name, f"{type(exc).__name__}: {exc}"))

    elapsed = time.monotonic() - started
    table = Table(title="Ingested", header_style="bold cyan")
    table.add_column("file")
    table.add_column("elements", justify="right")
    table.add_column("chunks", justify="right")
    for name, elements, chunks in results:
        table.add_row(name, str(elements), str(chunks))
    console.print(table)
    console.print(
        f"[ok]{len(results)} file(s)[/] in {elapsed:.1f}s"
        + (f"  [bad]{len(failures)} failed[/]" if failures else ""),
    )
    for name, err in failures:
        _err_console().print(f"  [bad]-[/] {name}: {err}")
    if failures and not results:
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(..., help="The search query."),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", help="How many results to return (default: DEFAULT_TOP_K).",
    ),
    rerank: bool = typer.Option(
        False, "--rerank", help="Force the cross-encoder rerank stage on.",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print raw JSON instead of a table.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show the library's own log output.",
    ),
) -> None:
    """Query the ingested corpus."""
    import json

    _quiet_logs(verbose)

    from atrium.config import env_int
    from atrium.retrieval import retrieve_and_rerank

    k = top_k if top_k is not None else env_int("DEFAULT_TOP_K")
    _, embedder, db = _build_runtime()

    reranker = None
    if rerank:
        from atrium.rerank_adapter import build_reranker

        reranker = build_reranker()
        if reranker is None:
            _err_console().print(
                "[warn]reranker unavailable — falling back to vector ranking. "
                "Install `atrium-rag[rerank]` or set RERANK_ENABLED=1.[/]",
            )

    with console.status("[accent]Searching…[/]"):
        vector = embedder.embed([query])[0]
        hits = retrieve_and_rerank(
            db, vector, query, k, reranker=reranker,
        )

    if as_json:
        console.print_json(json.dumps(hits, default=str))
        return

    if not hits:
        console.print("[warn]No results.[/] Have you ingested anything yet?")
        console.print("[dim]Try:[/] atrium ingest ./docs/")
        return

    table = Table(header_style="bold cyan", show_lines=False)
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("doc", no_wrap=True)
    table.add_column("text", overflow="fold")
    for i, hit in enumerate(hits, 1):
        text = (hit.get("text") or "").replace("\n", " ").strip()
        table.add_row(
            str(i),
            f"{hit.get('score', 0):.3f}",
            str(hit.get("doc_id", "")),
            text[:220],
        )
    console.print(table)


@app.command()
def reingest(
    path: str = typer.Argument(..., help="The document to re-chunk."),
    doc_id: str = typer.Option(..., "--doc-id", help="The existing document id to replace."),
    chunking_mode: str = typer.Option(
        "context_rich", "--chunking-mode",
        help="Chunking preset to re-chunk with: balanced, precise, context_rich.",
    ),
    chunking_config: Optional[str] = typer.Option(
        None, "--chunking-config",
        help="JSON object of chunking overrides.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show the library's own log output.",
    ),
) -> None:
    """Re-chunk an already-ingested document with a different strategy.

    The previous chunks for this document id are replaced, not mixed in
    — mixing granularities in one collection degrades retrieval.
    """
    import json

    from atrium.pipeline import ingest_document

    _quiet_logs(verbose)

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        _err_console().print(f"[bad]{file_path} is not a file[/]")
        raise typer.Exit(code=1)

    overrides: Optional[dict] = None
    if chunking_config:
        try:
            overrides = json.loads(chunking_config)
        except json.JSONDecodeError as exc:
            _err_console().print(f"[bad]--chunking-config is not valid JSON: {exc}[/]")
            raise typer.Exit(code=2) from None
        chunking_mode = "custom"

    parser, embedder, db = _build_runtime()
    try:
        db.ensure_collection("documents", embedder.dimension)
    except Exception as exc:
        _err_console().print(f"[bad]Could not prepare the vector store: {exc}[/]")
        raise typer.Exit(code=1) from None

    raw = file_path.read_bytes()
    mime = _detect_mime(raw, file_path.name)
    with console.status(f"[accent]Re-chunking[/] {file_path.name} → {chunking_mode}"):
        result = ingest_document(
            doc_id=doc_id,
            file_bytes=raw,
            filename=file_path.name,
            mime_type=mime,
            parser=parser,
            embedder=embedder,
            db=db,
            chunking_mode=chunking_mode,
            chunking_config=overrides,
        )
    console.print(
        f"[ok]Re-chunked[/] {doc_id}: {result.elements_parsed} elements → "
        f"{result.chunks_written} chunks [dim]({chunking_mode})[/]",
    )


@app.command()
def demo(
    path: Optional[str] = typer.Option(
        None, "--path", help="A document to ingest. Omit to use a built-in sample.",
    ),
    query: str = typer.Option(
        "What is this document about?", "--query", help="The query to run.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show the library's own log output.",
    ),
) -> None:
    """Run a self-contained end-to-end example.

    Ingests one document into the configured vector store and searches
    it back. This is the "does the whole thing actually work?" command.
    """
    from atrium.pipeline import ingest_document

    _quiet_logs(verbose)

    if path:
        sample = Path(path).expanduser()
        if not sample.is_file():
            _err_console().print(f"[bad]{sample} is not a file[/]")
            raise typer.Exit(code=1)
        raw = sample.read_bytes()
        filename = sample.name
    else:
        raw = _DEMO_DOCUMENT.encode("utf-8")
        filename = "atrium-demo.md"

    console.print(Panel(
        "[dim]Ingesting one document, then searching the corpus back.[/]",
        title="[accent]atrium demo[/]",
        border_style="cyan",
    ))

    parser, embedder, db = _build_runtime()
    try:
        db.ensure_collection("documents", embedder.dimension)
    except Exception as exc:
        _err_console().print(f"[bad]Could not prepare the vector store: {exc}[/]")
        raise typer.Exit(code=1) from None

    mime = _detect_mime(raw, filename)
    with console.status("[accent]Ingesting…[/]"):
        result = ingest_document(
            doc_id="atrium-demo",
            file_bytes=raw,
            filename=filename,
            mime_type=mime,
            parser=parser,
            embedder=embedder,
            db=db,
        )
    console.print(
        f"[ok]Ingested[/] {filename}: {result.elements_parsed} elements → "
        f"{result.chunks_written} chunks",
    )

    from atrium.retrieval import retrieve_and_rerank

    with console.status("[accent]Searching…[/]"):
        vector = embedder.embed([query])[0]
        hits = retrieve_and_rerank(db, vector, query, 3)

    if not hits:
        console.print("[warn]No results — the corpus looks empty.[/]")
        raise typer.Exit(code=1)

    table = Table(header_style="bold cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("score", justify="right")
    table.add_column("text", overflow="fold")
    for i, hit in enumerate(hits, 1):
        table.add_row(
            str(i),
            f"{hit.get('score', 0):.3f}",
            (hit.get("text") or "").replace("\n", " ").strip()[:200],
        )
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)"),
) -> None:
    """Run the FastAPI server with uvicorn. Reads .env from the cwd."""
    import uvicorn

    from atrium.config import env_str

    # `atrium` loads `.env` into the environment at import time, so
    # there is nothing to copy here — the FastAPI service sees exactly
    # the same values the CLI does.
    uvicorn.run(
        "atrium.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=env_str("LOG_LEVEL").lower(),
    )


@app.command()
def version() -> None:
    """Print the atrium version."""
    console.print(f"atrium {__version__}")


_DEMO_DOCUMENT = """\
# Quarterly Report

Revenue for Q3 reached 4.2 million dollars, a 17 percent increase over
Q2. The growth was driven primarily by the enterprise segment, which
now accounts for 61 percent of total revenue.

Operating costs grew 9 percent over the same period, giving a gross
margin of 74 percent. The company ended the quarter with 11.3 million
dollars in cash.

The engineering team shipped the new retrieval pipeline, which cut
median query latency from 340 milliseconds to 95 milliseconds.

Headcount grew from 84 to 91 employees, mostly in engineering and
customer success.

For the fourth quarter, management expects revenue of 4.6 million
dollars and continued margin improvement.
"""


if __name__ == "__main__":
    app()
