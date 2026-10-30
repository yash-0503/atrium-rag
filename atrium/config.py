"""
atrium/config.py

One resolver for every environment variable atrium understands, and one
place that decides which value wins when a variable is set in more than
one place.

Precedence
----------
    real environment  >  .env file  >  built-in default

This is the Docker/Kubernetes convention: an operator who exports
``API_KEY`` in the shell gets that value even when a stale ``.env`` sits
in the working directory, and a value in ``.env`` beats the packaged
default. Nothing else in the codebase is allowed to invent its own
default for a variable that also appears in :data:`DEFAULTS`.

Why this module exists
----------------------
Three concrete bugs it fixes. All three were verified by running the
code, not inferred:

1. **``.env`` was ignored by uvicorn.** There is no ``python-dotenv``
   dependency anywhere in the project, so an operator who set
   ``API_KEY`` in ``.env`` and then started the server with
   ``uvicorn atrium.main:app`` got a server with **no authentication on
   any route except ``/health``**. ``atrium serve`` appeared to work
   only because it copied the file into ``os.environ`` by hand;
   ``docker compose`` worked only because compose injects ``env_file:``.
   Every other entry point was wide open.

2. **The same variable had different defaults in different modules.**
   ``OLLAMA_BASE_URL`` had four — ``host.docker.internal:11434`` in
   ``embedder_config.py`` and ``worker.py``, ``http://localhost:11434``
   in ``main.py`` and the CLI. ``QDRANT_HOST`` had two — ``qdrant`` and
   ``localhost``. So ``atrium doctor`` probed ``localhost`` while the
   server connected to ``qdrant``: doctor could report "unreachable"
   against a perfectly healthy deployment.

3. **``atrium doctor`` lied about ``API_KEY``.** Doctor imported
   ``atrium.security``, which read ``os.getenv("API_KEY")`` at *import*
   time. Doctor never loaded ``.env``, so it reported "API_KEY is
   empty" no matter what the operator had configured.

The defaults below are laptop-correct (``localhost``, not
``host.docker.internal``). Container deployments override them through
compose ``env_file:`` or real environment variables — host names are
the operator's choice, not something the package should bake in.

Reading config
--------------
Two access patterns, both reading from the same table:

* :func:`env_str` / :func:`env_int` / :func:`env_float` /
  :func:`env_bool` read the *live* value. Use these for anything that
  can change at runtime — most importantly the bearer-token check,
  which must see a value set after import.
* :func:`get_settings` returns a frozen snapshot of the whole
  configuration. Use it for *structural* settings (which vector store,
  which host), which should not change mid-process, and for the
  programmatic API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "DEFAULTS",
    "Settings",
    "env_bool",
    "env_float",
    "env_int",
    "env_path",
    "env_str",
    "get_settings",
    "load_env_file",
    "parse_env_text",
    "read_env_file",
    "redacted_settings",
]


# ---------------------------------------------------------------------------
# Defaults — the single source of truth for every env var atrium reads.
#
# Keys are UPPER_SNAKE env var names; values are strings, because that is
# what the environment holds. Typed accessors convert on read.
#
# "localhost" is the default everywhere on purpose. The old defaults
# ("qdrant", "host.docker.internal") only resolve inside a Docker
# network, so a developer following the README on a laptop got
# connection errors for services that were running perfectly.
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, str] = {
    # -- vector store ------------------------------------------------------
    "VECTOR_DB": "qdrant",
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": "6333",
    "PG_CONN": "postgresql://user:pass@pgvector:5432/db",
    # -- embeddings / provider --------------------------------------------
    "OLLAMA_BASE_URL": "http://localhost:11434",
    # "bge-large" is the tag Ollama resolves. "bge-large-en-v1.5" — the
    # name in the original code — is not in Ollama's registry, so the
    # default pointed at a model that could never be pulled.
    "EMBED_MODEL": "bge-large",
    "EMBED_DIM": "1024",
    # -- vision ------------------------------------------------------------
    "VISION_ENABLED": "0",
    "VISION_MODEL": "minicpm-v4.6:1b",
    "VISION_TIMEOUT_SECONDS": "180",
    # -- OCR (opt-in: the `ocr` extra is not installed by default) --------
    "OCR_ENABLED": "0",
    "OCR_PROVIDER": "tesseract",
    "OCR_LANG": "eng",
    "TESSERACT_CMD": "",
    # -- rerank (opt-in: the `rerank` extra is not installed by default) ---
    "RERANK_ENABLED": "0",
    "RERANK_PROVIDER": "bge",
    "RERANK_MODEL": "",
    "RERANK_DEVICE": "",
    "RERANK_BACKEND": "",
    "RERANK_MAX_LENGTH": "512",
    "FLASHRANK_CACHE_DIR": "",
    "COHERE_API_KEY": "",
    # -- retrieval ---------------------------------------------------------
    "DEFAULT_TOP_K": "8",
    "DENSE_CONFIDENT_GAP": "0.05",
    "DENSE_MIN_SCORE": "0.55",
    # -- HTTP / security ---------------------------------------------------
    "API_KEY": "",
    "MAX_UPLOAD_BYTES": str(100 * 1024 * 1024),
    "ALLOWED_MIME_TYPES": "",
    "ALLOW_CHUNKED_UPLOADS": "0",
    "CORS_ALLOW_ORIGINS": "",
    # -- jobs / worker -----------------------------------------------------
    "JOB_DB_PATH": "jobs.db",
    "JOB_STALE_SECONDS": "1800",
    "INGEST_CONCURRENCY": "4",
    "WORKER_POLL_BASE": "1.0",
    "WORKER_POLL_MAX": "10.0",
    "WORKER_JOB_TIMEOUT": "0",
    # -- misc --------------------------------------------------------------
    "LOG_LEVEL": "INFO",
    "RETRIEVAL_LAB_MAX_SESSIONS": "16",
}


# ---------------------------------------------------------------------------
# .env file handling
# ---------------------------------------------------------------------------


# The shell environment as it was when this module was first imported —
# before any `.env` file was merged in. Capturing it here is what lets
# `load_env_file` honour "the operator's shell wins" even after a file
# has already contributed a value to `os.environ`.
_SHELL_ENV: frozenset = frozenset(os.environ)

# Keys whose current value came from a `.env` file rather than the
# shell. Re-loading a different file may replace these; it may never
# replace a shell-exported value.
_DOTENV_KEYS: set = set()


def parse_env_text(text: str) -> Dict[str, str]:
    """Parse ``.env``-shaped text into a dict.

    Rules:

    * blank lines and ``#`` comments are skipped
    * ``KEY=VALUE`` splits on the *first* ``=``, so values may contain ``=``
    * a matching pair of surrounding quotes is stripped
    * a leading ``export `` is accepted
    * a line with no ``=`` is ignored rather than raising
    """
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def read_env_file(path: Path) -> Dict[str, str]:
    """Read and parse a ``.env`` file. Missing/unreadable file → ``{}``."""
    try:
        return parse_env_text(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return {}


def env_path(target: Optional[Path] = None) -> Path:
    """Resolve which ``.env`` file to use.

    Order: explicit ``target`` > ``$RAG_INGEST_ENV`` > ``./.env``.
    """
    if target is not None:
        return Path(target)
    return Path(os.getenv("RAG_INGEST_ENV", "./.env"))


def load_env_file(
    path: Optional[Path] = None,
    override: bool = False,
) -> Dict[str, str]:
    """Load a ``.env`` file into ``os.environ``.

    Precedence: a real environment variable always beats the file. So
    an operator who exports ``API_KEY=...`` in the shell is not
    surprised by a stale ``.env`` overwriting it.

    Keys already sourced from a *previous* ``.env`` load are refreshed.
    That is what lets ``atrium doctor --env /other/.env`` point at a
    different file than the one ``atrium`` auto-loaded at import,
    without the shell losing its precedence.

    Returns the parsed file contents (empty dict when the file is
    absent). Safe to call more than once.
    """
    target = env_path(path)
    values = read_env_file(target) if target.exists() else {}
    for key, val in values.items():
        if not override and key in _SHELL_ENV:
            continue
        if key not in os.environ or key in _DOTENV_KEYS:
            os.environ[key] = val
            _DOTENV_KEYS.add(key)
    return values


# ---------------------------------------------------------------------------
# Typed accessors — the live view of the configuration
# ---------------------------------------------------------------------------


def env_str(name: str, default: Optional[str] = None) -> str:
    """Read a string setting.

    Resolution order: a non-empty real environment value, then a
    non-empty ``.env`` value (already merged into ``os.environ`` by
    :func:`load_env_file`), then :data:`DEFAULTS`, then ``default``.

    An *empty* environment value is treated as unset. That keeps
    ``OLLAMA_BASE_URL=`` in a ``.env`` from producing a broken URL, and
    it costs nothing for the keys whose default is legitimately empty
    (``API_KEY``, ``ALLOWED_MIME_TYPES``, ``COHERE_API_KEY``) — those
    still resolve to ``""``.
    """
    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        return raw.strip()
    if default is not None:
        return default
    return DEFAULTS.get(name, "")


def env_int(name: str, default: Optional[int] = None) -> int:
    """Read an integer setting. Unparseable values fall back."""
    raw = env_str(name)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    if default is not None:
        return default
    try:
        return int(DEFAULTS.get(name, "0"))
    except ValueError:
        return 0


def env_float(name: str, default: Optional[float] = None) -> float:
    """Read a float setting. Unparseable values fall back."""
    raw = env_str(name)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    if default is not None:
        return default
    try:
        return float(DEFAULTS.get(name, "0"))
    except ValueError:
        return 0.0


def env_bool(name: str, default: Optional[bool] = None) -> bool:
    """Read a boolean setting.

    Truthy: ``1 true yes on`` (case-insensitive).
    Falsy: ``0 false no off`` and the empty string.
    """
    raw = env_str(name)
    if raw:
        lowered = raw.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    if default is not None:
        return default
    return DEFAULTS.get(name, "0") in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """A frozen snapshot of the resolved configuration.

    Structural settings only — the ones that must not change while the
    process is running (which vector store, which host, which model).
    Live, per-request values (the bearer token) are read through the
    :func:`env_str` accessors instead.
    """

    # vector store
    vector_db: str = "qdrant"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    pg_conn: str = "postgresql://user:pass@pgvector:5432/db"

    # embeddings / provider
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "bge-large"
    embed_dim: int = 1024

    # vision
    vision_enabled: bool = False
    vision_model: str = "minicpm-v4.6:1b"
    vision_timeout_seconds: float = 180.0

    # OCR
    ocr_enabled: bool = False
    ocr_provider: str = "tesseract"
    ocr_lang: str = "eng"
    tesseract_cmd: str = ""

    # rerank
    rerank_enabled: bool = False
    rerank_provider: str = "bge"
    rerank_model: str = ""
    rerank_device: str = ""
    rerank_backend: str = ""
    rerank_max_length: int = 512
    flashrank_cache_dir: str = ""
    cohere_api_key: str = ""

    # retrieval
    default_top_k: int = 8
    dense_confident_gap: float = 0.05
    dense_min_score: float = 0.55

    # HTTP / security
    api_key: str = ""
    max_upload_bytes: int = 100 * 1024 * 1024
    allowed_mime_types: str = ""
    allow_chunked_uploads: bool = False
    cors_allow_origins: str = ""

    # jobs / worker
    job_db_path: str = "jobs.db"
    job_stale_seconds: int = 1800
    ingest_concurrency: int = 4
    worker_poll_base: float = 1.0
    worker_poll_max: float = 10.0
    worker_job_timeout: float = 0.0

    # misc
    log_level: str = "INFO"
    retrieval_lab_max_sessions: int = 16


def get_settings() -> Settings:
    """Return the resolved configuration as a frozen snapshot.

    Deliberately *not* cached: it is a handful of dict lookups, and a
    stale cache is a much worse bug than the cost. Call it once at
    startup and hold the value; do not call it in a hot loop.
    """
    allowed_mime = env_str("ALLOWED_MIME_TYPES")
    return Settings(
        vector_db=env_str("VECTOR_DB"),
        qdrant_host=env_str("QDRANT_HOST"),
        qdrant_port=env_int("QDRANT_PORT"),
        pg_conn=env_str("PG_CONN"),
        ollama_base_url=env_str("OLLAMA_BASE_URL"),
        embed_model=env_str("EMBED_MODEL"),
        embed_dim=env_int("EMBED_DIM"),
        vision_enabled=env_bool("VISION_ENABLED"),
        vision_model=env_str("VISION_MODEL"),
        vision_timeout_seconds=env_float("VISION_TIMEOUT_SECONDS"),
        ocr_enabled=env_bool("OCR_ENABLED"),
        ocr_provider=env_str("OCR_PROVIDER"),
        ocr_lang=env_str("OCR_LANG"),
        tesseract_cmd=env_str("TESSERACT_CMD"),
        rerank_enabled=env_bool("RERANK_ENABLED"),
        rerank_provider=env_str("RERANK_PROVIDER"),
        rerank_model=env_str("RERANK_MODEL"),
        rerank_device=env_str("RERANK_DEVICE"),
        rerank_backend=env_str("RERANK_BACKEND"),
        rerank_max_length=env_int("RERANK_MAX_LENGTH"),
        flashrank_cache_dir=env_str("FLASHRANK_CACHE_DIR"),
        cohere_api_key=env_str("COHERE_API_KEY"),
        default_top_k=env_int("DEFAULT_TOP_K"),
        dense_confident_gap=env_float("DENSE_CONFIDENT_GAP"),
        dense_min_score=env_float("DENSE_MIN_SCORE"),
        api_key=env_str("API_KEY"),
        max_upload_bytes=env_int("MAX_UPLOAD_BYTES"),
        allowed_mime_types=allowed_mime,
        allow_chunked_uploads=env_bool("ALLOW_CHUNKED_UPLOADS"),
        cors_allow_origins=env_str("CORS_ALLOW_ORIGINS"),
        job_db_path=env_str("JOB_DB_PATH"),
        job_stale_seconds=env_int("JOB_STALE_SECONDS"),
        ingest_concurrency=env_int("INGEST_CONCURRENCY"),
        worker_poll_base=env_float("WORKER_POLL_BASE"),
        worker_poll_max=env_float("WORKER_POLL_MAX"),
        worker_job_timeout=env_float("WORKER_JOB_TIMEOUT"),
        log_level=env_str("LOG_LEVEL").upper(),
        retrieval_lab_max_sessions=env_int("RETRIEVAL_LAB_MAX_SESSIONS"),
    )


def redacted_settings(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Return the settings as a dict with secret values masked.

    Used by ``atrium doctor`` and ``atrium configure --show`` so a
    support transcript never contains a live token.
    """
    current = settings or get_settings()
    secret_fields = {"api_key", "cohere_api_key", "pg_conn"}
    out: Dict[str, Any] = {}
    for f in fields(current):
        value = getattr(current, f.name)
        if f.name in secret_fields and value:
            value = "***set***" if f.name != "pg_conn" else "***set***"
        out[f.name] = value
    return out


def secret_keys() -> List[str]:
    """Env var names whose values must never be printed."""
    return ["API_KEY", "COHERE_API_KEY", "PG_CONN"]
