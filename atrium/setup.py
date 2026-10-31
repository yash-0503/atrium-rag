"""The setup wizard behind `atrium init`.

A first-run wizard has one job: leave the machine in a state where the
product works, without the user having to know what a vector database
is. This one looks at the machine, proposes a configuration, and shows
the result before anything is written.

The flow:

    scan     → what is installed, running, and reachable
    ask      → five short sections: store, embeddings, OCR, rerank, vision
    review   → a card of the result, then: write it / edit some fields /
               start over / cancel. Edits redo only the sections you pick.
    act      → pull a model, start a container, install an extra —
               each one behind its own yes/no
    write    → a .env holding the keys that differ from the defaults
    verify   → a real embedding call and a real store probe

Nothing is written to disk until the review is accepted, and the
verification step runs against what was actually written.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "brand",
    "Environment",
    "Endpoint",
    "Check",
    "detect_environment",
    "run_wizard",
    "embedder_from_values",
    "store_from_values",
    "verify",
]

# Endpoints worth looking for on a developer's machine.
CANDIDATE_ENDPOINTS = (
    ("http://localhost:11434/v1", "local server (Ollama-compatible)"),
    ("http://localhost:8000/v1", "local server on :8000"),
    ("http://localhost:1234/v1", "local server on :1234 (LM Studio)"),
    ("http://host.docker.internal:11434/v1", "host machine from a container"),
)

OPENAI_BASE_URL = "https://api.openai.com/v1"

# Everything the operator can choose is a choice, not a thing to type.
OCR_PROVIDERS = [
    ("rapidocr", "pip-only — no binary to install"),
    ("tesseract", "the Tesseract binary, if it is on PATH"),
]

OCR_LANGUAGES = [
    ("eng", "English"),
    ("ara", "Arabic"),
    ("chi_sim", "Chinese (simplified)"),
    ("jpn", "Japanese"),
    ("kor", "Korean"),
    ("deu", "German"),
    ("fra", "French"),
    ("spa", "Spanish"),
    ("rus", "Russian"),
]

RERANK_PROVIDERS = [
    ("bge", "BAAI cross-encoder, runs locally"),
    ("flashrank", "~30MB ONNX, no PyTorch"),
    ("cohere", "hosted API, needs a key"),
]

RERANK_MODELS = {
    "bge": ["BAAI/bge-reranker-large", "BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"],
    "flashrank": ["ms-marco-MiniLM-L-12-v2", "ms-marco-MiniLM-L-6-v2"],
    "cohere": ["rerank-english-v3.0", "rerank-multilingual-v3.0"],
}

VISION_MODELS = ["minicpm-v4.6:1b", "llama3.2-vision:11b", "qwen2.5vl:7b"]

# The dimensions of the models people actually use. Knowing the
# dimension saves a question; the verification step catches the rest.
MODEL_DIMENSIONS = {
    "bge-large": 1024,
    "bge-large-en-v1.5": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "bge-base": 768,
    "bge-base-en-v1.5": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "bge-small": 384,
    "bge-small-en-v1.5": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "nomic-embed-text": 768,
    "nomic-embed-text:v1.5": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "all-MiniLM-L6-v2": 384,
}

# Keys that must never be printed.
SECRET_KEYS = ("EMBED_API_KEY", "COHERE_API_KEY", "API_KEY", "PG_CONN")


def redact(values: Dict[str, str]) -> Dict[str, str]:
    """A copy of the values with the secrets replaced."""
    out = dict(values)
    for key in SECRET_KEYS:
        if out.get(key):
            out[key] = "set (hidden)"
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass
class Endpoint:
    """An embedding endpoint that answered."""

    base_url: str
    label: str
    models: List[str] = field(default_factory=list)
    detail: str = ""

    def has(self, model: str) -> bool:
        return any(m == model or m.startswith(model.split(":")[0]) for m in self.models)


@dataclass
class Environment:
    """What is on the machine."""

    ollama_installed: bool = False
    ollama_running: bool = False
    ollama_models: List[str] = field(default_factory=list)
    docker_installed: bool = False
    docker_running: bool = False
    qdrant_reachable: bool = False
    postgres_reachable: bool = False
    tesseract_installed: bool = False
    fastembed_installed: bool = False
    endpoints: List[Endpoint] = field(default_factory=list)

    def endpoint_for(self, model: str) -> Optional[Endpoint]:
        for endpoint in self.endpoints:
            if endpoint.has(model):
                return endpoint
        return None


def brand(console, subtitle: str = "") -> None:
    """The Atrium header, animated: this is the first thing `init` shows."""
    from atrium import __version__
    from atrium.ui import brand as _brand

    _brand(console, subtitle, __version__, animate=True)


def _which(binary: str) -> bool:
    return shutil.which(binary) is not None


def _run(cmd: Sequence[str], timeout: int = 10) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return 1, ""


def _probe_endpoint(base_url: str, api_key: str = "") -> Optional[Endpoint]:
    """Ask an endpoint what it is and what models it has."""
    try:
        import httpx
    except ImportError:
        return None

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=2.5) as client:
            resp = client.get(f"{base_url.rstrip('/')}/models", headers=headers)
        if resp.status_code >= 400:
            return None
        payload = resp.json()
        models = [
            str(item.get("id"))
            for item in (payload.get("data") or [])
            if item.get("id")
        ]
        return Endpoint(
            base_url=base_url.rstrip("/"),
            label=base_url,
            models=models,
            detail=f"{len(models)} models",
        )
    except Exception:
        return None


def detect_environment() -> Environment:
    """Look at the machine."""
    env = Environment()

    env.ollama_installed = _which("ollama")
    if env.ollama_installed:
        code, out = _run(["ollama", "list"], timeout=10)
        if code == 0:
            env.ollama_running = True
            for line in out.splitlines()[1:]:
                parts = line.split()
                if parts:
                    env.ollama_models.append(parts[0])

    env.docker_installed = _which("docker")
    if env.docker_installed:
        code, _ = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=15)
        env.docker_running = code == 0

    if env.docker_running:
        try:
            import httpx

            with httpx.Client(timeout=2.0) as client:
                resp = client.get("http://localhost:6333/healthz")
            env.qdrant_reachable = resp.status_code == 200
        except Exception:
            env.qdrant_reachable = False

    env.tesseract_installed = _which("tesseract")

    try:
        import fastembed  # noqa: F401

        env.fastembed_installed = True
    except ImportError:
        env.fastembed_installed = False

    for base_url, label in CANDIDATE_ENDPOINTS:
        endpoint = _probe_endpoint(base_url)
        if endpoint is not None:
            endpoint.label = label
            env.endpoints.append(endpoint)

    return env


# ---------------------------------------------------------------------------
# Building a runtime from a set of values
# ---------------------------------------------------------------------------


def embedder_from_values(values: Dict[str, str]):
    """The embedder a set of .env values describes."""
    provider = (values.get("EMBED_PROVIDER") or "http").strip().lower()
    model = values.get("EMBED_MODEL") or "bge-large"
    dim = int(values.get("EMBED_DIM") or 0)
    if provider in ("fastembed", "local", "inprocess", "in-process"):
        from atrium.local_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder(model=model, dim=dim or None)
    from atrium.adapters import OpenAICompatibleEmbedder

    clean = (values.get("EMBED_BASE_URL") or "").strip().rstrip("/")
    if clean.endswith("/v1"):  # strip only a trailing /v1, never one mid-URL
        clean = clean[:-3].rstrip("/")
    return OpenAICompatibleEmbedder(
        base_url=f"{clean}/v1",
        model=model,
        dim=dim or 1024,
        api_key=values.get("EMBED_API_KEY") or "",
    )


def store_from_values(values: Dict[str, str]):
    """The vector store a set of .env values describes."""
    kind = (values.get("VECTOR_DB") or "sqlite").strip().lower()
    if kind in ("sqlite", "local", "file"):
        from atrium.local_store import SqliteAdapter

        return SqliteAdapter(path=values.get("LOCAL_DB_PATH") or "atrium_local.db")
    if kind == "qdrant":
        from atrium.adapters import QdrantAdapter

        return QdrantAdapter(
            host=values.get("QDRANT_HOST") or "localhost",
            port=int(values.get("QDRANT_PORT") or 6333),
        )
    from atrium.adapters import PgVectorAdapter

    return PgVectorAdapter(conn_info=values.get("PG_CONN") or "")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """One thing that was checked, and what it said."""

    name: str
    ok: bool
    detail: str
    fix: str = ""



def kind_label(kind: str) -> str:
    """What a store is called in a sentence."""
    return {
        "qdrant": "qdrant",
        "pgvector": "postgres",
        "sqlite": "local file",
        "local": "local file",
        "file": "local file",
    }.get(kind, kind)


def verify(values: Dict[str, str]) -> List[Check]:
    """Check the configuration the way a user would experience it.

    The embedding check makes a real call and compares the dimension
    that comes back with the configured one: a server can be up, offer
    the model, and still return a different number of dimensions, which
    shows up much later as nonsense search results.
    """
    checks: List[Check] = []

    try:
        embedder = embedder_from_values(values)
    except Exception as exc:
        checks.append(
            Check(
                "embedding endpoint",
                False,
                f"{type(exc).__name__}: {exc}",
                "Install the extra for this provider, or choose another",
            )
        )
        embedder = None

    if embedder is not None:
        probe = getattr(embedder, "probe", None)
        try:
            if probe is not None:
                ok, detail = probe()
            else:
                ok, detail = embedder.health(), ""
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append(
            Check(
                "embedding endpoint",
                ok,
                detail
                or getattr(embedder, "base_url", "")
                or getattr(embedder, "model_name", ""),
                "Run `atrium init` to point at a reachable endpoint",
            )
        )

        if ok:
            try:
                vectors = embedder.embed(["atrium"])
                actual = len(vectors[0]) if vectors else 0
                expected = int(values.get("EMBED_DIM") or 0)
                if expected and actual != expected:
                    checks.append(
                        Check(
                            "embedding dimension",
                            False,
                            f"the model returns {actual} dimensions, EMBED_DIM says {expected}",
                            f"Set EMBED_DIM={actual} in .env",
                        )
                    )
                else:
                    checks.append(Check("embedding dimension", True, f"{actual} dimensions"))
            except Exception as exc:
                checks.append(
                    Check(
                        "embedding call",
                        False,
                        f"{type(exc).__name__}: {exc}",
                        "Check the model name and the API key",
                    )
                )

    store_kind = (values.get("VECTOR_DB") or "sqlite").lower()
    store_fix = "Start it, or set VECTOR_DB=sqlite for the local store"
    try:
        store = store_from_values(values)
        if store_kind in ("sqlite", "local", "file"):
            try:
                store.ensure_collection("documents", int(values.get("EMBED_DIM") or 1024))
                checks.append(Check("vector store", True, f"local file {store.path}"))
            except ValueError as exc:
                # The store holds vectors at a different dimension than
                # the model now produces. The vectors are useless to
                # the new model, so the store has to start again.
                checks.append(
                    Check(
                        "vector store",
                        False,
                        str(exc),
                        "The local store holds vectors the new model cannot "
                        "use. Start a new one — the old file is moved aside.",
                    )
                )
        else:
            healthy = store.health()
            checks.append(
                Check(
                    "vector store",
                    healthy,
                    kind_label(store_kind) if healthy else "not reachable",
                    store_fix,
                )
            )
    except Exception as exc:
        checks.append(
            Check(
                "vector store",
                False,
                f"{type(exc).__name__}: {exc}",
                store_fix,
            )
        )

    return checks


# ---------------------------------------------------------------------------
# .env writing
# ---------------------------------------------------------------------------


def _write_env_file(
    path: Path,
    values: Dict[str, str],
    defaults: Dict[str, str],
    sources: Optional[Dict[str, str]] = None,
) -> None:
    """Write the configuration.

    A key is written when it differs from the default *or* when the
    wizard chose it deliberately — a choice that happens to match the
    default is still a choice, and the file should show it.
    """
    sources = sources or {}
    lines = [
        "# --- atrium configuration ---",
        "# Written by `atrium init`. Edit freely; `atrium doctor` re-checks it.",
        "",
    ]
    for key, value in values.items():
        if value == defaults.get(key) and sources.get(key, "default") == "default":
            continue
        if key == "EMBED_API_KEY" and value:
            lines.append("# the key for the embedding endpoint")
        lines.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

# A server usually serves chat and vision models too; offering those as
# embedding models is a trap that only shows up at the first embed.
_EMBEDDING_HINTS = (
    "embed", "bge", "e5", "gte", "minilm", "mxbai", "paraphrase",
    "snowflake", "labse", "jina", "voyage",
)


def _looks_like_embedder(name: str) -> bool:
    lowered = (name or "").lower()
    return any(hint in lowered for hint in _EMBEDDING_HINTS)


def _is_local_url(url: str) -> bool:
    lowered = (url or "").lower()
    return any(
        host in lowered
        for host in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")
    )


def _on(value) -> bool:
    return str(value or "0").strip().lower() in ("1", "true", "yes", "on")


def _dimension_for(model: str, existing: Dict[str, str]) -> str:
    """The dimension of a model, asking only when it is not known.

    A previously configured dimension is reused only for the *same*
    model: a different, unknown model may have a different width.
    """
    name = (model or "").strip()
    bare = name.split(":")[0]
    for candidate in (name, bare, bare.lower(), name.lower()):
        if candidate in MODEL_DIMENSIONS:
            return str(MODEL_DIMENSIONS[candidate])
    if existing.get("EMBED_DIM") and existing.get("EMBED_MODEL") == name:
        return existing["EMBED_DIM"]
    import typer

    return str(typer.prompt("Dimensions", default="1024"))


def _menu(
    console,
    title: str,
    options: List[Tuple[str, str]],
    allow_other: bool = False,
    current: Optional[str] = None,
) -> str:
    """An arrow-key menu that returns the chosen label.

    `current` preselects an entry, so editing a section starts from
    what is already configured. Drawing and key handling live in
    atrium.ui, which also falls back to a numbered prompt when stdin is
    not a terminal.
    """
    from atrium import ui

    choices = list(options)
    if allow_other:
        choices.append(("type another name", "anything else"))
    default = next((i for i, (key, _) in enumerate(choices) if key == current), 0)
    index = ui.select(console, title, choices, default)
    if allow_other and index == len(choices) - 1:
        import typer

        return typer.prompt("name", default="").strip()
    return choices[index][0]


def _yes_no(console, title: str, default: bool = False, hint: str = "") -> bool:
    from atrium import ui

    options = [("Yes", hint), ("No", "")]
    return ui.select(console, title, options, 0 if default else 1) == 0


# ---------------------------------------------------------------------------
# The questions, one method per section
# ---------------------------------------------------------------------------

_SECTION_NAMES = ["Vector store", "Embeddings", "OCR", "Reranking", "Vision"]


class _Wizard:
    """The state of one wizard run, and the five sections that fill it.

    Seeded from the existing file, so re-running never silently drops a
    setting the operator made. Each section overwrites only its keys.
    """

    def __init__(
        self,
        env: Environment,
        existing: Dict[str, str],
        console,
        defaults: Optional[Dict[str, str]] = None,
    ) -> None:
        self.env = env
        self.defaults = dict(defaults or {})
        self.existing = dict(existing)
        self.console = console
        self.values: Dict[str, str] = dict(existing)
        self.sources: Dict[str, str] = {key: "existing" for key in existing}

    # -- state ------------------------------------------------------------

    def cur(self, key: str, default: str = "") -> str:
        """The value being edited, else the one from the file, else a default."""
        if key in self.values and self.values[key] != "":
            return self.values[key]
        if key in self.existing and self.existing[key] != "":
            return self.existing[key]
        return self.defaults.get(key, default)

    def set(self, key: str, value: str, source: str = "you") -> None:
        self.values[key] = value
        self.sources[key] = source

    def drop(self, *keys: str) -> None:
        """Remove stale provider-specific keys when switching backends."""
        for k in keys:
            self.values.pop(k, None)
            self.sources.pop(k, None)

    def reset(self) -> None:
        self.values = dict(self.existing)
        self.sources = {key: "existing" for key in self.existing}

    # -- sections ---------------------------------------------------------

    def ask_store(self) -> None:
        import typer

        store = _menu(
            self.console,
            "Vector store",
            [
                ("sqlite", "a local file — nothing to install"),
                ("qdrant", "a running Qdrant server"),
                ("pgvector", "Postgres with the pgvector extension"),
            ],
            current=self.cur("VECTOR_DB", "sqlite"),
        )
        self.set("VECTOR_DB", store, "chosen")
        if store == "qdrant":
            self.drop("PG_CONN", "LOCAL_DB_PATH")
            self.set(
                "QDRANT_HOST",
                typer.prompt("Host", default=self.cur("QDRANT_HOST", "localhost")),
                "chosen",
            )
            self.set(
                "QDRANT_PORT",
                str(typer.prompt("Port", default=self.cur("QDRANT_PORT", "6333"))),
                "chosen",
            )
        elif store == "pgvector":
            self.drop("QDRANT_HOST", "QDRANT_PORT", "LOCAL_DB_PATH")
            self.set(
                "PG_CONN",
                typer.prompt("Postgres DSN", default=self.cur("PG_CONN")),
                "chosen",
            )
        else:
            self.drop("QDRANT_HOST", "QDRANT_PORT", "PG_CONN")
            self.set("LOCAL_DB_PATH", self.cur("LOCAL_DB_PATH", "atrium_local.db"), "default")

    def ask_embeddings(self) -> None:
        import typer

        console, env = self.console, self.env

        base_now = self.cur("EMBED_BASE_URL")
        if self.cur("EMBED_PROVIDER", "http") == "fastembed":
            current = "in-process"
        elif base_now == OPENAI_BASE_URL:
            current = "OpenAI"
        elif base_now and not _is_local_url(base_now):
            current = "another server"
        else:
            current = "local server"

        provider = _menu(
            console,
            "Embeddings",
            [
                ("local server", "Ollama, LM Studio, vLLM, TEI"),
                ("OpenAI", "api.openai.com"),
                ("another server", "any OpenAI-compatible /v1"),
                ("in-process", "fastembed — no server at all"),
            ],
            current=current,
        )

        if provider == "in-process":
            self.drop("EMBED_BASE_URL", "EMBED_API_KEY")
            self.set("EMBED_PROVIDER", "fastembed", "chosen")
            self.set("EMBED_MODEL", "BAAI/bge-small-en-v1.5", "chosen")
            self.set("EMBED_DIM", "384", "chosen")
            return

        self.set("EMBED_PROVIDER", "http", "chosen")
        if provider == "OpenAI":
            base_url, needs_key = OPENAI_BASE_URL, True
        elif provider == "another server":
            default_url = base_now if (base_now and base_now != OPENAI_BASE_URL) else "http://localhost:11434/v1"
            base_url = typer.prompt("Base URL", default=default_url)
            needs_key = not _is_local_url(base_url)
        else:
            detected = env.endpoints[0].base_url if env.endpoints else "http://localhost:11434/v1"
            default_url = base_now if (base_now and _is_local_url(base_now)) else detected
            base_url = typer.prompt("Base URL", default=default_url)
            needs_key = not _is_local_url(base_url)
        self.set("EMBED_BASE_URL", base_url, "chosen")

        models: List[str] = []
        probed = _probe_endpoint(base_url)
        if probed is not None:
            models = list(probed.models)
        others: List[str] = []
        if models:
            options = []
            for name in models:
                dim = MODEL_DIMENSIONS.get(name) or MODEL_DIMENSIONS.get(name.split(":")[0])
                if _looks_like_embedder(name) or dim:
                    options.append((name, f"{dim}-d" if dim else ""))
                else:
                    others.append(name)
            if not options:
                options = [(name, "") for name in models[:6]]
                others = []
        else:
            options = [
                (name, f"{dim}-d")
                for name, dim in (
                    ("bge-large", 1024),
                    ("nomic-embed-text:v1.5", 768),
                    ("text-embedding-3-small", 1536),
                )
            ]
        model = _menu(
            console, "Model", options, allow_other=True, current=self.cur("EMBED_MODEL")
        )
        if others:
            console.print(
                "  [dim]also on this server, but not embedding models: "
                + ", ".join(others[:6]) + "[/]"
            )
        # Pass self.values so in-session dimension edits are preserved
        dim = _dimension_for(model, self.values)
        self.set("EMBED_MODEL", model, "chosen")
        self.set("EMBED_DIM", dim, "chosen")

        if needs_key:
            self.set(
                "EMBED_API_KEY",
                typer.prompt("API key", default=self.cur("EMBED_API_KEY"), hide_input=True),
                "chosen",
            )
        else:
            self.drop("EMBED_API_KEY")

    def ask_ocr(self) -> None:
        if _yes_no(
            self.console,
            "Enable OCR for scanned pages?",
            _on(self.cur("OCR_ENABLED")),
            "read text from scans and image-only documents",
        ):
            self.set("OCR_ENABLED", "1", "chosen")
            self.set(
                "OCR_PROVIDER",
                _menu(self.console, "OCR engine", OCR_PROVIDERS,
                      current=self.cur("OCR_PROVIDER", "rapidocr")),
                "chosen",
            )
            self.set(
                "OCR_LANG",
                _menu(self.console, "OCR language", OCR_LANGUAGES, allow_other=True,
                      current=self.cur("OCR_LANG", "eng")),
                "chosen",
            )
        else:
            self.set("OCR_ENABLED", "0", "chosen")

    def ask_rerank(self) -> None:
        import typer

        if _yes_no(
            self.console,
            "Enable cross-encoder reranking?",
            _on(self.cur("RERANK_ENABLED")),
            "re-score results: slower, more accurate",
        ):
            self.set("RERANK_ENABLED", "1", "chosen")
            provider = _menu(
                self.console, "Rerank engine", RERANK_PROVIDERS,
                current=self.cur("RERANK_PROVIDER", "bge"),
            )
            self.set("RERANK_PROVIDER", provider, "chosen")
            self.set(
                "RERANK_MODEL",
                _menu(
                    self.console,
                    "Rerank model",
                    [(name, "") for name in RERANK_MODELS.get(provider, [])],
                    allow_other=True,
                    current=self.cur("RERANK_MODEL"),
                ),
                "chosen",
            )
            if provider == "cohere":
                self.set(
                    "COHERE_API_KEY",
                    typer.prompt(
                        "Cohere API key",
                        default=self.cur("COHERE_API_KEY"),
                        hide_input=True,
                    ),
                    "chosen",
                )
        else:
            self.set("RERANK_ENABLED", "0", "chosen")

    def ask_vision(self) -> None:
        if _yes_no(
            self.console,
            "Enable vision models for images?",
            _on(self.cur("VISION_ENABLED")),
            "describe images with a vision model",
        ):
            self.set("VISION_ENABLED", "1", "chosen")
            self.set(
                "VISION_MODEL",
                _menu(
                    self.console,
                    "Vision model",
                    [(name, "") for name in VISION_MODELS],
                    allow_other=True,
                    current=self.cur("VISION_MODEL"),
                ),
                "chosen",
            )
        else:
            self.set("VISION_ENABLED", "0", "chosen")

    def sections(self):
        return [
            self.ask_store,
            self.ask_embeddings,
            self.ask_ocr,
            self.ask_rerank,
            self.ask_vision,
        ]

    def ask_all(self) -> None:
        for ask in self.sections():
            ask()


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

_SECTION_KEYS = {
    "VECTOR_DB", "LOCAL_DB_PATH", "QDRANT_HOST", "QDRANT_PORT", "PG_CONN",
    "EMBED_PROVIDER", "EMBED_BASE_URL", "EMBED_MODEL", "EMBED_DIM", "EMBED_API_KEY",
    "OCR_ENABLED", "OCR_PROVIDER", "OCR_LANG",
    "RERANK_ENABLED", "RERANK_PROVIDER", "RERANK_MODEL", "COHERE_API_KEY",
    "VISION_ENABLED", "VISION_MODEL",
}


def _review_sections(values: Dict[str, str]):
    """The configuration as human-readable sections (secrets hidden)."""

    def g(key: str, default: str = "") -> str:
        return values.get(key) or default

    store = g("VECTOR_DB", "sqlite")
    if store == "qdrant":
        store_rows = [
            ("Backend", "Qdrant"),
            ("Address", f"{g('QDRANT_HOST', 'localhost')}:{g('QDRANT_PORT', '6333')}"),
        ]
    elif store == "pgvector":
        store_rows = [
            ("Backend", "Postgres + pgvector"),
            ("Connection", "set (hidden)" if g("PG_CONN") else "not set"),
        ]
    else:
        store_rows = [
            ("Backend", "SQLite (local file)"),
            ("File", g("LOCAL_DB_PATH", "atrium_local.db")),
        ]

    if g("EMBED_PROVIDER", "http") == "fastembed":
        emb_rows = [
            ("Provider", "in-process (fastembed)"),
            ("Model", g("EMBED_MODEL", "?")),
            ("Dimensions", g("EMBED_DIM", "?")),
        ]
    else:
        emb_rows = [
            ("Provider", "OpenAI-compatible endpoint"),
            ("Endpoint", g("EMBED_BASE_URL", "?")),
            ("Model", g("EMBED_MODEL", "?")),
            ("Dimensions", g("EMBED_DIM", "?")),
            ("API key", "set (hidden)" if g("EMBED_API_KEY") else "not set"),
        ]

    if _on(values.get("OCR_ENABLED")):
        ocr_rows = [
            ("Status", "on"),
            ("Engine", g("OCR_PROVIDER", "rapidocr")),
            ("Language", g("OCR_LANG", "eng")),
        ]
    else:
        ocr_rows = [("Status", "off")]

    if _on(values.get("RERANK_ENABLED")):
        rerank_rows = [
            ("Status", "on"),
            ("Engine", g("RERANK_PROVIDER", "bge")),
            ("Model", g("RERANK_MODEL", "default")),
        ]
    else:
        rerank_rows = [("Status", "off")]

    if _on(values.get("VISION_ENABLED")):
        vision_rows = [("Status", "on"), ("Model", g("VISION_MODEL", "minicpm-v4.6:1b"))]
    else:
        vision_rows = [("Status", "off")]

    sections = [
        (_SECTION_NAMES[0], store_rows),
        (_SECTION_NAMES[1], emb_rows),
        (_SECTION_NAMES[2], ocr_rows),
        (_SECTION_NAMES[3], rerank_rows),
        (_SECTION_NAMES[4], vision_rows),
    ]

    others = [key for key in values if key not in _SECTION_KEYS]
    if others:
        sections.append(
            ("Other", [("Kept from .env", f"{len(others)} setting(s), unchanged")])
        )
    return sections


def _section_hint(sections, index: int) -> str:
    """A one-line summary of a section, for the edit checklist."""
    _, rows = sections[index]
    return " · ".join(value for _, value in rows[:3])


def _edit_single_key(w: _Wizard, console) -> None:
    """Pick any setting by name and give it a new value."""
    import typer

    from atrium import ui

    # Include all known section keys, defaults, and current values so every field is editable
    all_keys = sorted(set(_SECTION_KEYS) | set(w.defaults) | set(w.values))
    options = [
        (
            key,
            "set (hidden)"
            if key in SECRET_KEYS and w.cur(key)
            else (w.cur(key) or "not set"),
        )
        for key in all_keys
    ]
    options.append(("type another key", "any setting atrium reads"))
    index = ui.select(console, "Which setting?", options)
    if index == len(options) - 1:
        key = typer.prompt("Key").strip().upper()
    else:
        key = all_keys[index]
    if not key:
        return
    value = typer.prompt(
        key, default=w.cur(key, ""), hide_input=key in SECRET_KEYS
    )
    w.set(key, str(value).strip(), "you")


def _review_loop(w: _Wizard, target: Path, console) -> bool:
    """Show the result, and let the operator accept, edit, restart or cancel.

    Returns True when the configuration should be written.
    """
    from atrium import ui

    first = True
    while True:
        console.print()
        sections = _review_sections(w.values)
        ui.review_card(console, sections, title=f"Review · {target}", animate=first)
        first = False
        console.print()
        try:
            choice = ui.select(
                console,
                "Write this configuration?",
                [
                    ("Yes, write it", f"save to {target.name} and verify"),
                    ("Edit some fields", "change one or more sections"),
                    ("Start over", "go back to the first question"),
                    ("No, cancel", "write nothing"),
                ],
            )
        except KeyboardInterrupt:
            return False

        if choice == 0:
            return True
        if choice == 3:
            return False
        if choice == 2:
            w.reset()
            console.print()
            try:
                w.ask_all()
            except KeyboardInterrupt:
                return False
            continue

        # -- edit some fields ---------------------------------------------
        console.print()
        names = [
            (name, _section_hint(sections, i)) for i, name in enumerate(_SECTION_NAMES)
        ]
        names.append(("An individual setting", "change any key by name"))
        try:
            picked = ui.multiselect(console, "What do you want to change?", names)
            asks = w.sections()
            for i in picked:
                if i < len(asks):
                    console.print()
                    asks[i]()
            if len(asks) in picked:
                console.print()
                _edit_single_key(w, console)
        except KeyboardInterrupt:
            continue  # back to the review


# ---------------------------------------------------------------------------
# Non-interactive and setup actions
# ---------------------------------------------------------------------------


def _choose_non_interactive(env: Environment, existing: Dict[str, str], set_value) -> None:
    """Pick the best configuration the machine already offers."""
    if env.qdrant_reachable:
        set_value("VECTOR_DB", "qdrant", "detected")
        set_value("QDRANT_HOST", "localhost", "detected")
        set_value("QDRANT_PORT", "6333", "detected")
    else:
        set_value("VECTOR_DB", "sqlite", "detected")
        set_value("LOCAL_DB_PATH", "atrium_local.db", "detected")
    if env.fastembed_installed:
        set_value("EMBED_PROVIDER", "fastembed", "detected")
        set_value("EMBED_MODEL", "BAAI/bge-small-en-v1.5", "detected")
        set_value("EMBED_DIM", "384", "detected")
        return
    endpoint = env.endpoints[0] if env.endpoints else None
    set_value("EMBED_PROVIDER", "http", "detected")
    set_value(
        "EMBED_BASE_URL",
        endpoint.base_url if endpoint else "http://localhost:11434/v1",
        "detected",
    )
    set_value("EMBED_MODEL", existing.get("EMBED_MODEL", "bge-large"), "detected")
    set_value("EMBED_DIM", existing.get("EMBED_DIM", "1024"), "detected")


def _offer_actions(
    env: Environment,
    values: Dict[str, str],
    assume_yes: bool,
    console,
) -> None:
    """Offer to do the things that make the configuration work.

    Every one of them asks first. Nothing on the machine changes
    without the operator saying yes.
    """

    def offer(description: str, action) -> bool:
        approved = True if assume_yes else _yes_no(console, description, True)
        if not approved:
            return False
        try:
            action()
            console.print(f"[ok]done:[/] {description.split('?')[0].strip()}")
            return True
        except Exception as exc:
            console.print(f"[bad]could not:[/] {exc}")
            return False

    model = values.get("EMBED_MODEL", "")
    # Ollama reports "bge-large:latest"; the model is asked for as
    # "bge-large". Compare them as the same model.
    installed = {name.split(":")[0] for name in env.ollama_models}
    needs_pull = (
        values.get("EMBED_PROVIDER", "http") == "http"
        and model
        and env.ollama_installed
        and env.ollama_running
        and model.split(":")[0] not in installed
    )
    if needs_pull:
        offer(
            f"Pull the model {model}? (ollama pull {model})",
            lambda: subprocess.run(["ollama", "pull", model], check=True, timeout=900),
        )

    if values.get("VECTOR_DB") == "qdrant" and not env.qdrant_reachable:
        if env.docker_installed and not env.docker_running:
            offer(
                "Start Docker Desktop? (needed to run Qdrant)",
                lambda: subprocess.run(["open", "-a", "Docker"], check=True, timeout=60),
            )
            console.print(
                "[dim]Docker takes a moment to start; run `atrium doctor` again after it is up.[/]"
            )
        elif env.docker_installed:
            offer(
                "Start Qdrant? (docker compose --profile qdrant up -d qdrant)",
                lambda: subprocess.run(
                    ["docker", "compose", "--profile", "qdrant", "up", "-d", "qdrant"],
                    check=True,
                    timeout=300,
                ),
            )

    if values.get("EMBED_PROVIDER") == "fastembed" and not env.fastembed_installed:
        offer(
            "Install the local embedder? (pip install \"atrium-rag[local]\")",
            lambda: subprocess.run(
                [sys.executable, "-m", "pip", "install", "atrium-rag[local]"],
                check=True,
                timeout=900,
            ),
        )


# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------


def run_wizard(
    target: Path,
    *,
    defaults: Dict[str, str],
    existing: Dict[str, str],
    non_interactive: bool = False,
    assume_yes: bool = False,
    force: bool = False,
    as_json: bool = False,
    console=None,
    err_console=None,
) -> int:
    """Run the wizard. Returns a process exit code."""
    from atrium import ui

    console = console or _console()
    err_console = err_console or _err_console()

    if not non_interactive and not as_json:
        brand(console, f"setting up {target}")

    # Detection fills in defaults and decides which setup actions to
    # offer. It is not shown as a report: the operator came to answer a
    # few questions, not to read about their own laptop.
    if non_interactive:
        env = detect_environment()
    else:
        with ui.space_status(console, "Scanning this machine…"):
            env = detect_environment()

    w = _Wizard(env, existing, console, defaults=defaults)

    if non_interactive:
        _choose_non_interactive(env, existing, w.set)
    else:
        try:
            w.ask_all()
            if not _review_loop(w, target, console):
                err_console.print("[bad]Setup cancelled.[/] Nothing was written.")
                return 1
            _offer_actions(env, w.values, assume_yes, console)
        except KeyboardInterrupt:
            err_console.print("\n[bad]Setup cancelled.[/] Nothing was written.")
            return 1

    target_values = dict(w.values)

    # -- write and verify ---------------------------------------------------
    _write_env_file(target, target_values, defaults, w.sources)
    console.print(f"[ok]Wrote[/] {target}")

    with ui.space_status(console, "Verifying…"):
        checks = verify(target_values)
    ok = all(check.ok for check in checks)

    if as_json:
        console.print(
            json.dumps(
                {
                    "env": str(target),
                    "values": redact(target_values),
                    "checks": [
                        {"name": c.name, "ok": c.ok, "detail": c.detail, "fix": c.fix}
                        for c in checks
                    ],
                    "ok": ok,
                },
                indent=2,
            )
        )
        return 0 if ok else 1

    console.print()
    ui.reveal_checks(console, checks, animate=not non_interactive)

    if ok:
        console.print()
        ui.launch_ready(console, "All systems ready")
        console.print(
            "[dim]Next:[/] [key]atrium demo[/] to see it work, "
            "or [key]atrium serve[/] to run the API."
        )
    else:
        err_console.print()
        err_console.print("[bad]Some checks failed.[/] The configuration was still written:")
        for check in checks:
            if not check.ok and check.fix:
                err_console.print(f"  [key]{check.name}[/] — {check.fix}")
        # The one failure we can put right from here: a local store
        # whose vectors the new model cannot use.
        if _offer_new_store(target_values, checks, console, err_console):
            with ui.space_status(console, "Verifying…"):
                checks = verify(target_values)
            ok = all(check.ok for check in checks)
            console.print()
            ui.reveal_checks(console, checks, animate=False)
            if ok:
                console.print()
                ui.launch_ready(console, "All systems ready")
                return 0

        err_console.print(
            "[dim]Edit the file and run[/] [key]atrium doctor[/] [dim]to re-check.[/]"
        )
    return 0 if ok else 1



def _offer_new_store(
    values: Dict[str, str],
    checks: List[Check],
    console,
    err_console,
) -> bool:
    """Offer to start a fresh local store when the dimension changed.

    The vectors in the old store are a different length from the ones
    the new model produces, and the store will not mix them. Moving
    the file aside is the fix, and it keeps the old corpus around in
    case it was wanted.
    """
    import sys

    import typer

    if not sys.stdin.isatty():
        return False
    if (values.get("VECTOR_DB") or "sqlite").lower() not in ("sqlite", "local", "file"):
        return False
    store_check = next((c for c in checks if c.name == "vector store"), None)
    if store_check is None or store_check.ok or "dim" not in store_check.detail:
        return False
    path = Path(values.get("LOCAL_DB_PATH") or "atrium_local.db").expanduser()
    if not path.exists():
        return False
    if not typer.confirm(
        "Start a new local store? (the old file is moved aside)",
        default=True,
    ):
        return False
    backup = path.with_name(path.name + ".bak")
    try:
        path.rename(backup)
    except OSError as exc:
        err_console.print(f"[bad]Could not move {path}: {exc}[/]")
        return False
    console.print(f"[ok]Moved[/] {path.name} → {backup.name}")
    return True


def _console():
    from rich.console import Console

    from atrium.ui import THEME

    return Console(theme=THEME)


def _err_console():
    from rich.console import Console

    from atrium.ui import THEME

    return Console(theme=THEME, stderr=True)


def _version() -> str:
    from atrium import __version__

    return __version__