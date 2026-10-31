"""The setup wizard behind `atrium init`.

A first-run wizard has one job: leave the machine in a state where the
product works, without the user having to know what a vector database
is. The old wizard asked for thirty keys in a row and started nothing.
This one looks at the machine, proposes a configuration, fixes what it
can, and shows the result before writing anything.

The flow:

    detect  → what is installed, running, and reachable
    choose  → a vector store and an embedding provider
    act     → pull a model, start a container, install an extra —
              each one behind its own yes/no
    compose → a .env holding only the keys that differ from the defaults
    review  → a table of every key, editable, with `back` to any earlier step
    verify  → a real embedding call and a real store probe
    summary → what works, what does not, and the next three commands

Nothing is written to disk until the review step is accepted, and the
verification step runs against what was actually written.
"""
from __future__ import annotations

import json
import os
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

# Endpoints worth looking for on a developer's machine. Each one is
# probed for reachability and for the models it offers.
CANDIDATE_ENDPOINTS = (
    ("http://localhost:11434/v1", "local server (Ollama-compatible)"),
    ("http://localhost:8000/v1", "local server on :8000"),
    ("http://localhost:1234/v1", "local server on :1234 (LM Studio)"),
    ("http://host.docker.internal:11434/v1", "host machine from a container"),
)

OPENAI_BASE_URL = "https://api.openai.com/v1"

# The dimensions of the models people actually use. Knowing the
# dimension saves a question; the verification step catches the ones
# that are not in here.
# The options for the switches. Everything the operator can choose is
# a choice, not a thing to type: a name they have to remember is a
# question they can get wrong.
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

# Keys that must never be printed. The API key is the only one today,
# but the list is the place to add the next one.
SECRET_KEYS = ("EMBED_API_KEY", "COHERE_API_KEY", "API_KEY")


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
    """The Atrium header.

    A CLI that prints tables with no name on them looks like output
    from somewhere else. This is the one line that says which tool it
    is, and it is the first thing on the screen.
    """
    from rich.panel import Panel

    from atrium import __version__

    body = f"[accent]{__version__}[/]"
    if subtitle:
        body += f" [dim]·[/] {subtitle}"
    console.print(
        Panel(
            body,
            title="[bold accent]atrium[/]",
            title_align="left",
            border_style="accent",
            padding=(0, 1),
        )
    )


def _which(binary: str) -> bool:
    return shutil.which(binary) is not None


def _run(cmd: Sequence[str], timeout: int = 10) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
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

    base_url = (values.get("EMBED_BASE_URL") or "").strip()
    clean = base_url.replace("/v1", "").rstrip("/")
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


def verify(values: Dict[str, str]) -> List[Check]:
    """Check the configuration the way a user would experience it.

    The embedding check makes a real call and compares the dimension
    that comes back with the configured one. Listing the models is not
    enough: a server can be up, offer the model, and still return a
    different number of dimensions — which shows up much later as
    nonsense search results.
    """
    checks: List[Check] = []

    embedder = embedder_from_values(values)
    probe = getattr(embedder, "probe", None)
    if probe is not None:
        ok, detail = probe()
    else:
        ok, detail = embedder.health(), ""
    checks.append(
        Check(
            "embedding endpoint",
            ok,
            detail or getattr(embedder, "base_url", "") or getattr(embedder, "model_name", ""),
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

    try:
        store = store_from_values(values)
        kind = (values.get("VECTOR_DB") or "sqlite").lower()
        if kind in ("sqlite", "local", "file"):
            store.ensure_collection("documents", int(values.get("EMBED_DIM") or 1024))
            checks.append(Check("vector store", True, f"local file {store.path}"))
        else:
            healthy = store.health()
            checks.append(
                Check(
                    "vector store",
                    healthy,
                    kind if healthy else "not reachable",
                    "Start it, or set VECTOR_DB=sqlite for the local store",
                )
            )
    except Exception as exc:
        checks.append(
            Check(
                "vector store",
                False,
                f"{type(exc).__name__}: {exc}",
                "Start it, or set VECTOR_DB=sqlite for the local store",
            )
        )

    return checks


# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------


def _write_env_file(
    path: Path,
    values: Dict[str, str],
    defaults: Dict[str, str],
    sources: Optional[Dict[str, str]] = None,
) -> None:
    """Write the configuration, with a line saying where each key came from.

    A .env that lists thirty keys is unreadable, so a key is written
    when it differs from the default *or* when the wizard chose it
    deliberately — a choice that happens to match the default is still
    a choice, and the file should show it.
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



# Names that suggest a model embeds text. A server usually serves
# chat and vision models too, and offering those as embedding models
# is a trap: the request fails at the first embed, minutes later.
_EMBEDDING_HINTS = (
    "embed", "bge", "e5", "gte", "minilm", "mxbai", "paraphrase",
    "snowflake", "labse", "jina", "voyage",
)


def _looks_like_embedder(name: str) -> bool:
    lowered = (name or "").lower()
    return any(hint in lowered for hint in _EMBEDDING_HINTS)



def _is_local_url(url: str) -> bool:
    """Is this URL pointing at this machine?"""
    lowered = (url or "").lower()
    return any(
        host in lowered
        for host in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal")
    )


def _dimension_for(model: str, existing: Dict[str, str]) -> str:
    """The dimension of a model, asking only when it is not known.

    Most people know the model name and not the dimension, and the
    dimension is what has to be right — so it is looked up, and asked
    for only when the lookup cannot answer.
    """
    name = (model or "").strip()
    bare = name.split(":")[0]
    for candidate in (name, bare, bare.lower(), name.lower()):
        if candidate in MODEL_DIMENSIONS:
            return str(MODEL_DIMENSIONS[candidate])
    known = existing.get("EMBED_DIM")
    if known:
        return known
    import typer

    return str(typer.prompt("Dimensions", default="1024"))


def _menu(
    console,
    title: str,
    options: List[Tuple[str, str]],
    allow_other: bool = False,
    default: str = "1",
) -> str:
    """A menu the operator moves through, not one they type at.

    Arrow keys move the cursor, Enter chooses, and a digit jumps
    straight to an entry. When the input is not a terminal — CI, a
    pipe — it falls back to typing a number, because a menu that
    needs a keyboard is no use to a script.
    """
    import sys

    import typer

    choices = list(options)
    if allow_other:
        choices.append(("type another name", "anything else"))

    if not sys.stdin.isatty():
        console.print()
        console.print(f"[bold]{title}[/]")
        for index, (key, note) in enumerate(choices, start=1):
            console.print(f"  [dim]{index}.[/] {key:<34} [dim]{note}[/]")
        raw = typer.prompt("\u203a", default=default).strip()
        index = _safe_index(raw, len(choices)) - 1
        if allow_other and index == len(choices) - 1:
            return typer.prompt("name", default="").strip()
        return choices[max(0, min(index, len(choices) - 1))][0]

    return _menu_tty(title, choices, allow_other)


def _menu_tty(
    title: str,
    choices: List[Tuple[str, str]],
    allow_other: bool,
) -> str:
    """The arrow-key menu, drawn in place.

    Written with plain ANSI rather than rich, because the cursor has to
    be moved and the lines redrawn between keypresses, and rich's
    output model is a stream, not a screen.
    """
    import os
    import sys
    import termios
    import tty

    index = 0
    drawn = 0

    def render() -> None:
        nonlocal drawn
        if drawn:
            sys.stdout.write(f"\x1b[{drawn}A\x1b[J")
        lines = [f"\x1b[1m{title}\x1b[0m"]
        for position, (key, note) in enumerate(choices):
            marker = "\x1b[36m\u276f\x1b[0m" if position == index else " "
            weight = "\x1b[1m" if position == index else ""
            lines.append(f" {marker} {weight}{key:<34}\x1b[0m \x1b[2m{note}\x1b[0m")
        lines.append(" \x1b[2m\u2191\u2193 to move \u00b7 enter to choose\x1b[0m")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        drawn = len(lines)

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        render()
        while True:
            char = os.read(fd, 1)
            if char in (b"\r", b"\n"):
                break
            if char == b"\x03" or char == b"\x04":
                raise KeyboardInterrupt
            if char == b"\x1b":
                sequence = os.read(fd, 2)
                if sequence == b"[A":
                    index = (index - 1) % len(choices)
                elif sequence == b"[B":
                    index = (index + 1) % len(choices)
                elif sequence == b"[C":
                    index = (index + 1) % len(choices)
                elif sequence == b"[D":
                    index = (index - 1) % len(choices)
            elif char.isdigit():
                number = int(char)
                if 1 <= number <= len(choices):
                    index = number - 1
            else:
                continue
            render()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write("\n")
        sys.stdout.flush()

    if allow_other and index == len(choices) - 1:
        import typer

        return typer.prompt("name", default="").strip()
    return choices[index][0]



def _summary_line(values: Dict[str, str]) -> List[Tuple[str, str]]:
    """The handful of settings worth showing at a glance.

    Forty-five keys is a configuration dump. These five are the shape
    of the installation, and the rest is detail.
    """
    store = values.get("VECTOR_DB", "sqlite")
    store_note = {
        "sqlite": f"a local file ({values.get('LOCAL_DB_PATH', 'atrium_local.db')})",
        "qdrant": f"{values.get('QDRANT_HOST', 'localhost')}:{values.get('QDRANT_PORT', '6333')}",
        "pgvector": "postgres",
    }.get(store, store)

    provider = values.get("EMBED_PROVIDER", "http")
    if provider == "fastembed":
        embeddings = (
            f"{values.get('EMBED_MODEL', '?')} \u00b7 {values.get('EMBED_DIM', '?')}-d "
            "\u00b7 in-process"
        )
    else:
        embeddings = (
            f"{values.get('EMBED_BASE_URL', '?')} \u00b7 "
            f"{values.get('EMBED_MODEL', '?')} \u00b7 {values.get('EMBED_DIM', '?')}-d"
        )

    def _on(key: str) -> bool:
        return str(values.get(key, "0")).lower() in ("1", "true", "yes", "on")

    ocr = (
        "off"
        if not _on("OCR_ENABLED")
        else f"{values.get('OCR_PROVIDER', 'tesseract')} \u00b7 {values.get('OCR_LANG', 'eng')}"
    )
    rerank = (
        "off"
        if not _on("RERANK_ENABLED")
        else f"{values.get('RERANK_PROVIDER', 'bge')} \u00b7 "
             f"{values.get('RERANK_MODEL', '') or 'default'}"
    )
    vision = "off" if not _on("VISION_ENABLED") else values.get("VISION_MODEL", "minicpm-v4.6:1b")
    return [
        ("store", store_note),
        ("embeddings", embeddings),
        ("ocr", ocr),
        ("rerank", rerank),
        ("vision", vision),
    ]


def _ask_interactive(
    env: Environment,
    existing: Dict[str, str],
    set_value,
    console,
) -> None:
    """Six questions, most of them a single keypress.

    The order follows the dependencies: the store decides whether a
    host and a port are needed at all, and the embedding provider
    decides whether a key is needed at all. Nothing is asked that has a
    good default, nothing is asked twice, and every choice that has a
    list of possible answers is a list the operator picks from.
    """
    import typer

    # -- the vector store ------------------------------------------------
    store = _menu(
        console,
        "Vector store",
        [
            ("sqlite", "a local file \u2014 nothing to install"),
            ("qdrant", "a running Qdrant server"),
            ("pgvector", "Postgres with the pgvector extension"),
        ],
    )
    set_value("VECTOR_DB", store, "chosen")
    if store == "qdrant":
        set_value(
            "QDRANT_HOST",
            typer.prompt("Host", default=existing.get("QDRANT_HOST", "localhost")),
            "chosen",
        )
        set_value(
            "QDRANT_PORT",
            str(typer.prompt("Port", default=existing.get("QDRANT_PORT", "6333"))),
            "chosen",
        )
    elif store == "pgvector":
        set_value(
            "PG_CONN",
            typer.prompt("Postgres DSN", default=existing.get("PG_CONN", "")),
            "chosen",
        )
    else:
        set_value(
            "LOCAL_DB_PATH",
            existing.get("LOCAL_DB_PATH", "atrium_local.db"),
            "default",
        )

    # -- the embedding provider ------------------------------------------
    provider = _menu(
        console,
        "Embeddings",
        [
            ("local server", "Ollama, LM Studio, vLLM, TEI"),
            ("OpenAI", "api.openai.com"),
            ("another server", "any OpenAI-compatible /v1"),
            ("in-process", "fastembed \u2014 no server at all"),
        ],
    )

    if provider == "in-process":
        set_value("EMBED_PROVIDER", "fastembed", "chosen")
        set_value("EMBED_MODEL", "BAAI/bge-small-en-v1.5", "chosen")
        set_value("EMBED_DIM", "384", "chosen")
    else:
        set_value("EMBED_PROVIDER", "http", "chosen")
        if provider == "OpenAI":
            base_url = OPENAI_BASE_URL
            needs_key = True
        elif provider == "another server":
            base_url = typer.prompt(
                "Base URL",
                default=existing.get("EMBED_BASE_URL", "http://localhost:11434/v1"),
            )
            needs_key = not _is_local_url(base_url)
        else:
            detected = (
                env.endpoints[0].base_url if env.endpoints else "http://localhost:11434/v1"
            )
            base_url = typer.prompt(
                "Base URL", default=existing.get("EMBED_BASE_URL", detected)
            )
            needs_key = not _is_local_url(base_url)
        set_value("EMBED_BASE_URL", base_url, "chosen")

        # The models the endpoint actually offers, so the model is
        # picked rather than typed.
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
        model = _menu(console, "Model", options, allow_other=True)
        if others:
            console.print(
                "  [dim]also on this server, but not embedding models: "
                + ", ".join(others[:6]) + "[/]"
            )
        set_value("EMBED_MODEL", model, "chosen")
        set_value("EMBED_DIM", _dimension_for(model, existing), "chosen")

        if needs_key:
            set_value(
                "EMBED_API_KEY",
                typer.prompt(
                    "API key", default=existing.get("EMBED_API_KEY", ""), hide_input=True
                ),
                "chosen",
            )

    # -- OCR ---------------------------------------------------------------
    if typer.confirm("Enable OCR for scanned pages?", default=False):
        set_value("OCR_ENABLED", "1", "chosen")
        ocr_provider = _menu(console, "OCR provider", OCR_PROVIDERS)
        set_value("OCR_PROVIDER", ocr_provider, "chosen")
        language = _menu(console, "OCR language", OCR_LANGUAGES, allow_other=True)
        set_value("OCR_LANG", language, "chosen")
    else:
        set_value("OCR_ENABLED", "0", "chosen")

    # -- reranking ---------------------------------------------------------
    if typer.confirm("Enable cross-encoder reranking?", default=False):
        set_value("RERANK_ENABLED", "1", "chosen")
        rerank_provider = _menu(console, "Rerank provider", RERANK_PROVIDERS)
        set_value("RERANK_PROVIDER", rerank_provider, "chosen")
        model = _menu(
            console,
            "Rerank model",
            [(name, "") for name in RERANK_MODELS.get(rerank_provider, [])],
            allow_other=True,
        )
        set_value("RERANK_MODEL", model, "chosen")
        if rerank_provider == "cohere":
            set_value(
                "COHERE_API_KEY",
                typer.prompt(
                    "Cohere API key",
                    default=existing.get("COHERE_API_KEY", ""),
                    hide_input=True,
                ),
                "chosen",
            )
    else:
        set_value("RERANK_ENABLED", "0", "chosen")

    # -- vision ------------------------------------------------------------
    if typer.confirm("Enable vision models for images?", default=False):
        set_value("VISION_ENABLED", "1", "chosen")
        model = _menu(
            console,
            "Vision model",
            [(name, "") for name in VISION_MODELS],
            allow_other=True,
        )
        set_value("VISION_MODEL", model, "chosen")
    else:
        set_value("VISION_ENABLED", "0", "chosen")


def _review_table(values: Dict[str, str], sources: Dict[str, str]) -> str:
    """What the installation will look like, in two blocks.

    The first block is the shape of the thing — store, embeddings,
    OCR, rerank, vision. The second is the keys this run actually set.
    The forty other keys in the file are listed as a count, because a
    list of forty-five is not information anyone can act on.
    """
    interesting = {
        key: value
        for key, value in values.items()
        if sources.get(key, "existing") != "existing"
    }
    lines = []
    for label, note in _summary_line(values):
        lines.append(f"  {label:<12} {note}")

    if interesting:
        lines.append("")
        width = max(len(key) for key in interesting) + 2
        for key, value in interesting.items():
            shown = value if value else "(unset)"
            if key in SECRET_KEYS and value:
                shown = "set (hidden)"
            lines.append(f"  {key:<{width}} {shown}")

    others = len(values) - len(interesting)
    if others:
        lines.append("")
        lines.append(f"  [+ {others} other setting(s) already in .env, unchanged]")

    lines.append("")
    lines.append("  set KEY VALUE   change a value")
    lines.append("  unset KEY       remove a key")
    lines.append("  edit            open this file in $EDITOR")
    lines.append("  save            write it and continue")
    return "\n".join(lines)

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
    from rich.table import Table

    console = console or _console()
    err_console = err_console or _err_console()

    # Seeded from the existing file, so re-running the wizard never
    # silently drops a setting the operator made. The choices below
    # overwrite the keys they are about; everything else is left alone.
    values: Dict[str, str] = dict(existing)
    sources: Dict[str, str] = {key: "existing" for key in existing}

    def set_value(key: str, value: str, source: str = "you") -> None:
        values[key] = value
        sources[key] = source

    # -- 1. what is on this machine ---------------------------------------
    # Detection is used to fill in the defaults and to offer the setup
    # actions below. It is not shown: the operator came here to answer
    # a few questions, not to read a report about their own laptop.
    env = detect_environment()

    if non_interactive:
        _choose_non_interactive(env, existing, set_value)
    else:
        console.print()
        brand(console, f"setting up {target}")
        _ask_interactive(env, existing, set_value, console)

    # -- 4. actions -------------------------------------------------------
    if not non_interactive:
        _offer_actions(env, values, sources, assume_yes, console)

    # -- 5. review --------------------------------------------------------
    if non_interactive:
        target_values = dict(values)
    else:
        target_values = _review_loop(values, sources, target, console, err_console)
        if target_values is None:
            err_console.print("[bad]Setup cancelled.[/] Nothing was written.")
            return 1

    # -- 6. write and verify ---------------------------------------------
    _write_env_file(target, target_values, defaults, sources)
    console.print(f"[ok]Wrote[/] {target}")

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

    table = Table(title=None, show_header=True, header_style="dim")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for check in checks:
        table.add_row(
            check.name,
            "[ok]ok[/]" if check.ok else "[bad]failed[/]",
            check.detail,
        )
    console.print()
    console.print(table)

    if ok:
        console.print()
        console.print("[ok]Everything checks out.[/]")
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
        err_console.print(
            "[dim]Edit the file and run[/] [key]atrium doctor[/] [dim]to re-check.[/]"
        )
    return 0 if ok else 1


def _choose_non_interactive(
    env: Environment, existing: Dict[str, str], set_value
) -> None:
    """Pick the best configuration the machine already offers."""
    # The store: the local one needs nothing, so it is the default; a
    # reachable Qdrant is worth using when it is already there.
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


def _safe_index(raw: str, count: int) -> int:
    try:
        index = int(str(raw).strip())
    except (TypeError, ValueError):
        return 1
    return max(1, min(index, count))


def _offer_actions(
    env: Environment,
    values: Dict[str, str],
    sources: Dict[str, str],
    assume_yes: bool,
    console,
) -> None:
    """Offer to do the things that make the configuration work.

    Every one of them asks first. Nothing on the machine changes
    without the operator saying yes.
    """
    import typer

    def offer(description: str, action) -> bool:
        approved = True if assume_yes else typer.confirm(description, default=True)
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


def _review_loop(
    values: Dict[str, str],
    sources: Dict[str, str],
    target: Path,
    console,
    err_console,
):
    """Show the configuration, let it be changed, and only then write it."""
    import typer

    while True:
        console.print()
        console.print("[bold]Review[/] [dim]— nothing is written until you save[/]")
        console.print()
        console.print(_review_table(values, sources))
        console.print()
        try:
            raw = typer.prompt("›", default="save", show_default=False)
        except (EOFError, KeyboardInterrupt):
            return None

        command = raw.strip()
        lowered = command.lower()

        if lowered in ("save", "s", ""):
            return values
        if lowered in ("quit", "q", "cancel"):
            return None
        if lowered.startswith("set "):
            parts = command.split(" ", 2)
            if len(parts) == 3 and parts[1]:
                values[parts[1].strip()] = parts[2].strip()
                sources[parts[1].strip()] = "you"
                continue
            err_console.print("[bad]usage: set KEY VALUE[/]")
            continue
        if lowered.startswith("unset "):
            key = command[6:].strip()
            if key:
                values.pop(key, None)
                sources.pop(key, None)
                continue
            err_console.print("[bad]usage: unset KEY[/]")
            continue
        if lowered == "edit":
            _edit_in_editor(values)
            continue
        err_console.print(f"[bad]unknown command[/] {command!r}")


def _edit_in_editor(values: Dict[str, str]) -> None:
    """Hand the configuration to the user's editor, then read it back."""
    import tempfile

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, encoding="utf-8") as fh:
        for key, value in values.items():
            fh.write(f"{key}={value}\n")
        path = fh.name
    try:
        subprocess.run([editor, path], check=False)
        edited: Dict[str, str] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            edited[key.strip()] = value.strip()
        values.clear()
        values.update(edited)
    finally:
        Path(path).unlink(missing_ok=True)


def _console():
    from rich.console import Console

    return Console()


def _err_console():
    from rich.console import Console

    return Console(stderr=True)


def _version() -> str:
    from atrium import __version__

    return __version__
