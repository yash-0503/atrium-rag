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


def _review_table(values: Dict[str, str], sources: Dict[str, str]) -> str:
    rows = []
    for key, value in values.items():
        shown = value if value else "(unset)"
        if key in SECRET_KEYS and value:
            shown = "set (hidden)"
        rows.append(f"{key:<20} {shown:<42} {sources.get(key, 'default')}")
    body = "\n".join(rows)
    return (
        f"{body}\n\n"
        "set KEY VALUE   change a value\n"
        "unset KEY       remove a key\n"
        "edit            open this file in $EDITOR\n"
        "save            write it and continue"
    )


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
    import typer
    from rich.markup import escape
    from rich.panel import Panel
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

    # -- 1. detect --------------------------------------------------------
    env = detect_environment()

    if not non_interactive:
        console.print(
            Panel(
                f"atrium v{_version()}\n"
                "This checks your machine, sets up what it can, and writes a "
                "configuration you can edit before it is saved.\n"
                f"[dim]Writing to {target}[/]",
                title="[accent]atrium init[/]",
                border_style="cyan",
            )
        )

    detected = _detected_lines(env)
    if detected:
        console.print("[dim]On this machine:[/]")
        for line in detected:
            console.print(f"  {line}")
        console.print()

    # -- 2. the vector store ---------------------------------------------
    if non_interactive:
        store_choice = existing.get("VECTOR_DB", "sqlite")
    else:
        table = Table(title=None, show_header=True, header_style="dim")
        table.add_column("#")
        table.add_column("store")
        table.add_column("what it needs")
        table.add_row("1", "sqlite (local file)", "nothing — a file next to your project")
        table.add_row(
            "2",
            "qdrant",
            "a running Qdrant" + ("" if env.qdrant_reachable else " (not running now)"),
        )
        table.add_row("3", "pgvector", "a Postgres with the pgvector extension")
        console.print(table)
        raw = typer.prompt("Which vector store?", default="1")
        store_choice = {"1": "sqlite", "2": "qdrant", "3": "pgvector"}.get(raw.strip(), "sqlite")
        if store_choice == "pgvector":
            raw_conn = typer.prompt(
                "Postgres DSN",
                default=existing.get("PG_CONN", "postgresql://user:pass@localhost:5432/db"),
            )
            set_value("PG_CONN", raw_conn)
    set_value("VECTOR_DB", store_choice, "chosen")
    if store_choice in ("sqlite", "local", "file"):
        set_value("LOCAL_DB_PATH", existing.get("LOCAL_DB_PATH", "atrium_local.db"), "default")

    # -- 3. the embedding provider ---------------------------------------
    choices = _embedding_choices(env)
    if non_interactive:
        _choose_non_interactive(env, existing, set_value)
    else:
        table = Table(title=None, show_header=True, header_style="dim")
        table.add_column("#")
        table.add_column("embedding provider")
        table.add_column("what it needs")
        for index, (label, needs, _kind, _endpoint) in enumerate(choices, start=1):
            table.add_row(str(index), escape(label), escape(needs))
        console.print(table)
        raw = typer.prompt("Which embedding provider?", default="1")
        label, needs, kind, endpoint = choices[_safe_index(raw, len(choices)) - 1]
        if kind == "fastembed":
            set_value("EMBED_PROVIDER", "fastembed", "chosen")
            set_value("EMBED_MODEL", "BAAI/bge-small-en-v1.5", "chosen")
            set_value("EMBED_DIM", "384", "chosen")
        elif kind == "openai":
            set_value("EMBED_PROVIDER", "http", "chosen")
            set_value("EMBED_BASE_URL", OPENAI_BASE_URL, "chosen")
            set_value("EMBED_MODEL", "text-embedding-3-small", "chosen")
            set_value("EMBED_DIM", "1536", "chosen")
            raw_key = typer.prompt(
                "OpenAI API key", default=existing.get("EMBED_API_KEY", ""), hide_input=True
            )
            if raw_key:
                set_value("EMBED_API_KEY", raw_key)
        elif kind == "custom":
            set_value("EMBED_PROVIDER", "http", "chosen")
            set_value(
                "EMBED_BASE_URL",
                typer.prompt("Base URL (including /v1)", default=existing.get("EMBED_BASE_URL", "")),
                "chosen",
            )
            set_value(
                "EMBED_MODEL",
                typer.prompt("Model name", default=existing.get("EMBED_MODEL", "bge-large")),
                "chosen",
            )
            set_value(
                "EMBED_DIM",
                typer.prompt("Dimensions", default=existing.get("EMBED_DIM", "1024")),
                "chosen",
            )
            raw_key = typer.prompt(
                "API key (blank for local servers)",
                default=existing.get("EMBED_API_KEY", ""),
                hide_input=True,
            )
            if raw_key:
                set_value("EMBED_API_KEY", raw_key)
        else:
            set_value("EMBED_PROVIDER", "http", "chosen")
            set_value(
                "EMBED_BASE_URL",
                endpoint.base_url if endpoint else "http://localhost:11434/v1",
                "detected",
            )
            model = existing.get("EMBED_MODEL", "bge-large")
            if endpoint is not None and endpoint.models and model not in endpoint.models:
                model = endpoint.models[0]
            set_value("EMBED_MODEL", model, "detected")
            set_value("EMBED_DIM", existing.get("EMBED_DIM", "1024"), "default")

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


def _embedding_choices(env: Environment) -> List[Tuple[str, str, str, Optional[Endpoint]]]:
    """The embedding providers, as (label, needs, kind, endpoint) tuples."""
    choices: List[Tuple[str, str, str, Optional[Endpoint]]] = []
    for endpoint in env.endpoints:
        models = ", ".join(endpoint.models[:3]) if endpoint.models else "no models listed"
        choices.append((endpoint.label, f"reachable — {models}", "http", endpoint))
    choices.append(("local server (default URL)", "http://localhost:11434/v1", "http", None))
    choices.append(
        (
            "in-process (fastembed)",
            'nothing running — `pip install "atrium-rag[local]"`'
            if not env.fastembed_installed
            else "installed",
            "fastembed",
            None,
        )
    )
    choices.append(("OpenAI", "an API key", "openai", None))
    choices.append(("another OpenAI-compatible server", "a base URL and a model", "custom", None))
    return choices


def _safe_index(raw: str, count: int) -> int:
    try:
        index = int(str(raw).strip())
    except (TypeError, ValueError):
        return 1
    return max(1, min(index, count))


def _detected_lines(env: Environment) -> List[str]:
    lines = []
    if env.ollama_running:
        models = ", ".join(env.ollama_models[:4]) if env.ollama_models else "no models"
        lines.append(f"[ok]embedding server[/] running — {models}")
    elif env.ollama_installed:
        lines.append("[warn]embedding server[/] installed but not running")
    else:
        lines.append("[dim]embedding server[/] not installed")
    if env.qdrant_reachable:
        lines.append("[ok]qdrant[/] reachable")
    elif env.docker_installed:
        lines.append(
            "[warn]docker[/] installed"
            + ("" if env.docker_running else " but not running")
            + " — qdrant not reachable"
        )
    else:
        lines.append("[dim]docker[/] not installed")
    lines.append(
        "[ok]tesseract[/] installed"
        if env.tesseract_installed
        else "[dim]tesseract[/] not installed (OCR is off by default)"
    )
    lines.append(
        "[ok]fastembed[/] installed"
        if env.fastembed_installed
        else "[dim]fastembed[/] not installed (the `local` extra)"
    )
    return lines


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
    needs_pull = (
        values.get("EMBED_PROVIDER", "http") == "http"
        and model
        and env.ollama_installed
        and env.ollama_running
        and model not in env.ollama_models
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
