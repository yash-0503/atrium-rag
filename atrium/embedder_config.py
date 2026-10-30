"""
atrium/embedder_config.py

Single source of truth for embedding configuration shared by the HTTP
service (atrium/main.py) and the standalone worker (atrium/worker.py).

Before this module existed, the two paths constructed embedders
independently:
  - atrium/main.py:181 hardcoded `model="nomic-embed-text:v1.5", dim=768`
  - atrium/worker.py:131-138 used env vars (`EMBED_MODEL`,
    `EMBED_DIM`) with the worker defaults of bge-large-en-v1.5 / 1024

A vector upserted by the HTTP path (768-d) and a vector upserted by
the worker path (1024-d) cannot coexist in the same collection. The
two paths would create different collections on first run and any
mixed traffic would silently mismatch.

This module fixes that by exposing one `build_embedder()` factory
that both paths call, with one documented default.

Default: `bge-large` at 1024 dimensions. This matches the
2026-10-11 19-set A/B (bge-large wins +0.049 R@5 over nomic; the
v0.0.1 published headline at R@8 = 0.8976 was scored on this
embedder). Override via `EMBED_MODEL` / `EMBED_DIM` env vars for
A/B experiments.

The model *name* is `bge-large`, not `bge-large-en-v1.5` — the latter
is the HuggingFace repo name, not an Ollama registry tag, so it could
never be pulled. The 19-fixture benchmark itself ran on the
`bge-large` tag.

Fail-fast: if the collection's existing dimension disagrees with
the embedder's dimension, `db.ensure_collection("documents",
embedder.dimension)` raises a `ValueError` at startup (Qdrant) or
the startup check in the calling service raises a clearer
`RuntimeError` after the same check. The HTTP and worker boot
sequences both call `db.ensure_collection` so an incompatible
existing collection surfaces immediately, not on the first ingest.
"""
from __future__ import annotations

from typing import Optional

from atrium.adapters import OpenAICompatibleEmbedder
from atrium.config import DEFAULTS, env_int, env_str

# Documented v0.0.1 default. The 19-set A/B (memory
# `19set_embedder_ab_2026_10_11`) settled on bge-large over
# nomic-embed-text; promote embedder changes only on 19-set evidence.
#
# The value lives in atrium.config.DEFAULTS so the CLI wizard, the
# runtime and `atrium doctor` all agree on it. These module-level
# aliases stay for callers that import them directly — they name the
# *default*, so they read the defaults table rather than the
# environment.
DEFAULT_EMBED_MODEL = DEFAULTS["EMBED_MODEL"]
DEFAULT_EMBED_DIM = int(DEFAULTS["EMBED_DIM"])


def default_base_url() -> str:
    """Default OLLAMA_BASE_URL.

    `localhost` — not `host.docker.internal`. The old default only
    resolved inside a Docker network, so a developer following the
    README on a laptop got a DNS failure against a service that was
    running perfectly. Container deployments set OLLAMA_BASE_URL
    explicitly (compose `env_file:` or a real environment variable).
    """
    return env_str("OLLAMA_BASE_URL")


def build_embedder(
    base_url: Optional[str] = None,
) -> OpenAICompatibleEmbedder:
    """Construct the production embedder from env vars.

    `EMBED_MODEL` defaults to `bge-large`. `EMBED_DIM` defaults to
    1024. The dimension is a separate env var (not derived from the
    model name) so an operator with a non-default model can pin the
    dimension explicitly — relying on a model-name → dim lookup would
    silently produce wrong-dim vectors on Ollama if the local model
    name doesn't match the registry.

    The /v1 suffix is added if missing: `OpenAICompatibleEmbedder`
    calls /v1/embeddings but operators typically set OLLAMA_BASE_URL
    to the bare host.
    """
    raw_url = base_url if base_url is not None else default_base_url()
    clean_host = raw_url.replace("/v1", "").rstrip("/")
    return OpenAICompatibleEmbedder(
        base_url=f"{clean_host}/v1",
        model=env_str("EMBED_MODEL"),
        dim=env_int("EMBED_DIM"),
    )
