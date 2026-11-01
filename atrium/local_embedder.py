"""The in-process embedder: ONNX, no server, no GPU.

The `local` extra installs `fastembed`, which runs a quantised ONNX
model in this process. The weights are downloaded on first use and
cached — they are never bundled into the package, because a model is
tens of megabytes and belongs on the machine, not in a wheel.

    pip install "atrium-rag[local]"

With this in place, `atrium init` can configure a working product
with nothing else running: no Ollama, no API key, no server.

The default model is `BAAI/bge-small-en-v1.5` — 384 dimensions, about
130 MB — because the point of the local path is that it starts fast.
`BAAI/bge-base-en-v1.5` (768) and `BAAI/bge-large-en-v1.5` (1024) are
also available, and the embedding dimension has to match `EMBED_DIM`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

__all__ = ["FastEmbedEmbedder", "DEFAULT_LOCAL_MODEL"]

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"

# The models fastembed ships that are a drop-in for the HTTP path.
SUGGESTED_MODELS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}


class FastEmbedEmbedder:
    """Embeddings from a local ONNX model.

    Implements the same shape as `OpenAICompatibleEmbedder`, so every
    caller — the API, the worker, the retrieval lab, the CLI — works
    with it unchanged.
    """

    name = "fastembed"

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, dim: Optional[int] = None) -> None:
        self.model_name = model or DEFAULT_LOCAL_MODEL
        self._model = None
        self._dim = int(dim) if dim else self._lookup_dim(self.model_name)

    # -- model ----------------------------------------------------------

    @staticmethod
    def _supported() -> List[dict]:
        from fastembed import TextEmbedding

        return list(TextEmbedding.list_supported_models())

    def _lookup_dim(self, model: str) -> int:
        """The model's dimension.

        The pinned table wins: a model we know the width of must not
        change width because a dependency was upgraded. The catalogue is
        only consulted for a model we have not pinned.
        """
        if model in SUGGESTED_MODELS:
            return SUGGESTED_MODELS[model]
        try:
            for entry in self._supported():
                if entry.get("model") == model:
                    return int(entry.get("dim", 0))
        except Exception:
            pass
        return 0

    def _get(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    # -- the embedder protocol -------------------------------------------

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        from atrium.errors import AtriumSetupError

        try:
            # fastembed returns a generator of arrays, which cannot be
            # indexed — it has to be materialised before it can be
            # checked or converted.
            vectors = list(self._get().embed(list(texts)))
            self._check_dimension(vectors)
        except AtriumSetupError:
            raise
        except Exception as exc:
            raise AtriumSetupError(
                f"The local embedder could not load {self.model_name}: {exc}",
                hints=(
                    "Install the extra: `pip install \"atrium-rag[local]\"`",
                    "Or use an OpenAI-compatible endpoint instead: `atrium init`",
                ),
            ) from exc
        return [[float(v) for v in vector] for vector in vectors]

    def _check_dimension(self, vectors: Sequence[Sequence[float]]) -> None:
        from atrium.errors import AtriumSetupError

        """The model's output must match the pinned dimension.

        The dimension is pinned per model so that a fastembed release
        which changes a model's output cannot silently change the
        width of the vectors in the store — which the store would then
        refuse, with a message about the wrong thing.
        """
        if not vectors:
            return
        expected = self.dimension
        actual = len(vectors[0])
        if expected and actual != expected:
            raise AtriumSetupError(
                f"{self.model_name} returned {actual} dimensions, but "
                f"{expected} is pinned for it.",
                hints=(
                    "Pin a different model, or update the dimension in "
                    "atrium/local_embedder.py and in EMBED_DIM",
                ),
            )

    @property
    def dimension(self) -> int:
        if self._dim:
            return self._dim
        return self._lookup_dim(self.model_name)

    def health(self) -> bool:
        return self.probe()[0]

    def probe(self) -> tuple:
        """(available, detail)."""
        try:
            import fastembed  # noqa: F401
        except ImportError:
            return False, "the `local` extra is not installed"
        try:
            models = {entry.get("model") for entry in self._supported()}
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if self.model_name not in models:
            return False, f"{self.model_name} is not a supported model"
        return True, f"{self.model_name} ({self.dimension}-d), downloaded on first use"
