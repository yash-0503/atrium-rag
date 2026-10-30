"""
rerank_adapter.py

Cross-encoder reranking for RAG retrieval. Optional, opt-in — the
rest of the pipeline runs without it.

Why a reranker helps: a bi-encoder embedder (bge-large-en, nomic-embed,
OpenAI text-embedding-3) encodes the query and the document INDEPENDENTLY,
then compares them by cosine similarity. This is fast but loses
fine-grained query-document interaction: a chunk that talks about
"the sparrow's nest" might score lower than a chunk that mentions
"eagle" when the query is "sparrow", because the embedder's pooling
dilutes the signal.

A cross-encoder reranker reads the query AND document together as a
single (query, document) pair and outputs a relevance score. It's
much more accurate (~5-15 point recall@k gain in published benchmarks
on BEIR, TREC, etc.) but ~100x slower because every candidate needs
a forward pass. Standard pattern: vector search top-50, rerank to
top-5.

Adapter shape mirrors `OcrAdapterProtocol` / `VisionCaptionAdapter`:
a Protocol with the methods we need, plus concrete implementations
behind try/except import so a missing optional dep never crashes
the server. `LocalBgeReranker` requires sentence-transformers (in
`atrium-rag[rerank]`); `CohereReranker` requires the `cohere`
SDK + an API key. Auto-attach in main.py tries local first, then
Cohere if the env says so.

The Protocol is intentionally minimal so users can write their own
adapter (e.g. Jina, mixedbread, custom fine-tuned CrossEncoder).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol

from atrium.config import env_bool, env_int, env_str

log = logging.getLogger("ingest.rerank")


# Default local model. bge-reranker-base is 568M params / ~280MB on
# disk — small enough to ship as a "just works" opt-in default, large
# enough to materially move recall on real corpora. bge-reranker-large
# is ~1.3GB and noticeably more accurate; users with the disk budget
# can override via RERANK_MODEL=BAAI/bge-reranker-large.
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

# Default candidate budget: vector search returns top-50, reranker
# filters to top-K. 50 is the sweet spot on BEIR — past that the
# reranker starts getting too many obvious non-matches, and the cost
# grows linearly with N candidates.
DEFAULT_RERANK_INPUT_TOP_N = 50


class RerankCandidate:
    """A single retrieved chunk that the reranker re-scores. Kept as
    a typed dict-friendly container so callers can use either dicts
    (e.g. raw Qdrant points) or the dataclass — adapters accept
    both via duck typing.

    Fields:
      - id: stable per-chunk id (e.g. Qdrant point id, content_hash).
        Preserved through rerank so the caller can resolve results
        back to their source of truth.
      - text: the chunk text. Rerankers consume this; the rest of the
        payload (page_number, section_path, etc.) is opaque to the
        reranker and just gets passed through.
      - score: the ORIGINAL retrieval score (e.g. cosine similarity
        from vector search, or 1 - cosine_distance). Preserved so
        callers can choose to display it instead of the rerank score
        if they want.
      - payload: opaque dict of any other fields the caller wants
        threaded through. Carried untouched.
    """

    __slots__ = ("id", "text", "score", "payload")

    def __init__(self, id: str, text: str, score: float, payload: Optional[dict] = None):
        self.id = id
        self.text = text
        self.score = score
        self.payload = payload or {}


class RerankerAdapter(Protocol):
    """Rerank `candidates` by relevance to `query`, return top `top_k`
    in descending relevance order. Implementations must preserve the
    `id`, `text`, `score`, and `payload` fields on each returned
    candidate; only the order and (optionally) the `score` field
    change.

    `health()` is called at startup and on the `/health` endpoint so
    operators can see "reranker: ok / reranker: down" in the same
    probe that already reports DB and embedder status."""

    def rerank(self, query: str, candidates: list[RerankCandidate],
                top_k: int) -> list[RerankCandidate]: ...
    def health(self) -> bool: ...


class LocalBgeReranker:
    """Local CPU/GPU cross-encoder reranker via sentence-transformers.

    First call to `rerank()` loads the model. Loading time is ~5-30s
    on a typical CPU box (model download + JIT), so we lazy-load and
    cache. Subsequent calls reuse the loaded model. The CrossEncoder
    class from sentence-transformers is thread-safe for inference
    (it batches the inputs internally), so this adapter is safe to
    share across FastAPI request threads.

    Score normalization: CrossEncoder returns logits (unbounded). We
    squash them to [0, 1] with a sigmoid so the returned `score` is
    always in the same range as the cosine-similarity scores callers
    are used to from vector search. Without this, the JSON response
    shows scores in the [-10, +20] range which looks broken next to
    vector scores in [0, 1].
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL,
                 device: Optional[str] = None,
                 max_length: int = 512):
        self.model_name = model_name
        # Device: "cuda" if a GPU is available, else "cpu". Override
        # with RERANK_DEVICE for ops who want to force CPU (some
        # shared GPU hosts are slower for tiny reranker models than
        # the kernel-launch overhead would suggest).
        if device is None:
            device = env_str("RERANK_DEVICE")
        if not device:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        # CrossEncoder has a hard max length (token count) the
        # underlying transformer accepts. Truncation is on by default
        # but we surface it as a knob — operators with long legal/
        # medical chunks may want to raise this and accept the
        # memory cost.
        self.max_length = max_length
        self._model = None
        self._load_error: Optional[Exception] = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        try:
            from sentence_transformers import CrossEncoder
            log.info("Loading rerank model %s on %s...", self.model_name, self.device)
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            log.info("Rerank model loaded")
        except Exception as exc:
            self._load_error = exc
            raise

    def rerank(self, query: str, candidates: list[RerankCandidate],
                top_k: int) -> list[RerankCandidate]:
        if not candidates:
            return []
        if top_k <= 0:
            return []
        top_k = min(top_k, len(candidates))

        self._ensure_loaded()
        # CrossEncoder.predict accepts a list of (q, d) pairs and
        # returns a list of floats — one score per pair, in input
        # order. We don't need to do batching ourselves; the
        # CrossEncoder handles it internally.
        pairs = [(query, c.text) for c in candidates]
        try:
            raw_scores = self._model.predict(
                pairs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            # If inference fails (OOM, truncated model file, etc.),
            # fall back to the input order so the user still gets
            # results. Log loudly — silent fallback would mask a
            # real ops problem.
            log.warning("Rerank inference failed (%s: %s); "
                        "falling back to original order", type(exc).__name__, exc)
            return candidates[:top_k]

        # Squash logits to [0, 1] so the score field is comparable
        # to vector-search scores. Sigmoid is the standard cross-
        # encoder normalization (Cohere, Jina, bge docs all use it).
        try:
            import numpy as np
            scores = 1.0 / (1.0 + np.exp(-raw_scores))
            score_list = scores.tolist()
        except ImportError:
            # numpy is a hard dep of sentence-transformers so this
            # branch is essentially unreachable. Defensive only.
            score_list = [float(s) for s in raw_scores]

        scored = list(zip(candidates, score_list))
        # Sort desc by rerank score; stable sort preserves input order
        # on ties (important: the input order is the original
        # vector-search ranking, which is itself a useful signal).
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        return [
            RerankCandidate(
                id=c.id, text=c.text,
                score=float(s), payload=dict(c.payload),
            )
            for c, s in top
        ]

    def health(self) -> bool:
        if self._load_error is not None:
            return False
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False


class FlashRankReranker:
    """Lightweight ONNX-runtime cross-encoder reranker via flashrank.

    Trade-off vs `LocalBgeReranker`:
      - Tiny model footprint (~30 MB on disk, ~120 MB resident vs the
        ~568 MB / ~1.5 GB of BGE). Attractive in prod where the
        ~300 MB install footprint is a real budget constraint.
      - English-only by default. The default `ms-marco-MiniLM-L-6-v2`
        family loses substantial signal on CJK / Arabic / Cyrillic
        text. Use BGE-v2-m3 or Cohere for multilingual corpora; use
        FlashRank when the corpus is dominantly English.
      - Quality: roughly on par with BM25-style "closer to lexical
        match" reranking — markedly better than no rerank, markedly
        worse than bge-reranker-v2. Useful as a free-tier / OSS-tier
        default.

    Lazy-loaded: model is downloaded and `InferenceSession` compiled
    on the first rerank() call, then cached. First-call cost is
    ~5-15s on a warm cache, ~30s cold (model download); subsequent
    calls are pure CPU inference (~30ms per candidate on M-series).
    """

    DEFAULT_MODEL = "ms-marco-MiniLM-L-12-v2"

    def __init__(self, model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 max_length: int = 512,
                 log_level: str = "INFO"):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.cache_dir = cache_dir or (
            env_str("FLASHRANK_CACHE_DIR") or "/tmp/flashrank"
        )
        self.max_length = max_length
        self.log_level = log_level
        self._ranker: Optional[Any] = None
        self._load_error: Optional[Exception] = None

    def _ensure_loaded(self):
        if self._ranker is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        try:
            from flashrank import Ranker
            log.info("Loading FlashRank reranker %s into %s ...",
                     self.model_name, self.cache_dir)
            self._ranker = Ranker(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                max_length=self.max_length,
                log_level=self.log_level,
            )
        except Exception as exc:
            self._load_error = exc
            log.warning("FlashRank model load failed (%s: %s)",
                        type(exc).__name__, exc)
            raise

    def rerank(self, query: str, candidates: list[RerankCandidate],
                top_k: int) -> list[RerankCandidate]:
        if not candidates:
            return []
        top_k = min(top_k, len(candidates))
        try:
            self._ensure_loaded()
            from flashrank import RerankRequest
        except Exception:
            # Model failed to load — fall back to original order
            # so the caller still gets a result. Same fail-open
            # contract as the BGE adapter.
            return candidates[:top_k]

        # FlashRank mutates passage dicts in-place (assigns 'score'
        # then sorts by it). We pass {'id', 'text'} so we can map the
        # reranker's output back to the original RerankCandidate list.
        passages = [{"id": c.id, "text": c.text or ""} for c in candidates]
        try:
            ranked_passages = self._ranker.rerank(
                RerankRequest(query=query, passages=passages),
            )
        except Exception as exc:
            log.warning("FlashRank rerank call failed (%s: %s); "
                        "falling back to original order",
                        type(exc).__name__, exc)
            return candidates[:top_k]

        # Re-key candidates by id for O(1) lookup, then emit RerankCandidate
        # objects preserving the original payload.
        cand_by_id = {c.id: c for c in candidates}
        out: list[RerankCandidate] = []
        for rp in ranked_passages[:top_k]:
            cid = rp.get("id")
            orig = cand_by_id.get(cid)
            if orig is None:
                # Defensive: passage id present in rerank output but
                # missing from the input. Skip rather than fabricate.
                continue
            out.append(RerankCandidate(
                id=orig.id,
                text=orig.text,
                score=float(rp.get("score", 0.0)),
                payload=dict(orig.payload),
            ))
        return out

    def health(self) -> bool:
        if self._load_error is not None:
            return False
        try:
            self._ensure_loaded()
            return True
        except Exception:
            return False


class CohereReranker:
    """Cohere hosted reranker (rerank-english-v3.0 by default). Used
    when the operator sets `RERANK_PROVIDER=cohere` and provides
    `COHERE_API_KEY`. Latency is typically 100-300ms for 50 candidates
    on the v3 model — well within chat-RAG budgets, no local model
    weights to ship.

    Uses the official `cohere` SDK when available; falls back to a
    raw `httpx` POST if the SDK isn't installed (lighter dep tree).
    SDK is preferred because it handles retries, auth refresh, and
    rate-limit parsing — but the raw path keeps the door open for
    minimal-dep deployments that only want rerank and not the rest
    of the Cohere surface.
    """

    DEFAULT_MODEL = "rerank-english-v3.0"

    def __init__(self, api_key: str, model: Optional[str] = None):
        if not api_key:
            raise ValueError("CohereReranker requires a non-empty api_key")
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self._client = None
        self._sdk_load_error: Optional[Exception] = None
        try:
            import cohere
            self._client = cohere.Client(api_key=api_key)
        except Exception as exc:
            self._sdk_load_error = exc
            log.info("Cohere SDK not available (%s); using raw HTTP for rerank",
                     type(exc).__name__)

    def rerank(self, query: str, candidates: list[RerankCandidate],
                top_k: int) -> list[RerankCandidate]:
        if not candidates:
            return []
        top_k = min(top_k, len(candidates))

        if self._client is not None:
            return self._rerank_sdk(query, candidates, top_k)
        return self._rerank_http(query, candidates, top_k)

    def _rerank_sdk(self, query, candidates, top_k) -> list[RerankCandidate]:
        try:
            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=[c.text for c in candidates],
                top_n=top_k,
                return_documents=False,
            )
        except Exception as exc:
            log.warning("Cohere rerank failed (%s: %s); "
                        "falling back to original order", type(exc).__name__, exc)
            return candidates[:top_k]
        out = []
        for hit in response.results:
            idx = hit.index
            if 0 <= idx < len(candidates):
                c = candidates[idx]
                out.append(RerankCandidate(
                    id=c.id, text=c.text,
                    score=float(hit.relevance_score),
                    payload=dict(c.payload),
                ))
        return out

    def _rerank_http(self, query, candidates, top_k) -> list[RerankCandidate]:
        """Raw httpx path. The Cohere rerank endpoint is a single POST
        that takes the query, the documents, and the model name; no
        streaming, no auth refresh needed. The endpoint is stable
        across SDK versions, so the raw path is safe to use."""
        import httpx
        try:
            response = httpx.post(
                "https://api.cohere.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": [c.text for c in candidates],
                    "top_n": top_k,
                    "return_documents": False,
                },
                timeout=30.0,
            )
            response.raise_for_status()
        except Exception as exc:
            log.warning("Cohere HTTP rerank failed (%s: %s); "
                        "falling back to original order", type(exc).__name__, exc)
            return candidates[:top_k]

        data = response.json()
        out = []
        for hit in data.get("results", []):
            idx = hit.get("index")
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                c = candidates[idx]
                out.append(RerankCandidate(
                    id=c.id, text=c.text,
                    score=float(hit.get("relevance_score", 0.0)),
                    payload=dict(c.payload),
                ))
        return out

    def health(self) -> bool:
        if not self.api_key:
            return False
        # Lightweight probe — a 1-doc rerank call. The Cohere API
        # accepts an empty doc list (returns empty results) but
        # doesn't reliably validate auth on an empty body, so we
        # send a tiny real one. ~50-200ms on the v3 model.
        try:
            import httpx
            r = httpx.post(
                "https://api.cohere.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": "ping",
                    "documents": ["pong"],
                    "top_n": 1,
                },
                timeout=10.0,
            )
            return r.status_code == 200
        except Exception:
            return False


def build_reranker(force: bool = False) -> Optional[RerankerAdapter]:
    """Factory used by main.py. Dispatches on `RERANK_BACKEND` env var:

      RERANK_BACKEND=cohere    → CohereReranker  (needs COHERE_API_KEY)
      RERANK_BACKEND=flashrank → FlashRankReranker (lightweight ONNX)
      RERANK_BACKEND=bge-local → LocalBgeReranker (default; needs
                                  sentence-transformers and
                                  RERANK_ENABLED=1 to opt in)

    Backward compat: when `RERANK_BACKEND` is unset, the legacy
    `RERANK_PROVIDER=cohere` knob still works. When neither is set,
    the bge-local path runs only if `RERANK_ENABLED=1`.

    Returning None on no-reranker is the opt-in story: the server
    boots and works fine without a reranker; search just returns
    the vector-search ranking.

    `force=True` bypasses the `RERANK_ENABLED` gate. It is how a
    per-request override ("rerank this one query") beats the
    deployment-wide default. The gate still applies to the
    Cohere/FlashRank backends — those need their own credentials
    and models, not a flag.
    """
    backend = env_str("RERANK_BACKEND").lower().strip()
    provider = env_str("RERANK_PROVIDER").lower().strip()

    # Cohere (SaaS) — explicit provider knob or RERANK_BACKEND=cohere.
    if backend == "cohere" or provider == "cohere":
        api_key = env_str("COHERE_API_KEY")
        if not api_key:
            log.warning("Cohere reranker requested but COHERE_API_KEY is "
                        "empty; reranker disabled")
            return None
        try:
            return CohereReranker(api_key=api_key,
                                  model=env_str("RERANK_MODEL") or None)
        except Exception as exc:
            log.warning("CohereReranker failed to initialize (%s: %s); "
                        "reranker disabled", type(exc).__name__, exc)
            return None

    # FlashRank (lightweight local ONNX).
    if backend == "flashrank":
        try:
            return FlashRankReranker(
                model_name=env_str("RERANK_MODEL") or None,
                cache_dir=env_str("FLASHRANK_CACHE_DIR") or None,
                max_length=env_int("RERANK_MAX_LENGTH"),
            )
        except Exception as exc:
            log.warning("FlashRankReranker failed to initialize (%s: %s); "
                        "reranker disabled", type(exc).__name__, exc)
            return None

    # Default path: local bge-reranker, opt-in via RERANK_ENABLED=1.
    # `force` lets a per-request override switch it on for one call
    # without editing the deployment's env.
    if not force and not env_bool("RERANK_ENABLED"):
        return None

    try:
        model = env_str("RERANK_MODEL") or DEFAULT_RERANK_MODEL
        return LocalBgeReranker(model_name=model)
    except Exception as exc:
        # The actual model load is lazy — this only catches the
        # constructor failure. A later `health()` call or first
        # `rerank()` call will surface a model-load error, which
        # we log in `_ensure_loaded`.
        log.info("LocalBgeReranker not available (%s: %s); reranker disabled",
                 type(exc).__name__, exc)
        return None
