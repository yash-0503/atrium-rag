import asyncio
import base64
import logging
import os
import sys
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Import the existing embedder and DB adapters
from atrium.adapters import build_store
from atrium.config import data_path, env_bool, env_float, env_int, env_str
from atrium.embedder_config import build_embedder, default_base_url
from atrium.errors import AtriumSetupError
from atrium.job_store import JobStore
from atrium.pipeline import run_ingestion
from atrium.security import (
    check_content_length,
    check_mime,
    check_payload_size,
    log_redacted_traceback,
    max_upload_bytes,
    redacted_traceback,
    require_api_key_if_configured,
)
from atrium.universal_parser import UniversalParser

# Import the new modules
from atrium.vision_adapters import OllamaVisionAdapter

# numpy is only needed by the /retrieval_lab endpoints (they do their own
# cosine math on an in-memory matrix). Importing it eagerly costs ~40 ms
# of cold start for every `atrium serve` / `atrium ingest` that never
# touches the lab, so it is deferred to the handlers that use it and the
# one dataclass annotation is a forward reference.
if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Idempotent — uvicorn may have already configured the root.
logging.basicConfig(
    level=env_str("LOG_LEVEL").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("ingest.main")

# MIME detection lives in `atrium.mime` — a leaf module with no web
# framework in it, so the CLI can sniff a file without importing
# FastAPI. The private aliases below keep the historical
# `atrium.main._detect_mime` / `_match_signature` surface working.
from atrium.mime import (  # noqa: E402
    EXT_TO_MIME as _EXT_TO_MIME,
    SIGNATURES as _SIGNATURES,
    detect_mime as _detect_mime,
    looks_like_text as _looks_like_text,
    match_signature as _match_signature,
    mime_conflict as _mime_conflict,
    mime_from_extension as _mime_from_extension,
)

# These names are re-exported for backward compatibility: the private
# `atrium.main._detect_mime` / `_match_signature` surface predates the
# atrium.mime module, and tests import them from here.
_REEXPORTED_FOR_COMPAT = (
    _EXT_TO_MIME, _SIGNATURES, _looks_like_text, _mime_from_extension,
)


def _upload_filename(file: UploadFile) -> str:
    """The client's filename, or a safe placeholder.

    `UploadFile.filename` is `str | None`: a client that posts a
    multipart part without a `filename=` attribute gets None, which
    would otherwise flow into the corpus metadata as the literal
    string "None" and into the MIME detector as a missing name.
    """
    name = (file.filename or "").strip()
    return name or "upload.bin"


app = FastAPI(title="Atrium")
job_store = JobStore(db_path=env_str("JOB_DB_PATH"))
# Per-doc concurrency cap for batch ingest. BackgroundTasks fires one
# task per file and would otherwise allow unbounded parallelism, which
# on a self-hosted single-host deployment can starve Qdrant/Ollama.
INGEST_CONCURRENCY = env_int("INGEST_CONCURRENCY")
_ingest_semaphore = asyncio.Semaphore(INGEST_CONCURRENCY)

# Config
DB_TYPE = env_str("VECTOR_DB")

# The vector-DB adapter is built on first use rather than at import
# time. `qdrant_client` costs ~290 ms to import and `psycopg_pool`
# ~65 ms, and the PgVector pool opens real connections when it is
# constructed — so a process that imports `atrium.main` but never
# talks to the vector store (the test suite, `atrium doctor`,
# `atrium demo`) should not pay for either. The adapter is cached, so
# the cost is paid exactly once, on the first request that needs it.
_db: Optional[Any] = None




def _explain(exc: Exception) -> str:
    """An exception's message, with its ways out when it has them."""
    render = getattr(exc, "render", None)
    if callable(render):
        return render()
    return str(exc)



def _ingest_failure(exc: Exception) -> HTTPException:
    """Turn an ingest failure into the right status and the right message.

    A failed ingest must never answer 200. The status says what kind of
    failure it was; the message says what to do about it.

    * 422 — the document could not be parsed. The user's file is the
      problem, and the message says why.
    * 503 — a dependency is not there (the embedder, the store). The
      user's document may be fine; the deployment is not.
    * 500 — anything else. The message is redacted, so an internal
      path never reaches the client; the full traceback is in the log.
    """
    if isinstance(exc, AtriumSetupError):
        return HTTPException(503, _explain(exc))
    if isinstance(exc, ValueError):
        # A parse failure: the message is written for the person who
        # uploaded the file, so it is passed through as it is.
        return HTTPException(422, str(exc))
    log_redacted_traceback(exc, where="ingest")
    return HTTPException(500, f"ingestion failed: {redacted_traceback(exc)}")


def _abort_startup(message: str) -> None:
    """Print the reason and leave, without a stack trace.

    `os._exit` is deliberate. Uvicorn reports a failed startup as an
    exception, and a traceback for "the database is not running" buries
    the one sentence the operator needs. Nothing has started yet, so
    there is nothing to clean up.
    """
    print(message, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(1)


def _store_unreachable_message(exc: Exception) -> str:
    """What to say when the vector store is not there.

    Two ways out, always: start the store, or use the local one that
    needs nothing running.
    """
    store = DB_TYPE or "sqlite"
    if store == "qdrant":
        detail = (
            f"Qdrant is not reachable at {env_str('QDRANT_HOST')}:"
            f"{env_int('QDRANT_PORT')}."
        )
        hints = (
            "Start it: `docker compose --profile qdrant up -d qdrant`",
            "Or use the local store, which needs no server: "
            "set VECTOR_DB=sqlite in .env",
        )
    elif store == "pgvector":
        detail = "The Postgres server is not reachable."
        hints = (
            "Start Postgres, or point PG_CONN at it",
            "Or use the local store: set VECTOR_DB=sqlite in .env",
        )
    else:
        detail = f"The local store could not be opened: {exc}"
        hints = (
            "Check that LOCAL_DB_PATH is writable",
            "Or pick another store: `atrium init`",
        )
    return "\n".join([detail, "", *(f"  {hint}" for hint in hints)])


def _build_db():
    """Construct the vector-DB adapter (cached in `_db`)."""
    global _db
    if _db is None:
        _db = build_store()
    return _db


def _get_db():
    """Return the vector-DB adapter to use.

    A module-level `db` attribute wins if one is present — tests
    replace it wholesale with `monkeypatch.setattr("atrium.main.db",
    fake)`, and that replacement must be what the handlers talk to.
    Otherwise the lazily-built adapter is returned.
    """
    override = globals().get("db")
    if override is not None:
        return override
    return _build_db()


def __getattr__(name: str):
    """Module-level lazy attribute (PEP 562).

    `atrium.main.db` resolves to the vector-DB adapter on first access
    rather than at import time, so importing this module does not pull
    in `qdrant_client` / `psycopg_pool`. A plain `setattr(module,
    "db", ...)` — which is what monkeypatch does — shadows this hook
    entirely, so the existing test seam keeps working.
    """
    if name == "db":
        return _build_db()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Vision adapter — wires OllamaVisionAdapter into UniversalParser so
# image-only pages / embedded diagrams get captioned on parse. ON by
# default (this is the current behavior); set VISION_ENABLED=0 to opt
# out without code change (useful for A/B testing or when the vision
# model is down). Vision timeout is env-driven because the first call
# cold-loads the model — 60s default is too tight for minicpm-v4.6:1b
# on slow hardware (9 timeouts in the 2026-10-17 Test B run).
# (2026-10-17: vision was already unconditionally wired; this change
# adds the toggle + env-driven timeout, doesn't alter default behavior.)
raw_url = env_str("EMBED_BASE_URL")
# Strip /v1 if it exists so we have a clean base host
clean_host = raw_url.replace("/v1", "").rstrip("/")
vision = None
if env_bool("VISION_ENABLED"):
    vision = OllamaVisionAdapter(
        base_url=clean_host,
        model=env_str("VISION_MODEL"),
        timeout_seconds=env_float("VISION_TIMEOUT_SECONDS"),
    )
    log.info("Vision adapter enabled (model=%s, timeout=%ss)",
             env_str("VISION_MODEL"),
             env_float("VISION_TIMEOUT_SECONDS"))
else:
    log.info("Vision disabled (set VISION_ENABLED=1 to enable)")

# Opt-in OCR: only attempt to instantiate the Tesseract adapter when the
# user has explicitly opted in (OCR_ENABLED=1) AND has installed the
# the `ocr` extra + the system Tesseract binary. Tries the
# import inside a try/except so a missing pytesseract doesn't prevent
# the rest of the server from booting — the failure is logged once
# at startup and OCR fallback is silently unavailable thereafter.
#
# Default is OFF, matching the CLI wizard, the worker, and the `ocr`
# extra (which is not installed by default). It used to default to ON
# here, which meant every boot attempted `import pytesseract` — pulling
# in pandas for ~300 ms — and then logged a warning that the adapter
# could not be constructed anyway.
ocr_adapter = None
if env_bool("OCR_ENABLED"):
    try:
        from atrium.ocr_adapter import build_ocr_adapter
        ocr_adapter = build_ocr_adapter(
            lang=env_str("OCR_LANG"),
            tesseract_cmd=env_str("TESSERACT_CMD") or None,
        )
        log.info("OCR adapter enabled (provider=%s, lang=%s)",
                 env_str("OCR_PROVIDER") or "tesseract", env_str("OCR_LANG"))
    except Exception as exc:
        log.warning(
            "OCR_ENABLED=1 but the OCR adapter failed to initialize "
            "(%s: %s). OCR will not be used as a fallback. "
            "Install \"atrium-rag[ocr]\" (and the Tesseract binary for "
            "OCR_PROVIDER=tesseract) to enable it.",
            type(exc).__name__, exc,
        )
        ocr_adapter = None
else:
    log.info("OCR disabled (set OCR_ENABLED=1 + install atrium-rag[ocr] to enable)")

parser = UniversalParser(vision_adapter=vision, ocr_adapter=ocr_adapter)

# Embedder: shared configuration with the worker. See
# atrium/embedder_config.py for the default model + dimension. Both the
# HTTP service and the worker construct the embedder through
# `build_embedder()` so they cannot drift on model or dim.
try:
    embedder = build_embedder(base_url=default_base_url())
except AtriumSetupError as exc:
    # No embedding endpoint is configured. That is a setup problem, not
    # a crash: the process must still be able to start far enough to
    # say so, so the message is kept and the lifespan reports it.
    embedder = None
    EMBEDDER_SETUP_ERROR = exc
    log.error("%s", exc.render())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: verify the vector collection's dim matches
    the embedder's dim on startup. The Qdrant adapter raises
    ValueError on dim mismatch; the PgVector adapter does the same
    (see atrium/adapters.py). A misconfigured collection that disagrees
    with the embedder's dim surfaces here, not on the first ingest.

    `db.ensure_collection` is idempotent — it no-ops when the
    collection already exists with the right dim and creates it
    when missing. The lifespan context runs at app startup, AFTER
    module import, so tests that import `atrium.main` (e.g.
    `tests/test_v0_suite.py`) don't trigger the network call.
    """
    if embedder is None:
        # The embedding endpoint is not configured. Say exactly what to
        # do and stop, instead of failing later with a traceback.
        _abort_startup(EMBEDDER_SETUP_ERROR.render())

    # The embedding endpoint is probed, not required. A service that
    # refuses to start because the embedder is briefly down is worse
    # than one that starts, says so, and answers with the reason.
    probe = getattr(embedder, "probe", None)
    if callable(probe):
        try:
            reachable, detail = probe()
        except Exception as exc:
            # A probe that misbehaves is not a reason to refuse to
            # start; the service would answer with the real error.
            reachable, detail = False, f"{type(exc).__name__}: {exc}"
        if not reachable:
            log.warning(
                "the embedding endpoint is not answering (%s). The service "
                "is up, but /ingest and /search will fail until it is: "
                "run `atrium init` to point at a reachable one.",
                detail,
            )
    log.info(
        "job store: %s", data_path("JOB_DB_PATH", "jobs.db"),
    )
    log.info(
        "vector store: %s (%s)", DB_TYPE, data_path("LOCAL_DB_PATH", "atrium_local.db"),
    )
    try:
        _get_db().ensure_collection("documents", embedder.dimension)
        log.info(
            "ensured collection 'documents' with dim=%d (embedder=%s)",
            embedder.dimension, type(embedder).__name__,
        )
    except ValueError:
        # A dimension disagreement. The adapter's own message already
        # names both numbers ("collection 'documents' exists with dim
        # 768, but config requests 1024"), so don't overwrite it —
        # just add the two ways out.
        log.error(
            "ensure_collection failed at startup: the vector store's "
            "dimension disagrees with the embedder's (dim=%d). Either "
            "re-create the collection with that dim, or point EMBED_DIM "
            "at the dim the collection already has.",
            embedder.dimension,
        )
        raise
    except Exception as exc:
        # Anything else is a connection-level failure: DNS, refused
        # connection, TLS, auth. The old message blamed the dimension
        # for *every* startup failure, so an unresolvable host name
        # was reported as a dim mismatch — advice that could not fix
        # the actual problem.
        log.error(
            "could not reach the vector store at startup: %s: %s. "
            "Check that the database is running and that VECTOR_DB / "
            "QDRANT_HOST / QDRANT_PORT / PG_CONN point at it.",
            type(exc).__name__, exc,
        )
        _abort_startup(_store_unreachable_message(exc))
    try:
        yield
    finally:
        # The store holds a connection pool — for pgvector, worker
        # threads and open connections. Nothing closed it, so a reload
        # or a shutdown left them behind.
        try:
            _get_db().close()
        except Exception as exc:  # pragma: no cover - shutdown path
            log.warning("could not close the vector store: %s", exc)


app = FastAPI(title="Atrium", lifespan=lifespan)


@app.exception_handler(AtriumSetupError)
async def _setup_error_handler(request: Request, exc: AtriumSetupError):
    """A setup problem reaches the client as a sentence.

    The status is 503: the service is fine, the dependency is not.
    """
    return JSONResponse(status_code=503, content={"detail": exc.render()})


# Bearer-token gate. Skipped when API_KEY is empty (dev default).
# See atrium.security for the matching + extraction logic.
from starlette.middleware.base import BaseHTTPMiddleware


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Per-request bearer-token check. Wraps `require_api_key_if_configured`
    so the FastAPI Depends machinery isn't needed in every handler."""

    async def dispatch(self, request, call_next):
        try:
            require_api_key_if_configured(request)
        except HTTPException as exc:
            # Use JSONResponse so the bearer-rejection contract stays
            # uniform with the rest of the API.
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers or {},
            )
        return await call_next(request)


app.add_middleware(APIKeyMiddleware)


# CORS. Off by default: an empty CORS_ALLOW_ORIGINS means no
# cross-origin headers are emitted at all, which is the right default
# for an API that may sit behind bearer auth. Set it to a
# comma-separated list of origins (e.g. "https://app.example.com") to
# let a browser client call it.
_cors_origins = [o.strip() for o in env_str("CORS_ALLOW_ORIGINS").split(",") if o.strip()]
if _cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    log.info("CORS enabled for origins: %s", ", ".join(_cors_origins))


@app.get("/health")
def health():
    """Liveness + dependency probe. Returns 200 if the API is up and the
    vector DB + embedder are reachable; 503 otherwise (used by compose
    healthcheck and any external uptime monitor)."""
    db_ok = bool(_get_db().health())
    embedder_ok = bool(embedder.health())
    status_code = 200 if (db_ok and embedder_ok) else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if status_code == 200 else "degraded",
            "db": db_ok,
            "embedder": embedder_ok,
            "vector_db_type": DB_TYPE,
        },
    )

class SearchRequest(BaseModel):
    """Body of ``POST /search``.

    Every field except ``query`` is optional. The defaults come from
    the server's configuration, so a client that only sends a query
    gets the same behaviour as the deployment was tuned for.
    """

    query: str = Field(..., description="Natural-language query.")
    top_k: Optional[int] = Field(
        None,
        description=(
            "How many results to return. Defaults to DEFAULT_TOP_K (8). "
            "The topk sweep on the 19-fixture benchmark showed k=8 "
            "captures 5 more queries than k=5 for 1.6x the scoring cost."
        ),
    )
    rerank: Optional[bool] = Field(
        None,
        description=(
            "Force the cross-encoder rerank stage on (true) or off (false). "
            "Omit to follow the server's RERANK_ENABLED setting. Turning "
            "it on raises the ranking quality on most corpora but "
            "regresses on a few (the reranker is trained on web QA, not "
            "on forms, slides, math or OCR'd scans), so it is opt-in."
        ),
    )
    candidate_k: Optional[int] = Field(
        None,
        description=(
            "How many candidates the reranker scores before truncation. "
            "Default 30. Only used when the rerank stage is on."
        ),
    )
    score_threshold: Optional[float] = Field(
        None,
        description="Minimum cosine similarity for a chunk to be returned.",
    )


@app.post("/search")
async def search(body: SearchRequest):
    """Search the ingested corpus.

    This is the corpus-level retrieval endpoint — the HTTP counterpart
    of ``atrium search``. It embeds the query and then runs the same
    ``atrium.retrieval.retrieve_and_rerank`` pipeline the benchmark
    harness uses: wide candidate fetch → optional cross-encoder rerank
    → parent dedup → parent resolution.

    ``/retrieval_lab/search`` is deliberately a *different* thing: it
    scores against an in-RAM cosine matrix for a single uploaded
    document, with no vector store, no BM25 and no reranker. Use this
    endpoint to query what you have actually ingested.

    Per-request overrides beat the server's configuration, which beats
    the built-in defaults:

    * ``top_k`` beats ``DEFAULT_TOP_K``
    * ``rerank`` beats ``RERANK_ENABLED``
    """
    from atrium.retrieval import retrieve_and_rerank

    query = (body.query or "").strip()
    if not query:
        raise HTTPException(422, "query is empty")

    top_k = body.top_k if body.top_k is not None else env_int("DEFAULT_TOP_K")
    if top_k <= 0 or top_k > 100:
        raise HTTPException(422, "top_k must be between 1 and 100")
    if body.candidate_k is not None and body.candidate_k <= 0:
        raise HTTPException(422, "candidate_k must be positive")
    if body.score_threshold is not None and not -1.0 <= body.score_threshold <= 1.0:
        raise HTTPException(
            422, "score_threshold must be between -1.0 and 1.0 (cosine similarity)",
        )

    # Reranker. `body.rerank` is a tri-state: True forces it on, False
    # forces it off, None follows RERANK_ENABLED.
    reranker = None
    want_rerank = body.rerank if body.rerank is not None else env_bool("RERANK_ENABLED")
    if want_rerank:
        from atrium.rerank_adapter import build_reranker

        reranker = build_reranker(force=True)
        if reranker is None:
            if body.rerank is True:
                # The caller asked for reranking explicitly. Silently
                # returning unreranked results under a `rerank: true`
                # request would be a lie.
                raise HTTPException(
                    503,
                    "reranker requested but unavailable — install `atrium-rag[rerank]` "
                    "and set RERANK_ENABLED=1 (or RERANK_BACKEND=cohere with "
                    "COHERE_API_KEY)",
                )
            log.warning(
                "RERANK_ENABLED=1 but no reranker could be built; "
                "serving the vector ranking for this query",
            )

    started = time.monotonic()
    try:
        query_vector = (await asyncio.to_thread(embedder.embed, [query]))[0]
    except Exception as exc:
        log.warning("search embed failed: %s", exc)
        raise HTTPException(503, f"Embedder unavailable: {_explain(exc)}") from exc

    try:
        hits = await asyncio.to_thread(
            retrieve_and_rerank,
            _get_db(),
            query_vector,
            query,
            top_k,
            reranker=reranker,
            candidate_k=body.candidate_k,
            score_threshold=body.score_threshold,
        )
    except Exception as exc:
        log.warning("search retrieval failed: %s", exc)
        raise HTTPException(503, f"Retrieval failed: {exc}") from exc

    results = []
    for rank, hit in enumerate(hits, start=1):
        results.append({
            "rank": rank,
            "score": float(hit.get("score", 0.0)),
            "chunk_id": hit.get("chunk_id"),
            "doc_id": hit.get("doc_id", ""),
            "text": hit.get("text", ""),
            "page_number": hit.get("page_number"),
            "section_path": hit.get("section_path") or [],
            "element_type": hit.get("element_type", ""),
            "chunking_strategy": hit.get("chunking_strategy", ""),
            "is_parent": hit.get("is_parent", False),
            "parent_id": hit.get("parent_id"),
            "parent_text": hit.get("parent_text"),
            "parent_page_number": hit.get("parent_page_number"),
            "parent_section_path": hit.get("parent_section_path") or [],
        })

    # How many chunks the store holds. Without it, an empty result is
    # ambiguous: an empty corpus and a query that matched nothing look
    # the same, and "is my data even in there?" has no answer.
    try:
        corpus_chunks = _get_db().count_chunks()
    except Exception:
        corpus_chunks = None

    return {
        "query": query,
        "top_k": top_k,
        "rerank": reranker is not None,
        "count": len(results),
        "corpus_chunks": corpus_chunks,
        "took_ms": int((time.monotonic() - started) * 1000),
        "results": results,
    }


def _parse_chunking_overrides(
    chunking_mode: Optional[str],
    chunking_config: Optional[str],
):
    """Validate a per-request chunking override pair.

    Returns ``(mode, config)``. Raises ``HTTPException(422)`` naming
    the offending field. The validation happens *before* any parsing or
    embedding, so a bad mode is a rejected request rather than a
    failed background job the caller only learns about later.
    """
    import json

    from atrium.smart_chunker import resolve_chunk_config

    mode = (chunking_mode or "auto").strip() or "auto"
    overrides = None
    if chunking_config:
        try:
            overrides = json.loads(chunking_config)
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"chunking_config is not valid JSON: {exc}") from exc
        if not isinstance(overrides, dict):
            raise HTTPException(422, "chunking_config must be a JSON object")
    try:
        resolve_chunk_config(mode, overrides)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return mode, overrides


async def _run_one_ingestion(
    file_bytes: bytes,
    filename: str,
    doc_id: Optional[str],
    chunking_mode: str = "auto",
    chunking_config: Optional[dict] = None,
) -> dict:
    """One file's worth of ingest, gated by the concurrency semaphore.
    Runs run_ingestion in a thread so the event loop isn't blocked while
    PyMuPDF / Qdrant / Ollama do their blocking work."""
    # Belt-and-suspenders: the request handler checks Content-Length
    # before reading, but a client could lie about it. Verify the
    # actual byte count here against MAX_UPLOAD_BYTES (env-driven).
    check_payload_size(len(file_bytes))
    mime = _detect_mime(file_bytes, filename)
    check_mime(mime)
    job_id = str(uuid.uuid4())
    final_doc_id = doc_id or str(uuid.uuid4())
    job_store.create(job_id, final_doc_id)
    try:
        async with _ingest_semaphore:
            await asyncio.to_thread(
                run_ingestion, job_id, final_doc_id, file_bytes, filename, mime,
                parser, embedder, _get_db(), job_store,
                chunking_mode, chunking_config,
            )
    except HTTPException:
        raise
    except Exception as exc:
        # A dependency that is missing, a document that will not parse,
        # or anything else — each with the status that says which.
        raise _ingest_failure(exc) from exc
    record = job_store.get(job_id) or {}
    return {
        "job_id": job_id,
        "doc_id": final_doc_id,
        "filename": filename,
        # Pages the OCR engine could not read. Present only when there
        # were some, so the response shape is unchanged otherwise.
        **(
            {"ocr_failures": record["ocr_failures"]}
            if record.get("ocr_failures")
            else {}
        ),
    }


@app.post("/ingest")
async def ingest(
    request: Request,
    bg: BackgroundTasks,
    files: list[UploadFile] = File(..., description="One or more files to ingest"),
    doc_ids: Optional[str] = Form(None, description="Optional comma-separated doc_ids aligned with files order"),
    chunking_mode: str = Form(
        "auto",
        description=(
            "Chunking strategy for this request: auto, balanced, precise, "
            "context_rich, or custom (with chunking_config)."
        ),
    ),
    chunking_config: Optional[str] = Form(
        None,
        description=(
            "JSON object of ChunkConfig overrides, e.g. "
            "'{\"target_chars\": 900}'. Only used with chunking_mode=custom."
        ),
    ),
):
    """Ingest one or more files. Each file becomes its own job, capped at
    INGEST_CONCURRENCY in flight at once. Returns one record per file.

    A single file is the same call with one entry in `files`.

    `chunking_mode` / `chunking_config` override the server's chunking
    for this request only. The strategy that actually ran is recorded
    on every chunk as `chunking_strategy`, so a corpus with mixed
    strategies stays auditable.
    """
    # Reject oversize uploads BEFORE any bytes are read into memory.
    # check_content_length() inspects Content-Length; check_payload_size()
    # re-verifies after the read in case the client lied about length.
    check_content_length(request.headers.get("content-length"))
    if not files:
        raise HTTPException(422, "No files provided")
    mode, config = _parse_chunking_overrides(chunking_mode, chunking_config)
    requested_doc_ids: List[Optional[str]] = (
        [d.strip() for d in doc_ids.split(",")] if doc_ids else [None] * len(files)
    )
    if len(requested_doc_ids) < len(files):
        requested_doc_ids.extend([None] * (len(files) - len(requested_doc_ids)))

    # How many files one request may carry. Each file is read into
    # memory while it is ingested, so this is what bounds a request's
    # footprint.
    max_files = env_int("MAX_FILES_PER_REQUEST")
    if max_files and len(files) > max_files:
        raise HTTPException(
            422,
            f"this request carries {len(files)} files; the limit is "
            f"{max_files} (MAX_FILES_PER_REQUEST)",
        )

    # The size of each file is known from the multipart headers, so an
    # oversize file is rejected before its bytes are read into memory.
    limit = max_upload_bytes()
    for upload in files:
        size = getattr(upload, "size", None)
        if size is not None and limit and size > limit:
            raise HTTPException(
                413,
                f"upload exceeds MAX_UPLOAD_BYTES={limit} "
                f"(file {upload.filename!r} is {size} bytes)",
            )

    # Each file is read inside its own ingest, under the concurrency
    # semaphore. Reading them all up front meant the request held every
    # file in memory at once, so the memory a request needed grew with
    # the number of files in it rather than with the concurrency limit.
    async def ingest_one(upload, wanted_doc_id):
        raw = await upload.read()
        return await _run_one_ingestion(
            raw, _upload_filename(upload), wanted_doc_id, mode, config,
        )

    # `return_exceptions=True` because one bad file must not discard the
    # other files' results — their work has already happened, and the
    # client needs to know which of them landed.
    settled = await asyncio.gather(
        *[
            ingest_one(upload, wanted)
            for upload, wanted in zip(files, requested_doc_ids)
        ],
        return_exceptions=True,
    )

    results: List[dict] = []
    failures: List[HTTPException] = []
    for upload, outcome in zip(files, settled):
        name = _upload_filename(upload)
        if isinstance(outcome, BaseException):
            failure = (
                outcome
                if isinstance(outcome, HTTPException)
                else HTTPException(500, "ingestion failed")
            )
            failures.append(failure)
            results.append({
                "job_id": None,
                "doc_id": None,
                "filename": name,
                "error": failure.detail,
            })
        else:
            results.append(outcome)

    if not failures:
        return {"jobs": results}
    if len(failures) == len(results):
        # Every file failed: answer with the first file's status, so a
        # client that checks the status code sees a failure.
        raise failures[0]
    # Some succeeded and some did not. 207 says exactly that; a 200
    # would tell the client the whole batch landed.
    return JSONResponse(status_code=207, content={"jobs": results})


# --- Async enqueue -------------------------------------------------------
#
# POST /jobs is the queue contract: accept the document, hand it to a
# worker, return 202 + job_id immediately. /ingest (above) is the
# inline contract: parse, embed and store before answering. Both exist
# because they answer different questions — "I want this searchable
# now" versus "queue this for the worker pool".
#
# The endpoint was specified by tests/test_v0_suite.py and never
# implemented; the worker it was meant to feed has been calling
# JobStore.claim_next() / release_stale_claims(), which did not exist.
# Both are implemented now.

# A queued document is held in the job store as base64 in a SQLite row
# until a worker picks it up, so it gets a tighter cap than the inline
# path: the whole document sits in the job DB. 50 MB is the ceiling the
# v0 contract test asserts.
_ASYNC_JOB_MAX_BYTES = 50 * 1024 * 1024


@app.post("/jobs", status_code=202)
async def enqueue_job(
    request: Request,
    file: UploadFile = File(..., description="The document to ingest asynchronously."),
    doc_id: Optional[str] = Form(None, description="Optional document id."),
    chunking_mode: str = Form("auto"),
    chunking_config: Optional[str] = Form(None),
    ocr_override: Optional[bool] = Form(None),
):
    """Queue one document for asynchronous ingestion. Returns 202 + job_id.

    The document is written to the job store with the options that
    apply to it; a worker (`python -m atrium.worker`) claims queued
    jobs and runs the parse → chunk → embed → store pipeline. Poll
    `GET /jobs/{job_id}` for the status.

    Unlike `/ingest`, this endpoint does *not* process the document in
    the request: the caller gets its job_id back immediately, which is
    the point — a 300-page PDF takes minutes to parse, and holding an
    HTTP connection open for that is how a client times out and then
    uploads the same file three times.
    """
    check_content_length(request.headers.get("content-length"))
    mode, config = _parse_chunking_overrides(chunking_mode, chunking_config)

    file_bytes = await file.read()

    # The document is stored in the job DB until a worker claims it,
    # so it gets its own cap rather than the streaming upload cap.
    limit = min(env_int("MAX_UPLOAD_BYTES"), _ASYNC_JOB_MAX_BYTES)
    if len(file_bytes) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"upload exceeds the async job limit of {limit} bytes (Max 50MB)",
        )

    filename = _upload_filename(file)
    mime = _detect_mime(file_bytes, filename)
    declared = (file.content_type or "").strip()
    # The conflict check reads the *content*, not the filename. On a host
    # without libmagic, `_detect_mime` falls back to the extension table
    # — so a file named report.pdf that contains HTML would "detect" as
    # application/pdf and sail through. `_match_signature` looks at the
    # bytes alone, which is the only tier that can catch the lie.
    content_mime = _match_signature(file_bytes)
    if content_mime and _mime_conflict(declared, content_mime):
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported content: declared {declared!r} but the bytes "
                f"look like {content_mime!r}"
            ),
        )
    if _mime_conflict(declared, mime):
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported content: declared {declared!r} but the bytes "
                f"look like {mime!r}"
            ),
        )
    check_mime(mime)

    job_id = str(uuid.uuid4())
    final_doc_id = doc_id or str(uuid.uuid4())
    job_store.enqueue(
        job_id,
        final_doc_id,
        {
            "file_bytes_b64": base64.b64encode(file_bytes).decode("ascii"),
            "filename": filename,
            "mime": mime,
            "ocr_override": ocr_override,
            "chunking_mode": mode,
            "chunking_config": config,
        },
    )
    return {"job_id": job_id, "doc_id": final_doc_id, "filename": filename}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    record = job_store.get(job_id)
    if record is None: raise HTTPException(404)
    # Preserve the original wire shape (status, chunks_written, error) for SDK compat.
    return {k: record[k] for k in ("status", "chunks_written", "error") if record[k] is not None}


@app.post("/preview")
async def preview_chunks(
    request: Request,
    file: UploadFile,
    chunking_mode: str = Form("auto"),
    chunking_config: Optional[str] = Form(None),
):
    """Dry-run endpoint: Parses the document and returns chunks without embedding or storing them."""
    check_content_length(request.headers.get("content-length"))
    # Validate the override *outside* the try: the handler below turns
    # any exception into a 200 with an `error` body, and a rejected
    # request must not come back as a success.
    mode, config = _parse_chunking_overrides(chunking_mode, chunking_config)
    try:
        from atrium.smart_chunker import chunk_with_mode

        file_bytes = await file.read()
        check_payload_size(len(file_bytes))
        upload_name = _upload_filename(file)
        mime = _detect_mime(file_bytes, upload_name)
        check_mime(mime)

        # 1. Parse using the new universal parser
        raw_elements = parser.parse(file_bytes, upload_name, mime)

        # 2. Smart Chunk — 2026-10-17: was `chunk_elements` (legacy direct
        # call). Now routes through the dispatcher so /preview matches
        # /jobs + /ingest + /retrieval_lab/prepare. mode='auto' resolves
        # to PRESETS['precise'] after the P0 dispatcher fix.
        smart_chunks = chunk_with_mode(
            raw_elements, doc_id="preview", file_name=upload_name,
            mode=mode, config=config,
        )
        
        # 3. Format for the UI
        from atrium.smart_chunker import resolve_chunk_config

        strategy = resolve_chunk_config(mode, config).strategy
        # The response is the whole document's chunks. A 100 MB document
        # is thousands of them, and the caller asked to look at the
        # structure, not to download the corpus.
        total_chunks = len(smart_chunks)
        preview_cap = env_int("MAX_PREVIEW_CHUNKS")
        truncated = False
        if preview_cap and total_chunks > preview_cap:
            smart_chunks = smart_chunks[:preview_cap]
            truncated = True
        formatted_chunks = []
        for sc in smart_chunks:
            formatted_chunks.append({
                "element_type": sc.element_type,
                "text": sc.text,
                "embed_text": sc.embed_text,
                "section_path": sc.source.section_path,
                "source": {
                    "page_number": sc.source.page_number,
                    "page_numbers": getattr(sc.source, 'page_numbers', [sc.source.page_number]
                                            if sc.source.page_number else []),
                },
                "content_hash": sc.content_hash,
                "chunking_strategy": strategy,
            })

        return {
            "filename": upload_name,
            "total_chunks": total_chunks,
            "truncated": truncated,
            "chunking_mode": mode,
            "chunking_strategy": strategy,
            "chunks": formatted_chunks
        }
    except HTTPException:
        # A rejected request (size, MIME, chunking override) keeps its
        # own status. It must not come back as a success.
        raise
    except Exception as e:
        # The document could not be parsed or chunked. Log the full
        # traceback server-side for ops, but return a redacted message
        # to the caller so internal file paths / source lines don't leak.
        log_redacted_traceback(e, where="preview_chunks")
        raise HTTPException(422, f"could not parse or chunk the document: {redacted_traceback(e)}") from e

@app.get("/visualize", response_class=HTMLResponse)
def get_visualizer_ui():
    """Serves the gorgeous Tailwind UI directly from FastAPI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document Chunk Visualizer</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #1f2937; }
            ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #6b7280; }
        </style>
    </head>
    <body class="bg-gray-900 text-gray-100 min-h-screen p-8 font-sans">
        <div class="max-w-7xl mx-auto">
            <header class="mb-10 flex justify-between items-end border-b border-gray-800 pb-6">
                <div>
                    <h1 class="text-4xl font-bold text-white mb-2">Chunk Visualizer</h1>
                    <p class="text-gray-400">See exactly how the parser slices your layout before vector ingestion.</p>
                </div>
                <div class="flex items-center gap-4">
                    <input type="file" id="fileInput" class="hidden" />
                    <label for="fileInput" class="cursor-pointer bg-blue-600 hover:bg-blue-500 text-white px-6 py-3 rounded-lg font-semibold transition-colors shadow-lg">
                        Select Document
                    </label>
                </div>
            </header>

            <div id="loading" class="hidden py-20 text-center text-gray-400">
                <svg class="animate-spin h-10 w-10 mx-auto mb-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                <p class="text-lg">Parsing geometry and extracting text (dry run)...</p>
            </div>

            <div id="stats" class="hidden mb-8 flex gap-4 text-sm font-mono"></div>
            <div id="grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"></div>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const grid = document.getElementById('grid');
            const loading = document.getElementById('loading');
            const stats = document.getElementById('stats');

            const badgeColors = {
                'heading': 'bg-purple-900/50 text-purple-300 border-purple-700',
                'paragraph': 'bg-gray-800 text-gray-300 border-gray-600',
                'table': 'bg-blue-900/50 text-blue-300 border-blue-700',
                'image_caption': 'bg-emerald-900/50 text-emerald-300 border-emerald-700'
            };

            fileInput.addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;

                grid.innerHTML = '';
                stats.classList.add('hidden');
                loading.classList.remove('hidden');

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/preview', { method: 'POST', body: formData });
                    const data = await response.json();
                    loading.classList.add('hidden');
                    
                    if (data.error) {
                        grid.innerHTML = `<div class="col-span-full text-red-400 bg-red-900/20 p-4 rounded-lg border border-red-800">${data.error}</div>`;
                        return;
                    }

                    stats.classList.remove('hidden');
                    stats.innerHTML = `
                        <div class="bg-gray-800 px-4 py-2 rounded-lg border border-gray-700 shadow">File: <span class="font-bold text-white">${data.filename}</span></div>
                        <div class="bg-gray-800 px-4 py-2 rounded-lg border border-gray-700 shadow">Total Chunks: <span class="font-bold text-white">${data.total_chunks}</span></div>
                    `;

                    data.chunks.forEach((chunk, index) => {
                        const badgeClass = badgeColors[chunk.element_type] || badgeColors['paragraph'];
                        const card = document.createElement('div');
                        card.className = "bg-gray-800/80 rounded-xl border border-gray-700 overflow-hidden flex flex-col hover:border-gray-500 transition-colors shadow-lg";
                        
                        card.innerHTML = `
                            <div class="px-4 py-3 border-b border-gray-700 bg-gray-900/80 flex justify-between items-center">
                                <span class="text-xs font-mono text-gray-500">#${index + 1}</span>
                                <span class="px-2.5 py-1 rounded-full text-xs font-medium border ${badgeClass} uppercase tracking-wider">
                                    ${chunk.element_type.replace('_', ' ')}
                                </span>
                            </div>
                            <div class="p-4 flex-grow overflow-y-auto h-48 text-sm font-serif leading-relaxed whitespace-pre-wrap">${
                                chunk.text.replace(/</g, "&lt;").replace(/>/g, "&gt;")
                            }</div>
                            <div class="px-4 py-3 border-t border-gray-700 bg-gray-900/80 flex justify-between text-xs text-gray-400 font-mono">
                                <span>Page ${chunk.source.page_number}</span>
                                <span>Hash: ${chunk.content_hash.substring(0, 8)}</span>
                            </div>
                        `;
                        grid.appendChild(card);
                    });
                } catch (err) {
                    loading.classList.add('hidden');
                    grid.innerHTML = `<div class="col-span-full text-red-400">Failed to connect to server.</div>`;
                }
            });
        </script>
    </body>
    </html>
    """

# Retrieval Lab — RAM-only parse + chunk + embed + cosine-search playground.
# Flow: upload → /retrieval_lab/prepare parses/chunks/embeds in RAM under a
# session_id → /retrieval_lab/search runs cosine sim against the session's
# stored embeddings. No Qdrant, no jobs.db, no worker. Sessions LRU-evict
# at _RETRIEVAL_LAB_MAX_SESSIONS so a long-lived process doesn't grow
# unbounded. The global `parser` from module load is reused, so env-flag
# toggles (OCR_ENABLED, VISION_ENABLED) are honored here too.

# Cap on concurrently-held lab sessions. 17th prepare evicts the LRU one.
_RETRIEVAL_LAB_MAX_SESSIONS = env_int("RETRIEVAL_LAB_MAX_SESSIONS")


@dataclass
class _RetrievalLabSession:
    """In-memory state for one Retrieval Lab session.

    - `chunks` is the full chunk list as it came back from the chunker —
      used for the chunk grid display in the UI.
    - `kept_chunks` is the parallel list to `embeddings`: the subset of
      chunks we actually embedded. With v0 (flat) chunking this is the
      same as `chunks`; with v1 (parent-child) parents are excluded
      from the embedding matrix but kept in `chunks` for display.
    - `embeddings` is an L2-normalized (N, dim) float32 matrix. The
      search path computes `embeddings @ q_vec` for cosine sim.
    - `bm25_index` is the per-session sparse retriever used by the
      opt-in hybrid search path. Built at prepare time so the search
      endpoint can fuse dense + sparse ranks via RRF without paying
      indexing cost per query. Cheap: indexing is O(N*L), ~ms for
      typical lab corpora.
    """
    session_id: str
    filename: str
    doc_id: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    kept_chunks: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Optional["np.ndarray"] = None
    bm25_index: Optional[Any] = None


_retrieval_lab_sessions: "OrderedDict[str, _RetrievalLabSession]" = OrderedDict()


def _retrieval_lab_get(session_id: str) -> Optional[_RetrievalLabSession]:
    """Get a session and refresh its LRU position. Returns None if missing."""
    sess = _retrieval_lab_sessions.get(session_id)
    if sess is not None:
        _retrieval_lab_sessions.move_to_end(session_id)
    return sess


def _retrieval_lab_put(session_id: str, sess: _RetrievalLabSession) -> None:
    """Store a session and evict the oldest ones over the caps.

    Two caps: the number of sessions, and the number of chunks they
    hold between them. The second one is the one that matters — sixteen
    sessions of a large document is a lot of memory, and the session
    count alone did not bound it.
    """
    _retrieval_lab_sessions[session_id] = sess
    _retrieval_lab_sessions.move_to_end(session_id)
    while len(_retrieval_lab_sessions) > _RETRIEVAL_LAB_MAX_SESSIONS:
        _retrieval_lab_sessions.popitem(last=False)

    total_cap = env_int("MAX_LAB_TOTAL_CHUNKS")
    while total_cap and _lab_total_chunks() > total_cap:
        oldest = next(iter(_retrieval_lab_sessions), None)
        if oldest is None or oldest == session_id:
            break
        _retrieval_lab_sessions.popitem(last=False)


def _lab_total_chunks() -> int:
    """How many chunks the lab is holding."""
    return sum(len(sess.chunks) for sess in _retrieval_lab_sessions.values())


@app.post("/retrieval_lab/prepare")
async def retrieval_lab_prepare(
    request: Request,
    file: UploadFile,
    chunking_mode: Optional[str] = Form(None, description="Chunking mode: 'auto' (default), 'balanced', 'precise', 'context_rich', or 'custom'. Currently all modes delegate to the same chunker; the mode is accepted for forward-compat."),
):
    """Upload a doc, chunk + embed it in-memory, return chunks + session_id.

    Uses the same parser + chunker + embedder as the prod /ingest path so
    what you see in the lab is what production produces. Embeddings are
    L2-normalized so cosine sim is a single matmul. No persistence —
    the session is evicted on LRU pressure or process restart, by design
    (the lab is meant to be stateless vs any production DB).
    """
    check_content_length(request.headers.get("content-length"))
    import numpy as np

    try:
        from atrium.smart_chunker import chunk_with_mode
    except ImportError:
        # smart_chunker shim missing in this build — fall back to the
        # legacy chunk_elements so the endpoint still works.
        from atrium.smart_chunker import chunk_elements
        chunk_with_mode = None  # type: ignore[assignment]

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "file is empty")
    check_payload_size(len(file_bytes))
    upload_name = _upload_filename(file)
    mime = _detect_mime(file_bytes, upload_name)
    check_mime(mime)

    # Parse + chunk using the same code path /ingest uses. Runs in a thread
    # so the event loop isn't blocked on PyMuPDF / OCR / vision calls.
    def _parse_and_chunk():
        raw = parser.parse(file_bytes, upload_name, mime)
        if chunk_with_mode is not None:
            return chunk_with_mode(
                raw, doc_id="lab", file_name=upload_name,
                mode=chunking_mode or "auto", config=None,
            )
        return chunk_elements(raw, doc_id="lab", file_name=upload_name)

    try:
        smart_chunks = await asyncio.to_thread(_parse_and_chunk)
    except Exception as exc:
        log.warning("retrieval_lab/prepare parse/chunk failed: %s", exc)
        raise HTTPException(422, f"Parse or chunk failed: {exc}") from exc

    # The lab holds a document's chunks *and* their embeddings in memory,
    # for as long as the session lives. A document with ten thousand
    # chunks is ten thousand vectors, and sixteen such sessions is a
    # hundred megabytes of process. The cap is applied before the
    # embedding call, so the cost is bounded too.
    lab_cap = env_int("MAX_LAB_CHUNKS_PER_SESSION")
    truncated = False
    if lab_cap and len(smart_chunks) > lab_cap:
        log.warning(
            "retrieval_lab/prepare: %d chunks, keeping the first %d",
            len(smart_chunks), lab_cap,
        )
        smart_chunks = smart_chunks[:lab_cap]
        truncated = True

    # Project smart_chunker.ChunkRecord into a flat dict for the JSON wire.
    # Mirrors the chunk shape /preview returns minus embed_text internals.
    chunks_out = []
    kept_chunks = []
    embed_texts = []
    for sc in smart_chunks:
        et = getattr(sc, "embed_text", None) or sc.text
        chunk_dict = {
            "chunk_id": sc.chunk_id,
            "text": sc.text,
            "embed_text": et,
            "element_type": sc.element_type,
            "page_number": sc.source.page_number,
            "section_path": sc.source.section_path or [],
            "content_hash": sc.content_hash,
        }
        chunks_out.append(chunk_dict)
        kept_chunks.append(chunk_dict)
        embed_texts.append(et)

    # Embed all chunks in one call. If the corpus is empty, store a
    # zero-width matrix with the right dimensionality so the cosine-sim
    # code in /retrieval_lab/search short-circuits cleanly.
    if not embed_texts:
        embeddings = np.zeros((0, embedder.dimension), dtype=np.float32)
    else:
        try:
            embeddings = np.asarray(
                embedder.embed(embed_texts), dtype=np.float32,
            )
        except Exception as exc:
            log.warning("retrieval_lab/prepare embed failed: %s", exc)
            # The ways out travel with the message: a 503 that says
            # "not answering" and nothing else leaves the operator
            # guessing.
            raise HTTPException(
                503, f"Embedder unavailable: {_explain(exc)}"
            ) from exc

        # L2-normalize so cosine sim reduces to a single matmul. bge-large
        # returns normalized vectors in practice, but we normalize defensively
        # because the contract is "the search endpoint computes cosine sim"
        # and any drift in normalization would silently break the ranking.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embeddings = (embeddings / norms).astype(np.float32)

    # Build a per-session BM25 sparse index over the same text. The
    # search endpoint fuses dense + sparse ranks via RRF when the
    # caller opts into `hybrid=true`; building at prepare time keeps
    # the search endpoint off the indexing hot path. Indexing cost is
    # O(N*L) — sub-millisecond at lab-corpus scale.
    bm25_index = None
    if kept_chunks:
        try:
            from atrium.bm25 import BM25Index
            bm25_index = BM25Index()
            for i, c in enumerate(kept_chunks):
                # chunk_id is the public identifier; BM25 stores it
                # opaque. meta lets the hybrid fusion path reconstruct
                # a full hit shape (text, page_number, section_path,
                # element_type) for BM25-only hits without a second
                # lookup round-trip.
                bm25_index.add_doc(
                    c["chunk_id"], c["text"] or "",
                    meta={
                        "id": c["chunk_id"],
                        "chunk_id": c["chunk_id"],
                        "doc_id": "lab",
                        "text": c["text"] or "",
                        "page_number": c.get("page_number"),
                        "section_path": c.get("section_path") or [],
                        "element_type": c.get("element_type", ""),
                        "rank_index": i,  # for fallback hit-shape build
                    },
                )
        except Exception as exc:
            log.warning("retrieval_lab/prepare BM25 index failed: %s", exc)
            bm25_index = None

    session_id = str(uuid.uuid4())
    _retrieval_lab_put(session_id, _RetrievalLabSession(
        session_id=session_id,
        filename=_upload_filename(file),
        doc_id="lab",
        chunks=chunks_out,
        kept_chunks=kept_chunks,
        embeddings=embeddings,
        bm25_index=bm25_index,
    ))

    return {
        "session_id": session_id,
        "truncated": truncated,
        "filename": _upload_filename(file),
        "mime": mime,
        "chunking_mode": chunking_mode or "auto",
        "n_chunks": len(chunks_out),
        "n_indexed": len(embed_texts),
        "chunks": chunks_out,
    }


@app.post("/retrieval_lab/search")
async def retrieval_lab_search(
    session_id: str = Form(..., description="Returned by /retrieval_lab/prepare"),
    query: str = Form(..., description="Natural-language query"),
    top_k: int = Form(8, description="Number of results to return. Default 8 — topk_sweep showed k=8 captures +5 queries over k=5 on the 19-set for 1.6x scoring cost; k=10 only adds +2 more."),
    hybrid: bool = Form(False, description=(
        "When true, fuse dense cosine-sim with BM25 sparse via Reciprocal "
        "Rank Fusion (RRF, k=60). Opt-in: dense-only is the default. "
        "2026-10-17 regression finding — turning hybrid on globally "
        "regressed the 19-set mean R@5 0.8158 → 0.80 and dropped 1 "
        "fixture below threshold (BM25 tokenization fails on CJK / "
        "Arabic / scanned-caption text). Pass true only for "
        "controlled experiments on corpora where lexical match helps."
    )),
):
    """Embed the query, cosine-sim against the session's stored embeddings,
    return the top-K chunks in descending similarity order.

    By default this is pure dense retrieval — no reranker, no BM25, no
    hybrid RRF, no parent resolution. That's the point: you want to see
    what the embedder alone thinks, so you can decide whether downstream
    pipeline stages are helping or hurting.

    With `hybrid=true`, dense cosine sim is fused with BM25 sparse via
    Reciprocal Rank Fusion (Cormack et al., 2009; k=60). The same
    candidate budget is used for both sides (top_k * 5 by default), so
    the fused list doesn't grow beyond what either ranker would have
    surfaced. 2026-10-17 finding: hybrid helps 4 fixtures (academic
    papers, DOCX/ODT proposals) but destroys 3 (Arabic Wikipedia -0.60,
    Scanned PDF -0.33, BERT -0.14). The tokenization failure mode
    (whitespace splitter breaks CJK / Arabic) is global, so it's an
    opt-in experiment flag, not a default behavior.
    """
    if not query.strip():
        raise HTTPException(422, "query is empty")
    if top_k <= 0 or top_k > 100:
        raise HTTPException(422, "top_k must be 1..100")

    import numpy as np

    sess = _retrieval_lab_get(session_id)
    if sess is None:
        raise HTTPException(
            404,
            "session not found (expired, evicted, or never created). "
            "Re-upload the doc to /retrieval_lab/prepare.",
        )

    kept_chunks = sess.kept_chunks
    embeddings = sess.embeddings
    if len(kept_chunks) == 0 or embeddings is None or embeddings.shape[0] == 0:
        return {"query": query, "top_k": top_k, "results": [], "hybrid": hybrid}

    # Embed the query. The embedder call is blocking (it talks to Ollama),
    # so run it in a thread.
    try:
        q_vec = np.asarray(
            (await asyncio.to_thread(embedder.embed, [query]))[0],
            dtype=np.float32,
        )
    except Exception as exc:
        log.warning("retrieval_lab/search embed failed: %s", exc)
        raise HTTPException(503, f"Embedder unavailable: {_explain(exc)}") from exc

    # L2-normalize the query, then dot with all chunk embeddings. Both sides
    # are unit vectors after the prep step + this line, so the product is
    # pure cosine sim. Result is a length-N vector of similarities.
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm > 0:
        q_vec = q_vec / q_norm
    sims = (embeddings @ q_vec).astype(float)

    n = len(sims)
    k = min(top_k, n)

    if not hybrid or sess.bm25_index is None:
        # Dense-only path. Pure cosine sim, no fusion.
        top_idx = np.argsort(-sims)[:k]
        results = []
        for rank, idx in enumerate(top_idx, start=1):
            i = int(idx)
            c = kept_chunks[i]
            results.append({
                "rank": rank,
                "score": float(sims[i]),
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "page_number": c["page_number"],
                "section_path": c["section_path"],
                "element_type": c["element_type"],
            })
        return {
            "query": query,
            "top_k": top_k,
            "hybrid": False,
            "results": results,
        }

    # Hybrid path (RRF). Pull a wider candidate set from each ranker
    # (top_k * 5) so the fused list isn't capped by either side's
    # precision at exactly top_k. RRF fuses by rank, so we only need
    # the rank ordering, not the raw score.
    cand_k = min(n, max(k * 5, 30))
    dense_ranks = np.argsort(-sims)[:cand_k]
    # BM25 ranks for the same query.
    sparse_hits = sess.bm25_index.search_with_meta(query, top_k=cand_k) or []

    RRF_K = 60  # Cormack et al. 2009 — same constant as atrium/retrieval.py
    fused: dict[str, dict[str, Any]] = {}

    # Dense contribution. Each dense hit gets 1/(RRF_K + dense_rank).
    for dense_rank, idx in enumerate(dense_ranks, start=1):
        i = int(idx)
        c = kept_chunks[i]
        cid = c["chunk_id"]
        rrf_contrib = 1.0 / (RRF_K + dense_rank)
        cur = fused.get(cid)
        if cur is None:
            fused[cid] = {
                "chunk_id": cid,
                "text": c["text"],
                "page_number": c["page_number"],
                "section_path": c["section_path"],
                "element_type": c["element_type"],
                "rrf_score": rrf_contrib,
                "rrf_rank_dense": dense_rank,
                "rrf_rank_sparse": None,
                "dense_score": float(sims[i]),
            }
        else:
            cur["rrf_score"] += rrf_contrib
            cur["rrf_rank_dense"] = dense_rank
            cur["dense_score"] = float(sims[i])

    # Sparse (BM25) contribution. Each BM25 hit gets 1/(RRF_K + sparse_rank).
    # Hits present only here are promoted to a full hit shape using the
    # metadata we stored at prepare time (text, page_number, etc.) — no
    # second round-trip needed.
    for sparse_rank, (cid, _bm25_score, bm25_meta) in enumerate(sparse_hits, start=1):
        rrf_contrib = 1.0 / (RRF_K + sparse_rank)
        cur = fused.get(cid)
        if cur is None:
            # BM25-only hit. Use the chunk's text/page if we have the
            # chunk in kept_chunks; otherwise reconstruct from bm25_meta.
            text = bm25_meta.get("text", "")
            page_number = bm25_meta.get("page_number")
            section_path = bm25_meta.get("section_path") or []
            element_type = bm25_meta.get("element_type", "")
            # kept_chunks is keyed by chunk_id; if the BM25 hit is in
            # the lab corpus, prefer the lab-side text/page (it's the
            # exact one the user sees in the chunk grid).
            rank_index = bm25_meta.get("rank_index")
            if isinstance(rank_index, int) and 0 <= rank_index < len(kept_chunks):
                c = kept_chunks[rank_index]
                if c.get("chunk_id") == cid:
                    text = c.get("text", text)
                    page_number = c.get("page_number", page_number)
                    section_path = c.get("section_path") or section_path
                    element_type = c.get("element_type", element_type)
            fused[cid] = {
                "chunk_id": cid,
                "text": text,
                "page_number": page_number,
                "section_path": section_path,
                "element_type": element_type,
                "rrf_score": rrf_contrib,
                "rrf_rank_dense": None,
                "rrf_rank_sparse": sparse_rank,
                "dense_score": None,
            }
        else:
            cur["rrf_score"] += rrf_contrib
            cur["rrf_rank_sparse"] = sparse_rank

    # Sort by fused RRF score desc, truncate to top_k. The score
    # surfaced in the response is the RRF score (a small positive
    # number that grows as the ranker agreement strengthens). This
    # is more honest than reporting a stale cosine sim that no
    # longer reflects the ranker's view of relevance.
    ranked = sorted(
        fused.values(), key=lambda h: h["rrf_score"], reverse=True,
    )[:k]
    results = []
    for rank, h in enumerate(ranked, start=1):
        results.append({
            "rank": rank,
            "score": float(h["rrf_score"]),
            "chunk_id": h["chunk_id"],
            "text": h["text"],
            "page_number": h["page_number"],
            "section_path": h["section_path"],
            "element_type": h["element_type"],
            "rrf_rank_dense": h["rrf_rank_dense"],
            "rrf_rank_sparse": h["rrf_rank_sparse"],
        })

    return {
        "query": query,
        "top_k": top_k,
        "hybrid": True,
        "results": results,
    }


@app.get("/retrieval_lab", response_class=HTMLResponse)
def retrieval_lab_ui():
    """Single-page UI: file picker → chunking mode → embed+chunk button →
    chunk grid → query input → search results. Uses the same Tailwind
    aesthetic as /visualize so the two pages feel like one tool.

    The "embed and chunk" button posts the file to /retrieval_lab/prepare
    which parses, chunks, and embeds in RAM, returning a session_id. The
    search box then queries that session via /retrieval_lab/search.
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Retrieval Lab</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #1f2937; }
            ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #6b7280; }
        </style>
    </head>
    <body class="bg-gray-900 text-gray-100 min-h-screen p-8 font-sans">
        <div class="max-w-7xl mx-auto">
            <header class="mb-8 flex justify-between items-end border-b border-gray-800 pb-6">
                <div>
                    <h1 class="text-4xl font-bold text-white mb-2">Retrieval Lab</h1>
                    <p class="text-gray-400">RAM-only playground: parse, chunk, embed, search. No vector DB.</p>
                </div>
                <a href="/visualize" class="text-sm text-gray-400 hover:text-gray-200">Chunk Visualizer &rarr;</a>
            </header>

            <section class="mb-8 bg-gray-800/50 rounded-lg p-6 border border-gray-700">
                <h2 class="text-lg font-semibold text-white mb-4">1. Prepare a document</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <div class="md:col-span-2">
                        <label class="block text-xs text-gray-400 uppercase mb-2">Document</label>
                        <input type="file" id="fileInput" class="block w-full text-sm text-gray-300
                            file:mr-4 file:py-2 file:px-4 file:rounded file:border-0
                            file:text-sm file:font-semibold file:bg-blue-600 file:text-white
                            hover:file:bg-blue-500 file:cursor-pointer" />
                    </div>
                    <div>
                        <label class="block text-xs text-gray-400 uppercase mb-2">Chunking mode</label>
                        <select id="chunkingMode" class="w-full bg-gray-900 border border-gray-700 rounded
                            text-gray-200 text-sm px-3 py-2 focus:outline-none focus:border-blue-500">
                            <option value="auto" selected>Auto (default)</option>
                            <option value="balanced">Balanced</option>
                            <option value="precise">Precise</option>
                            <option value="context_rich">Context Rich</option>
                            <option value="custom">Custom</option>
                        </select>
                    </div>
                </div>
                <div class="flex items-center gap-3">
                    <button id="prepareBtn" class="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2
                        rounded font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                        Embed and Chunk
                    </button>
                    <span id="prepareStatus" class="text-sm text-gray-400"></span>
                </div>
            </section>

            <section id="chunkSection" class="mb-8 hidden">
                <h2 class="text-lg font-semibold text-white mb-4">
                    2. Chunks <span id="chunkCount" class="text-gray-400 text-base font-normal"></span>
                </h2>
                <div id="chunkGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4"></div>
            </section>

            <section id="searchSection" class="mb-8 hidden bg-gray-800/50 rounded-lg p-6 border border-gray-700">
                <h2 class="text-lg font-semibold text-white mb-4">3. Search</h2>
                <div class="flex gap-3 mb-3">
                    <input id="queryInput" type="text" placeholder="Type a query..."
                        class="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-2 text-gray-200
                        focus:outline-none focus:border-blue-500" />
                    <button id="searchBtn" class="bg-emerald-600 hover:bg-emerald-500 text-white px-6 py-2
                        rounded font-semibold transition-colors">Search</button>
                </div>
                <div class="flex items-center gap-2 mb-4">
                    <input id="hybridToggle" type="checkbox" class="h-4 w-4 rounded border-gray-600 bg-gray-900
                        text-emerald-500 focus:ring-emerald-500 focus:ring-offset-gray-800" />
                    <label for="hybridToggle" class="text-sm text-gray-300 select-none cursor-pointer">
                        Hybrid (dense + BM25 via RRF)
                    </label>
                    <span class="text-xs text-gray-500 ml-2">— opt-in; default is dense-only (2026-10-17 regression: hybrid regresses Arabic / Scanned / BERT)</span>
                </div>
                <div id="results" class="space-y-3"></div>
            </section>

            <div id="loading" class="hidden py-12 text-center text-gray-400">
                <svg class="animate-spin h-8 w-8 mx-auto mb-3 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <p id="loadingText" class="text-sm">Working...</p>
            </div>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const chunkingMode = document.getElementById('chunkingMode');
            const prepareBtn = document.getElementById('prepareBtn');
            const prepareStatus = document.getElementById('prepareStatus');
            const chunkSection = document.getElementById('chunkSection');
            const chunkCount = document.getElementById('chunkCount');
            const chunkGrid = document.getElementById('chunkGrid');
            const searchSection = document.getElementById('searchSection');
            const queryInput = document.getElementById('queryInput');
            const searchBtn = document.getElementById('searchBtn');
            const hybridToggle = document.getElementById('hybridToggle');
            const results = document.getElementById('results');
            const loading = document.getElementById('loading');
            const loadingText = document.getElementById('loadingText');

            let sessionId = null;

            const badgeColors = {
                'heading': 'bg-purple-900/50 text-purple-300 border-purple-700',
                'paragraph': 'bg-gray-800 text-gray-300 border-gray-600',
                'table': 'bg-blue-900/50 text-blue-300 border-blue-700',
                'image_caption': 'bg-emerald-900/50 text-emerald-300 border-emerald-700'
            };

            function setLoading(on, text) {
                if (on) { loading.classList.remove('hidden'); loadingText.textContent = text || 'Working...'; }
                else { loading.classList.add('hidden'); }
            }

            prepareBtn.addEventListener('click', async () => {
                const file = fileInput.files[0];
                if (!file) { prepareStatus.textContent = 'Pick a file first.'; return; }
                prepareBtn.disabled = true;
                prepareStatus.textContent = '';
                chunkSection.classList.add('hidden');
                searchSection.classList.add('hidden');
                chunkGrid.innerHTML = '';
                results.innerHTML = '';
                setLoading(true, 'Parsing, chunking, and embedding...');

                const fd = new FormData();
                fd.append('file', file);
                fd.append('chunking_mode', chunkingMode.value);

                try {
                    const resp = await fetch('/retrieval_lab/prepare', { method: 'POST', body: fd });
                    const data = await resp.json();
                    if (!resp.ok) {
                        prepareStatus.textContent = data.detail || ('Error ' + resp.status);
                        return;
                    }
                    sessionId = data.session_id;
                    chunkCount.textContent = `(${data.n_chunks} chunks, ${data.n_indexed} indexed)`;
                    chunkGrid.innerHTML = '';
                    for (const c of data.chunks) {
                        const card = document.createElement('div');
                        card.className = 'bg-gray-800 border border-gray-700 rounded-lg overflow-hidden flex flex-col';
                        const badgeClass = badgeColors[c.element_type] || 'bg-gray-700 text-gray-300 border-gray-600';
                        card.innerHTML = `
                            <div class="px-3 py-2 border-b border-gray-700 flex justify-between items-center text-xs">
                                <span class="px-2 py-0.5 rounded border ${badgeClass}">${c.element_type}</span>
                                <span class="text-gray-500 font-mono">Page ${c.page_number ?? '-'}</span>
                            </div>
                            <div class="p-3 text-xs font-serif leading-relaxed whitespace-pre-wrap flex-grow overflow-y-auto h-32 text-gray-300">${
                                (c.text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                            }</div>
                        `;
                        chunkGrid.appendChild(card);
                    }
                    chunkSection.classList.remove('hidden');
                    searchSection.classList.remove('hidden');
                    prepareStatus.textContent = 'Session ' + sessionId.slice(0, 8) + ' ready.';
                    queryInput.focus();
                } catch (err) {
                    prepareStatus.textContent = 'Network error: ' + err;
                } finally {
                    setLoading(false);
                    prepareBtn.disabled = false;
                }
            });

            searchBtn.addEventListener('click', async () => {
                if (!sessionId) { results.innerHTML = '<div class="text-red-400">Prepare a doc first.</div>'; return; }
                const q = queryInput.value.trim();
                if (!q) return;
                searchBtn.disabled = true;
                setLoading(true, 'Embedding query and ranking...');
                results.innerHTML = '';

                const fd = new FormData();
                fd.append('session_id', sessionId);
                fd.append('query', q);
                fd.append('top_k', '8');
                fd.append('hybrid', hybridToggle && hybridToggle.checked ? 'true' : 'false');

                try {
                    const resp = await fetch('/retrieval_lab/search', { method: 'POST', body: fd });
                    const data = await resp.json();
                    if (!resp.ok) {
                        results.innerHTML = `<div class="text-red-400">${data.detail || ('Error ' + resp.status)}</div>`;
                        return;
                    }
                    if (!data.results || data.results.length === 0) {
                        results.innerHTML = '<div class="text-gray-400">No results.</div>';
                        return;
                    }
                    const modeLabel = data.hybrid ? 'hybrid RRF' : 'dense';
                    const header = `<div class="text-xs text-gray-500 mb-2">Ranked by: <span class="text-emerald-400 font-mono">${modeLabel}</span></div>`;
                    results.insertAdjacentHTML('beforeend', header);
                    for (const r of data.results) {
                        const card = document.createElement('div');
                        card.className = 'bg-gray-800 border border-gray-700 rounded p-4';
                        const badgeClass = badgeColors[r.element_type] || 'bg-gray-700 text-gray-300 border-gray-600';
                        const rankInfo = (r.rrf_rank_dense || r.rrf_rank_sparse)
                            ? `<span class="text-gray-500 font-mono ml-2 text-xs">dense=${r.rrf_rank_dense ?? '-'} bm25=${r.rrf_rank_sparse ?? '-'}</span>`
                            : '';
                        card.innerHTML = `
                            <div class="flex justify-between items-center mb-2 text-xs">
                                <div class="flex items-center gap-2">
                                    <span class="text-gray-500 font-mono">#${r.rank}</span>
                                    <span class="px-2 py-0.5 rounded border ${badgeClass}">${r.element_type}</span>
                                    <span class="text-gray-500">Page ${r.page_number ?? '-'}</span>
                                    ${rankInfo}
                                </div>
                                <span class="text-emerald-400 font-mono">score ${r.score.toFixed(4)}</span>
                            </div>
                            <div class="text-sm font-serif leading-relaxed whitespace-pre-wrap text-gray-200">${
                                (r.text || '').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                            }</div>
                        `;
                        results.appendChild(card);
                    }
                } catch (err) {
                    results.innerHTML = '<div class="text-red-400">Network error: ' + err + '</div>';
                } finally {
                    setLoading(false);
                    searchBtn.disabled = false;
                }
            });

            queryInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') searchBtn.click();
            });
        </script>
    </body>
    </html>
    """
