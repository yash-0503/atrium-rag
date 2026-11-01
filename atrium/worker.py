"""
atrium/worker.py

Standalone async ingestion worker. Drains the JobStore queue by claiming
pending jobs, running `run_ingestion` on each, and updating status.

Run with:  python -m atrium.worker
Scale by spawning more processes — Postgres SKIP LOCKED or SQLite
BEGIN IMMEDIATE make claim atomic across workers.
"""

from __future__ import annotations

import base64
import logging
import signal
import sqlite3
import sys
import time
import traceback
import uuid

from atrium.adapters import build_store
from atrium.config import data_path, env_bool, env_float, env_int, env_str
from atrium.embedder_config import build_embedder, default_base_url
from atrium.job_store import JobStore
from atrium.pipeline import run_ingestion
from atrium.universal_parser import UniversalParser
from atrium.vision_adapters import OllamaVisionAdapter

# Idempotent on re-import; matches main.py's basicConfig shape.
logging.basicConfig(
    level=env_str("LOG_LEVEL").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("ingest.worker")


# Stable for the lifetime of the process; written to jobs.claimed_by
# so operators can tell which worker is processing what.
WORKER_ID = f"worker-{uuid.uuid4().hex[:8]}"

# Polling cadence when the queue is empty. Linear backoff up to POLL_MAX,
# resets to POLL_BASE the moment a job is claimed.
POLL_BASE = env_float("WORKER_POLL_BASE")
POLL_MAX = env_float("WORKER_POLL_MAX")

# How long a `running` job can sit before another worker reclaims it.
STALE_SECONDS = env_int("JOB_STALE_SECONDS")

# Per-job hard timeout. 0 = disabled (rely on stale-claim recovery).
JOB_TIMEOUT_SECONDS = env_float("WORKER_JOB_TIMEOUT")

# How many times a job may be attempted before it is given up on. A
# document that hangs the parser used to be released and retried for
# ever; it now fails and stays failed.
MAX_JOB_ATTEMPTS = env_int("MAX_JOB_ATTEMPTS")

# How long a finished job row is kept. The row is the record of what
# happened; the payload it carried was the user's document.
JOB_RETENTION_SECONDS = env_float("JOB_RETENTION_SECONDS")


_shutdown = False


def _handle_signal(signum, _frame) -> None:
    global _shutdown
    log.info("received signal %s; finishing current job then exiting", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# Each worker process boots its own parser/embedder/db — restartable
# without coordinating with the API.
def _build_parser_and_ocr():
    """Returns (parser, ocr_adapter_or_None). Mirrors atrium/main.py."""
    raw_url = env_str("EMBED_BASE_URL")
    clean_host = raw_url.replace("/v1", "").rstrip("/")

    # Vision is gated on VISION_ENABLED here too. The worker used to
    # construct the vision adapter unconditionally, which meant a
    # deployment that had deliberately turned vision off (main.py
    # respects the flag) still had the worker calling a vision model
    # on every image-only page.
    vision = None
    if env_bool("VISION_ENABLED"):
        vision = OllamaVisionAdapter(
            base_url=clean_host,
            model=env_str("VISION_MODEL"),
            timeout_seconds=env_float("VISION_TIMEOUT_SECONDS"),
        )

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
                "(%s: %s). OCR will be unavailable in this worker.",
                type(exc).__name__, exc,
            )
            ocr_adapter = None

    parser = UniversalParser(vision_adapter=vision, ocr_adapter=ocr_adapter)
    return parser, ocr_adapter


def _build_embedder():
    # Shared with the HTTP service via build_embedder() so model + dim
    # cannot drift between the two ingest paths.
    return build_embedder(base_url=default_base_url())


def _build_db():
    # The same factory the API uses. It used to be a copy that had
    # drifted: with the default `VECTOR_DB=sqlite` this built a
    # Postgres adapter while the API built a SQLite one, so the worker
    # and the service wrote to different databases.
    return build_store()


def _build_job_store():
    # SQLite only. The job store is a single file; run several workers
    # against the same one by putting it on a shared volume. (It used
    # to read JOB_DB_URL here and hand the value to sqlite3.connect(),
    # which created a file literally named "postgresql://…".)
    #
    # The path is resolved to an absolute one, because the API and this
    # worker have to be looking at the same file even when they do not
    # share a working directory.
    return JobStore(db_path=str(data_path("JOB_DB_PATH", "jobs.db")))


def _build_parser_for_call(global_parser, ocr_adapter, ocr_override):
    # Swap in a sibling parser when the payload specifies per-job OCR.
    if ocr_override is None:
        return global_parser
    chosen = ocr_adapter if ocr_override else None
    return UniversalParser(
        vision_adapter=global_parser.vision_adapter,
        ocr_adapter=chosen,
    )



def _arm_job_timeout() -> None:
    """Start the per-job watchdog, if one is configured."""
    if JOB_TIMEOUT_SECONDS <= 0:
        return

    def _on_timeout(signum, frame):
        raise TimeoutError(
            f"the job exceeded WORKER_JOB_TIMEOUT={JOB_TIMEOUT_SECONDS}s"
        )

    try:
        signal.signal(signal.SIGALRM, _on_timeout)
        signal.setitimer(signal.ITIMER_REAL, JOB_TIMEOUT_SECONDS)
    except (AttributeError, ValueError):  # pragma: no cover - platform
        log.warning(
            "this platform has no interval timer; the per-job timeout is off"
        )


def _disarm_job_timeout() -> None:
    """Stop the per-job watchdog."""
    if JOB_TIMEOUT_SECONDS <= 0:
        return
    try:
        signal.setitimer(signal.ITIMER_REAL, 0)
    except (AttributeError, ValueError):  # pragma: no cover - platform
        pass


def _process_one(job: dict, parser, ocr_adapter, embedder, db, job_store) -> None:
    """Run one claimed job end-to-end. Updates job status on success/failure.
    The job is already status='running' (set by claim_next)."""
    job_id = job["job_id"]
    doc_id = job["doc_id"]
    payload = job.get("payload") or {}

    # Payload contract:
    #   file_bytes_b64: str,   base64-encoded original upload bytes
    #   filename:        str,
    #   mime:            str,  libmagic-detected or "application/octet-stream"
    #   ocr_override:    bool | None,
    try:
        file_bytes = base64.b64decode(payload["file_bytes_b64"])
    except (KeyError, TypeError, ValueError) as exc:
        # Corrupt or missing payload — can't re-process, mark failed.
        log.error("job %s: bad payload (%s); marking failed", job_id, exc)
        job_store.update(job_id, status="failed", error=f"bad payload: {exc}")
        return

    filename = payload.get("filename", "<unknown>")
    mime = payload.get("mime", "application/octet-stream")
    ocr_override = payload.get("ocr_override")
    chunking_mode = payload.get("chunking_mode", "auto")
    chunking_config = payload.get("chunking_config")

    parser_for_this_call = _build_parser_for_call(parser, ocr_adapter, ocr_override)

    log.info(
        "job %s: doc_id=%s filename=%s mime=%s ocr=%s chunking_mode=%s",
        job_id, doc_id, filename, mime, ocr_override, chunking_mode,
    )

    try:
        # The per-job timeout. It is a signal-based watchdog: the worker
        # is single-threaded while it ingests, so an interval timer can
        # interrupt a job that has stopped making progress — a
        # pathological PDF, a zip bomb, a regex that backtracks. The
        # timeout used to be read and then ignored, which meant one bad
        # document could wedge a worker for ever.
        _arm_job_timeout()
        run_ingestion(
            job_id, doc_id, file_bytes, filename, mime,
            parser_for_this_call, embedder, db, job_store,
            chunking_mode=chunking_mode, chunking_config=chunking_config,
        )
        log.info("job %s: complete", job_id)
    except TimeoutError:
        log.error(
            "job %s: timed out after %ss; the document is not being ingested",
            job_id, JOB_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        # run_ingestion already updates status to failed; we wrap here
        # so ops gets the traceback.
        log.error(
            "job %s: failed (%s: %s)\n%s",
            job_id, type(exc).__name__, exc, traceback.format_exc(),
        )
    finally:
        _disarm_job_timeout()


def main() -> int:
    log.info("worker %s starting", WORKER_ID)
    parser, ocr_adapter = _build_parser_and_ocr()
    embedder = _build_embedder()
    db = _build_db()
    job_store = _build_job_store()

    if not db.health():
        log.error("vector DB unhealthy at startup; exiting")
        return 1
    if not embedder.health():
        log.error("embedder unhealthy at startup; exiting")
        return 1
    db.ensure_collection("documents", embedder.dimension)
    log.info(
        "worker %s ready (stale=%ds, poll=%.1f..%.1fs, job store=%s)",
        WORKER_ID, STALE_SECONDS, POLL_BASE, POLL_MAX,
        data_path("JOB_DB_PATH", "jobs.db"),
    )

    backoff = POLL_BASE
    while not _shutdown:
        # Reclaim stale claims BEFORE claiming fresh work. Otherwise a
        # worker that just booted would race against a slow worker
        # still mid-job and could double-claim.
        recovered = job_store.release_stale_claims(STALE_SECONDS, MAX_JOB_ATTEMPTS)
        if recovered:
            log.warning("recovered %d stale job(s) (worker crash / hang)", recovered)

        try:
            job = job_store.claim_next(WORKER_ID)
        except sqlite3.OperationalError as exc:
            # The job store is one SQLite file. The API enqueues into
            # it while this worker claims from it, and a contended
            # write raises "database is locked". That used to kill the
            # worker process; now it is a pause.
            log.warning("could not claim a job (%s); retrying", exc)
            time.sleep(min(backoff, 5.0))
            continue
        if job is None:
            # Empty queue — sleep with linear backoff so an idle
            # single-host install doesn't burn CPU.
            time.sleep(backoff)
            backoff = min(backoff * 1.5, POLL_MAX)
            continue

        # Got a job — reset backoff so the next empty poll reacts fast.
        backoff = POLL_BASE
        _process_one(job, parser, ocr_adapter, embedder, db, job_store)

        # Housekeeping: the job rows are a record of what happened, but
        # the payloads in them were the user's documents.
        purged = job_store.purge_terminal_jobs(JOB_RETENTION_SECONDS)
        if purged:
            log.info("purged %d finished job(s) older than %ss", purged, JOB_RETENTION_SECONDS)

    log.info("worker %s shutting down cleanly", WORKER_ID)
    job_store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())