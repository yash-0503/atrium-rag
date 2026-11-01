"""
atrium/security.py

HTTP-layer security: upload size cap, MIME allowlist, bearer-token
auth, stack-trace redaction. All knobs are env-driven (see
`.env.example`). Fails closed when API_KEY is non-empty: every
endpoint except /health and /visualize requires Bearer auth.

Every setting is read **live** through `atrium.config` rather than
snapshotted at import time. That matters for two reasons:

  - `.env` is loaded into the process environment when `atrium` is
    imported, so a value written there is visible here — but only if we
    read it after that point.
  - `atrium doctor` imports this module to report the operator's
    security posture. When the values were module-level constants
    captured at import, doctor ran in a process that had not loaded
    `.env` yet and therefore reported "API_KEY is empty" for every
    configuration, including ones that had a key.
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import List, Optional, Set

from fastapi import HTTPException, Request, status

from atrium.config import env_bool, env_int, env_str

log = logging.getLogger("ingest.security")


# Live accessors. Each call re-reads the environment, so a value that
# changes after import is picked up without a restart.


def max_upload_bytes() -> int:
    """Upload size cap in bytes. 100 MB default."""
    return env_int("MAX_UPLOAD_BYTES")


def allowed_mime_types() -> Set[str]:
    """The MIME allowlist. Empty = any detected MIME accepted.

    Set to a comma-separated list (`application/pdf,text/markdown`) to
    lock ingestion to a known set.
    """
    raw = env_str("ALLOWED_MIME_TYPES")
    return {m.strip() for m in raw.split(",") if m.strip()}


def api_key() -> str:
    """Bearer-token auth key. Empty = auth disabled."""
    return env_str("API_KEY")


# Routes that stay open even when API_KEY is set. /health for compose
# healthcheck + uptime monitors; /visualize is a static page.
#
# The schema browsers are NOT in the list. With a key set, /docs,
# /openapi.json and /redoc describe the whole API surface — every
# route, every field — to anyone who asks. They are still reachable
# when no key is set, which is the case they are for.
_OPEN_ROUTES = frozenset({"/health", "/visualize"})


def check_content_length(content_length_header: Optional[str]) -> None:
    """Reject requests that advertise a body larger than MAX_UPLOAD_BYTES.
    Checked on the Content-Length header before any bytes flow in;
    uvicorn/h11 backpressure covers the lying-hostile case.
    """
    if content_length_header is None:
        # No content-length: reject by default unless the operator
        # has set ALLOW_CHUNKED_UPLOADS=1 explicitly. Chunked uploads
        # are a streaming attack surface and the server has no reason
        # to accept them for a normal document-ingestion workflow.
        if not env_bool("ALLOW_CHUNKED_UPLOADS"):
            raise HTTPException(
                status_code=status.HTTP_411_LENGTH_REQUIRED,
                detail="Content-Length header is required for uploads",
            )
        return
    try:
        length = int(content_length_header)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid Content-Length",
        ) from None
    limit = max_upload_bytes()
    if length > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"upload exceeds MAX_UPLOAD_BYTES={limit} "
                f"(got Content-Length={length})"
            ),
        )


def check_payload_size(actual_size: int) -> None:
    """Belt-and-suspenders check on the actual byte count after the
    upload is read. Some clients lie about Content-Length."""
    limit = max_upload_bytes()
    if actual_size > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"upload exceeds MAX_UPLOAD_BYTES={limit}",
        )


# MIME allowlist


def check_mime(mime: str) -> None:
    """Reject a detected MIME that's outside the allowlist. Empty
    allowlist = accept everything (dev default)."""
    allowed = allowed_mime_types()
    if not allowed:
        return
    if mime not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"unsupported mime {mime!r}; "
                f"allowed: {sorted(allowed)}"
            ),
        )


# API key auth


def _unauthorized() -> HTTPException:
    # RFC 7235 wants a WWW-Authenticate header on 401 responses; uvicorn
    # doesn't set one for plain JSONResponses, but including the challenge
    # in the detail keeps well-behaved clients on the right path.
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="missing or invalid Authorization header (expected 'Bearer <API_KEY>')",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_bearer(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if not auth:
        return None
    # Match `Bearer <token>` ignoring case + surrounding whitespace.
    m = re.match(r"^\s*[Bb][Ee][Aa][Rr][Ee][Rr]\s+(.+?)\s*$", auth)
    if not m:
        return None
    return m.group(1).strip()


def require_api_key_if_configured(request: Request) -> None:
    """FastAPI dependency. No-op when API_KEY is empty (dev default).
    Raises 401 when API_KEY is set and the bearer token doesn't match.

    Imported into route signatures as `dependencies=[Depends(require_api_key_if_configured)]`
    or wrapped as middleware (see atrium/main.py). The middleware form is
    used so /health and /visualize stay open without per-route opt-outs.
    """
    expected = api_key()
    if not expected:
        return
    if request.url.path in _OPEN_ROUTES:
        return
    token = _extract_bearer(request)
    if token is None:
        raise _unauthorized()
    # Constant-time-ish compare. secrets.compare_digest handles short
    # tokens correctly; using `!=` would leak length via timing on a
    # network attacker.
    import hmac

    if not hmac.compare_digest(token, expected):
        raise _unauthorized()


# Error-message redaction


def redacted_traceback(exc: BaseException) -> str:
    """Return a traceback string with file paths / line numbers / source
    lines stripped. Use in HTTP responses so a stack trace doesn't leak
    internal layout to callers, but operators can still get the full
    thing from the server log.
    """
    return "".join(
        traceback.format_exception_only(type(exc), exc)
    ).strip()


def log_redacted_traceback(exc: BaseException, where: str) -> None:
    """Server-side: emit the full traceback to the log (so operators
    can debug) but never echo it back to the client."""
    log.error(
        "uncaught exception in %s: %s\n%s",
        where,
        type(exc).__name__,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )


# Path traversal
# -------------
#
# There is no path-traversal guard here, and there is not one in the
# codebase at all. The one that used to be here (`safe_resolve`) was
# never called: the HTTP API takes uploads, not paths, and the CLI
# runs as the operator and may legitimately be pointed anywhere on
# their own disk. Confining the CLI to the working directory would
# break `atrium ingest ~/Documents/papers`, and a guard that is never
# applied protects nothing. The traversal risk that *was* real — the
# CLI's file walk following a symlinked directory out of the tree it
# was pointed at — is fixed where it belongs, in the walk itself
# (`atrium/cli.py::_walk_files`).


# Diagnostics


def diagnostics() -> List[str]:
    """Return human-readable notes for `atrium doctor`. Helps the
    operator spot a permissive security posture before deploying."""
    notes: List[str] = []
    if not api_key():
        notes.append(
            "API_KEY is empty — every endpoint is open. Set API_KEY in .env for any non-dev deploy.",
        )
    if not allowed_mime_types():
        notes.append(
            "ALLOWED_MIME_TYPES is empty — any MIME the detector returns is accepted. "
            "Set this for hostile upload surfaces.",
        )
    limit = max_upload_bytes()
    if limit > 500 * 1024 * 1024:
        notes.append(
            f"MAX_UPLOAD_BYTES={limit} is > 500 MB — make sure your reverse proxy "
            "is also limiting requests, or a single client can OOM the server.",
        )
    return notes