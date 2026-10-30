"""
atrium/mime.py

MIME detection, with no web framework and no heavy dependencies.

This used to live inside `atrium.main`, which meant the CLI had to
import FastAPI (~220 ms) just to work out what a file was. The
detection logic has no business depending on an HTTP framework, and
`atrium ingest` / `atrium search` are CLI commands that never serve a
request. So it lives here now: a leaf module with nothing but the
standard library (plus `python-magic` when it happens to be
installed), importable from the CLI, the API and the worker alike.

`atrium.main` re-exports the private names (`_detect_mime`,
`_match_signature`, ...) so existing imports and tests keep working.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "EXT_TO_MIME",
    "detect_mime",
    "looks_like_text",
    "match_signature",
    "mime_conflict",
    "mime_from_extension",
]


# libmagic fallback: deferred-import so envs without it can still boot,
# and an extension→mime table for the `application/octet-stream` case
# that surfaces in CI / dev.
EXT_TO_MIME = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".html": "text/html",
    ".htm":  "text/html",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".csv":  "text/csv",
    ".json": "application/json",
    ".epub": "application/epub+zip",
    ".rtf":  "application/rtf",
    ".xls":  "application/vnd.ms-excel",
    ".odt":  "application/vnd.oasis.opendocument.text",
    ".ods":  "application/vnd.oasis.opendocument.spreadsheet",
    ".odp":  "application/vnd.oasis.opendocument.presentation",
}


# Content signatures — formats identifiable from their leading bytes
# alone. This is the dependency-free fallback for environments where
# libmagic can't load (CI, slim Docker images, dev machines missing the
# native library). Kept as a plain table so it costs nothing to import.
SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"{\\rtf", "application/rtf"),
    # Legacy OLE2 compound documents (.doc / .xls / .ppt).
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "application/x-ole-storage"),
)

# Whitespace control characters are legal in text; every other control
# character marks the bytes as binary.
_TEXT_WHITESPACE = "\t\n\r\x0b\x0c"


def looks_like_text(content: bytes) -> bool:
    """True when the bytes decode as UTF-8 and carry no control
    characters beyond ordinary whitespace.

    The control-character check is what separates real text from a
    binary blob that happens to decode cleanly by luck — e.g. six
    low bytes are valid UTF-8 but are not prose.
    """
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return not any(
        (ch < " " or ch == "\x7f") and ch not in _TEXT_WHITESPACE
        for ch in text
    )


def match_signature(content: bytes) -> Optional[str]:
    """Identify a file's format from its leading bytes.

    Returns the mime string, or None when nothing matches. None means
    "opaque" — not "unsupported" — so the caller can distinguish a
    file we couldn't classify from one we deliberately reject.
    """
    if not content:
        return None
    for prefix, mime in SIGNATURES:
        if content.startswith(prefix):
            return mime
    if looks_like_text(content):
        return "text/plain"
    return None


def mime_from_extension(filename: str) -> str:
    """Coarse extension-only mime guess. Returns the canonical mime
    string for known extensions, or `application/octet-stream` for
    anything we don't recognize. The caller should treat the latter
    as 'unsupported' and surface a 415 to the user."""
    if not filename:
        return "application/octet-stream"
    lower = filename.lower()
    # rsplit on '.' so '.tar.gz' is matched against the last extension only.
    dot = lower.rfind(".")
    if dot < 0:
        return "application/octet-stream"
    ext = lower[dot:]
    return EXT_TO_MIME.get(ext, "application/octet-stream")


def detect_mime(file_bytes: bytes, filename: str = "") -> str:
    """Detect mime type with a four-tier fallback:
      1. libmagic (most accurate — looks at magic bytes).
      2. Filename extension table.
      3. Content signature table (`match_signature`).
      4. `application/octet-stream` (caller treats this as unsupported).

    libmagic is wrapped in try/except so a missing system library
    doesn't 500 the request — it just falls through to the extension
    table. The extension fallback is the only one that ever fires in
    CI / local-dev environments without libmagic installed, which is
    why the error used to be so visible there.

    The signature tier is the backstop for the case neither of the
    first two covers: a request with no usable filename, or a file
    whose extension we don't recognize. Without it, a real PDF arrives
    as `application/octet-stream` and the parser rejects it outright
    (`Unsupported mime_type`) rather than routing it.

    Tier order note: the extension table outranks the signature
    because it names a specific member of a container family (OOXML
    and ODF are all ZIPs) and the parser's dispatch table routes on
    those exact mime strings.
    """
    try:
        import magic as _magic
        detected = _magic.from_buffer(file_bytes, mime=True)
    except ImportError:
        # No libmagic — skip straight to extension fallback.
        detected = "application/octet-stream"
    except Exception:
        # libmagic present but raised (corrupt input, permission, etc.).
        # Be defensive: still try the extension path before giving up.
        detected = "application/octet-stream"

    # When libmagic returns the catch-all 'octet-stream' we can't trust
    # it — it usually means libmagic didn't recognize the format. The
    # extension table is more reliable for our supported set since the
    # parser only knows the formats we list in EXT_TO_MIME.
    if detected == "application/octet-stream" and filename:
        ext_mime = mime_from_extension(filename)
        if ext_mime != "application/octet-stream":
            return ext_mime

    # Last resort before giving up: classify from the bytes themselves.
    if detected == "application/octet-stream":
        detected = match_signature(file_bytes) or "application/octet-stream"

    return detected


def mime_conflict(declared: str, detected: str) -> bool:
    """True when the client's declared content type contradicts the bytes.

    A file named `report.pdf` whose body is HTML is the classic case:
    the browser (or a naive client) says `application/pdf`, the bytes
    say `text/html`. Serving that to a parser is how a PDF upload turns
    into an HTML parse.

    The comparison is deliberately lenient — it only fires on a
    *category* disagreement, not on an exact-string one:

    * `application/octet-stream` on either side means "unknown", not a
      conflict. Every curl upload without `-H 'Content-Type: …'` sends
      this, and rejecting it would break the ordinary case.
    * `application/zip` never conflicts: OOXML and ODF files are ZIP
      containers, so libmagic reports the container while the client
      declares the specific member type.
    * Same top-level type never conflicts: `text/markdown` vs
      `text/plain`, `application/pdf` vs `application/x-pdf`. Clients
      and libmagic disagree about the subtype often enough that
      rejecting on subtype alone would be a false-positive machine.
    """
    if not declared or not detected:
        return False
    if declared == "application/octet-stream" or detected == "application/octet-stream":
        return False
    if declared == detected:
        return False
    if detected in ("application/zip", "application/x-zip-compressed"):
        return False
    return declared.split("/", 1)[0] != detected.split("/", 1)[0]
