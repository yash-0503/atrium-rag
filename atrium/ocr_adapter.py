"""
ocr_adapter.py

Optional, opt-in OCR adapters for scanned pages and image-only documents.
The rest of the system runs fine without this — users can install
`atrium-rag[ocr]` (and the host-side Tesseract binary) anytime.

Two providers, selected by the ``OCR_PROVIDER`` env var:

* ``tesseract`` (default) — `TesseractOCRAdapter`. Needs the Python
  wrapper *and* the Tesseract binary on PATH. Mature, broad language
  coverage.
* ``rapidocr`` — `RapidOCRAdapter`. pip-only, no system binary. PP-OCRv4
  packaged as ONNX Runtime; the models ship inside the wheel, so there
  is no download on first use. Better on CJK / Indic / mixed-script
  scans, where Tesseract gives up.

Both implement the OcrAdapterProtocol from universal_parser.py:
    transcribe(image_bytes, lang=None) -> str
    health() -> bool

Use :func:`build_ocr_adapter` to construct the right one from the
environment; it is what `atrium.main`, `atrium.worker` and the CLI all
call, so the three entry points cannot disagree about which provider
is in play.

Tesseract's constructor probes the binary eagerly so a missing install
fails with an actionable RuntimeError at server boot, not a confusing
traceback on first upload.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

log = logging.getLogger("ingest.ocr")

__all__ = ["TesseractOCRAdapter", "RapidOCRAdapter", "build_ocr_adapter"]


class TesseractOCRAdapter:
    """Synchronous single-image text extraction via Tesseract. Use this
    when the vision adapter isn't configured (no Ollama vision model,
    no hosted API) and you still want scanned content indexed as text.

    Returns plain text — no descriptions, no metadata, no coordinates.
    The vision adapter is the richer path (caption + chart/diagram
    description); this is the deterministic fallback for users who
    either don't have a vision model or want a pure-OCR pipeline.
    """

    def __init__(self, lang: str = "eng", tesseract_cmd: Optional[str] = None,
                 timeout: float = 30.0):
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract is required for OCR. Install with: "
                "pip install \"atrium-rag[ocr]\""
            ) from exc

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            from atrium.config import env_str

            configured = env_str("TESSERACT_CMD")
            if configured:
                pytesseract.pytesseract.tesseract_cmd = configured

        # Probe the binary NOW so a missing install fails with a clear,
        # actionable message at server boot — not a confusing
        # TesseractNotFoundError on the first scanned upload.
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "Tesseract binary not found on PATH. Install it (apt/brew/apk) "
                "or set the TESSERACT_CMD "
                f"env var to its full path. Underlying error: {exc}"
            ) from exc

        self.lang = lang
        self.timeout = timeout
        log.info("TesseractOCRAdapter ready (lang=%s)", lang)

    def transcribe(self, image_bytes: bytes, lang: Optional[str] = None) -> str:
        """Extract text via OCR. Never throws — always returns a string
        (possibly empty) so a problematic image becomes an empty caption
        chunk rather than a crashed job. OCR is synchronous by design:
        pages are processed sequentially, a single page at 150 dpi is
        sub-second, and queueing adds no value here."""
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            return pytesseract.image_to_string(
                img, lang=lang or self.lang, timeout=self.timeout,
            ).strip()
        except Exception as exc:
            log.warning("OCR transcription failed (%s: %s)", type(exc).__name__, exc)
            return ""

    def health(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False


class RapidOCRAdapter:
    """PP-OCRv4 via ONNX Runtime. No system binary, no model download.

    The models ship inside the `rapidocr-onnxruntime` wheel, so the
    first call does not reach out to the network — which is the whole
    reason to prefer this provider over Tesseract in a container.

    Language coverage is the trade-off: the bundled recognition model
    covers Chinese and English. `OCR_LANG` is accepted for interface
    parity with Tesseract and is ignored when it names a language the
    model cannot read; the adapter logs that once at construction
    rather than silently producing garbage.
    """

    #: Languages the bundled PP-OCRv4 recognition model covers.
    SUPPORTED_LANGS = frozenset({"eng", "en", "chi", "ch", "zh", "zho"})

    def __init__(self, lang: str = "eng", timeout: float = 30.0):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "rapidocr-onnxruntime is required for OCR_PROVIDER=rapidocr. "
                "Install with: pip install atrium-rag[ocr]"
            ) from exc

        if lang and lang.strip().lower() not in self.SUPPORTED_LANGS:
            log.warning(
                "OCR_PROVIDER=rapidocr with OCR_LANG=%s: the bundled PP-OCRv4 "
                "model reads Chinese and English only. Use "
                "OCR_PROVIDER=tesseract for other languages.",
                lang,
            )

        # Constructing the engine loads the ONNX models (~1-2 s). It is
        # done here rather than lazily so a broken install fails at boot
        # with a clear message, matching Tesseract's behaviour.
        # ONNX Runtime's default is one thread per core, per session.
        # Three sessions (detector, classifier, recogniser) times ten
        # cores is thirty threads, and a process that is also holding
        # another library's thread pool can exhaust the limit and take
        # the process down with a segfault. OCR is not throughput
        # critical on a page, so the sessions are pinned to a small
        # number of threads.
        threads = max(1, int(os.environ.get("ATRIUM_OCR_THREADS", "2")))
        try:
            self._engine = RapidOCR(
                det_intra_op_num_threads=threads,
                det_inter_op_num_threads=1,
                cls_intra_op_num_threads=threads,
                cls_inter_op_num_threads=1,
                rec_intra_op_num_threads=threads,
                rec_inter_op_num_threads=1,
            )
        except Exception as exc:
            raise RuntimeError(
                f"RapidOCR failed to initialize ({type(exc).__name__}: {exc}). "
                "Reinstall with: pip install --force-reinstall atrium-rag[ocr]"
            ) from exc

        self.lang = lang
        self.timeout = timeout
        log.info("RapidOCRAdapter ready (lang=%s)", lang)

    def transcribe(self, image_bytes: bytes, lang: Optional[str] = None) -> str:
        """Extract text via PP-OCRv4. Never throws — returns a string.

        The engine returns one entry per detected text line
        ([box, text, score]); the lines are joined with newlines in
        reading order.
        """
        try:
            result, _elapse = self._engine(image_bytes)
        except Exception as exc:
            log.warning("RapidOCR transcription failed (%s: %s)", type(exc).__name__, exc)
            return ""
        if not result:
            return ""
        lines = []
        for item in result:
            # Each item is [box, text, score]; be tolerant of shapes
            # that differ (a bare string, a dict) rather than crashing
            # the ingest on an unexpected engine version.
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = item[1]
            elif isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = item
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
        return "\n".join(lines)

    def health(self) -> bool:
        return self._engine is not None


def build_ocr_adapter(
    provider: Optional[str] = None,
    lang: Optional[str] = None,
    tesseract_cmd: Optional[str] = None,
) -> TesseractOCRAdapter | RapidOCRAdapter:
    """Build the OCR adapter named by ``OCR_PROVIDER``.

    Raises RuntimeError with an actionable message when the provider is
    unknown or its dependencies are missing — the caller decides
    whether that is fatal (the API) or a warning (the worker).
    """
    from atrium.config import env_str

    chosen = (provider or env_str("OCR_PROVIDER") or "tesseract").strip().lower()
    language = lang or env_str("OCR_LANG") or "eng"

    if chosen == "rapidocr":
        return RapidOCRAdapter(lang=language)
    if chosen in ("tesseract", ""):
        return TesseractOCRAdapter(
            lang=language,
            tesseract_cmd=tesseract_cmd or (env_str("TESSERACT_CMD") or None),
        )
    raise RuntimeError(
        f"Unknown OCR_PROVIDER {chosen!r}. Expected 'tesseract' or 'rapidocr'."
    )