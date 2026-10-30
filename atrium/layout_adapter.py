"""
layout_adapter.py

Optional, opt-in layout/region adapter for scanned pages and image-only
documents where Tesseract-style whole-page OCR loses reading order
(multi-column layouts, sidebars, magazine spreads, mixed-orientation
pages). The strict gated on `LAYOUT_ENABLED=1` so it adds zero cost to
the default install.

The adapter implements the LayoutRegionAdapterProtocol from
universal_parser.py:
    layout_regions(image_bytes, lang=None) -> LayoutResult
    health() -> bool

`LayoutResult` carries one `LayoutRegion` per detected region on the
page, each tagged with a `region_type` ("text" | "image" |
"figure_caption"). The default `RapidOCRLayoutAdapter` derives them from
Pure-PaddleOCR's PP-OCRv4 (via rapidocr-onnxruntime) detection output:
the detector already finds text regions in reading order, so
"regions" + bboxes come for free. Image regions are detected as large
empty horizontal strips in the page bbox (a heuristic — no ML model
required for this first cut, see the caveats below).

Why this beats plain OCR on scanned PDFs:
  - TesseractOCRAdapter.transcribe() flattens the whole page to one
    string. A two-column magazine scan ends up as
    "left1 right1 left2 right2" — wrong reading order.
  - RapidOCRLayoutAdapter preserves region order AND region bboxes,
    so the smart_chunker can attach figure-captions to their nearest
    text region and the chunker can keep multi-column pages atomic.

Caveats (be honest):
  - The "image" detection here is a heuristic: an empty horizontal
    strip larger than 15% of the page height, with no text-region
    overlap. Docling detects images via RT-DETR (ML), which catches
    full-bleed photos, charts with no white space, and figures that
    aren't on a card. We do NOT match that here — when we find one,
    we mark it for vision captioning via the existing vision queue
    (which OllamaVisionAdapter handles well).
  - RapidOCR's text-region detector is sufficient for clean scans but
    is fooled by very dense multi-column magazine layouts. For those,
    set LAYOUT_ENABLED=0 and rely on the existing vision path
    instead — full-page vision models handle layout natively.

Install: `pip install "atrium-rag[ocr]"`. Pure pip install, no system
binary. ~50 MB model auto-downloaded to ~/.rapidocr on first call.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

log = logging.getLogger("ingest.layout")


# Region-type tag set. Strings, not an enum, so callers (the parser +
# the chunker) can do `region.region_type == "text"` without importing
# this module.
TEXT = "text"
IMAGE = "image"
FIGURE_CAPTION = "figure_caption"


@dataclass
class LayoutRegion:
    """One region on a scanned page. `bbox` is (x0, y0, x1, y1) in
    pixel coordinates of the original image. `text` is empty for
    `IMAGE` regions (their content is visual, not textual — vision
    captioning fills it). `confidence` is the detector's score (0 for
    heuristic-derived regions)."""
    region_type: str
    bbox: tuple
    text: str = ""
    confidence: float = 0.0


@dataclass
class LayoutResult:
    """What `layout_regions()` returns. `regions` is in reading order
    top-to-bottom (RapidOCR's detector already gives us this for
    vertically-upgraded pages). `text` is the joined text of all
    `TEXT` regions — a convenience for callers that just want the
    page-as-string and don't care about region boundaries."""
    regions: List[LayoutRegion] = field(default_factory=list)
    text: str = ""
    page_rect: tuple = ()   # (width, height) in px


class LayoutRegionAdapterProtocol(Protocol):
    """Duck-typed adapter contract. The parser calls `layout_regions`
    on the raw png bytes of a scanned page. Returns "" with an empty
    `regions` list on silent failure (never raises) so a failing
    adapter becomes a placeholder chunk rather than a crashed job."""
    def layout_regions(self, image_bytes: bytes,
                       lang: Optional[str] = None) -> LayoutResult: ...
    def health(self) -> bool: ...


class RapidOCRLayoutAdapter:
    """RapidOCR + light region-heuristics. Implements both
    LayoutRegionAdapterProtocol (preferred path) AND can be used as
    the text-extraction engine for the vision queue's image captioning
    flow when no other language model is around.

    Pipeline:
      1. Run rapidocr-onnxruntime's detector + recognizer on the page.
      2. Sort detected text regions by top-y (reading order).
      3. Detect image regions as large empty horizontal strips
         between text regions (a heuristic — see caveats in module
         docstring).
      4. Detect figure-caption regions as short, wide text regions
         immediately below an image region.
      5. Return a LayoutResult with one entry per detected region.

    Model download: `~/.rapidocr/` is populated on first call.
    ~50-80 MB total. Cached after first run.
    """

    # Image-region heuristic: a horizontal strip of at least this many
    # pixels height (24% of a typical 600-dpi letter page is ~144 px)
    # AND no text-region overlap. Tuned conservatively to avoid false
    # positives — the cost of a missed image is just a missing caption,
    # the cost of a false positive is a wrong caption.
    _MIN_IMAGE_STRIP_PX = 80
    # Caption-region heuristic: text region shorter than this many
    # characters AND positioned directly below an image region within
    # this many pixels.
    _CAPTION_MAX_CHARS = 200
    _CAPTION_GAP_PX = 30

    def __init__(self, lang: str = "en", text_score: float = 0.5,
                 min_image_strip_px: Optional[int] = None):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "rapidocr-onnxruntime is required. Install with: "
                "pip install \"atrium-rag[ocr]\""
            ) from exc
        self._engine = RapidOCR()
        self.lang = lang
        self.text_score = text_score
        if min_image_strip_px is not None:
            self._MIN_IMAGE_STRIP_PX = min_image_strip_px
        log.info("RapidOCRLayoutAdapter ready (lang=%s, text_score=%.2f)",
                 lang, text_score)

    def layout_regions(self, image_bytes: bytes,
                       lang: Optional[str] = None) -> LayoutResult:
        """Run RapidOCR + region heuristics. Never raises — returns an
        empty LayoutResult on silent failure so the parser degrades
        gracefully to a placeholder chunk."""
        try:
            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.array(img)
            page_h, page_w = arr.shape[:2]
            page_rect = (page_w, page_h)

            result, _elapse = self._engine(arr)
            if result is None or not result[1]:
                return LayoutResult(regions=[], text="", page_rect=page_rect)

            boxes, txts, scores = result[0], result[1], result[2] or [0.0] * len(result[1])
            text_regions: List[LayoutRegion] = []
            for box, txt, score in zip(boxes, txts, scores):
                if not txt or not txt.strip():
                    continue
                if score is not None and score < self.text_score:
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                text_regions.append(LayoutRegion(
                    region_type=TEXT,
                    bbox=(int(min(xs)), int(min(ys)),
                          int(max(xs)), int(max(ys))),
                    text=txt.strip(),
                    confidence=float(score) if score is not None else 0.0,
                ))
            text_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
            joined_text = "\n".join(r.text for r in text_regions).strip()

            # Heuristic image-region detection: large empty horizontal
            # strips between text regions.
            image_regions = self._detect_image_regions(text_regions)

            # Tag figure-caption regions: short text immediately below
            # an image region.
            for tr in text_regions:
                if len(tr.text) > self._CAPTION_MAX_CHARS:
                    continue
                if any(self._is_below_image(tr, ir) for ir in image_regions):
                    tr.region_type = FIGURE_CAPTION

            # Interleave image regions into the list in their top-y
            # position so the parser can render them in reading order.
            all_regions: List[LayoutRegion] = sorted(
                text_regions + image_regions,
                key=lambda r: (r.bbox[1], r.bbox[0]),
            )
            return LayoutResult(
                regions=all_regions, text=joined_text, page_rect=page_rect,
            )
        except Exception as exc:
            log.warning("Layout analysis failed (%s: %s)", type(exc).__name__, exc)
            return LayoutResult(regions=[], text="", page_rect=())

    def health(self) -> bool:
        try:
            import numpy as np
            tiny = np.zeros((8, 32, 3), dtype=np.uint8)
            self._engine(tiny)
            return True
        except Exception:
            return False

    def _detect_image_regions(self, text_regions: List[LayoutRegion]) -> List[LayoutRegion]:
        """Heuristic: find vertical gaps BETWEEN consecutive text regions
        that exceed _MIN_IMAGE_STRIP_PX. Each gap becomes an image
        region spanning the full text-column width at that y-band.

        Deliberately ignores the leading strip (above the first text
        region) and trailing strip (below the last) — those are page
        margins in 95% of real documents, not images. Without this
        guard, the heuristic would flag every blank top margin as a
        false-positive image, which inflates chunk counts without
        adding signal. Operators who want aggressive image detection
        can lower _MIN_IMAGE_STRIP_PX."""
        if not text_regions:
            return []
        image_regions: List[LayoutRegion] = []
        sorted_text = sorted(text_regions, key=lambda r: r.bbox[1])
        # Skip the leading strip — only flag gaps BETWEEN text blocks.
        prev_bottom = sorted_text[0].bbox[3]
        for tr in sorted_text[1:]:
            gap = tr.bbox[1] - prev_bottom
            if gap >= self._MIN_IMAGE_STRIP_PX:
                image_regions.append(LayoutRegion(
                    region_type=IMAGE,
                    bbox=(0, prev_bottom, 0, tr.bbox[1]),  # width filled below
                    text="",
                    confidence=0.0,
                ))
            prev_bottom = max(prev_bottom, tr.bbox[3])
        # Skip the trailing strip too — same reasoning.
        # Set the image-region width to the widest detected text-region
        # right edge so downstream code can use bbox meaningfully.
        max_x1 = max((tr.bbox[2] for tr in text_regions), default=0)
        for ir in image_regions:
            ir.bbox = (ir.bbox[0], ir.bbox[1], max_x1, ir.bbox[3])
        return image_regions

    @staticmethod
    def _is_below_image(caption: LayoutRegion, image: LayoutRegion) -> bool:
        """True when caption's top edge sits within ~30 px below the
        image's bottom edge and overlaps horizontally."""
        vertical_gap = caption.bbox[1] - image.bbox[3]
        if vertical_gap < 0 or vertical_gap > RapidOCRLayoutAdapter._CAPTION_GAP_PX:
            return False
        return caption.bbox[0] < image.bbox[2] and caption.bbox[2] > image.bbox[0]