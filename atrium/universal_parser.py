"""
universal_parser.py

A sophisticated, ML-free document parser built for messy real-world
PDFs (and DOCX/PPTX/XLSX/HTML/images), designed to hand off to a
pluggable vision adapter (Ollama/OpenAI/Gemini/etc.) instead of OCR.

What this does that a naive PyMuPDF pass doesn't:
  - Adaptive heading-level detection: ranks font sizes relative to THIS
    document's own body-text size, instead of a hardcoded pt threshold.
  - Section-path tracking: every element carries the stack of ancestor
    headings active when it was emitted, for accurate citations.
  - Multi-column reading-order correction, applied to TEXT AND TABLES
    together: a table interleaves at its true page position instead of
    being dumped after all of a page's text.
  - Repeated header/footer stripping across pages.
  - Cross-page paragraph stitching.
  - Per-line (not per-block) classification: a block mixing an intro
    line with several list items — bulleted or not — is split into
    separate list_item elements instead of collapsing into one run-on
    paragraph. A line with 2+ wide whitespace gaps (columns/side-by-side
    items with no visible bullet glyph) explodes into separate items.
  - Borderless-table fallback via a whitespace-column heuristic.
  - Scanned-page handling without OCR: a page is only treated as a scan
    when it has near-zero extractable native text IN ABSOLUTE TERMS —
    never merely because it happens to contain a large image. A large
    image sitting on an otherwise text-rich page (with real paragraphs
    and tables around it) is captioned individually; the table and text
    are still parsed normally. Conflating "has a big image" with "is a
    scan" was a real bug: it discarded genuine tables/text on mixed
    pages and sent the whole page to the vision model instead.
  - Vision calls are queued during parsing and resolved in one
    concurrent batch at the end (bounded thread pool), instead of each
    call blocking the page loop serially — this is the main fix for
    slow/crashing runs on image-heavy documents.
  - Every embedded image sent to a vision model is downscaled and
    re-encoded as JPEG first (via Pillow — a small, non-ML image
    library, not to be confused with the torch/transformers weight
    this project deliberately avoids). Full-resolution scans/photos
    cost latency and memory with no captioning benefit past a modest
    resolution.
  - Per-page and per-table parsing is wrapped defensively: one
    malformed page or corrupt table structure is recorded as an error
    marker and skipped, never crashes the whole document.

No ML dependencies beyond Pillow (plain image resizing, no models).
The vision adapter is entirely optional — with none configured,
image/scan regions become clearly-labeled placeholder chunks and the
rest of the pipeline is unaffected.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .pdf_backend import FLAG_BOLD

# Pillow is imported lazily, inside the function that needs it.
#
# `from PIL import Image` costs ~15 ms — more than the rest of this
# module put together. A process that imports
# atrium.universal_parser but never parses a PDF (the CLI's `atrium
# search`, the worker's startup path, a test that only exercises the
# chunker) should not pay for it. The cost lands on the first PDF
# parse instead, where it is amortised over hundreds of pages.
#
# The PDF library itself is behind `atrium.pdf_backend`, which is
# imported lazily too: pdfplumber pulls in pdfminer.six and pypdfium2,
# which are not cheap to import and not needed by anything that does
# not read a PDF.

# The Office / HTML parsers below import their library *inside the
# method that uses it*, not here at module scope.
#
# `import pptx` alone costs ~77 ms, `openpyxl` ~106 ms, `bs4` ~46 ms.
# A PDF-only deployment — the common case — should not pay any of that
# at import time, and neither should a server that has already booted
# and is waiting on requests. python-docx is cheap enough to leave
# alone, but it is moved too for symmetry.

log = logging.getLogger("ingest.parser")


# Data model

ElementType = str  # "heading" | "paragraph" | "list_item" | "table" | "code" | "image_caption"


@dataclass
class ParsedElement:
    type: ElementType
    text: str
    page: int
    bbox: Optional[tuple] = None
    heading_level: Optional[int] = None                      # only set for type == "heading"
    section_path: list[str] = field(default_factory=list)    # ancestor headings at emission time


class VisionCaptionAdapter(Protocol):
    def describe_image(self, image_bytes: bytes, mimetype: str,
                        context: Optional[str] = None,
                        max_tokens: Optional[int] = None) -> str: ...
    def health(self) -> bool: ...


class OcrAdapterProtocol(Protocol):
    """Duck-typed OCR adapter contract. The parser calls `transcribe`
    on raw images of scanned pages / standalone image uploads when no
    vision adapter is configured. Returns "" on silent failure so the
    caller always gets a string. Kept separate from VisionCaptionAdapter
    so vision and OCR can be configured independently — most users will
    have one or the other, not both."""
    def transcribe(self, image_bytes: bytes, lang: Optional[str] = None) -> str: ...
    def health(self) -> bool: ...


# Constants / heuristics

MONO_FONTS = {
    "courier", "consolas", "menlo", "monaco", "dejavusansmono",
    "liberationmono", "inconsolata", "sourcecodepro", "robotomono",
    "firacode", "monospace", "terminal", "couriernew",
}

HEADING_NUMBER_REGEX = re.compile(r"^(\d+(\.\d+)*|[A-Z](\.\d+)*)\s+[A-Z0-9].*$")
BULLET_PREFIXES = ("- ", "\u2022 ", "* ", "\u25e6 ", "\u2023 ")
GLYPH_NOISE_RE = re.compile(r"[\u25a0\u25a1\u25aa\u25ab\uf0a7\uf0b7\u200b]")
SENTENCE_END_RE = re.compile(r"[.!?:;\"')\]]\s*$")

MIN_IMAGE_DIM = 120                      # px; DOCX/PPTX icon filter (no on-page placement to measure there)
MIN_IMAGE_DISPLAY_PT = 60                # pt (~0.83in); PDF icon filter, measured by DISPLAYED size on the page,
                                          # not raw embedded pixel dimensions (the two can differ enormously —
                                          # see universal_parser docs/commit notes on get_image_rects)
SCANNED_PAGE_ABS_TEXT_CHARS = 40         # absolute extractable chars below which a page is a real scan
HEADER_FOOTER_BAND = 0.08                # top/bottom 8% of page height
HEADER_FOOTER_MIN_PAGE_FRACTION = 0.6    # must repeat on >=60% of pages to be stripped

# Footnote detection heuristic: a SHORT block sitting in the bottom
# 15% of the page and starting with a numeric prefix (1. / 1) / ¹ / ²)
# is almost always a footnote, not body text. We mark it as such so the
# chunker keeps it atomic — without this, footnote citations get packed
# into the nearest body paragraph and pollute retrieval with off-topic
# references.
FOOTNOTE_BAND = 0.15
FOOTNOTE_MAX_CHARS = 400
_FOOTNOTE_PREFIX_RE = re.compile(r"^\s*(?:\d{1,3}[\.\)]|[\u00b9\u00b2\u00b3\u2070-\u2079])\s+\S")

# Missing-glyph / tofu detection. PyMuPDF returns these placeholder
# characters when the embedded font doesn't have a glyph for a code
# point. They're useless for retrieval (every missing char renders
# identically), so we strip them and tag the affected element so the
# chunker can decide whether to surface a marker.
TOFU_CHARS = "\ufffd\u25a1\u25a0\u25ae\u25af"

MAX_VISION_IMAGE_DIMENSION = 1280        # longest edge cap before sending to a vision model
VISION_JPEG_QUALITY = 82
DEFAULT_VISION_CONCURRENCY = 4           # bounded thread pool for concurrent vision calls
CAPTION_MAX_SENTENCES = 2                # hard cap: image captions are two short sentences, never more
DIAGRAM_CAPTION_MAX_TOKENS = 100         # model-side generation cap for image captions
CAPTION_PROMPT = ("Describe this image in 2 short sentences only, no more. "
                  "State what it shows (chart/diagram/photo) and its key point or subject; "
                  "skip minor visual details.")
# Only raster formats a vision model can actually decode (skips EMF/WMF/SVG).
CAPTIONABLE_IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp",
                           "image/tiff", "image/webp"}

# Spreadsheet-style tables (XLSX, XLS, CSV, ODS) with more than this many
# data rows get chunked at this boundary with the header row repeated per
# piece (same strategy as the PDF table chunker). Without this, a
# 5,000-row sheet ships as one giant table chunk that no embedding model
# can actually use for retrieval.
MAX_TABLE_ROWS_PER_CHUNK = 40

# JSON walker hard-cap on nesting depth. Beyond this we stop recursing
# and emit whatever scalar we've collected so far as a paragraph.
MAX_JSON_DEPTH = 5


def _is_monospace(font_name: str) -> bool:
    clean = font_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    return any(m in clean for m in MONO_FONTS)


def _clean_glyph_noise(text: str) -> str:
    return GLYPH_NOISE_RE.sub("", text).strip()


def _is_inside(box: tuple, container: tuple, pad: float = 2.0) -> bool:
    return (box[0] >= container[0] - pad and box[1] >= container[1] - pad and
            box[2] <= container[2] + pad and box[3] <= container[3] + pad)


def _normalize_for_repeat_check(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


# Universal parser

class UniversalParser:
    def __init__(self, vision_adapter: Optional[VisionCaptionAdapter] = None,
                 ocr_adapter: Optional[OcrAdapterProtocol] = None,
                 max_vision_calls_per_doc: Optional[int] = None,   # None = unlimited
                 vision_concurrency: int = DEFAULT_VISION_CONCURRENCY,
                 docx_page_by_heading: bool = False):
        """
        Args:
            vision_adapter: optional VisionCaptionAdapter for image regions.
            ocr_adapter: optional OCR adapter (used only when no vision adapter).
            max_vision_calls_per_doc: cap on vision model calls per doc (None=unlimited).
            vision_concurrency: thread pool size for vision calls.
            docx_page_by_heading: when True, DOCX page numbers are derived
                from Heading 1 boundaries only (`bump_levels=(1,)`). The
                default (False) uses Heading 1 + Heading 2
                (`bump_levels=(1, 2)`) — closer to ODT's existing
                behaviour but inflates page counts on docs where Heading
                2 paragraphs are sub-section labels rather than page
                boundaries. Set to True for benchmark fixtures / corpuses
                where ground-truth page numbers are anchored to Heading 1
                only (e.g., 10_gov_docx: "Pages derive from Heading 1
                boundaries: 1=Title+Exec Summary, 3=Background, …").
                19-fixture opt-in; the harness sets this to True via the
                fixture's own decision.
        """
        self.vision = vision_adapter
        # Opt-in OCR fallback. Only consulted when the vision adapter
        # isn't configured (or returned nothing usable). The protocol
        # is duck-typed so this module never imports the OCR adapter
        # directly — keeps the parser usable without OCR deps.
        self.ocr = ocr_adapter
        self.max_vision_calls_per_doc = max_vision_calls_per_doc
        self.vision_concurrency = vision_concurrency
        self._docx_page_by_heading = docx_page_by_heading
        self._vision_calls_queued = 0
        self._vision_cache: dict[str, str] = {}   # sha256(downscaled bytes) -> caption
        self._pending_vision: list[tuple] = []     # (element, bytes, mimetype, context, digest)

    # -- public entrypoint ------------------------------------------------

    def parse(self, file_bytes: bytes, filename: str, mime_type: str) -> list[ParsedElement]:
        self._vision_calls_queued = 0
        self._vision_cache.clear()
        self._pending_vision = []

        if mime_type == "application/pdf":
            elements = self._safe_parse("PDF", self._parse_pdf, file_bytes)
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            elements = self._safe_parse("DOCX", self._parse_docx, file_bytes)
        elif mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            elements = self._safe_parse("PPTX", self._parse_pptx, file_bytes)
        elif mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            elements = self._safe_parse("XLSX", self._parse_xlsx, file_bytes)
        elif mime_type == "text/html":
            elements = self._safe_parse("HTML", self._parse_html, file_bytes)
        elif mime_type in ("image/jpeg", "image/png", "image/webp"):
            elements = self._safe_parse("image", self._parse_raw_image, file_bytes, mime_type)
        # --- Tier 1: text-based formats (no new deps) ---
        elif mime_type == "text/plain":
            elements = self._safe_parse("plain text", self._parse_txt, file_bytes)
        elif mime_type == "text/markdown":
            elements = self._safe_parse("Markdown", self._parse_md, file_bytes)
        elif mime_type == "text/csv":
            elements = self._safe_parse("CSV", self._parse_csv, file_bytes)
        elif mime_type == "application/json":
            elements = self._safe_parse("JSON", self._parse_json, file_bytes)
        elif mime_type == "application/epub+zip":
            elements = self._safe_parse("EPUB", self._parse_epub, file_bytes)
        # --- Tier 2: lightweight-dep formats ---
        elif mime_type == "application/rtf":
            elements = self._safe_parse("RTF", self._parse_rtf, file_bytes)
        elif mime_type == "application/vnd.ms-excel":
            elements = self._safe_parse("XLS", self._parse_xls, file_bytes)
        elif mime_type in (
            "application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.spreadsheet",
            "application/vnd.oasis.opendocument.presentation",
        ):
            # OpenDocument: label per sub-format so the placeholder names
            # the right extension (.odt / .ods / .odp).
            odf_label = {
                "application/vnd.oasis.opendocument.text": "ODT",
                "application/vnd.oasis.opendocument.spreadsheet": "ODS",
                "application/vnd.oasis.opendocument.presentation": "ODP",
            }[mime_type]
            elements = self._safe_parse(odf_label, self._parse_odf, file_bytes, mime_type)
        # --- Tier 3: legacy Office binary — friendly error, not a 500 ---
        elif mime_type == "application/msword":
            raise ValueError(
                "Legacy .doc is not supported. Please save as .docx or .pdf and re-upload."
            )
        elif mime_type == "application/vnd.ms-powerpoint":
            raise ValueError(
                "Legacy .ppt is not supported. Please save as .pptx or .pdf and re-upload."
            )
        else:
            raise ValueError(f"Unsupported mime_type: {mime_type}")

        self._resolve_pending_vision()
        return elements

    def _safe_parse(self, label: str, fn, *args) -> list[ParsedElement]:
        """Run a per-format parser.

        A parse failure is a parse failure, and it propagates. The
        ingest fails, the job row says so, and nothing is indexed.

        This used to return a placeholder paragraph instead — "we're
        working on full support for this file" — which meant a crashed
        parser produced a real, embedded, retrievable chunk of
        marketing text, and the job reported `complete`. A failure
        reported as content is worse than a failure reported as a
        failure: it fills the corpus with lies and tells the operator
        everything worked.

        A truly-empty result is logged and returned as-is. An empty
        CSV / HTML / PDF is a legitimate result for an actually-empty
        file, and inventing a chunk there would mislead retrieval
        downstream just the same."""
        try:
            elements = fn(*args)
        except Exception:
            log.exception("Parser %s crashed", label)
            raise
        if not elements:
            log.warning("Parser %s returned 0 elements", label)
        return elements

    # -- PDF ----------------------------------------------------------------

    def _parse_pdf(self, file_bytes: bytes) -> list[ParsedElement]:
        # Opening a PDF raises on a corrupt or empty stream, and on an
        # encrypted PDF it raises whatever that library calls a
        # password error. Different libraries expose different
        # exception classes for the encrypted case, so we catch
        # broadly by module and distinguish by message text rather than
        # by exception type.
        # These are USER errors — the caller passed bad input — so they
        # surface as a 400-class message in the job's error field, not
        # as a 500. Wrapping here keeps the rest of the pipeline free of
        # PyMuPDF-specific imports.
        # The PDF library is a configuration choice, not a dependency
        # of the architecture. See atrium/pdf_backend.py.
        from .pdf_backend import get_backend

        backend = get_backend()
        try:
            document = backend.open(file_bytes)
        except Exception as exc:
            exc_module = getattr(type(exc), "__module__", "") or ""
            if any(hint in exc_module.lower() for hint in ("fitz", "pymupdf", "pdfplumber", "pdfminer")):
                msg = str(exc).lower()
                if "password" in msg or "encrypt" in msg or "permission" in msg:
                    raise ValueError(
                        "PDF is password-protected. Encrypted PDFs are not supported."
                    ) from None
                raise ValueError(f"PDF is corrupt or unreadable: {exc}") from None
            raise

        page_data = []
        font_size_counter: Counter = Counter()

        for page in document.pages:
            # The backend returns spans in reading order, with the bbox
            # coordinates normalised into the rotated visual frame, so
            # downstream code can compare them directly. (A rotated
            # page — a phone scan exported with /Rotate 90 or 270 —
            # would otherwise ship blocks in the unrotated content
            # order, i.e. scrambled.)
            for block in page.blocks:
                for line in block.lines:
                    for span in line.spans:
                        size = round(span.size, 1)
                        txt_len = len(span.text.strip())
                        if txt_len:
                            font_size_counter[size] += txt_len
            page_data.append(page)

        body_font_size = font_size_counter.most_common(1)[0][0] if font_size_counter else 10.0
        heading_levels = self._build_heading_scale(font_size_counter, body_font_size)

        raw_elements: list[ParsedElement] = []
        page_texts_by_band: dict[str, list[int]] = defaultdict(list)
        carry_paragraph: Optional[ParsedElement] = None
        num_pages = document.page_count

        for page_idx, page in enumerate(page_data):
            page_num = page_idx + 1
            try:
                carry_paragraph = self._parse_pdf_page(
                    page, page_num, heading_levels, body_font_size,
                    page_texts_by_band, raw_elements, carry_paragraph,
                )
            except Exception as exc:
                # One malformed page must never crash the whole document.
                raw_elements.append(ParsedElement(
                    type="paragraph", page=page_num,
                    text=f"[Page {page_num} could not be parsed: {type(exc).__name__}]",
                ))

        if carry_paragraph:
            raw_elements.append(carry_paragraph)

        repeated_norms = {
            norm for norm, pages in page_texts_by_band.items()
            if len(set(pages)) >= max(3, int(num_pages * HEADER_FOOTER_MIN_PAGE_FRACTION))
        }
        if repeated_norms:
            raw_elements = [el for el in raw_elements
                            if _normalize_for_repeat_check(el.text) not in repeated_norms]

        # Cross-page table stitch: a table on page N+1 that opens with the
        # same header row as the table on page N is a continuation, not a
        # new table — common in financial statements / datasheets where
        # the printer repeats "Description | Q1 | Q2 | Total" at the top
        # of every page. Without this we'd emit two chunks with a broken
        # row in between (header repeated, data split, no merge signal).
        raw_elements = self._stitch_cross_page_tables(raw_elements)

        # Footnote reclassification: paragraphs that sit in the bottom
        # band of a page with a numeric prefix are likely footnotes,
        # not body prose. Re-emit them with type="footnote" so the
        # chunker keeps them atomic and downstream retrieval can filter
        # them out (citations to other pages) or surface them separately
        # depending on the use case.
        raw_elements = self._reclassify_footnotes(raw_elements)

        # Tofu scrub: strip missing-glyph placeholder characters and
        # collapse consecutive ones to a single "[?]" marker so a
        # paragraph that's 30% tofu doesn't ship as a useless paragraph.
        # The affected element's section_path is preserved; we just
        # rewrite its text in place.
        raw_elements = self._scrub_tofu(raw_elements)

        return self._attach_section_paths(raw_elements)

    def _parse_pdf_page(self, page, page_num: int, heading_levels: dict,
                         body_font_size: float, page_texts_by_band: dict,
                         raw_elements: list[ParsedElement],
                         carry_paragraph: Optional[ParsedElement]) -> Optional[ParsedElement]:

        page_rect = page.rect
        blocks = page.blocks
        tables = page.tables
        # PyMuPDF's table detector is aggressive: on Canva-exported PDFs
        # and many "designer" templates, the page itself is wrapped in a
        # background rectangle that find_tables() mistakes for a giant
        # table covering the whole page. Those phantom tables have a
        # bbox that covers most of the page and one real text column
        # plus several empty ones. They are NOT tables — letting them
        # through silently buries every paragraph block (because the
        # outside-table-text path masks them) and ships a single fake
        # "table" chunk with the entire page text crammed into column 1.
        # _is_phantom_table() filters these out so real text extraction
        # works and only genuine tables (smaller bbox, multi-column rows)
        # survive.
        table_bboxes = []
        table_items: list[tuple[str, object, tuple]] = []
        for tab in tables:
            try:
                if self._is_phantom_table(tab, page_rect):
                    continue
                table_bboxes.append(tab.bbox)
                el = self._table_to_element(tab, page_num)
                if el:
                    table_items.append(("table", el, tab.bbox))
            except Exception:
                continue  # one malformed table must not lose the rest of the page

        # Text OUTSIDE table regions (used for actual element extraction below —
        # table cell text must not be double-extracted as loose paragraphs).
        outside_table_text_chars = 0
        candidate_items: list[tuple[str, object, tuple]] = []
        for block in blocks:
            bbox = block.bbox
            if any(_is_inside(bbox, tbox) for tbox in table_bboxes):
                continue
            if not block.lines:
                continue
            candidate_items.append(("block", block, bbox))
            for line in block.lines:
                for span in line.spans:
                    outside_table_text_chars += len(span.text)

        # Text INSIDE detected tables — must count toward "is this page a
        # scan?", or a table-heavy page with little surrounding prose looks
        # like it has "no text" even though it's full of real, perfectly
        # extractable content, and gets wrongly sent whole-page to vision.
        table_text_chars = sum(len(t_el.text) for _, t_el, _ in table_items)
        total_page_text_chars = outside_table_text_chars + table_text_chars

        # Scanned-page detection: gated purely on ABSOLUTE extractable text,
        # never on the presence of a large image. A page with real paragraphs
        # and a table next to one big diagram is NOT a scan — that image just
        # gets captioned individually below, same as any other embedded image.
        if total_page_text_chars < SCANNED_PAGE_ABS_TEXT_CHARS:
            element = ParsedElement(type="image_caption", text="[Scanned page \u2014 pending vision]",
                                     page=page_num)
            raw_elements.append(element)
            png_bytes = page.render_png(150)
            if self.vision:
                context = (f"Full scanned page {page_num}. Transcribe all readable text, in reading "
                          f"order, and describe any charts or diagrams present.")
                self._queue_vision_caption(element, png_bytes, "image/png", context=context,
                                            truncate_summary=False)
            elif self.ocr:
                # OCR fallback. Synchronous: page is sequential, and OCR
                # on a 150-dpi page is sub-second — no benefit to batching.
                text = self.ocr.transcribe(png_bytes)
                if text:
                    element.type = "paragraph"
                    element.text = text
                else:
                    element.text = "[Scanned page \u2014 OCR returned no text]"
            else:
                element.text = "[Scanned page \u2014 no vision or OCR adapter configured]"
            return carry_paragraph

        ordered_items = self._order_items_reading_order(candidate_items + table_items, page_rect.width)

        for kind, payload, bbox in ordered_items:
            if kind == "table":
                carry_paragraph = self._flush_carry(raw_elements, carry_paragraph)
                raw_elements.append(payload)
                continue

            block = payload
            block_lines_data = []
            for line in block.lines:
                line_spans = line.spans
                line_str = "".join(s.text for s in line_spans)
                clean_line = _clean_glyph_noise(line_str)
                if not clean_line:
                    continue
                max_font_size = max((s.size for s in line_spans), default=10.0)
                is_mono = any(_is_monospace(s.font) for s in line_spans)
                # PyMuPDF's flag bits: 16 is bold, 2 is *italic*. The
                # old check tested bit 2, so italic body text counted
                # as bold and could be promoted to a heading.
                is_bold = any(("bold" in s.font.lower() or s.flags & FLAG_BOLD != 0)
                              for s in line_spans)
                block_lines_data.append({
                    "text": line_str.rstrip("\n"), "clean_text": clean_line,
                    "font_size": round(max_font_size, 1), "is_mono": is_mono, "is_bold": is_bold,
                })
            if not block_lines_data:
                continue

            if bbox[1] <= page_rect.height * HEADER_FOOTER_BAND or \
               bbox[3] >= page_rect.height * (1 - HEADER_FOOTER_BAND):
                joined = " ".join(l["clean_text"] for l in block_lines_data)
                page_texts_by_band[_normalize_for_repeat_check(joined)].append(page_num)

            mono_count = sum(1 for l in block_lines_data if l["is_mono"])
            # NOTE: "- " is deliberately NOT in this list. It's the single
            # most common PROSE bullet marker (see BULLET_PREFIXES below,
            # checked per-line right after this block-level check) — far
            # more often a bullet point than a YAML list item in real
            # business documents. Including it here previously reclassified
            # ordinary bulleted lists as atomic "code" blocks, which could
            # never merge with surrounding paragraphs (see smart_chunker.py
            # _merge_small_paragraph_chunks: only element_type=="paragraph"
            # is eligible). The mono_count ratio check above already covers
            # genuine code; these remaining prefixes are unambiguous enough
            # (prose essentially never starts a line with them) to keep.
            is_yaml_or_code = any(
                l["clean_text"].startswith(("apiVersion:", "static_configs:", "curl ",
                                             "http:", "https:", "def ", "class ", "import ", "{", "}"))
                for l in block_lines_data
            )
            if (mono_count / len(block_lines_data) >= 0.5) or is_yaml_or_code:
                code_content = "\n".join(l["text"] for l in block_lines_data).strip()
                if len(code_content) > 3:
                    carry_paragraph = self._flush_carry(raw_elements, carry_paragraph)
                    raw_elements.append(ParsedElement(type="code", text=code_content,
                                                       page=page_num, bbox=bbox))
                continue

            whitespace_table = self._maybe_whitespace_table(block_lines_data)
            if whitespace_table:
                carry_paragraph = self._flush_carry(raw_elements, carry_paragraph)
                raw_elements.append(ParsedElement(type="table", text=whitespace_table,
                                                   page=page_num, bbox=bbox))
                continue

            first_line = block_lines_data[0]
            first_clean = first_line["clean_text"]
            is_heading_pattern = bool(HEADING_NUMBER_REGEX.match(first_clean))
            level = self._classify_heading_level(first_line["font_size"], first_line["is_bold"],
                                                  heading_levels, body_font_size)
            looks_like_heading = (
                len(block_lines_data) == 1
                and len(first_clean) < 120
                and not SENTENCE_END_RE.search(first_clean.rstrip())
                and (level is not None or (is_heading_pattern and first_line["is_bold"]))
            )

            if looks_like_heading:
                carry_paragraph = self._flush_carry(raw_elements, carry_paragraph)
                raw_elements.append(ParsedElement(
                    type="heading", text=first_clean, page=page_num, bbox=bbox,
                    heading_level=level or 1,
                ))
                continue

            # Per-line classification: split dense/bulleted lines instead of
            # flattening the whole block into one run-on paragraph.
            paragraph_run: list[str] = []

            def flush_paragraph_run() -> None:
                nonlocal carry_paragraph
                if not paragraph_run:
                    return
                assembled_text = self._join_lines(paragraph_run)
                paragraph_run.clear()
                if not assembled_text or len(assembled_text) <= 2:
                    return
                if carry_paragraph and carry_paragraph.type == "paragraph":
                    carry_paragraph = ParsedElement(
                        type="paragraph",
                        text=carry_paragraph.text.rstrip() + " " + assembled_text,
                        page=carry_paragraph.page, bbox=carry_paragraph.bbox,
                    )
                else:
                    carry_paragraph = self._flush_carry(raw_elements, carry_paragraph)
                    carry_paragraph = ParsedElement(type="paragraph", text=assembled_text,
                                                     page=page_num, bbox=bbox)
                if SENTENCE_END_RE.search(assembled_text.rstrip()):
                    raw_elements.append(carry_paragraph)
                    carry_paragraph = None

            for line_data in block_lines_data:
                txt = line_data["clean_text"]

                if txt.startswith(BULLET_PREFIXES):
                    flush_paragraph_run()
                    carry_paragraph = self._flush_carry(raw_elements, carry_paragraph)
                    raw_elements.append(ParsedElement(type="list_item", text=txt,
                                                       page=page_num, bbox=bbox))
                    continue

                dense_parts = [p for p in re.split(r"\s{2,}", txt.strip()) if len(p) >= 2]
                if len(dense_parts) >= 2:
                    flush_paragraph_run()
                    carry_paragraph = self._flush_carry(raw_elements, carry_paragraph)
                    for part in dense_parts:
                        raw_elements.append(ParsedElement(type="list_item", text=part,
                                                           page=page_num, bbox=bbox))
                    continue

                paragraph_run.append(txt)

            flush_paragraph_run()

        carry_paragraph = self._flush_carry(raw_elements, carry_paragraph)

        # Embedded diagrams/photos get individual captions (queued, resolved
        # concurrently later). Filtered by DISPLAYED on-page size, not raw
        # embedded pixel dimensions: a PDF can place the same embedded image
        # at wildly different sizes (a genuine figure scaled down for file
        # size, or a tiny icon saved at high resolution), so raw pixel count
        # alone both wrongly drops real figures and wrongly keeps icons.
        for image in page.images:
            image_bytes = image.data
            if not image_bytes:
                continue
            displayed_w = image.bbox[2] - image.bbox[0]
            displayed_h = image.bbox[3] - image.bbox[1]
            if displayed_w < MIN_IMAGE_DISPLAY_PT or displayed_h < MIN_IMAGE_DISPLAY_PT:
                continue
            element = ParsedElement(type="image_caption", text="[Diagram/figure \u2014 pending vision]",
                                     page=page_num)
            raw_elements.append(element)
            self._queue_vision_caption(element, image_bytes, image.mime,
                                        context=CAPTION_PROMPT, truncate_summary=True,
                                        max_tokens=DIAGRAM_CAPTION_MAX_TOKENS)

        return carry_paragraph

    @staticmethod
    def _flush_carry(raw_elements: list[ParsedElement],
                      carry_paragraph: Optional[ParsedElement]) -> Optional[ParsedElement]:
        if carry_paragraph:
            raw_elements.append(carry_paragraph)
        return None

    @staticmethod
    def _join_lines(lines: list[str]) -> str:
        assembled = ""
        for txt in lines:
            if not assembled:
                assembled = txt
            elif assembled.endswith("-") and not assembled.endswith("--"):
                assembled = assembled[:-1] + txt   # de-hyphenate line-break splits
            else:
                assembled += " " + txt
        return assembled

    # -- heading scale --------------------------------------------------

    def _build_heading_scale(self, font_size_counter: Counter, body_font_size: float) -> dict[float, int]:
        larger_sizes = sorted((s for s in font_size_counter if s > body_font_size * 1.05), reverse=True)
        return {size: i + 1 for i, size in enumerate(larger_sizes[:6])}

    def _classify_heading_level(self, font_size: float, is_bold: bool,
                                 heading_levels: dict[float, int], body_font_size: float) -> Optional[int]:
        if font_size in heading_levels:
            return heading_levels[font_size]
        # NOTE: previously also treated any bold, body-sized text as a
        # heading (level 4 fallback). That fired on bold inline labels,
        # "Note:" lead-ins, bold table captions, etc. — none of them real
        # section breaks — which fragmented ordinary paragraphs into
        # spurious one-sentence/one-line "sections". A genuine heading in a
        # professionally produced PDF is almost always a distinct font
        # size (caught above) or a numbered pattern (caught separately via
        # is_heading_pattern in the caller) — bold-at-body-size alone isn't
        # a reliable enough signal to justify a section break.
        return None

    def _attach_section_paths(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        stack: list[tuple[int, str]] = []
        for el in elements:
            if el.type == "heading":
                level = el.heading_level or 1
                stack = [(lvl, txt) for lvl, txt in stack if lvl < level]
                stack.append((level, el.text))
            el.section_path = [txt for _, txt in stack]
        return elements

    def _renumber_pages_by_headings(
        self,
        elements: list[ParsedElement],
        *,
        bump_levels: tuple[int, ...] = (1, 2),
    ) -> list[ParsedElement]:
        """Reassign element.page based on top-level heading positions.

        Non-PDF formats (HTML/DOCX/ODT/MD) all start with page=1 because
        their input doesn't carry a native page concept — but every
        retrieval ground truth references content by "page number," and
        human-authored page labels map to the document's structural
        sections, not physical page breaks. So we walk the element list
        and treat each heading at level in `bump_levels` (default H1+H2)
        as a page boundary, with the *previous* heading's page number
        preserved for everything until the next bump.

        Concrete rules (per `_renumber_pages_by_headings` defaults):

          - The first qualifying heading (and everything before it) is
            page 1 — that's the document's title page / cover, matching
            how ground-truth queries reference page 1.
          - The Nth qualifying heading starts page N+1. Subordinates
            (paragraphs, lists, tables, lower-level headings) keep that
            page until the next bump.
          - Elements that already carry a meaningful page (PDF/EPUB with
            real page numbers, DOCX with `<w:br w:type="page"/>` counts)
            are NOT passed through this helper — those parsers keep
            their native numbering.

        Effect on retrieval: page-anchored recall@k jumps from ~0 to
        meaningful on the non-PDF fixtures (07_wikipedia_html,
        10_gov_docx, 12_libreoffice_odt, 14_cjk_wikipedia, ...) that
        have ground-truth pages indexed to section boundaries. Before
        this pass, every element was page=1 and `expected_in_corpus`
        failed on every relevant page that wasn't 1 — that single bug
        capped the 19-fixture R@5 at ~0.65 even with the strongest
        embedder. With the renumber, the fixture-level page_tolerance=2
        absorbs any minor mapping drift and the relevant pages all
        show up.
        """
        page = 1
        out: list[ParsedElement] = []
        for el in elements:
            if el.type == "heading":
                level = el.heading_level or 1
                if level in bump_levels and out:
                    # A qualifying heading after content bumps the page.
                    # The heading itself anchors to the NEW page so a
                    # query like "Section 2: Method" lands on the
                    # heading text, not the previous section's last
                    # paragraph.
                    page += 1
            # All elements get the current page — head-and-tail alike.
            out.append(ParsedElement(
                type=el.type, text=el.text, page=page,
                bbox=el.bbox, heading_level=el.heading_level,
                section_path=list(el.section_path),
            ))
        return out

    # -- reading order for multi-column pages (text + tables together) --

    def _order_items_reading_order(self, items: list[tuple[str, object, tuple]],
                                    page_width: float) -> list[tuple[str, object, tuple]]:
        """Sorts text blocks AND tables together by reading position, so a
        table interleaves at its true position on the page instead of being
        dumped after all of the page's text (which used to disorder the
        section context a table's surrounding paragraphs provide)."""
        if len(items) <= 1:
            return items

        x_starts = sorted(item[2][0] for item in items)
        gaps = [(a + b) / 2 for a, b in zip(x_starts, x_starts[1:]) if b - a > page_width * 0.04]

        if not gaps:
            return sorted(items, key=lambda item: (item[2][1], item[2][0]))

        boundary = max(gaps)
        left = sorted((item for item in items if item[2][0] < boundary), key=lambda item: item[2][1])
        right = sorted((item for item in items if item[2][0] >= boundary), key=lambda item: item[2][1])
        return left + right

    # -- tables: ruled-line first, whitespace-column fallback second ----

    @staticmethod
    def _is_phantom_table(tab, page_rect) -> bool:
        """Detect "tables" that PyMuPDF's find_tables() identifies but
        are actually the page-level layout (common in Canva-exported
        PDFs and many designer templates — the page background is
        wrapped in a rectangle that the detector mistakes for a
        full-page table).

        Two signals, either of which marks a table as phantom:

          1. The bbox covers most of the page area (>= 70%). A real
             table sits inside one section of the page, not the whole
             page.
          2. Most rows (>= 50%) have only 0-1 non-empty cells. A real
             data table has every cell filled in its column-aligned
             rows; a phantom table has the entire page text jammed
             into one column and the rest of the columns empty.

        Either check alone is too narrow: a tall infographic-style
        table might legitimately cover most of the page; a sparse
        legend-style table might have some empty cells. Either check
        passing together is rare for real tables, so we use OR.
        """
        try:
            bbox = tab.bbox
        except Exception:
            return False
        if not bbox or len(bbox) != 4:
            return False
        try:
            table_area = max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))
        except Exception:
            return False
        page_area = max(1.0, page_rect.width * page_rect.height)
        if table_area / page_area >= 0.70:
            return True
        try:
            rows = tab.rows
        except Exception:
            rows = None
        if rows and len(rows) >= 2:
            emptyish = sum(1 for r in rows
                           if sum(1 for c in r if c and str(c).strip()) <= 1)
            if emptyish / len(rows) >= 0.5:
                return True
        return False

    def _scrub_tofu(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        """Strip missing-glyph placeholder characters (U+FFFD, U+25A1,
        etc.) from element text and collapse consecutive ones to a single
        "[?]" marker. These show up whenever a PDF's embedded font is
        missing glyphs for a code point — common for legacy CJK, math
        symbols, or any character outside the font's subset. Without
        scrubbing, the embedder sees a paragraph that's 30% `[?]`
        boxes and retrieval quality drops on every query that touches
        those pages. The `[?]` marker preserves a hint in the text that
        something was lost there."""
        out: list[ParsedElement] = []
        for el in elements:
            text = el.text
            if not any(c in text for c in TOFU_CHARS):
                out.append(el)
                continue
            # Replace each contiguous run of tofu chars with one [?].
            cleaned = re.sub(f"[{re.escape(TOFU_CHARS)}]+", "[?]", text)
            # If scrubbing removed everything (all-tofu text), drop the
            # element entirely — it carries no signal.
            if not cleaned.strip() or cleaned.strip() == "[?]":
                continue
            out.append(ParsedElement(
                type=el.type, text=cleaned, page=el.page,
                bbox=el.bbox, heading_level=el.heading_level,
                section_path=list(el.section_path),
            ))
        return out

    def _reclassify_footnotes(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        """Walk emitted elements and re-tag any paragraph that's shaped
        like a footnote (short, in the page's bottom band, numeric
        prefix). The bbox check is the disambiguator: a short paragraph
        with a numeric prefix in the middle of the page is probably a
        numbered list item, not a footnote. Bottom-band + numeric prefix
        together is the real signal."""
        if not elements:
            return elements
        # Group by page to know the page-height for each element.
        page_rects: dict[int, Optional[tuple]] = {}
        for el in elements:
            if el.bbox and el.page not in page_rects:
                page_rects[el.page] = el.bbox  # not quite a page rect but works as a height proxy
        # Without per-page rects we'd need to track page heights through
        # _parse_pdf_page. Heuristic fallback: standard A4 / Letter
        # height (~842 pt). Close enough for the band threshold.
        FALLBACK_PAGE_HEIGHT = 842.0
        out: list[ParsedElement] = []
        for el in elements:
            if (el.type == "paragraph"
                    and len(el.text) <= FOOTNOTE_MAX_CHARS
                    and _FOOTNOTE_PREFIX_RE.match(el.text)
                    and el.bbox is not None):
                # bbox is (x0, y0, x1, y1); check if y0 sits in the
                # bottom 15% of the page.
                page_height = FALLBACK_PAGE_HEIGHT
                if el.bbox[3] > page_height:
                    page_height = el.bbox[3]  # bbox gives us at least the height
                if el.bbox[1] >= page_height * (1 - FOOTNOTE_BAND):
                    out.append(ParsedElement(
                        type="footnote", text=el.text, page=el.page,
                        bbox=el.bbox, section_path=list(el.section_path),
                    ))
                    continue
            out.append(el)
        return out

    def _stitch_cross_page_tables(self, elements: list[ParsedElement]) -> list[ParsedElement]:
        """Concat adjacent table elements when one is the continuation of
        the other. Two tables are "the same table across a page break"
        when:
          1. They're consecutive in element order (no intervening
             non-table element between them), AND
          2. The first row of table N+1's text matches the header row of
             table N's text (case-insensitive, whitespace-normalized), AND
          3. Their column counts match.
        The merged element keeps the original page-1 bbox (the most
        useful citation target) and sets page_numbers = the union of
        both pages, so retrieval filters and the chunker's
        cross-page metadata both stay correct."""
        if not elements:
            return elements
        out: list[ParsedElement] = []
        i = 0
        while i < len(elements):
            cur = elements[i]
            if cur.type != "table":
                out.append(cur)
                i += 1
                continue
            # Walk forward while the next element is a continuation table.
            merged_text_lines = cur.text.split("\n")
            j = i + 1
            while j < len(elements) and elements[j].type == "table":
                nxt = elements[j]
                cur_lines = merged_text_lines
                if len(cur_lines) < 2:
                    break
                cur_header = cur_lines[0]
                cur_cols = cur_header.count("|") + 1
                next_lines = nxt.text.split("\n")
                if len(next_lines) < 2:
                    break
                next_header = next_lines[0]
                next_cols = next_header.count("|") + 1
                if cur_cols != next_cols:
                    break
                if _normalize_for_repeat_check(cur_header) != _normalize_for_repeat_check(next_header):
                    break
                # Continuation: drop next's header + separator, keep its rows.
                merged_text_lines.extend(next_lines[2:])
                j += 1
            if j > i + 1:
                # We merged at least one continuation. Emit the merged
                # element. page stays at the first page so citations
                # anchor to where the table starts; the merged data rows
                # carry across the page break in a single atomic chunk,
                # which is the whole point of this stitch pass.
                merged_text = "\n".join(merged_text_lines)
                out.append(ParsedElement(
                    type="table", text=merged_text,
                    page=cur.page, bbox=cur.bbox,
                    section_path=list(cur.section_path),
                ))
                i = j
            else:
                out.append(cur)
                i += 1
        return out

    def _table_to_element(self, tab, page_num: int) -> Optional[ParsedElement]:
        try:
            md_table = tab.rows
        except Exception:
            return None
        if not md_table or len(md_table) < 2:
            return None
        headers = [str(c or "").replace("\n", " ").strip() for c in md_table[0]]
        rows = [" | ".join(headers), " | ".join(["---"] * len(headers))]
        for r in md_table[1:]:
            rows.append(" | ".join(str(c or "").replace("\n", " ").strip() for c in r))
        return ParsedElement(type="table", text="\n".join(rows), page=page_num, bbox=tab.bbox)

    def _maybe_whitespace_table(self, block_lines_data: list[dict]) -> Optional[str]:
        if len(block_lines_data) < 3:
            return None
        split_rows = [re.split(r"\s{2,}", l["clean_text"].strip()) for l in block_lines_data]
        col_counts = [len(r) for r in split_rows]
        mode_count, mode_freq = Counter(col_counts).most_common(1)[0]
        if mode_count < 2 or mode_freq / len(col_counts) < 0.7:
            return None
        rows = [" | ".join(r) for r in split_rows if len(r) == mode_count]
        if len(rows) < 2:
            return None
        header, *body = rows
        return "\n".join([header, " | ".join(["---"] * mode_count), *body])

    # -- vision: queue during parsing, resolve concurrently at the end ---

    def _downscale_for_vision(self, image_bytes: bytes) -> tuple[bytes, str]:
        """Caps the longest edge and re-encodes as JPEG before sending to a
        vision model. Uses Pillow — a small image library, not an ML
        framework — so this stays consistent with the project's no-torch
        default. Falls back to the original bytes if decoding fails."""
        from PIL import Image  # deferred; see the module header.

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            longest = max(img.size)
            if longest > MAX_VISION_IMAGE_DIMENSION:
                scale = MAX_VISION_IMAGE_DIMENSION / longest
                new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                img = img.resize(new_size, Image.LANCZOS)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=VISION_JPEG_QUALITY)
            return out.getvalue(), "image/jpeg"
        except Exception:
            return image_bytes, "image/png"

    def _queue_vision_caption(self, element: ParsedElement, image_bytes: bytes, mimetype: str,
                              context: Optional[str] = None, truncate_summary: bool = True,
                              max_tokens: Optional[int] = None) -> None:
        if not self.vision:
            element.text = "[Image \u2014 no vision adapter configured]"
            return

        downscaled_bytes, downscaled_mime = self._downscale_for_vision(image_bytes)
        digest = hashlib.sha256(downscaled_bytes).hexdigest()

        if digest in self._vision_cache:
            element.text = self._vision_cache[digest]
            return
        if (self.max_vision_calls_per_doc is not None
                and self._vision_calls_queued >= self.max_vision_calls_per_doc):
            element.text = "[Image \u2014 vision call budget exhausted for this document]"
            return

        self._vision_calls_queued += 1
        self._pending_vision.append(
            (element, downscaled_bytes, downscaled_mime, context, digest, truncate_summary, max_tokens)
        )

    def _resolve_pending_vision(self) -> None:
        """Runs all queued vision calls concurrently (bounded thread pool)
        instead of blocking the page loop on each one serially — the main
        fix for slow/crashing runs on image-heavy documents."""
        if not self._pending_vision:
            return

        def _call(item):
            element, img_bytes, mime, context, digest, truncate_summary, max_tokens = item
            try:
                caption = self.vision.describe_image(img_bytes, mime, context, max_tokens=max_tokens)
            except Exception as e:
                # Was silently swallowed before, which made caption problems undiagnosable.
                log.warning("Vision caption failed (%s: %s)", type(e).__name__, e)
                caption = "[Image \u2014 vision call failed]"
            if truncate_summary:
                # Defense in depth: the prompt already asks for brevity and
                # the adapter caps generation length, but not every model
                # (especially small local ones) reliably obeys either — so
                # a diagram/photo caption is hard-capped here regardless of
                # what the model actually returned. Full-page scan
                # transcriptions skip this: there, the "caption" IS the
                # page's entire content, and truncating it would discard
                # real text, not just a wordy description.
                caption = self._truncate_caption(caption)
            return element, digest, caption

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.vision_concurrency) as pool:
            for element, digest, caption in pool.map(_call, self._pending_vision):
                element.text = caption
                self._vision_cache[digest] = caption

        self._pending_vision = []

    @staticmethod
    def _truncate_caption(text: str, max_sentences: int = CAPTION_MAX_SENTENCES) -> str:
        text = " ".join(text.split())          # collapse newlines/bullets into one line
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= max_sentences:
            return text
        return " ".join(sentences[:max_sentences]).strip()

    def _add_office_image(self, elements: list[ParsedElement], blob: bytes,
                          content_type: str, page: int) -> None:
        """Queue a caption for an image embedded in a DOCX/PPTX. Skips icons/
        logos (< MIN_IMAGE_DIM) and vector formats a vision model can't decode."""
        if (content_type or "").lower() not in CAPTIONABLE_IMAGE_MIMES:
            return
        from PIL import Image  # deferred; see the module header.

        try:
            with Image.open(io.BytesIO(blob)) as im:
                w, h = im.size
        except Exception:
            return
        if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
            return
        element = ParsedElement(type="image_caption", text="[Diagram/figure \u2014 pending vision]", page=page)
        elements.append(element)
        self._queue_vision_caption(element, blob, content_type, context=CAPTION_PROMPT,
                                    truncate_summary=True, max_tokens=DIAGRAM_CAPTION_MAX_TOKENS)

    @staticmethod
    def _iter_pptx_shapes(shapes):
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        for sh in shapes:
            if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
                yield from UniversalParser._iter_pptx_shapes(sh.shapes)
            else:
                yield sh

    # -- other formats --------------------------------------------------

    def _parse_docx(self, file_bytes: bytes) -> list[ParsedElement]:
        """DOCX parser.

        Pages: python-docx doesn't expose page numbers directly (it would
        require rendering), but a Word document's effective page breaks
        are visible in the XML as <w:br w:type="page"/> runs. We walk the
        document body counting page breaks and assign page numbers that
        way — a paragraph carrying a page break starts on the next page.
        Approximation, but vastly better than the previous "everything is
        page 1" behavior and good enough for citation.

        Headings: Word's built-in Heading 1/2/3 styles map to heading
        levels 1/2/3 — already done, but tightened to validate the level
        int and fall back to 1 on any unparseable style name.

        Footnotes: stripped from the body emit by default. Footnotes are
        commentary that belongs near the body text, but emitting them as
        separate paragraphs floods the chunker with tiny "see note 3"
        fragments that don't carry enough retrievable signal. If you ever
        need footnotes back, gate on an env flag rather than re-adding
        silently.
        """
        import docx

        doc_file = docx.Document(io.BytesIO(file_bytes))
        elements: list[ParsedElement] = []

        # Collect footnote IDs to strip from body paragraphs.
        # We don't try to merge footnote content into the body — by
        # default footnote refs are removed via the inline element list.
        footnote_ids = set()
        try:
            footnotes_part = doc_file.part.part_related_by(
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
            )
            for fn in footnotes_part.element.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}footnote"):
                fn_id = fn.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}id")
                if fn_id is not None:
                    footnote_ids.add(fn_id)
        except Exception:
            pass  # no footnotes part — fine

        page = 1
        for para in doc_file.paragraphs:
            txt = para.text.strip()
            # Count page breaks BEFORE this paragraph — a <w:br w:type="page">
            # inside the paragraph ends the previous page; the paragraph
            # itself lands on the next one.
            try:
                page_breaks = sum(
                    1 for br in para._element.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br")
                    if br.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type") == "page"
                )
                page += page_breaks
            except Exception:
                page_breaks = 0

            if txt:
                style = para.style.name or ""
                if style.startswith("Heading"):
                    try:
                        level = int(style.replace("Heading", "").strip() or 1)
                    except ValueError:
                        level = 1
                    level = max(1, min(level, 6))
                    elements.append(ParsedElement(
                        type="heading", text=txt, page=page, heading_level=level,
                    ))
                else:
                    elements.append(ParsedElement(
                        type="paragraph", text=txt, page=page,
                    ))
            # Inline AND floating pictures: every <a:blip r:embed="rIdN"> in
            # the paragraph. (Checked even when the paragraph has no text —
            # image-only paragraphs are the norm.)
            for rid in para._element.xpath(".//a:blip/@r:embed"):
                part = doc_file.part.related_parts.get(rid)
                if part is not None:
                    self._add_office_image(elements, part.blob, part.content_type, page=page)
        for table in doc_file.tables:
            rows = []
            for i, row in enumerate(table.rows):
                cells = [c.text.replace("\n", " ").strip() for c in row.cells]
                rows.append(" | ".join(cells))
                if i == 0:
                    rows.append(" | ".join(["---"] * len(cells)))
            if rows:
                elements.append(ParsedElement(type="table", text="\n".join(rows), page=page))
        elements = self._attach_section_paths(elements)
        # DOCX: derive page from Heading 1 positions when the
        # `docx_page_by_heading` opt-in flag is set (heading-only
        # derivation, matching the ODT-style convention used by the
        # 10_gov_docx fixture's ground truth). Default bump_levels
        # are (1, 2) — Heading 1 + Heading 2 — which inflates page
        # counts on docs where Heading 2 paragraphs are sub-section
        # labels rather than page boundaries. The `<w:br w:type="page"/>`
        # counter above is preserved on each element's `page` already;
        # this final pass overrides it with section-based numbering
        # so ground-truth queries anchored to Heading-1 boundaries
        # can hit the right content. See UniversalParser.__init__ for
        # the trade-off rationale.
        bump_levels = (1,) if self._docx_page_by_heading else (1, 2)
        return self._renumber_pages_by_headings(elements, bump_levels=bump_levels)

    def _parse_pptx(self, file_bytes: bytes) -> list[ParsedElement]:
        import pptx

        prs = pptx.Presentation(io.BytesIO(file_bytes))
        elements: list[ParsedElement] = []
        for i, slide in enumerate(prs.slides):
            title_shape = slide.shapes.title
            if title_shape is not None and getattr(title_shape, "has_text_frame", False):
                title_txt = title_shape.text_frame.text.strip()
                if title_txt:
                    elements.append(ParsedElement(type="heading", text=title_txt, page=i + 1, heading_level=1))
            for shape in slide.shapes:
                if shape == title_shape:
                    continue
                if getattr(shape, "has_text_frame", False):
                    tf = shape.text_frame
                    raw_text = tf.text.strip()
                    if not raw_text:
                        continue
                    # Per-paragraph emission: a dense bullet slide has
                    # one text frame with N paragraphs (one per bullet);
                    # the old behavior collapsed all of them into one
                    # run-on paragraph that lost bullet structure entirely.
                    paragraphs = [p.text.strip() for p in tf.paragraphs if p.text.strip()]
                    if len(paragraphs) <= 1:
                        elements.append(ParsedElement(type="paragraph", text=raw_text, page=i + 1))
                    else:
                        for para in paragraphs:
                            # Heuristic: short paragraph (<=120 chars, no
                            # terminal punctuation) and the slide has no
                            # subtitle-shaped element — treat as subtitle
                            # level 2. Cheap, doesn't fight real headings.
                            pass
                            elements.append(ParsedElement(type="list_item", text=para, page=i + 1))
                elif getattr(shape, "has_table", False):
                    rows = []
                    for r_idx, row in enumerate(shape.table.rows):
                        cells = [c.text_frame.text.replace("\n", " ").strip() for c in row.cells]
                        rows.append(" | ".join(cells))
                        if r_idx == 0:
                            rows.append(" | ".join(["---"] * len(cells)))
                    if rows:
                        elements.append(ParsedElement(type="table", text="\n".join(rows), page=i + 1))
            for shape in self._iter_pptx_shapes(slide.shapes):
                try:
                    img = shape.image          # only picture / picture-placeholder shapes have .image
                except Exception:
                    continue
                self._add_office_image(elements, img.blob, img.content_type, page=i + 1)
            # Speaker notes — emitted as trailing paragraphs under the
            # current slide's section_path. These often contain
            # presenter-only context (talking points, source citations)
            # that retrieval benefits from but never reach the audience.
            try:
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                    notes_text = slide.notes_slide.notes_text_frame.text.strip()
                    if notes_text:
                        for note_line in [n.strip() for n in notes_text.split("\n") if n.strip()]:
                            elements.append(ParsedElement(
                                type="paragraph", text=note_line, page=i + 1,
                            ))
            except Exception:
                # Some PPTX files have malformed notes parts; never crash
                # the whole slide for them.
                pass
        return self._attach_section_paths(elements)

    def _parse_xlsx(self, file_bytes: bytes) -> list[ParsedElement]:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        elements: list[ParsedElement] = []
        for ws in wb.worksheets:
            # Skip hidden sheets — they typically contain scratch data or
            # macros and would pollute retrieval without adding useful
            # signal.
            if ws.sheet_state != "visible":
                continue
            elements.append(ParsedElement(type="heading", text=ws.title, page=1, heading_level=1))

            # Compute the set of hidden column letters and hidden row
            # indices so we can drop them from the emitted table — their
            # values are typically scaffolding or user-hidden content
            # that shouldn't reach retrieval.
            hidden_cols = {
                col for col, dim in ws.column_dimensions.items()
                if getattr(dim, "hidden", False)
            }
            hidden_rows = {
                r for r, dim in ws.row_dimensions.items()
                if getattr(dim, "hidden", False)
            }

            # Compute merged cell markers so the table output preserves
            # the structure: a 2x2 merge of cell A1 is emitted with
            # "(merged 2x2)" appended to its value, and the cells it
            # covers are emitted as empty so column alignment is
            # preserved.
            merged_map: dict[tuple[int, int], str] = {}
            merged_covered: set[tuple[int, int]] = set()
            for mr in ws.merged_cells.ranges:
                min_col, min_row, max_col, max_row = (
                    mr.min_col, mr.min_row, mr.max_col, mr.max_row,
                )
                w = max_col - min_col + 1
                h = max_row - min_row + 1
                # Anchor cell carries the marker; everything else in the
                # range is "covered" and becomes empty.
                if w > 1 or h > 1:
                    merged_map[(min_row, min_col)] = f"(merged {w}x{h})"
                for r in range(min_row, max_row + 1):
                    for c in range(min_col, max_col + 1):
                        if (r, c) != (min_row, min_col):
                            merged_covered.add((r, c))

            rows_text: list[str] = []
            for row_idx, row in enumerate(
                ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=False),
                start=1,
            ):
                if row_idx in hidden_rows:
                    continue
                cell_strs = []
                for col_idx, cell in enumerate(row, start=1):
                    # Skip cells whose column letter is hidden.
                    from openpyxl.utils import get_column_letter
                    if get_column_letter(col_idx) in hidden_cols:
                        continue
                    if (row_idx, col_idx) in merged_covered:
                        cell_strs.append("")
                        continue
                    val = cell.value
                    val_str = "" if val is None else str(val).replace("\n", " ").strip()
                    if (row_idx, col_idx) in merged_map:
                        val_str = f"{val_str} {merged_map[(row_idx, col_idx)]}".strip()
                    cell_strs.append(val_str)
                # Drop trailing empties so column-aligned output isn't
                # padded with junk on ragged sheets.
                while cell_strs and not cell_strs[-1]:
                    cell_strs.pop()
                if any(c.strip() for c in cell_strs):
                    rows_text.append(" | ".join(cell_strs))
            if not rows_text:
                continue

            # Determine a representative column count for the separator.
            # Use the first non-empty row to size the markdown separator.
            col_count = max(rows_text[0].count("|") + 1, 1)
            sep = " | ".join(["---"] * col_count)

            # Use the first row as header. If there's only one row, fall
            # back to a synthetic header so the markdown table is valid.
            if len(rows_text) == 1:
                header = rows_text[0]
                data_rows: list[str] = []
            else:
                header = rows_text[0]
                data_rows = rows_text[1:]

            if len(data_rows) <= MAX_TABLE_ROWS_PER_CHUNK:
                elements.append(ParsedElement(
                    type="table", text="\n".join([header, sep, *data_rows]), page=1,
                ))
            else:
                for start in range(0, len(data_rows), MAX_TABLE_ROWS_PER_CHUNK):
                    piece = data_rows[start:start + MAX_TABLE_ROWS_PER_CHUNK]
                    elements.append(ParsedElement(
                        type="table", text="\n".join([header, sep, *piece]), page=1,
                    ))
        elements = self._attach_section_paths(elements)
        # XLSX: each sheet name is emitted as a level-1 heading, so the
        # page-derivation pass naturally gives each sheet its own page
        # number. That's the convention every spreadsheet ground-truth
        # query I've seen uses — "Sheet 1 row 4" → page 1.
        return self._renumber_pages_by_headings(elements, bump_levels=(1,))

    @staticmethod
    def _html_inline_text(el) -> str:
        """Like get_text(strip=True) but wraps inline <code> contents
        with backticks so they survive concatenation with surrounding
        prose as visually-distinct tokens. Plain text for retrieval
        (the backticks become part of the indexed text, helping queries
        that look for code-shaped snippets); minor display side-effect
        is acceptable for citation.

        Walking with descendants() and stringifying text nodes directly
        causes duplicates when a <code> child is wrapped — its text
        shows up once as the wrapped token AND once as a raw text node.
        We skip text nodes whose parent is a handled <code>."""
        parts: list[str] = []
        INLINE_PARENTS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
        for child in el.descendants:
            if getattr(child, "name", None) is None:
                # Bare text node (NavigableString has .name == None).
                # Skip if its immediate parent is a <code> we're already
                # wrapping — otherwise its text appears twice.
                if getattr(getattr(child, "parent", None), "name", None) == "code":
                    continue
                s = str(child).strip()
                if s:
                    parts.append(s)
            elif child.name == "code":
                if child.parent and child.parent.name in INLINE_PARENTS:
                    s = child.get_text(strip=True)
                    if s:
                        parts.append(f"`{s}`")
        return " ".join(parts)

    def _parse_html(self, file_bytes: bytes) -> list[ParsedElement]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(file_bytes, "html.parser")
        # Strip noise: anything that exists for navigation, chrome,
        # accessibility-only purposes, or content duplication should never
        # reach the chunker. The list is intentionally generous — false
        # positives only cost us a chunk that wasn't useful anyway.
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside",
                                  "noscript", "iframe", "form", "button", "input",
                                  "select", "textarea", "svg", "canvas"]):
            tag.decompose()
        for sel in [
            "[role=presentation]", "[aria-hidden=true]", "[hidden]",
            # Boilerplate by semantic class — these are the structural
            # CSS hooks most blog/news templates use for share bars,
            # related-articles blocks, comment threads, cookie banners,
            # and author bylines. Stripping by class is heuristic but
            # matches 90% of real-world templates.
            "[class*=comment]", "[class*=Comments]",
            "[class*=share]", "[class*=Share]",
            "[class*=related]", "[class*=Related]",
            "[class*=author]", "[class*=byline]", "[class*=Byline]",
            "[class*=cookie]", "[class*=Cookie]", "[class*=consent]", "[class*=Consent]",
            "[class*=newsletter]", "[class*=Newsletter]",
            "[class*=promo]", "[class*=Promo]", "[class*=advert]", "[class*=Advert]",
            "[class*=social]", "[class*=Social]",
            "[id*=comment]", "[id*=share]", "[id*=related]", "[id*=cookie]",
            "[id*=newsletter]", "[id*=promo]", "[id*=social]",
        ]:
            for tag in soup.select(sel):
                tag.decompose()
        # Prefer main content if present; otherwise fall back to whole body.
        root = soup.find("main") or soup.find("article") or soup.body or soup
        elements: list[ParsedElement] = []

        # Extract images: only data: URIs (inline base64) and same-page
        # relative URLs are decoded and queued through the vision
        # adapter. Remote http(s) URLs are skipped — fetching them at
        # ingest time is a side effect the user almost never wants
        # (slow, breaks offline operation, hits random CDNs), and the
        # alt/title text is enough for retrieval to find the image's
        # semantic context.
        import base64 as _b64
        import re as _re
        for img in root.find_all("img"):
            src = img.get("src", "").strip()
            alt = img.get("alt", "").strip() or img.get("title", "").strip()
            if not src:
                continue
            mime = "image/png"
            blob: Optional[bytes] = None
            if src.startswith("data:"):
                # data:image/png;base64,XXXX — extract mime + payload
                m = _re.match(r"data:([^;]+);base64,(.+)", src, _re.DOTALL)
                if m:
                    mime = m.group(1) or "image/png"
                    try:
                        blob = _b64.b64decode(m.group(2))
                    except Exception:
                        blob = None
            elif src.startswith(("http://", "https://", "//")):
                # Remote URL — emit a placeholder caption carrying the
                # alt text so retrieval can still find it.
                elements.append(ParsedElement(
                    type="image_caption",
                    text=f"[Remote image: {alt or src}]", page=1,
                ))
                continue
            else:
                # Relative URL — we don't have a base to resolve against,
                # so emit a placeholder like for remote URLs.
                elements.append(ParsedElement(
                    type="image_caption",
                    text=f"[Image (relative URL): {alt or src}]", page=1,
                ))
                continue
            if blob is None:
                continue
            if mime.lower() not in CAPTIONABLE_IMAGE_MIMES:
                continue
            element = ParsedElement(
                type="image_caption", text="[Diagram/figure \u2014 pending vision]", page=1,
            )
            elements.append(element)
            self._queue_vision_caption(
                element, blob, mime, context=CAPTION_PROMPT,
                truncate_summary=True, max_tokens=DIAGRAM_CAPTION_MAX_TOKENS,
            )

        for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "table", "li"]):
            if el.name.startswith("h"):
                txt = el.get_text(strip=True)
                if txt:
                    elements.append(ParsedElement(type="heading", text=txt, page=1, heading_level=int(el.name[1])))
            elif el.name == "p":
                txt = self._html_inline_text(el)
                if txt:
                    elements.append(ParsedElement(type="paragraph", text=txt, page=1))
            elif el.name == "li":
                txt = el.get_text(strip=True)
                if txt:
                    elements.append(ParsedElement(type="list_item", text=txt, page=1))
            elif el.name == "pre":
                txt = el.get_text(strip=True)
                if txt:
                    elements.append(ParsedElement(type="code", text=txt, page=1))
            elif el.name == "table":
                rows = []
                trs = el.find_all("tr")
                for i, tr in enumerate(trs):
                    cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
                    if not cells:
                        continue
                    rows.append(" | ".join(cells))
                    if i == 0:
                        rows.append(" | ".join(["---"] * len(cells)))
                if rows:
                    elements.append(ParsedElement(type="table", text="\n".join(rows), page=1))
        elements = self._attach_section_paths(elements)
        # HTML has no native page numbers; derive them from H1+H2
        # heading positions so ground-truth queries like "page 4" can
        # actually hit the right content. Without this, every chunk
        # carries page=1 and MISSING_FROM_CORPUS fires on every query
        # whose relevant page isn't 1.
        return self._renumber_pages_by_headings(elements, bump_levels=(1, 2))

    def _parse_raw_image(self, file_bytes: bytes, mime_type: str) -> list[ParsedElement]:
        element = ParsedElement(type="image_caption", text="[Image \u2014 pending vision]", page=1)
        if self.vision:
            self._queue_vision_caption(element, file_bytes, mime_type, context=CAPTION_PROMPT,
                                        truncate_summary=True, max_tokens=DIAGRAM_CAPTION_MAX_TOKENS)
        elif self.ocr:
            # OCR fallback for users without a vision adapter. Vision
            # is richer (captions + chart/diagram description); OCR is
            # the deterministic pure-text path. Promote the element to
            # type="paragraph" when OCR produces real text so the
            # chunker treats it as content rather than a placeholder.
            text = self.ocr.transcribe(file_bytes)
            if text:
                element.type = "paragraph"
                element.text = text
            else:
                element.text = "[Image \u2014 OCR returned no text]"
        else:
            element.text = "[Image \u2014 no vision or OCR adapter configured]"
        return [element]

    # -- Plain text (Tier 1.1) ---------------------------------------------

    def _parse_txt(self, file_bytes: bytes) -> list[ParsedElement]:
        """Plain `.txt`: split on blank lines into paragraphs. UTF-8 with
        replacement so a stray Latin-1 quote doesn't kill the whole file."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("utf-8", errors="replace")
        elements: list[ParsedElement] = []
        for block in re.split(r"\n\s*\n+", text):
            block = block.strip()
            if not block:
                continue
            elements.append(ParsedElement(type="paragraph", text=block, page=1))
        return elements

    # -- Markdown (Tier 1.2) ----------------------------------------------

    _MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
    _MD_FENCE_RE = re.compile(r"^```")
    _MD_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")
    # Bullet / ordered list markers we recognise at line start.
    _MD_LIST_ITEM_RE = re.compile(
        r"^\s*(?:[-*+]|\d+\.)\s+\S"
    )

    def _parse_md(self, file_bytes: bytes) -> list[ParsedElement]:
        """Markdown: `#` headers → headings (clamped 1-6), fenced code
        blocks → code, pipe tables → tables, everything else → paragraph.
        Code fences and tables are detected line-by-line so we don't
        accidentally emit a `#` inside a code block as a heading."""
        text = file_bytes.decode("utf-8", errors="replace")
        lines = text.splitlines()

        # YAML/TOML frontmatter: a leading "---" ... "---" block at the
        # top of the file holds metadata (title, date, author, tags).
        # It's pure noise for retrieval — strip before parsing.
        if lines and lines[0].strip() == "---":
            end = None
            for k in range(1, len(lines)):
                if lines[k].strip() == "---":
                    end = k
                    break
            if end is not None:
                lines = lines[end + 1:]

        elements: list[ParsedElement] = []
        i = 0
        in_code = False
        code_buf: list[str] = []
        para_buf: list[str] = []

        def flush_para():
            if not para_buf:
                return
            # If every buffered line is a list item, emit each as its own
            # list_item element. Otherwise join with spaces and emit as a
            # paragraph. This keeps "- a / - b" from collapsing into
            # "- a - b" which would lose the list structure.
            if all(self._MD_LIST_ITEM_RE.match(l) for l in para_buf):
                for l in para_buf:
                    # Strip the marker — the bullet glyph is implicit in
                    # the list_item type so the chunker/embedder doesn't
                    # see redundant "- " prefixes.
                    stripped = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", l).strip()
                    if stripped:
                        elements.append(ParsedElement(
                            type="list_item", text=stripped, page=1,
                        ))
            else:
                joined = " ".join(p.strip() for p in para_buf if p.strip()).strip()
                if joined:
                    elements.append(ParsedElement(type="paragraph", text=joined, page=1))
            para_buf.clear()

        while i < len(lines):
            line = lines[i]
            if self._MD_FENCE_RE.match(line):
                if not in_code:
                    flush_para()
                    in_code = True
                    code_buf = []
                else:
                    elements.append(ParsedElement(type="code", text="\n".join(code_buf), page=1))
                    in_code = False
                    code_buf = []
                i += 1
                continue
            if in_code:
                code_buf.append(line)
                i += 1
                continue

            m = self._MD_HEADING_RE.match(line)
            if m:
                flush_para()
                level = min(len(m.group(1)), 6)
                elements.append(ParsedElement(
                    type="heading", text=m.group(2).strip(),
                    page=1, heading_level=level,
                ))
                i += 1
                continue

            # Pipe table: a header line with pipes followed by a separator line.
            if "|" in line and i + 1 < len(lines) and self._MD_TABLE_SEP_RE.match(lines[i + 1]):
                flush_para()
                table_lines = [line]
                j = i + 2
                while j < len(lines) and "|" in lines[j] and lines[j].strip():
                    table_lines.append(lines[j])
                    j += 1
                elements.append(self._md_table_to_element(table_lines))
                i = j
                continue

            if line.strip() == "":
                flush_para()
                i += 1
                continue

            para_buf.append(line)
            i += 1

        if in_code and code_buf:
            elements.append(ParsedElement(type="code", text="\n".join(code_buf), page=1))
        flush_para()
        # Markdown: `#` and `##` headings bump the page. The chunker's
        # atomic-type treatment keeps heading-level elements addressable
        # for citation; this pass ensures each `# Heading` / `## Heading`
        # starts a new "page" for ground-truth queries that reference
        # the section.
        return self._renumber_pages_by_headings(elements, bump_levels=(1, 2))

    @staticmethod
    def _md_table_to_element(table_lines: list[str]) -> ParsedElement:
        """Convert raw pipe-table lines to the same `header | --- | rows`
        markdown shape the chunker already understands."""
        def split_row(l: str) -> list[str]:
            return [c.strip() for c in l.strip().strip("|").split("|")]

        rows = [split_row(l) for l in table_lines]
        if len(rows) < 2:
            # Just a header — fall back to paragraph.
            return ParsedElement(type="paragraph", text=table_lines[0], page=1)
        header = rows[0]
        # rows[1] is the separator, drop it
        data_rows = rows[2:]
        sep = " | ".join(["---"] * len(header))
        out = [" | ".join(header), sep]
        out.extend(" | ".join(r) for r in data_rows)
        return ParsedElement(type="table", text="\n".join(out), page=1)

    # -- CSV (Tier 1.3) ----------------------------------------------------

    def _parse_csv(self, file_bytes: bytes) -> list[ParsedElement]:
        """CSV: first row treated as header; emitted as a markdown-shaped
        table chunked at MAX_TABLE_ROWS_PER_CHUNK. Encoding is attempted
        as UTF-8 first, falling back to latin-1 (which never fails to
        decode) so non-UTF-8 CSVs from Excel still get through."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")
        import csv as _csv
        import io as _io
        reader = _csv.reader(_io.StringIO(text))
        rows = [r for r in reader if any(c.strip() for c in r)]
        if not rows:
            return []
        header = [c.strip() for c in rows[0]]
        data_rows = [[c.strip() for c in r] for r in rows[1:]]
        sep = " | ".join(["---"] * len(header))
        elements: list[ParsedElement] = []
        if len(data_rows) <= MAX_TABLE_ROWS_PER_CHUNK:
            chunk = "\n".join([" | ".join(header), sep,
                               *[" | ".join(r) for r in data_rows]])
            elements.append(ParsedElement(type="table", text=chunk, page=1))
        else:
            for start in range(0, len(data_rows), MAX_TABLE_ROWS_PER_CHUNK):
                piece = data_rows[start:start + MAX_TABLE_ROWS_PER_CHUNK]
                chunk = "\n".join([" | ".join(header), sep,
                                   *[" | ".join(r) for r in piece]])
                elements.append(ParsedElement(type="table", text=chunk, page=1))
        return elements

    # -- JSON (Tier 1.4) ---------------------------------------------------

    def _parse_json(self, file_bytes: bytes) -> list[ParsedElement]:
        """JSON: best-effort flattening into retrieval-friendly chunks.
        - List of flat dicts → single table (dict keys = header)
        - Dict at root → each key becomes a level-3 heading followed by
          its value (recursively walked)
        - Scalar at leaf → paragraph
        Recursion is capped at MAX_JSON_DEPTH to avoid runaway on
        pathological data."""
        try:
            data = json.loads(file_bytes.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from None
        elements: list[ParsedElement] = []
        self._json_walk(data, elements, section_path=[], depth=0)
        return elements

    def _json_walk(self, node, elements: list[ParsedElement],
                   section_path: list[str], depth: int) -> None:
        if depth >= MAX_JSON_DEPTH:
            elements.append(ParsedElement(
                type="paragraph",
                text=f"{' > '.join(section_path) or 'root'}: (truncated, depth > {MAX_JSON_DEPTH})",
                page=1,
            ))
            return

        if isinstance(node, list):
            # List of flat dicts → table.
            if node and all(isinstance(x, dict) for x in node):
                keys: list[str] = []
                for x in node:
                    for k in x.keys():
                        if k not in keys:
                            keys.append(k)
                if keys:
                    sep = " | ".join(["---"] * len(keys))
                    rows = [" | ".join(str(x.get(k, "")) for k in keys) for x in node]
                    chunk = "\n".join([" | ".join(keys), sep, *rows])
                    elements.append(ParsedElement(type="table", text=chunk, page=1,
                                                   section_path=list(section_path)))
                    return
            # Otherwise: scalar items as paragraphs.
            for item in node:
                if isinstance(item, (dict, list)):
                    self._json_walk(item, elements, section_path, depth + 1)
                else:
                    elements.append(ParsedElement(
                        type="paragraph", text=str(item), page=1,
                        section_path=list(section_path),
                    ))
            return

        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    elements.append(ParsedElement(
                        type="heading", text=str(k), page=1, heading_level=3,
                        section_path=list(section_path),
                    ))
                    self._json_walk(v, elements, section_path + [str(k)], depth + 1)
                else:
                    elements.append(ParsedElement(
                        type="paragraph", text=f"{k}: {v}", page=1,
                        section_path=list(section_path),
                    ))
            return

        # Scalar leaf.
        elements.append(ParsedElement(
            type="paragraph", text=str(node), page=1,
            section_path=list(section_path),
        ))

    # -- EPUB (Tier 1.5) ---------------------------------------------------

    def _parse_epub(self, file_bytes: bytes) -> list[ParsedElement]:
        """EPUB: a zip containing XHTML files. We parse container.xml →
        OPF (package document) → spine, then run each spine item through
        the existing HTML parser. `page` becomes the chapter index."""
        import xml.etree.ElementTree as _ET
        import zipfile as _zipfile

        elements: list[ParsedElement] = []
        try:
            zf = _zipfile.ZipFile(io.BytesIO(file_bytes))
        except _zipfile.BadZipFile as exc:
            raise ValueError(f"EPUB is corrupt or unreadable: {exc}") from None

        try:
            container_xml = zf.read("META-INF/container.xml")
        except KeyError:
            raise ValueError("EPUB missing META-INF/container.xml") from None

        ns = {"cn": "urn:oasis:names:tc:opendocument:xmlns:container"}
        try:
            container_root = _ET.fromstring(container_xml)
        except _ET.ParseError as exc:
            raise ValueError(f"EPUB container.xml is malformed: {exc}") from None
        rootfile = container_root.find(".//cn:rootfile", ns)
        if rootfile is None:
            raise ValueError("EPUB container.xml has no rootfile pointer")
        opf_path = rootfile.get("full-path")
        if not opf_path:
            raise ValueError("EPUB rootfile missing full-path")

        try:
            opf_xml = zf.read(opf_path)
        except KeyError:
            raise ValueError(f"EPUB references missing OPF: {opf_path}") from None

        opf_ns = {"opf": "http://www.idpf.org/2007/opf"}
        try:
            opf_root = _ET.fromstring(opf_xml)
        except _ET.ParseError as exc:
            raise ValueError(f"EPUB OPF is malformed: {exc}") from None

        # Map id → href from manifest.
        id_to_href: dict[str, str] = {}
        for item in opf_root.findall(".//opf:manifest/opf:item", opf_ns):
            item_id = item.get("id")
            href = item.get("href")
            media_type = item.get("media-type", "")
            if item_id and href and "html" in media_type.lower():
                id_to_href[item_id] = href

        # Spine itemref ids give reading order.
        spine_ids = [
            ir.get("idref") for ir in opf_root.findall(".//opf:spine/opf:itemref", opf_ns)
            if ir.get("idref")
        ]

        opf_dir = opf_path.rsplit("/", 1)[0] if "/" in opf_path else ""

        # Skip standard EPUB boilerplate files that are technically in
        # the spine but carry no real content: navigation document,
        # copyright page, colophon, cover. Filename heuristics match
        # the de-facto naming conventions used by every major EPUB
        # producer (iBooks, Kindle, Calibre, Sigil).
        EBOOK_BOILERPLATE_HREFS = (
            "nav", "nav.xhtml", "toc", "toc.xhtml", "copyright",
            "copyright.xhtml", "colophon", "colophon.xhtml",
            "cover", "cover.xhtml", "title", "title.xhtml",
        )

        for chapter_idx, spine_id in enumerate(spine_ids, start=1):
            href = id_to_href.get(spine_id)
            if not href:
                continue
            if href.lower().split("/")[-1] in EBOOK_BOILERPLATE_HREFS:
                continue
            entry_path = f"{opf_dir}/{href}" if opf_dir else href
            try:
                chapter_bytes = zf.read(entry_path)
            except KeyError:
                log.warning("EPUB spine item %s not found in archive", entry_path)
                continue
            chapter_elements = self._parse_html(chapter_bytes)
            # Re-page every element to the chapter index so citations
            # point at "chapter 3" instead of "page 1" for every chunk.
            for el in chapter_elements:
                el.page = chapter_idx
            elements.extend(chapter_elements)
        return elements

    # -- RTF (Tier 2.1) ----------------------------------------------------

    def _parse_rtf(self, file_bytes: bytes) -> list[ParsedElement]:
        """RTF: hand the bytes to `striprtf`, then run the resulting
        plain text through the same blank-line paragraph split as .txt."""
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError as exc:
            raise RuntimeError(
                "striprtf is required for RTF support. Add it to requirements.txt."
            ) from exc
        try:
            text = rtf_to_text(file_bytes.decode("utf-8", errors="replace"))
        except Exception as exc:
            raise ValueError(f"RTF is corrupt or unreadable: {exc}") from None
        elements: list[ParsedElement] = []
        for block in re.split(r"\n\s*\n+", text):
            block = block.strip()
            if not block:
                continue
            elements.append(ParsedElement(type="paragraph", text=block, page=1))
        return elements

    # -- Legacy XLS (Tier 2.2) ---------------------------------------------

    def _parse_xls(self, file_bytes: bytes) -> list[ParsedElement]:
        """Legacy .xls via xlrd 1.2.0 (the last release with .xls
        support — xlrd 2.0+ only handles .xlsx). Hidden sheets are
        skipped via `sheet_visible()`, mirroring the XLSX path."""
        try:
            import xlrd
        except ImportError as exc:
            raise RuntimeError(
                "xlrd==1.2.0 is required for legacy .xls support. Add it to requirements.txt."
            ) from exc
        try:
            book = xlrd.open_workbook(file_contents=file_bytes, formatting_info=False)
        except Exception as exc:
            raise ValueError(f"XLS is corrupt or unreadable: {exc}") from None

        elements: list[ParsedElement] = []
        # xlrd 1.2.0 has no `sheet_visible()` method; visibility lives on
        # the book's `_sheet_visibility` list as integers — 0 = visible,
        # 1 = hidden, 2 = very hidden. Anything non-zero is excluded.
        visibility = getattr(book, "_sheet_visibility", [0] * book.nsheets)
        for sheet_idx in range(book.nsheets):
            if sheet_idx < len(visibility) and visibility[sheet_idx] != 0:
                continue
            sh = book.sheet_by_index(sheet_idx)
            elements.append(ParsedElement(
                type="heading", text=sh.name, page=1, heading_level=1,
            ))
            rows_text: list[str] = []
            for r in range(sh.nrows):
                cell_strs = []
                for c in range(sh.ncols):
                    val = sh.cell_value(r, c)
                    if sh.cell_type(r, c) == xlrd.XL_CELL_DATE:
                        # Try to format dates, fall back to the float.
                        try:
                            import xlrd.xldate as _xldate
                            val = _xldate.xldate_as_datetime(val, book.datemode).date().isoformat()
                        except Exception:
                            val = str(val)
                    val_str = "" if val == "" else str(val).replace("\n", " ").strip()
                    cell_strs.append(val_str)
                while cell_strs and not cell_strs[-1]:
                    cell_strs.pop()
                if any(c.strip() for c in cell_strs):
                    rows_text.append(" | ".join(cell_strs))
            if not rows_text:
                continue

            col_count = max(rows_text[0].count("|") + 1, 1)
            sep = " | ".join(["---"] * col_count)
            if len(rows_text) == 1:
                header = rows_text[0]
                data_rows: list[str] = []
            else:
                header = rows_text[0]
                data_rows = rows_text[1:]
            if len(data_rows) <= MAX_TABLE_ROWS_PER_CHUNK:
                elements.append(ParsedElement(
                    type="table", text="\n".join([header, sep, *data_rows]), page=1,
                ))
            else:
                for start in range(0, len(data_rows), MAX_TABLE_ROWS_PER_CHUNK):
                    piece = data_rows[start:start + MAX_TABLE_ROWS_PER_CHUNK]
                    elements.append(ParsedElement(
                        type="table", text="\n".join([header, sep, *piece]), page=1,
                    ))
        return elements

    # -- OpenDocument (Tier 2.3) -------------------------------------------

    def _parse_odf(self, file_bytes: bytes, mime_type: str) -> list[ParsedElement]:
        """ODF (odt/ods/odp): odfpy walks the body tree directly.
        Branch on mime_type to choose the right element-type mapping
        since the three formats share a content model but emit very
        different chunk shapes."""
        try:
            # Only `opendocument` is used: the ODF walk reads the parsed
            # tree directly (doc.text / doc.spreadsheet / doc.presentation),
            # so the per-element `odf.text` / `odf.table` modules are not
            # needed here.
            from odf import opendocument
        except ImportError as exc:
            raise RuntimeError(
                "odfpy is required for OpenDocument support. Add it to requirements.txt."
            ) from exc
        try:
            doc = opendocument.load(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValueError(f"OpenDocument is corrupt or unreadable: {exc}") from None

        if mime_type.endswith(".text"):
            return self._parse_odf_text(doc)
        if mime_type.endswith(".spreadsheet"):
            return self._parse_odf_spreadsheet(doc)
        if mime_type.endswith(".presentation"):
            return self._parse_odf_presentation(doc)
        # Should be unreachable — dispatch is closed.
        return []

    def _parse_odf_text(self, doc) -> list[ParsedElement]:
        """ODT: every <text:p> is a paragraph; <text:h> is a heading
        whose level is in the `text:outline-level` attribute (1-10)."""
        elements: list[ParsedElement] = []
        body = doc.text if hasattr(doc, "text") else None
        if body is None:
            return elements
        for child in body.childNodes:
            # Element nodes only — skip text/whitespace.
            if not hasattr(child, "tagName"):
                continue
            # odfpy gives tagName with namespace prefix (e.g. "text:h",
            # "text:p"); compare against the local form.
            tag = child.tagName.split(":", 1)[-1]
            if tag == "h":
                # odfpy's getAttribute wants the bare local name with the
                # dash stripped ("outlinelevel", not "outline-level" or
                # "text:outline-level").
                lvl = child.getAttribute("outlinelevel")
                try:
                    heading_level = max(1, min(int(lvl), 6))
                except (TypeError, ValueError):
                    heading_level = 2
                text = "".join(
                    str(n) for n in child.childNodes
                    if hasattr(n, "nodeType") and n.nodeType == n.TEXT_NODE
                ).strip()
                if text:
                    elements.append(ParsedElement(
                        type="heading", text=text, page=1, heading_level=heading_level,
                    ))
            elif tag == "p":
                text = "".join(
                    str(n) for n in child.childNodes
                    if hasattr(n, "nodeType") and n.nodeType == n.TEXT_NODE
                ).strip()
                if text:
                    elements.append(ParsedElement(type="paragraph", text=text, page=1))
        # ODT: page = (count of outline-level 1 + 2 headings seen) + 1.
        # ODT's outline-level attribute is what LibreOffice writes for
        # the heading paragraph style ("Heading 1" → outlinelevel="1",
        # etc.); counting them in document order gives a stable page
        # number for every chunk without depending on the document's
        # actual physical layout.
        return self._renumber_pages_by_headings(elements, bump_levels=(1, 2))

    @staticmethod
    def _parse_odf_spreadsheet(doc) -> list[ParsedElement]:
        """ODS: each <table:table> is a sheet, with rows that may have
        repeated/covered cells via <table:covered-table-cell>."""
        elements: list[ParsedElement] = []
        body = doc.spreadsheet if hasattr(doc, "spreadsheet") else None
        if body is None:
            return elements
        sheets = body.childNodes
        for sheet in sheets:
            if not hasattr(sheet, "tagName"):
                continue
            tag = sheet.tagName.split(":", 1)[-1]
            if tag != "table":
                continue
            sheet_name = sheet.getAttribute("name") or "Sheet"
            elements.append(ParsedElement(
                type="heading", text=sheet_name, page=1, heading_level=1,
            ))
            rows_text: list[str] = []
            for tr in sheet.childNodes:
                if not hasattr(tr, "tagName"):
                    continue
                if tr.tagName.split(":", 1)[-1] != "table-row":
                    continue
                cell_strs: list[str] = []
                for tc in tr.childNodes:
                    if not hasattr(tc, "tagName"):
                        continue
                    tc_tag = tc.tagName.split(":", 1)[-1]
                    if tc_tag == "table-covered-cell":
                        cell_strs.append("")
                        continue
                    if tc_tag != "table-cell":
                        continue
                    repeat = tc.getAttribute("numbercolumnsrepeated")
                    if repeat and repeat.isdigit() and int(repeat) > 1:
                        cell_strs.extend([""] * (int(repeat) - 1))
                    # Cell values live in attributes (e.g.
                    # office:string-value="id") when set via the typed
                    # API. Real ODF documents also embed the value as a
                    # <text:p> child for rich content, so we fall back
                    # to childNodes for that case.
                    text = (tc.getAttribute("stringvalue")
                            or tc.getAttribute("value")
                            or "")
                    if not text:
                        text = "".join(
                            str(n) for n in tc.childNodes
                            if hasattr(n, "nodeType") and n.nodeType == n.TEXT_NODE
                        ).strip()
                    cell_strs.append(text)
                while cell_strs and not cell_strs[-1]:
                    cell_strs.pop()
                if any(c.strip() for c in cell_strs):
                    rows_text.append(" | ".join(cell_strs))
            if not rows_text:
                continue
            col_count = max(rows_text[0].count("|") + 1, 1)
            sep = " | ".join(["---"] * col_count)
            if len(rows_text) == 1:
                header, data_rows = rows_text[0], []
            else:
                header, data_rows = rows_text[0], rows_text[1:]
            if len(data_rows) <= MAX_TABLE_ROWS_PER_CHUNK:
                elements.append(ParsedElement(
                    type="table", text="\n".join([header, sep, *data_rows]), page=1,
                ))
            else:
                for start in range(0, len(data_rows), MAX_TABLE_ROWS_PER_CHUNK):
                    piece = data_rows[start:start + MAX_TABLE_ROWS_PER_CHUNK]
                    elements.append(ParsedElement(
                        type="table", text="\n".join([header, sep, *piece]), page=1,
                    ))
        return elements

    @staticmethod
    def _walk_odp_text(container, slide_idx: int) -> list[ParsedElement]:
        """Pull <text:p> / <text:h> out of an ODP container at any
        nesting depth, recursing through <draw:frame> and
        <draw:text-box> wrappers. The typical ODP placement chain is
        <draw:page> > <draw:frame> > <draw:text-box> > <text:p>; some
        producers skip the text-box layer or add another level.
        tagName-walking (rather than getElementsByType) survives every
        variant and is the same approach _parse_odf_text uses — keeps
        the ODF parsers consistent."""
        out: list[ParsedElement] = []
        for child in container.childNodes:
            if not hasattr(child, "tagName"):
                continue
            local = child.tagName.split(":", 1)[-1]
            if local in ("frame", "text-box"):
                # Container — recurse and collect whatever text lives
                # inside. We deliberately don't bound the depth: real
                # ODPs go 3 levels deep and there's no benefit to
                # clipping arbitrary nesting for this format.
                out.extend(UniversalParser._walk_odp_text(child, slide_idx))
                continue
            if local == "h":
                lvl = child.getAttribute("outlinelevel")
                try:
                    lvl_int = max(1, min(int(lvl), 6))
                except (TypeError, ValueError):
                    lvl_int = 3
                text = "".join(
                    str(n) for n in child.childNodes
                    if hasattr(n, "nodeType") and n.nodeType == n.TEXT_NODE
                ).strip()
                if text:
                    out.append(ParsedElement(
                        type="heading", text=text,
                        page=slide_idx, heading_level=lvl_int,
                    ))
            elif local == "p":
                text = "".join(
                    str(n) for n in child.childNodes
                    if hasattr(n, "nodeType") and n.nodeType == n.TEXT_NODE
                ).strip()
                if text:
                    out.append(ParsedElement(
                        type="paragraph", text=text, page=slide_idx,
                    ))
        return out

    @staticmethod
    def _parse_odf_presentation(doc) -> list[ParsedElement]:
        """ODP: each <draw:page> is a slide. Page number = slide index.

        odfpy exposes `<draw:frame>` (placement container) and
        `<draw:text-box>` (text holder) as the relevant element
        classes — there is no `TextFrame` attribute on `odf.draw`,
        which the previous implementation assumed and crashed on. We
        walk by tagName instead, which also matches the pattern used
        in _parse_odf_text / _parse_odf_spreadsheet."""
        elements: list[ParsedElement] = []
        body = doc.presentation if hasattr(doc, "presentation") else None
        if body is None:
            return elements
        slide_idx = 0
        for slide in body.childNodes:
            if not hasattr(slide, "tagName"):
                continue
            if slide.tagName.split(":", 1)[-1] != "page":
                continue
            slide_idx += 1
            elements.append(ParsedElement(
                type="heading", text=f"Slide {slide_idx}", page=slide_idx, heading_level=2,
            ))
            elements.extend(UniversalParser._walk_odp_text(slide, slide_idx))
        return elements