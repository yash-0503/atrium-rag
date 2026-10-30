"""The PDF backend.

The parser's *logic* — heading detection, reading order, bands, section
paths, table serialisation, list and code detection, boilerplate
removal, the scanned-page path — is ours and is independent of any
particular PDF library. Only seven primitives come from a library:

    open the document            page size
    text with geometry           tables
    render a page to an image    embedded images and their bytes

This module puts a seam around exactly those seven, so the library
becomes a configuration choice rather than a dependency of the
architecture.

**Why the seam exists.** The PDF library is the one place where
licence and speed pull against each other, so it is a configuration
choice rather than a decision baked into the architecture:

    PyMuPDF          AGPL-3.0     fastest, C, best text extraction
    pdfplumber       MIT          pdfminer.six + pypdfium2
    pdfminer.six     MIT          the layout engine under pdfplumber
    pypdfium2        BSD-3        PDFium bindings, C, fast rendering
    RapidOCR         Apache-2.0   the OCR path, already used
    Pillow           HPND         image handling

**pdfplumber is the default**, because it is permissively licensed and
covers all seven primitives, and because it reports rowspan and
colspan on tables — which the parser's table output is otherwise
missing.

PyMuPDF remains available for the deployments that want its speed.
It is AGPL-3.0, and AGPL §13 reaches a service offered over a
network, so it is an opt-in extra rather than a runtime dependency.
Atrium itself is MIT under either choice.

The backends are not interchangeable in behaviour. They extract text
with different geometry, order blocks differently, and infer bold and
monospace differently. The benchmark in `eval_v2/` is the instrument
for measuring that difference rather than guessing at it.

Select a backend with the `ATRIUM_PDF_BACKEND` environment variable:
`pdfplumber` (default), `pymupdf`.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

__all__ = [
    "backend_licence",
    "PdfSpan",
    "PdfLine",
    "PdfBlock",
    "PdfTable",
    "PdfImage",
    "PdfPage",
    "PdfDocument",
    "PdfBackend",
    "get_backend",
    "available_backends",
    "BACKEND_ENV_VAR",
]

BACKEND_ENV_VAR = "ATRIUM_PDF_BACKEND"

# The default backend. pdfplumber is MIT licensed; PyMuPDF is AGPL-3.0
# and is opt-in.
DEFAULT_BACKEND = "pdfplumber"

# PyMuPDF's span flag bits, reproduced here so the backends agree on
# what a flag means.
FLAG_SUPERSCRIPT = 1
FLAG_ITALIC = 2
FLAG_SERIF = 4
FLAG_MONOSPACE = 8
FLAG_BOLD = 16

# Font names that mean monospace. PyMuPDF reports a flag for this;
# pdfminer reports only the name.
_MONO_HINTS = (
    "courier", "consolas", "menlo", "monaco", "dejavusansmono",
    "liberationmono", "nsimsun", "notosansmono", "sourcecodepro",
    "roboto mono", "sfmono", "firacode", "terminal", "couriernew",
    "consola", "cambria mono", "aptos mono", "mesquite", "monospaced",
)

_BOLD_HINTS = ("bold", "black", "heavy", "semibold", "demibold", "extrabold", "ultrabold")


@dataclass
class PdfSpan:
    """A run of text with one font, one size, one position."""

    text: str
    bbox: tuple[float, float, float, float]
    size: float = 10.0
    font: str = ""
    flags: int = 0


@dataclass
class PdfLine:
    """A line of text, made of spans."""

    bbox: tuple[float, float, float, float]
    spans: list[PdfSpan] = field(default_factory=list)


@dataclass
class PdfBlock:
    """A block of text. PyMuPDF groups nearby lines into blocks."""

    bbox: tuple[float, float, float, float]
    lines: list[PdfLine] = field(default_factory=list)


@dataclass
class PdfTable:
    """A detected table and its cell grid."""

    bbox: tuple[float, float, float, float]
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class PdfImage:
    """An embedded image on the page, with its bytes."""

    bbox: tuple[float, float, float, float]
    data: bytes = b""
    mime: str = "image/png"


@dataclass
class _Rect:
    """A page rectangle, with the attributes the parser expects."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class PdfPage:
    """One page, normalised."""

    number: int
    width: float
    height: float
    blocks: list[PdfBlock] = field(default_factory=list)
    tables: list[PdfTable] = field(default_factory=list)
    images: list[PdfImage] = field(default_factory=list)
    # Set by the backend that produced the page, for diagnostics.
    backend: str = ""
    # Renders the page to PNG bytes at a given dpi. Supplied by the
    # backend, because rendering is the one primitive that needs the
    # library's own page object.
    renderer: Optional[Any] = None

    @property
    def rect(self) -> _Rect:
        return _Rect(0.0, 0.0, self.width, self.height)

    def render_png(self, dpi: int = 150) -> bytes:
        """The page as PNG bytes."""
        if self.renderer is None:
            return b""
        return self.renderer(dpi)


@dataclass
class PdfDocument:
    """A normalised document."""

    pages: list[PdfPage] = field(default_factory=list)
    backend: str = ""

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PdfBackend(Protocol):
    """A PDF library, behind our seven primitives."""

    name: str
    license: str

    def open(self, data: bytes) -> PdfDocument:
        ...


# ---------------------------------------------------------------------------
# PyMuPDF — AGPL-3.0, the optional fast backend
# ---------------------------------------------------------------------------


class PyMuPdfBackend:
    """PyMuPDF (AGPL-3.0), wrapped in the normalised shapes.

    This is a faithful translation of what the parser used to do
    inline: the same block grouping, the same flags, the same table
    grids. It exists so the two backends can be compared on the same
    footing, and so the PyMuPDF path keeps working exactly as it did.
    """

    name = "pymupdf"
    license = "AGPL-3.0"

    def open(self, data: bytes) -> PdfDocument:
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        pages: list[PdfPage] = []
        for index in range(len(doc)):
            pages.append(self._page(doc[index], index + 1))
        return PdfDocument(pages=pages, backend=self.name)

    def _page(self, page: Any, number: int) -> PdfPage:
        page_dict = page.get_text("dict", sort=True)
        blocks: list[PdfBlock] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            lines: list[PdfLine] = []
            for line in block.get("lines", []):
                spans = [
                    PdfSpan(
                        text=span.get("text", ""),
                        bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                        size=float(span.get("size", 10.0)),
                        font=str(span.get("font", "")),
                        flags=int(span.get("flags", 0)),
                    )
                    for span in line.get("spans", [])
                ]
                if spans:
                    lines.append(
                        PdfLine(bbox=tuple(line.get("bbox", (0, 0, 0, 0))), spans=spans)
                    )
            if lines:
                blocks.append(PdfBlock(bbox=tuple(block.get("bbox", (0, 0, 0, 0))), lines=lines))

        tables: list[PdfTable] = []
        try:
            for table in page.find_tables():
                grid = table.extract()
                rows = [
                    ["" if cell is None else str(cell).strip() for cell in row]
                    for row in (grid or [])
                ]
                if rows:
                    tables.append(PdfTable(bbox=tuple(table.bbox), rows=rows))
        except Exception:
            pass

        images: list[PdfImage] = []
        try:
            for image in page.get_images(full=True):
                xref = image[0]
                try:
                    extracted = page.parent.extract_image(xref)
                    data_bytes = extracted.get("image") or b""
                    mime = f"image/{extracted.get('ext', 'png')}"
                except Exception:
                    data_bytes = b""
                    mime = "image/png"
                for rect in page.get_image_rects(xref):
                    images.append(
                        PdfImage(
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            data=data_bytes,
                            mime=mime,
                        )
                    )
        except Exception:
            pass

        rect = page.rect
        return PdfPage(
            number=number,
            width=float(rect.width),
            height=float(rect.height),
            blocks=blocks,
            tables=tables,
            images=images,
            backend=self.name,
            renderer=lambda dpi=150: self.render(page, dpi),
        )

    def render(self, page: Any, dpi: int = 150) -> bytes:
        """Render a page to PNG bytes."""
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")


# ---------------------------------------------------------------------------
# pdfplumber — the permissive default, MIT
# ---------------------------------------------------------------------------


class PdfPlumberBackend:
    """pdfplumber (MIT), wrapped in the normalised shapes.

    pdfplumber is the permissive default because it covers all seven
    primitives with one dependency tree — pdfminer.six for the layout
    and text geometry, pypdfium2 for rendering — and because its table
    extraction reports rowspan and colspan, which the parser does not
    currently carry through.

    The differences from PyMuPDF, which the benchmark has to measure:

    * **Text geometry.** pdfminer reports a character's size from the
      text matrix, which is not always the same number PyMuPDF reports
      for the same text. The heading thresholds are derived from the
      size distribution, so they move with it.
    * **Bold and monospace.** There are no flags; both are inferred
      from the font name.
    * **Block grouping.** PyMuPDF merges nearby lines into a block.
      pdfplumber does not group, so blocks here are the lines that
      pdfminer's layout analysis identifies as belonging together.
    * **Reading order.** PyMuPDF emits visual blocks; pdfminer emits
      objects in content-stream order. The parser's own reading-order
      pass runs on top of either.
    """

    name = "pdfplumber"
    license = "MIT"

    def open(self, data: bytes) -> PdfDocument:
        import pdfplumber

        pages: list[PdfPage] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for index, page in enumerate(pdf.pages):
                pages.append(self._page(page, index + 1))
        return PdfDocument(pages=pages, backend=self.name)

    def _page(self, page: Any, number: int) -> PdfPage:
        blocks = self._blocks(page)
        tables = self._tables(page)
        images = self._images(page)
        return PdfPage(
            number=number,
            width=float(page.width),
            height=float(page.height),
            blocks=blocks,
            tables=tables,
            images=images,
            backend=self.name,
            renderer=lambda dpi=150: self.render(page, dpi),
        )

    def _blocks(self, page: Any) -> list[PdfBlock]:
        """Group the page's text into blocks.

        PyMuPDF merges nearby lines into a block; pdfplumber does not
        group at all. Blocks are the unit the parser's band detection
        and reading-order pass work on, so they have to be grouped
        here, by geometry: consecutive lines that start at roughly the
        same left edge and are not separated by a vertical gap.

        A line is a run of characters, and a span is a run of
        characters in one font at one size — the same subdivision
        PyMuPDF reports.
        """
        try:
            raw_lines = page.extract_text_lines(return_chars=True) or []
        except Exception:
            raw_lines = []

        lines: list[PdfLine] = []
        for raw in raw_lines:
            chars = raw.get("chars") or []
            spans = self._spans_from_chars(chars)
            if not spans:
                text = str(raw.get("text", "") or "")
                if not text.strip():
                    continue
                spans = [
                    PdfSpan(
                        text=text,
                        bbox=(
                            float(raw.get("x0", 0)),
                            float(raw.get("top", 0)),
                            float(raw.get("x1", 0)),
                            float(raw.get("bottom", 0)),
                        ),
                        size=float(raw.get("size", 10.0) or 10.0),
                        font=str(raw.get("fontname", "") or ""),
                        flags=_flags_for(str(raw.get("fontname", "") or "")),
                    )
                ]
            lines.append(
                PdfLine(
                    bbox=(
                        float(raw.get("x0", 0)),
                        float(raw.get("top", 0)),
                        float(raw.get("x1", 0)),
                        float(raw.get("bottom", 0)),
                    ),
                    spans=spans,
                )
            )

        return self._group_lines_into_blocks(lines)

    @staticmethod
    def _spans_from_chars(chars: list[dict]) -> list[PdfSpan]:
        """Split a line's characters into spans of one font and size.

        The characters have to be reassembled with their spaces. A PDF
        stores glyphs and their positions, not the space characters
        between them: pdfminer reports each glyph with its box and the
        gap to the next one. Joining the glyph texts directly gives
        "BERT:Pre-trainingofDeep" with no spaces at all, which breaks
        the token alignment against the canonical text and leaves every
        chunk unreadable.

        A space is therefore inserted wherever the horizontal gap
        between two glyphs is wider than a space would be.
        """
        spans: list[PdfSpan] = []
        for group in _group_by(
            chars,
            lambda c: (
                str(c.get("fontname", "") or ""),
                round(float(c.get("size", 0) or 0), 1),
            ),
        ):
            text = _join_glyphs(group)
            if not text:
                continue
            spans.append(
                PdfSpan(
                    text=text,
                    bbox=(
                        min(float(c["x0"]) for c in group),
                        min(float(c["top"]) for c in group),
                        max(float(c["x1"]) for c in group),
                        max(float(c["bottom"]) for c in group),
                    ),
                    size=float(group[0].get("size", 10.0) or 10.0),
                    font=str(group[0].get("fontname", "") or ""),
                    flags=_flags_for(str(group[0].get("fontname", "") or "")),
                )
            )
        return spans

    @staticmethod
    def _group_lines_into_blocks(lines: list[PdfLine]) -> list[PdfBlock]:
        """Merge lines that belong to the same paragraph.

        Two lines join the same block when they start at roughly the
        same left edge and are separated by less than about one line
        height. A wider vertical gap is a paragraph break, and a
        different left edge is a column change.
        """
        blocks: list[PdfBlock] = []
        current: list[PdfLine] = []
        block_x0: Optional[float] = None
        prev_bottom: Optional[float] = None

        def flush() -> None:
            nonlocal current, block_x0, prev_bottom
            if current:
                blocks.append(
                    PdfBlock(
                        bbox=(
                            min(l.bbox[0] for l in current),
                            min(l.bbox[1] for l in current),
                            max(l.bbox[2] for l in current),
                            max(l.bbox[3] for l in current),
                        ),
                        lines=list(current),
                    )
                )
            current = []
            block_x0 = None
            prev_bottom = None

        for line in lines:
            height = max(1.0, line.bbox[3] - line.bbox[1])
            same_column = block_x0 is None or abs(line.bbox[0] - block_x0) <= 30.0
            close = prev_bottom is None or (line.bbox[1] - prev_bottom) <= height * 0.8
            if current and not (same_column and close):
                flush()
            if block_x0 is None:
                block_x0 = line.bbox[0]
            current.append(line)
            prev_bottom = line.bbox[3]
        flush()
        return blocks

    def _tables(self, page: Any) -> list[PdfTable]:
        """Detect tables with pdfplumber.

        `find_tables()` gives the bounding box; `extract()` gives the
        grid. Cells that span rows or columns are repeated in the grid,
        which is how the parser learns about a merged header.
        """
        tables: list[PdfTable] = []
        try:
            found = page.find_tables()
        except Exception:
            return []
        for table in found:
            try:
                grid = table.extract()
            except Exception:
                continue
            rows = [
                ["" if cell is None else str(cell).strip() for cell in row]
                for row in (grid or [])
            ]
            if rows:
                tables.append(PdfTable(bbox=tuple(table.bbox), rows=rows))
        return tables

    def _images(self, page: Any) -> list[PdfImage]:
        """The images embedded on the page.

        The bytes are produced by rendering the page and cropping to
        the image's box, rather than by pulling the raw stream out of
        the PDF. A raw stream can be DCT or a colour-managed JPEG that
        the vision adapter would then have to decode; a cropped PNG is
        always something the adapter can take.
        """
        images: list[PdfImage] = []
        try:
            entries = page.images or []
        except Exception:
            return []
        if not entries:
            return []
        try:
            rendered = page.to_image(resolution=150).original
        except Exception:
            return []
        for entry in entries:
            try:
                x0 = float(entry.get("x0", 0))
                top = float(entry.get("top", 0))
                x1 = float(entry.get("x1", 0))
                bottom = float(entry.get("bottom", 0))
            except (TypeError, ValueError):
                continue
            if x1 <= x0 or bottom <= top:
                continue
            crop = rendered.crop(
                (int(x0), int(top), int(x1), int(bottom))
            )
            buffer = io.BytesIO()
            try:
                crop.save(buffer, format="PNG")
            except Exception:
                continue
            images.append(
                PdfImage(
                    bbox=(x0, top, x1, bottom),
                    data=buffer.getvalue(),
                    mime="image/png",
                )
            )
        return images

    def render(self, page: Any, dpi: int = 150) -> bytes:
        """Render a page to PNG bytes."""
        image = page.to_image(resolution=dpi).original
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )


def _join_glyphs(chars: list[dict]) -> str:
    """Reassemble glyphs into text, restoring the spaces between them.

    The threshold is relative to the glyph width, because a space in
    an 8pt font is narrower than one in a 20pt font, and a fixed
    threshold either loses spaces in small text or invents them in
    large text.
    """
    if not chars:
        return ""
    parts: list[str] = []
    previous: Optional[dict] = None
    for char in chars:
        if previous is not None:
            gap = float(char.get("x0", 0) or 0) - float(previous.get("x1", 0) or 0)
            width = float(previous.get("width", 0) or 0)
            if gap > max(0.35, width * 0.22):
                parts.append(" ")
        parts.append(str(char.get("text", "")))
        previous = char
    return "".join(parts)


def _group_by(items: list[dict], key) -> list[list[dict]]:
    """Group consecutive items that share a key, keeping the order."""
    groups: list[list[dict]] = []
    current_key: Any = object()
    for item in items:
        item_key = key(item)
        if not groups or item_key != current_key:
            groups.append([])
            current_key = item_key
        groups[-1].append(item)
    return groups


def _flags_for(font_name: str) -> int:
    """Infer PyMuPDF-style flags from a font name.

    pdfminer reports the font name and nothing else, so bold,
    monospace and italic have to be read off the name. The conventions
    are the usual ones: "Bold" and "Black" are bold, "Courier" and
    friends are monospace, "Italic" and "Oblique" are italic.
    """
    name = (font_name or "").lower()
    flags = 0
    if any(hint in name for hint in _BOLD_HINTS):
        flags |= FLAG_BOLD
    if any(hint in name for hint in _MONO_HINTS):
        flags |= FLAG_MONOSPACE
    if "italic" in name or "oblique" in name:
        flags |= FLAG_ITALIC
    return flags


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, Any] = {
    "pymupdf": PyMuPdfBackend,
    "pdfplumber": PdfPlumberBackend,
}


def available_backends() -> list[str]:
    """The backends that can be selected."""
    return sorted(_BACKENDS)


def backend_licence(name: Optional[str] = None) -> str:
    """The licence of the selected backend, for the report header."""
    name = name or os.environ.get(BACKEND_ENV_VAR, DEFAULT_BACKEND)
    backend = _BACKENDS.get(name.strip().lower())
    return getattr(backend, "license", "unknown")


def get_backend(name: Optional[str] = None) -> PdfBackend:
    """The selected backend.

    `ATRIUM_PDF_BACKEND` chooses it; `pdfplumber` is the default. MIT
    code that can import an AGPL library is not itself AGPL, but
    shipping the AGPL library inside an MIT distribution would be, so
    the permissive backend is the one that ships by default.
    """
    name = name or os.environ.get(BACKEND_ENV_VAR, DEFAULT_BACKEND)
    key = name.strip().lower()
    if key not in _BACKENDS:
        raise ValueError(
            f"unknown PDF backend {name!r}; available: {', '.join(available_backends())}"
        )
    return _BACKENDS[key]()
