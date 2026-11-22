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

# -- pdfplumber word and column detection -----------------------------------
#
# pdfminer has no page-layout model: it groups characters by baseline
# only, so text at the same height in two columns comes back as ONE
# line spanning the full page width. PyMuPDF's structured-text device
# does a layout analysis and never lets a block cross a column gutter.
#
# The gutter, however, is just whitespace, and whitespace is measurable
# from the character boxes pdfminer already reports. These constants
# drive that measurement.

# pdfminer's own word segmentation. Its default of 3.0pt is too loose
# for a LaTeX PDF, where words are positioned rather than separated by
# space glyphs: at 3.0pt a whole line merges into one "word". At 2.0pt
# the segmentation matches PyMuPDF's on the same page.
WORD_X_TOLERANCE = 2.0

# pdfminer's default line-clustering tolerance, in points.
LINE_Y_TOLERANCE = 3.0

# A gap wider than this is a column gutter rather than a space.
# column gutter rather than a space. Measured on the corpus: the gutter
# between two columns is about seven times the space between two
# words, while a full-width paragraph never exceeds twice it.
GUTTER_MIN_GAP_PT = 8.0

# The vertical whitespace profile is computed on horizontal strips of
# this height. A strip that holds any text counts as a "text row".
COLUMN_ROW_PT = 2.0

# A column of the page is a gutter when it holds text on at most this
# fraction of the text rows. Some slack is needed because a full-width
# title, an abstract or an equation crosses the gutter on a few rows.
COLUMN_MAX_COVERAGE = 0.25

# A gutter narrower than this fraction of the page width is noise.
COLUMN_MIN_WIDTH_FRAC = 0.010

# Both sides of a gutter must carry real text: this share of the page's
# characters, and at least this many. Without it a page margin or a
# stray indented block reads as a column.
COLUMN_MIN_CHAR_SHARE = 0.15
COLUMN_MIN_CHARS = 20

# At most this many column splits per page.
COLUMN_MAX_SPLITS = 3



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

        Three things make this match PyMuPDF closely.

        **Words come from pdfminer's own segmentation**, not from a
        local gap rule. A PDF stores glyphs and their positions, not
        the spaces between them, so the spaces have to be put back;
        pdfminer already does that, and doing it again with a local
        threshold is what corrupted word boundaries before.

        **Text is split per column.** pdfminer groups characters by
        baseline with no notion of a column, so on a two-column page
        it returns one line holding both columns. The gutter between
        them is whitespace, and whitespace is measurable, so the page
        is cut at its gutters before the text is read.

        A line is only cut at a gutter when the gap there really is a
        gutter. A title page has a full-width abstract above a
        two-column body, and cutting the abstract at the gutter would
        tear one paragraph in half.
        """
        words = self._words(page)
        if not words:
            return []
        chars = self._chars(page)
        bands = self._column_bands(chars, page)

        all_lines = self._word_lines(words)
        segments: list[list[dict]] = []
        for line in all_lines:
            segments.extend(_split_at_gutters(line, bands))

        blocks: list[PdfBlock] = []
        for band in bands:
            band_words: list[dict] = []
            for segment in segments:
                centre = (float(segment[0]["x0"]) + float(segment[-1]["x1"])) / 2.0
                if band[0] <= centre < band[1]:
                    band_words.extend(segment)
            if not band_words:
                continue
            lines = self._lines_from_words(band_words)
            blocks.extend(self._group_lines_into_blocks(lines))
        return blocks

    @staticmethod
    def _chars(page: Any) -> list[dict]:
        """The page's characters, in reading order."""
        try:
            return page.chars or []
        except Exception:
            return []

    @staticmethod
    def _words(page: Any) -> list[dict]:
        """pdfminer's word segmentation, with the font attributes attached.

        The font name and size come along so a word can be grouped into
        a span; pdfminer reports no style flags, so those still come
        from the font name.
        """
        try:
            return page.extract_words(
                x_tolerance=WORD_X_TOLERANCE,
                extra_attrs=["fontname", "size"],
            ) or []
        except Exception:
            return []

    @staticmethod
    def _raw_lines(page: Any) -> list[dict]:
        """pdfminer's own line grouping, with the characters of each line.

        pdfminer knows about rotated text and text direction, so its
        line grouping is better than anything re-derived from the
        character boxes.
        """
        try:
            return page.extract_text_lines(return_chars=True) or []
        except Exception:
            return []

    @staticmethod
    def _column_bands(chars: list[dict], page: Any) -> list[tuple[float, float]]:
        """The page's column bands, left to right.

        The gutter between two columns is a vertical strip of the page
        that holds almost no text. It is found by sweeping horizontal
        strips of the page and counting, for each x position, how many
        of those strips hold text there — a column of x positions that
        is empty on nearly every strip is a gutter.

        The profile is built from characters, not words: a word's box
        spans the gaps between its letters, so a word reaching into the
        gutter hides it. Characters show the true whitespace.

        A little slack is allowed because a full-width heading crosses
        the gutter on one or two strips. Margins are excluded, a gutter
        has to have real text on both sides, and only the widest few
        gutters are kept.
        """
        if not chars:
            return [(0.0, float(page.width))]

        min_x = int(min(c["x0"] for c in chars))
        max_x = int(max(c["x1"] for c in chars))
        span = max_x - min_x + 2

        rows: dict[int, list[dict]] = {}
        for char in chars:
            rows.setdefault(int(char["top"] / COLUMN_ROW_PT), []).append(char)
        n_rows = len(rows)
        if n_rows < 4:
            return [(0.0, float(page.width))]

        # A difference array over x: each strip that holds text in some
        # x range adds 1 there, and a single prefix sum gives the
        # number of strips holding text at each x.
        diff = [0] * (span + 2)
        for row_chars in rows.values():
            intervals = sorted((int(c["x0"]), int(c["x1"])) for c in row_chars)
            merged: list[tuple[int, int]] = []
            for a, b in intervals:
                if merged and a <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            for a, b in merged:
                lo = max(0, a - min_x)
                hi = min(span - 1, b - min_x)
                if hi >= lo:
                    diff[lo] += 1
                    diff[hi + 1] -= 1

        coverage: list[int] = []
        running = 0
        for value in diff:
            running += value
            coverage.append(running)

        runs: list[tuple[int, int]] = []
        start: Optional[int] = None
        for offset in range(span):
            if coverage[offset] / n_rows <= COLUMN_MAX_COVERAGE:
                start = offset if start is None else start
            elif start is not None:
                runs.append((start, offset - 1))
                start = None
        if start is not None:
            runs.append((start, span - 1))

        min_width = page.width * COLUMN_MIN_WIDTH_FRAC
        total = len(chars)
        splits = []
        for a, b in runs:
            x_a, x_b = a + min_x, b + min_x
            if x_a <= min_x + 2 or x_b >= max_x - 2:
                continue  # a page margin, not a gutter
            if (x_b - x_a) < min_width:
                continue
            cut = (x_a + x_b) / 2.0
            # A gutter separates two columns, so both sides must carry
            # real text. Without this a wide margin or a stray indented
            # block reads as a column, and the page is then read in the
            # wrong order.
            left = sum(1 for c in chars if float(c["x1"]) <= cut)
            right = sum(1 for c in chars if float(c["x0"]) >= cut)
            if left < COLUMN_MIN_CHAR_SHARE * total or right < COLUMN_MIN_CHAR_SHARE * total:
                continue
            if min(left, right) < COLUMN_MIN_CHARS:
                continue
            splits.append((cut, b - a))
        splits.sort(key=lambda item: -item[1])
        cuts = sorted(x for x, _ in splits[:COLUMN_MAX_SPLITS])

        edges = [0.0] + cuts + [float(page.width)]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    @staticmethod
    def _word_lines(words: list[dict]) -> list[list[dict]]:
        """Group words into lines.

        Upright words are clustered by their vertical position and
        sorted left to right within a cluster, which is pdfminer's own
        rule for deciding what sits on the same line. Rotated words —
        a marginal stamp, a vertical label — are clustered along their
        own axis instead, so they do not get mixed into the horizontal
        lines above them.
        """
        if not words:
            return []
        upright = [w for w in words if w.get("upright", True)]
        rotated = [w for w in words if not w.get("upright", True)]

        lines: list[list[dict]] = []
        for cluster in _cluster_by_doctop(upright, LINE_Y_TOLERANCE):
            cluster.sort(key=lambda w: float(w["x0"]))
            lines.append(cluster)
        if rotated:
            for cluster in _cluster_by_position(rotated, LINE_Y_TOLERANCE, "x0"):
                cluster.sort(key=lambda w: float(w["top"]))
                lines.append(cluster)
        return lines

    @staticmethod
    def _lines_from_words(words: list[dict]) -> list[PdfLine]:
        """The lines the words make up."""
        lines: list[PdfLine] = []
        for cluster in PdfPlumberBackend._word_lines(words):
            line = PdfPlumberBackend._line_from_words(cluster)
            if line:
                lines.append(line)
        return lines

    @staticmethod
    def _line_from_words(words: list[dict]) -> Optional[PdfLine]:
        """One line, built from the words that make it up.

        Words are grouped into spans of one font and one size, so the
        parser can still tell a bold run from body text.
        """
        if not words:
            return None
        groups = _group_by(
            words,
            lambda w: (
                str(w.get("fontname", "") or ""),
                round(float(w.get("size", 0) or 0), 1),
            ),
        )
        spans: list[PdfSpan] = []
        for index, group in enumerate(groups):
            text = " ".join(str(w.get("text", "") or "") for w in group)
            if not text.strip():
                continue
            # The parser joins spans without a separator, so the space
            # between two font groups has to sit inside a span.
            if index + 1 < len(groups):
                text += " "
            font = str(group[0].get("fontname", "") or "")
            spans.append(
                PdfSpan(
                    text=text,
                    bbox=(
                        min(float(w["x0"]) for w in group),
                        min(float(w["top"]) for w in group),
                        max(float(w["x1"]) for w in group),
                        max(float(w["bottom"]) for w in group),
                    ),
                    size=float(group[0].get("size", 10.0) or 10.0),
                    font=font,
                    flags=_flags_for(font),
                )
            )
        if not spans:
            return None
        return PdfLine(
            bbox=(
                min(s.bbox[0] for s in spans),
                min(s.bbox[1] for s in spans),
                max(s.bbox[2] for s in spans),
                max(s.bbox[3] for s in spans),
            ),
            spans=spans,
        )

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


def _split_at_gutters(line: list[dict],
                      bands: list[tuple[float, float]]) -> list[list[dict]]:
    """Split a line where the whitespace is a column gutter.

    A line on a two-column page holds both columns, because pdfminer
    groups characters by baseline. The two columns are separated by the
    gutter, and the gutter is much wider than the space between two
    words — measured on the corpus, seven times wider.

    A line that is one full-width paragraph has ordinary word spacing
    all the way along, and cutting it at the column position would
    tear the paragraph in half. Splitting only where the gap is a real
    gutter tells the two cases apart.

    The split also only happens at a column boundary: a wide gap
    somewhere else on the page — a diagram drawn with individually
    placed letters, say — is not a column and must not be broken up.
    """
    if len(line) < 3:
        return [line]
    cuts = [b[0] for b in bands[1:]]
    if not cuts:
        return [line]

    segments: list[list[dict]] = [[line[0]]]
    for previous, word in zip(line, line[1:]):
        gap = float(word["x0"]) - float(previous["x1"])
        at_boundary = any(previous["x1"] <= cut <= word["x0"] for cut in cuts)
        if at_boundary and gap > GUTTER_MIN_GAP_PT:
            segments.append([word])
        else:
            segments[-1].append(word)
    return segments


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


def _cluster_by_doctop(items: list[dict], tolerance: float) -> list[list[dict]]:
    """Group items that sit on the same line, the way pdfminer does.

    The vertical positions are clustered with a tolerance, and items
    within a cluster belong to the same line. This is pdfminer's own
    rule, reimplemented so the backend does not depend on a private
    pdfminer API.
    """
    return _cluster_by_position(items, tolerance, "doctop")


def _cluster_by_position(items: list[dict], tolerance: float,
                         key: str) -> list[list[dict]]:
    """Cluster items by one coordinate, within a tolerance."""
    if not items:
        return []

    def position(item: dict) -> float:
        value = item.get(key)
        if value is None:
            value = item.get("top", 0)
        return float(value or 0)

    values = sorted({position(item) for item in items})
    clusters: list[list[float]] = [[values[0]]]
    last = values[0]
    for value in values[1:]:
        if value <= last + tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
        last = value
    assignment: dict[float, int] = {}
    for index, cluster in enumerate(clusters):
        for value in cluster:
            assignment[value] = index

    groups: dict[int, list[dict]] = {}
    for item in items:
        groups.setdefault(assignment[position(item)], []).append(item)
    return [groups[index] for index in sorted(groups)]


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

