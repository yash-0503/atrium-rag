"""
tsr_adapter.py

Optional, opt-in Table Structure Recognition (TSR) adapter for tables
where the heuristic `find_tables()` + whitespace-column fallback loses
to Docling's TableFormer. Specifically:

  - merged-cell financial tables (Apple 10-K style)
  - tables with multi-row / multi-column spanning headers
  - borderless tables that PyMuPDF's detector misses entirely

The default install ships only the Protocol. Operators plug in their
own TSR implementation (UniTable, Microsoft's TableTransformer, Surya
Tables, etc.) by subclassing `TSRAdapterProtocol` and passing it to
UniversalParser(tsr_adapter=...). Off by default — the heuristic
parser does the right work on the 90% of tables that aren't
deliberately pathological, and turning TSR on adds latency + a model
download.

The parser decides WHEN to call the TSR adapter based on three
heuristics computed from the heuristic `find_tables()` output:
  1. High column-count variance across rows (a row has 3 cells, the
     next has 6 cells → merged cells).
  2. Multi-row headers (the first 1-2 rows have fewer cells than the
     body — typical of merged-cell column headers).
  3. Empty cells surrounded by populated cells (a row with one
     empty cell sandwiched between populated ones — characteristic
     of misdetected merged-cell tables).

If any signal fires AND `tsr_adapter` is set, the parser crops the
table bbox from the page pixmap and calls
`tsr_adapter.recognize_table(cropped_image_bytes, bbox, hint_rows)`.
The adapter returns better-structured rows, which the parser uses
verbatim. If the adapter isn't configured, the parser logs a single
INFO message naming the first suspicious table and continues with the
heuristic result — operators see "table X may be improved with
TSR_ENABLED=1" in logs and can choose to flip the flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

log = logging.getLogger("ingest.tsr")


@dataclass
class RecognizedTable:
    """What `recognize_table()` returns. `rows` is a list of lists
    matching the existing `_table_rows_to_element` contract (first row
    is the header, rest are body). `confidence` is the model's
    self-reported per-table score (0.0 if the adapter doesn't report
    one). `bbox` echoes the input so callers can confirm the crop
    landed where they expected."""
    rows: List[List[str]] = field(default_factory=list)
    confidence: float = 0.0
    bbox: tuple = ()


class TSRAdapterProtocol(Protocol):
    """Duck-typed adapter contract. The parser calls `recognize_table`
    on the cropped pixmap bytes of one suspicious table plus the
    heuristic rows as a hint. Returns a RecognizedTable with the
    adapter's best-effort row structure, or an empty rows list on
    silent failure (the parser falls back to the heuristic)."""
    def recognize_table(self, image_bytes: bytes, bbox: tuple,
                         hint_rows: Optional[List[List[str]]] = None,
                         page_num: Optional[int] = None) -> RecognizedTable: ...
    def health(self) -> bool: ...


class StubTSRAdapter:
    """No-op adapter. Returns the heuristic rows unchanged. Useful as
    a default that satisfies `TSR_ENABLED=1` without loading any model
    — operators can flip the flag without committing to a heavyweight
    model install, see the suspicious-table warnings in logs, and
    upgrade to a real TSR implementation later by passing that class
    instead.

    Operators wanting a real TSR should subclass `TSRAdapterProtocol`
    with one of:
      - `UniTableAdapter` (planned) — Microsoft's UniTable, MIT,
        ~80 MB ONNX, <500 ms per table on CPU.
      - Any class implementing the Protocol using their preferred
        model — TabStruct, TableTransformer, Surya Tables, etc.
    """

    def recognize_table(self, image_bytes: bytes, bbox: tuple,
                         hint_rows: Optional[List[List[str]]] = None,
                         page_num: Optional[int] = None) -> RecognizedTable:
        """Pass-through: return the heuristic rows as-is. Operators
        upgrading to a real model should override this."""
        rows: List[List[str]] = []
        if hint_rows:
            for row in hint_rows:
                rows.append([str(c) if c is not None else "" for c in row])
        return RecognizedTable(rows=rows, confidence=0.0, bbox=bbox)

    def health(self) -> bool:
        return True