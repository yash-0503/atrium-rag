"""
smart_chunker.py

Turns a ParsedElement stream (from universal_parser.py) into
citation-ready ChunkRecords.

Design choices that matter for chunk quality:
  - Headings are NOT emitted as standalone chunks. A lone heading like
    "III. Writing introductions" carries almost no signal for an
    embedding model and only pollutes the vector index. Instead, a
    heading's text becomes a breadcrumb prefix attached to whatever
    content follows it (`embed_text`), while the chunk's own `text`
    stays clean for display/citation. A heading is only ever emitted
    as its own chunk in the rare case it's followed by nothing (EOF or
    another heading) — so real content is never silently lost.
  - Packing is section-aware: a paragraph buffer flushes whenever
    `section_path` changes, even if under target_chars, so one chunk
    never silently mixes two different sections/topics.
  - Packing is sentence-aware, not a blind character slice: overflow
    is resolved at sentence boundaries, and overlap between adjacent
    chunks carries the last 1-2 complete sentences, never a mid-word
    or mid-sentence fragment.
  - A single element longer than target_chars is hard-split at
    sentence boundaries into multiple chunks (falling back to a raw
    character split only for a single pathological run-on sentence
    with no punctuation at all) — nothing ships as one giant fuzzy
    embedding.
  - Merged-chunk metadata reflects every element it spans: `page_number`
    is the first page touched, and `page_numbers` in SourceMetadata
    lists every distinct page the chunk actually covers, so citations
    and page filters don't silently point at the wrong page for packs
    that cross a page boundary. bbox is only kept when the chunk stays
    on a single page; a fabricated bbox across pages would be wrong.
  - Tables, code blocks, and image captions are never split mid-content
    by the packer — they're atomic units (a genuinely huge table is
    still split by row-groups, repeating the header row in each piece).
  - content_hash covers the chunk text AND its full source metadata
    (page(s), bbox, section_path) — not text alone — so a paragraph
    that moves pages during a re-ingest is correctly flagged as
    changed even though the words are identical.

Token note: embedding models are token-sensitive, not char-sensitive.
Defaults below stay char-based per the "no new dependency" preference
(no tiktoken), but CHARS_PER_TOKEN_ESTIMATE is exposed so callers can
reason about an approximate token budget without adding a tokenizer
dependency; tune DEFAULT_TARGET_CHARS against your actual embedding
model's practical window if retrieval quality suggests it.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field, replace
from typing import List, Optional

from atrium.universal_parser import ParsedElement

log = logging.getLogger("ingest.chunker")

DEFAULT_TARGET_CHARS = 1000
DEFAULT_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 250              # paragraph chunks below this merge into ANY adjacent paragraph
                                    # chunk, even across a real section boundary — see
                                    # _merge_small_paragraph_chunks for why section purity is
                                    # deliberately traded off here
MAX_TABLE_ROWS_PER_CHUNK = 40
ATOMIC_TYPES = ("table", "code", "image_caption", "footnote")   # heading handled separately, see chunk_elements
CONTEXT_BREADCRUMB_DEPTH = 2       # how many ancestor headings to prefix onto a chunk
CHARS_PER_TOKEN_ESTIMATE = 4       # rough, no-tokenizer-dependency approximation

# Caption/note attachment: a SHORT paragraph sitting immediately next to an
# atomic chunk (table / code / image_caption) on the same page is almost
# always a caption (above) or a note (below), not a real standalone
# paragraph. Without this, every table/caption pair becomes two useless
# single-vector chunks, and tiny trailing "see layout problems" sentences
# after a table ship as their own paragraphs. Bounds are deliberately
# conservative — anything that crosses them is treated as a real paragraph
# that should run with surrounding prose, not attach to a nearby atomic
# chunk. Two signals are used together: the text must be short, AND it must
# not end with terminal sentence punctuation (captions and labels almost
# never do — "Sample tables", "Table 5: ...", "see layout problems)" — while
# a real standalone paragraph almost always does). Length alone is too
# loose: any genuine paragraph under ~120 chars would be mis-eaten.
CAPTION_NOTE_MAX_CHARS = 90
CAPTION_NOTE_MAX_SENTENCES = 2
_TERMINAL_PUNCT_RE = re.compile(r"[.!?]\s*$")

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# CJK sentences are not whitespace-separated — "第一句。这是第二句。" has no
# space between the period and the next character. The split point is the
# terminal punctuation itself followed by any non-whitespace character.
CJK_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])(?=\S)")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
                   "e.g", "i.e", "fig", "no", "vol", "st", "approx"}


@dataclass
class SourceMetadata:
    file_name: str
    page_number: int                       # first page this chunk touches
    bbox: Optional[tuple]                  # only set when the chunk stays on one page
    section_path: list[str]
    page_numbers: list[int] = field(default_factory=list)   # every distinct page spanned


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    text: str                # clean content only — what you'd show/cite
    embed_text: str          # text prefixed with a heading breadcrumb — what you embed
    element_type: str
    source: SourceMetadata
    char_span: tuple[int, int]
    content_hash: str
    embedding: Optional[list[float]] = None
    # Parent-child plumbing for Strategy E. Both default to safe no-ops
    # so every existing caller (preview, /ingest, /retrieval_lab) keeps
    # working unchanged. When the parent-child strategy is used, parents
    # get child_ids populated and children get parent_id set. Flat
    # strategies (A/B/C/D) leave these as None / [] and the retrieval
    # layer treats the chunks as v0 (no parent resolution).
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    # Which strategy produced this chunk. Stamped by `chunk_with_mode`
    # (and `chunk_default`) after the strategy function returns, so the
    # producer is recorded on the chunk itself rather than being
    # re-derived by whoever consumes it. Carried through to the
    # persisted payload as `chunking_strategy`.
    chunking_strategy: str = ""


# ChunkConfig + PRESETS
# The public surface advertises 5 strategies A-E and 3
# named presets (balanced / precise / context_rich). Both are data, not
# code branches: a single `chunk_with_mode(elements, doc_id, file_name,
# mode, config)` reads the strategy name out of the resolved config and
# dispatches to the corresponding function. Adding a new strategy means
# adding one function and one entry to STRATEGY_DISPATCH — the public
# API doesn't change.

# Strategy identifiers. Used as the `strategy` field on ChunkConfig and
# as the keys of STRATEGY_DISPATCH below. Add a new strategy by:
#   1. Writing a `chunk_<name>(elements, doc_id, file_name, cfg) -> list[ChunkRecord]`
#   2. Adding the name to STRATEGY_DISPATCH
# That's the whole patch — presets, the public API, and the auto router
# pick it up without changes.
STRATEGY_BASELINE          = "baseline"           # A — pure sliding window, fastest path
STRATEGY_SENTENCE_BOUNDED  = "sentence_bounded"   # B — split on sentence boundaries, then pack
STRATEGY_RECURSIVE_TOKEN   = "recursive_token"    # C — langchain-style hierarchical split
STRATEGY_STRUCTURE_AWARE   = "structure_aware"    # D — respect element types, headings, atomic units
STRATEGY_PARENT_CHILD      = "parent_child"       # E — produce parents (context) + children (retrieval)

_STRATEGY_NAMES = (
    STRATEGY_BASELINE,
    STRATEGY_SENTENCE_BOUNDED,
    STRATEGY_RECURSIVE_TOKEN,
    STRATEGY_STRUCTURE_AWARE,
    STRATEGY_PARENT_CHILD,
)


@dataclass(frozen=True)
class ChunkConfig:
    """Immutable parameter set consumed by all 5 strategies.

    Fields unused by a given strategy are ignored — e.g. `parent_target_chars`
    is only meaningful for `parent_child`; the baseline strategy reads
    only `target_chars` and `overlap_chars`. This keeps the dataclass
    flat (one shape, not 5 specialized subclasses) and lets presets be
    plain dicts / instances that can be serialized for /ingest and the
    custom-mode API.

    Defaults match the legacy `chunk_elements(target_chars=1000,
    overlap_chars=150)` call signature so the structure_aware strategy
    (which is the new default) is byte-equivalent to the old code path.
    Speed and memory profile unchanged for the default config.
    """
    target_chars: int = DEFAULT_TARGET_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS
    min_chunk_chars: int = 100
    strategy: str = STRATEGY_STRUCTURE_AWARE
    # Parent-child knobs (Strategy E only).
    parent_target_chars: int = 3000
    parent_overlap_chars: int = 300
    # Sentence-bounded knobs (Strategy B only).
    sentence_pack: bool = True
    # Structure-aware knobs (Strategy D only). Currently no extra knobs —
    # the existing heading/atomic/caption logic is unconditional. Reserved
    # for future per-doc overrides without breaking the config shape.
    preserve_atomic_units: bool = True
    # Audit label: which preset (or "auto" / "custom") this config was
    # resolved from. Informational only — no strategy reads it — but it
    # makes a config self-describing when it's logged or serialized.
    mode: str = "auto"


# Public preset table. Keys are the preset names; values
# are the ChunkConfig instances used by `chunk_with_mode` when the
# caller passes `mode="balanced"`, `mode="precise"`, etc. Add a new
# preset by adding one line here — no other code changes.
#
# Choice rationale:
#   - `balanced`  → baseline (Strategy A) at the legacy 1000/150
#                  defaults. Fastest, smallest memory. The hot path.
#   - `precise`   → structure_aware (Strategy D) at 700/100. The
#                  structure_aware strategy is what the old chunk_elements
#                  already does, just with tighter packing for dense
#                  technical docs. Smallest change from baseline.
#   - `context_rich` → parent_child (Strategy E) with parents at 3000
#                  chars, children at 1000. Most expensive (2x chunks)
#                  but produces context for citation. Opt-in for users
#                  who need surrounding context in their results.
PRESETS: dict = {
    "balanced": ChunkConfig(
        target_chars=1000, overlap_chars=150,
        strategy=STRATEGY_BASELINE,
        mode="balanced",
    ),
    "precise": ChunkConfig(
        target_chars=700, overlap_chars=100,
        strategy=STRATEGY_STRUCTURE_AWARE,
        mode="precise",
    ),
    "context_rich": ChunkConfig(
        target_chars=1000, overlap_chars=150,
        parent_target_chars=3000, parent_overlap_chars=300,
        strategy=STRATEGY_PARENT_CHILD,
        mode="context_rich",
    ),
}

# The production default, pinned. `chunk_default` and `DEFAULT_CHUNK_CONFIG`
# are the documented back-compat surface for call sites that
# predate the mode dispatcher. It is bound to PRESETS["precise"] — the
# structure-aware strategy at 700/100 chars, which is what the 19-fixture
# benchmark measures.
DEFAULT_CHUNK_CONFIG: ChunkConfig = PRESETS["precise"]


def _validate_chunking_mode(mode: Optional[str]) -> str:
    """Backward-compat: accept the legacy 5-mode names AND preset names.
    Returns the canonical mode string the rest of the dispatcher uses.

    Accepts:
      - None / ""                 → "auto"
      - "auto"                    → "auto"
      - "balanced" / "precise" / "context_rich"  → pass through
      - "custom"                  → pass through (caller must supply config)
      - legacy names (A/B/C/D/E)  → "A" → "baseline", "B" → "sentence_bounded",
                                     "C" → "recursive_token", "D" → "structure_aware",
                                     "E" → "parent_child"
    """
    if mode is None or mode == "":
        return "auto"
    if mode in ("auto", "balanced", "precise", "context_rich", "custom"):
        return mode
    legacy_map = {
        "A": STRATEGY_BASELINE,
        "B": STRATEGY_SENTENCE_BOUNDED,
        "C": STRATEGY_RECURSIVE_TOKEN,
        "D": STRATEGY_STRUCTURE_AWARE,
        "E": STRATEGY_PARENT_CHILD,
    }
    if mode in legacy_map:
        return legacy_map[mode]
    raise ValueError(
        f"chunking_mode must be one of ('auto', 'balanced', 'precise', "
        f"'context_rich', 'custom', 'A', 'B', 'C', 'D', 'E'), got {mode!r}"
    )


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _content_hash(text: str, source: SourceMetadata) -> str:
    payload = "|".join([
        text, source.file_name, str(source.page_number), str(source.page_numbers),
        str(source.bbox), ">".join(source.section_path),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _with_context(text: str, breadcrumb_headings: list[str]) -> str:
    if not breadcrumb_headings:
        return text
    breadcrumb = " > ".join(breadcrumb_headings[-CONTEXT_BREADCRUMB_DEPTH:])
    return f"[{breadcrumb}]\n{text}"


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Latin/Cyrillic/Greek scripts use whitespace as a sentence boundary,
    # CJK does not. Pick the splitter based on whether CJK terminal
    # punctuation is present — the two regexes are independent so a mixed
    # document falls into the CJK path (where the absence of whitespace
    # between Latin sentences is harmless; the period-with-space pattern
    # wouldn't fire anyway on "Hello world. Next sentence." inside a CJK
    # document because there's no CJK punctuation driving the choice).
    if any(c in text for c in "。！？"):
        raw_splits = CJK_SENTENCE_SPLIT_RE.split(text)
    else:
        raw_splits = SENTENCE_SPLIT_RE.split(text)
    merged: list[str] = []
    for part in raw_splits:
        if merged:
            prev = merged[-1]
            last_word = re.split(r"\s+", prev.strip())[-1].rstrip(".").lower()
            if last_word in _ABBREVIATIONS:
                merged[-1] = prev + " " + part
                continue
        merged.append(part)
    return [p for p in merged if p.strip()]


def _looks_like_caption_or_note(text: str) -> bool:
    """True when `text` reads as a caption or footnote fragment rather than
    a real standalone paragraph.

    Two signals, both required:
      1. Length: must be short enough that a real paragraph wouldn't fit
         comfortably (CAPTION_NOTE_MAX_CHARS). Real paragraphs that wander
         next to tables are almost always longer than this.
      2. Shape: must NOT end with terminal sentence punctuation (.!?).
         Captions and labels — "Sample tables", "Table 5: ...", "see
         layout problems)" — almost never do, while genuine paragraphs
         almost always do. This is the signal that distinguishes a 60-char
         caption from a 60-char real paragraph ending with a period.

    The risk of a real-but-short standalone paragraph getting eaten by an
    adjacent atomic chunk is bounded by these two checks together, not
    either alone: a genuine ~80-char paragraph without terminal punctuation
    is rare in practice, and even when it happens the cost (it joins the
    table chunk) is small compared to the default failure mode (it ships
    as a single-sentence chunk with no retrievable signal).
    """
    text = text.strip()
    if not text or len(text) > CAPTION_NOTE_MAX_CHARS:
        return False
    if _TERMINAL_PUNCT_RE.search(text):
        return False
    sentences = _split_sentences(text)
    return 1 <= len(sentences) <= CAPTION_NOTE_MAX_SENTENCES


def _pack_sentences(sentences: list[str], spans: list[tuple[int, int]],
                     target_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
    """Greedily pack sentences into groups under target_chars. A single
    sentence larger than target_chars (a run-on with no internal
    punctuation) is the only case that falls back to a raw character
    split, since there's no punctuation left to split on."""
    groups: list[tuple[str, int, int]] = []
    current: list[str] = []
    current_spans: list[tuple[int, int]] = []

    def flush_group() -> None:
        if not current:
            return
        groups.append((" ".join(current), current_spans[0][0], current_spans[-1][1]))

    i = 0
    while i < len(sentences):
        sent, span = sentences[i], spans[i]

        if len(sent) > target_chars and not current:
            for j in range(0, len(sent), target_chars):
                piece = sent[j:j + target_chars]
                groups.append((piece, span[0] + j, span[0] + j + len(piece)))
            i += 1
            continue

        current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
        if current and current_len + 1 + len(sent) > target_chars:
            flush_group()
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap_chars or len(overlap_sents) >= 2:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s) + 1
            # If the overlap seed alone would already overflow once `sent` is
            # added back, keep looping with a non-empty `current` forever —
            # flush_group() re-emits the same seed as a "new" group each pass
            # (i is never advanced) and the result list grows without bound.
            # Dropping the seed here guarantees `current` becomes falsy, which
            # forces the plain-append branch below and advances `i`.
            if overlap_len + 1 + len(sent) > target_chars:
                overlap_sents = []
            current = overlap_sents.copy()
            current_spans = current_spans[-len(overlap_sents):] if overlap_sents else []
            continue  # retry adding this sentence to the fresh, overlap-seeded buffer

        current.append(sent)
        current_spans.append(span)
        i += 1

    flush_group()
    return groups


def _common_section_prefix(a: list[str], b: list[str]) -> list[str]:
    """Longest shared ancestor path — e.g. ['Ch1','Topic 1.1'] + ['Ch1','Topic 1.2']
    -> ['Ch1']. Used when merging bridges a real section boundary: the
    merged chunk no longer belongs to one specific subtopic exclusively,
    so its breadcrumb should point at the nearest ancestor both share
    (never a heading that only one side actually sits under)."""
    out: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def _combine_paragraph_chunks(prev: ChunkRecord, nxt: ChunkRecord) -> ChunkRecord:
    text = prev.text.rstrip() + "\n\n" + nxt.text.lstrip()
    pages = sorted(set(prev.source.page_numbers) | set(nxt.source.page_numbers))
    bbox = prev.source.bbox if prev.source.bbox == nxt.source.bbox else None
    section_path = (prev.source.section_path if prev.source.section_path == nxt.source.section_path
                    else _common_section_prefix(prev.source.section_path, nxt.source.section_path))
    source = SourceMetadata(file_name=prev.source.file_name, page_number=pages[0], bbox=bbox,
                             section_path=section_path, page_numbers=pages)
    return ChunkRecord(
        doc_id=prev.doc_id, chunk_id=str(uuid.uuid4()), text=text,
        embed_text=_with_context(text, source.section_path),
        element_type="paragraph", source=source,
        char_span=(prev.char_span[0], nxt.char_span[1]),
        content_hash=_content_hash(text, source),
    )


def _merge_small_paragraph_chunks(chunks: list[ChunkRecord], target_chars: int,
                                    min_chars: int) -> list[ChunkRecord]:
    """
    Section-aware/sentence-aware packing (_process_run/_pack_sentences)
    already fills chunks up to target_chars WITHIN one contiguous run of
    paragraph/list_item elements. But a run ends the moment an atomic
    element (table/code/image_caption) or a heading appears — even when the
    "section" on both sides is identical, or the heading itself only has
    1-2 sentences of real content under it. Two situations previously
    shipped unusably tiny chunks:
      1. A diagram/table between two short paragraphs of the SAME section
         split what should be one chunk into three.
      2. A document with many small REAL sections (each a legitimately
         differently-sized/numbered heading, just with little content
         under it) produced one tiny chunk per section — e.g. a 130-char,
         two-sentence chunk for each of ten short subsections.

    This is a second, global pass over the fully assembled chunk list. It
    merges two ADJACENT chunks whenever both are element_type == "paragraph"
    (table/code/image_caption/heading chunks are never touched, matching
    ATOMIC_TYPES elsewhere in this module), at least one side is < min_chars,
    and the combined text still fits target_chars.

    Deliberate trade-off: this does NOT require the two chunks to share the
    same section_path. A 150-char, two-real-sentence chunk under its own
    genuine heading is just as useless for retrieval as a 20-char fragment
    from a false heading — so below min_chars, section purity is traded for
    a chunk that's actually large enough to carry retrievable signal. When
    a merge does cross a real section boundary, the merged chunk's
    section_path becomes the common ancestor of both sides (see
    _common_section_prefix), never either side's specific breadcrumb alone.
    If this merges too aggressively for your content (e.g. many short but
    individually complete/standalone sections), lower MIN_CHUNK_CHARS
    rather than reintroducing a same-section requirement here.
    """
    if not chunks:
        return chunks
    merged: list[ChunkRecord] = [chunks[0]]
    for c in chunks[1:]:
        prev = merged[-1]
        mergeable_types = prev.element_type == "paragraph" and c.element_type == "paragraph"
        if not mergeable_types:
            merged.append(c)
            continue

        combined_len = len(prev.text) + 2 + len(c.text)   # +2 for the "\n\n" joiner
        either_small = len(prev.text) < min_chars or len(c.text) < min_chars

        if either_small and combined_len <= target_chars:
            merged[-1] = _combine_paragraph_chunks(prev, c)
        else:
            merged.append(c)
    return merged


def _process_run(run_elements: list[ParsedElement], breadcrumb: list[str],
                  target_chars: int, overlap_chars: int, doc_id: str,
                  file_name: str, chunks: list[ChunkRecord]) -> None:
    """A 'run' is a contiguous sequence of paragraph/list_item elements
    sharing one section_path. Join -> sentence-split -> pack, so packing
    is both section-aware (callers only ever pass one section's worth of
    elements) and sentence-aware (never slices mid-sentence)."""
    if not run_elements:
        return

    joined_parts: list[str] = []
    offsets: list[tuple[int, int, ParsedElement]] = []
    cursor = 0
    for el in run_elements:
        start = cursor
        joined_parts.append(el.text)
        cursor += len(el.text)
        offsets.append((start, cursor, el))
        cursor += 1  # account for the "\n" join separator

    joined_text = "\n".join(joined_parts)
    sentences = _split_sentences(joined_text)

    spans: list[tuple[int, int]] = []
    search_from = 0
    for s in sentences:
        idx = joined_text.find(s, search_from)
        if idx == -1:
            idx = search_from
        spans.append((idx, idx + len(s)))
        search_from = idx + len(s)

    groups = _pack_sentences(sentences, spans, target_chars, overlap_chars)

    for group_text, group_start, group_end in groups:
        touched = [el for (s, e, el) in offsets if e > group_start and s < group_end]
        pages = sorted({el.page for el in touched}) or [run_elements[0].page]
        bbox = touched[0].bbox if len(pages) == 1 and touched and touched[0].bbox else None
        text = group_text.strip()
        source = SourceMetadata(file_name=file_name, page_number=pages[0], bbox=bbox,
                                 section_path=list(breadcrumb), page_numbers=pages)
        chunks.append(ChunkRecord(
            doc_id=doc_id, chunk_id=str(uuid.uuid4()), text=text,
            embed_text=_with_context(text, breadcrumb),
            element_type="paragraph", source=source,
            char_span=(group_start, group_end),
            content_hash=_content_hash(text, source),
        ))


def chunk_elements(elements: list[ParsedElement], doc_id: str, file_name: str,
                    target_chars: int = DEFAULT_TARGET_CHARS,
                    overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    run_buffer: list[ParsedElement] = []
    pending_context: list[str] = []            # heading breadcrumb not yet attached to a chunk
    pending_orphan_headings: list[ParsedElement] = []   # consecutive headings not yet resolved
    pending_caption: Optional[ParsedElement] = None     # short paragraph held to prepend onto next atomic

    def flush_run() -> None:
        nonlocal run_buffer
        _process_run(run_buffer, pending_context, target_chars, overlap_chars,
                     doc_id, file_name, chunks)
        run_buffer = []

    def _prev_chunk_is_atomic_on_same_page(page: int) -> bool:
        return (bool(chunks) and chunks[-1].element_type in ATOMIC_TYPES
                and chunks[-1].source.page_number == page)

    def _next_is_atomic_on_same_page(idx: int) -> bool:
        if idx + 1 >= len(elements):
            return False
        nxt = elements[idx + 1]
        return nxt.type in ATOMIC_TYPES and nxt.page == elements[idx].page

    for idx, el in enumerate(elements):

        if el.type == "heading":
            # A heading breaks any pending caption context — a heading
            # between the caption and the table means they're not actually
            # adjacent in reading order.
            pending_caption = None
            flush_run()
            pending_context = list(el.section_path)
            # Tentatively orphaned — resolved (and discarded) the moment any
            # real content follows, below. A whole nested run (H1 -> H2 ->
            # H3 -> paragraph) must NOT emit one standalone chunk per level;
            # only the case where the run runs off the end of the document
            # with nothing ever following it needs to surface anything.
            pending_orphan_headings.append(el)
            continue

        # Any non-heading element resolves the pending run as "not
        # orphaned" — those headings successfully attached as context to
        # whatever is about to be processed below.
        pending_orphan_headings = []

        if el.type in ATOMIC_TYPES:
            flush_run()
            for piece_text, piece_source in _atomic_pieces(el, file_name):
                # If a caption has been waiting for THIS atomic chunk on the
                # same page, prepend it now — both to the display text (so
                # retrieval sees the caption with the table) and to
                # embed_text (so the embedding carries the caption context
                # too). char_span grows to cover the caption length.
                if pending_caption and pending_caption.page == el.page:
                    caption_text = pending_caption.text.strip()
                    caption_with_ctx = (_with_context(caption_text, pending_caption.section_path)
                                        if pending_caption.section_path else caption_text)
                    new_text = caption_text + "\n\n" + piece_text
                    new_embed = caption_with_ctx.rstrip() + "\n\n" + piece_text
                    chunks.append(ChunkRecord(
                        doc_id=doc_id, chunk_id=str(uuid.uuid4()), text=new_text,
                        embed_text=new_embed, element_type=el.type,
                        source=piece_source, char_span=(0, len(new_text)),
                        content_hash=_content_hash(new_text, piece_source),
                    ))
                    pending_caption = None
                    continue
                chunks.append(ChunkRecord(
                    doc_id=doc_id, chunk_id=str(uuid.uuid4()), text=piece_text,
                    embed_text=_with_context(piece_text, pending_context),
                    element_type=el.type, source=piece_source,
                    char_span=(0, len(piece_text)),
                    content_hash=_content_hash(piece_text, piece_source),
                ))
            continue

        # paragraph / list_item: a SHORT element sitting immediately next to
        # an atomic chunk on the same page is a caption (above) or a note
        # (below), not a real standalone paragraph — attach it to the atomic
        # chunk rather than letting it ship as its own single-vector chunk.
        if _looks_like_caption_or_note(el.text):
            # Note case: previous chunk is atomic on the same page.
            if _prev_chunk_is_atomic_on_same_page(el.page):
                prev = chunks[-1]
                note = el.text.strip()
                new_text = prev.text + "\n\n" + note
                new_embed = prev.embed_text.rstrip() + "\n\n" + note
                new_span = (prev.char_span[0], prev.char_span[1] + len(note) + 2)
                chunks[-1] = ChunkRecord(
                    doc_id=prev.doc_id, chunk_id=str(uuid.uuid4()), text=new_text,
                    embed_text=new_embed, element_type=prev.element_type,
                    source=prev.source, char_span=new_span,
                    content_hash=_content_hash(new_text, prev.source),
                )
                continue
            # Caption case: next element is atomic on the same page — flush
            # whatever's in the run, hold this element as a pending caption,
            # and let the atomic branch above pick it up.
            if _next_is_atomic_on_same_page(idx):
                flush_run()
                pending_caption = el
                continue

        # paragraph / list_item: accumulate into the current run, but a
        # section change ends the run immediately even under target_chars —
        # two different sections never get packed into the same vector.
        if run_buffer and run_buffer[-1].section_path != el.section_path:
            flush_run()
        run_buffer.append(el)

    # Drop any still-pending caption at end of document — without a target
    # atomic chunk to attach to, it would just be a paragraph; let it run as
    # one by adding it to the buffer and flushing.
    if pending_caption:
        run_buffer.append(pending_caption)
        pending_caption = None

    flush_run()

    # The document ended on one or more headings with nothing ever
    # following them (a trailing outline, or a genuinely empty final
    # section). Surface exactly ONE combined chunk for the whole run —
    # never one standalone chunk per heading level. The ancestor breadcrumb
    # comes from the run's own FIRST heading's section_path (minus itself),
    # not from whatever pending_context happened to be left over from an
    # earlier, unrelated section.
    if pending_orphan_headings:
        combined_text = " > ".join(h.text for h in pending_orphan_headings)
        first, last = pending_orphan_headings[0], pending_orphan_headings[-1]
        ancestor_context = first.section_path[:-1]
        source = SourceMetadata(file_name=file_name, page_number=last.page, bbox=last.bbox,
                                 section_path=ancestor_context, page_numbers=[last.page])
        chunks.append(ChunkRecord(
            doc_id=doc_id, chunk_id=str(uuid.uuid4()), text=combined_text,
            embed_text=_with_context(combined_text, ancestor_context),
            element_type="heading", source=source,
            char_span=(0, len(combined_text)),
            content_hash=_content_hash(combined_text, source),
        ))

    return _merge_small_paragraph_chunks(chunks, target_chars, MIN_CHUNK_CHARS)


def _atomic_pieces(el: ParsedElement, file_name: str):
    source = SourceMetadata(file_name=file_name, page_number=el.page, bbox=el.bbox,
                             section_path=el.section_path, page_numbers=[el.page])
    if el.type != "table":
        yield el.text, source
        return

    lines = el.text.split("\n")
    if len(lines) <= MAX_TABLE_ROWS_PER_CHUNK + 2:
        yield el.text, source
        return

    header, sep, *rows = lines
    for i in range(0, len(rows), MAX_TABLE_ROWS_PER_CHUNK):
        piece_rows = rows[i:i + MAX_TABLE_ROWS_PER_CHUNK]
        yield "\n".join([header, sep, *piece_rows]), source

# Strategy implementations + dispatch
# Five chunking strategies (A-E). Each is a top-level
# function with the same signature so the dispatcher can pick one by
# name from the resolved ChunkConfig. The `structure_aware` strategy
# IS the existing `chunk_elements` — the dispatcher wraps it without
# touching the function. The other four are new add-on code that does
# NOT modify `chunk_elements` or any of its helpers.

def _chunk_baseline(elements: list[ParsedElement], doc_id: str,
                    file_name: str, cfg: ChunkConfig) -> list[ChunkRecord]:
    """Strategy A — pure sliding window. Fastest path. No heading
    detection, no atomic-unit logic, no caption attachment. Just
    concatenates element text and slices on target_chars with overlap.

    Use this when speed and memory are the priority and the document
    doesn't need structure-aware handling. PDF text-extraction
    pipelines and bulk-ingest jobs default here.
    """
    chunks: list[ChunkRecord] = []
    buffer: list[str] = []
    buf_len = 0
    char_offset = 0
    page_for_buffer: Optional[int] = None
    section_for_buffer: list[str] = []

    def _flush() -> None:
        nonlocal buffer, buf_len, char_offset
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if not text:
            buffer, buf_len, char_offset = [], 0, 0
            return
        source = SourceMetadata(
            file_name=file_name,
            page_number=page_for_buffer if page_for_buffer is not None else 0,
            bbox=None,
            section_path=list(section_for_buffer),
            page_numbers=[page_for_buffer] if page_for_buffer is not None else [],
        )
        span = (char_offset, char_offset + len(text))
        chunks.append(ChunkRecord(
            doc_id=doc_id,
            chunk_id=str(uuid.uuid4()),
            text=text,
            embed_text=_with_context(text, section_for_buffer),
            element_type="paragraph",
            source=source,
            char_span=span,
            content_hash=_content_hash(text, source),
        ))
        # Keep the tail as overlap for the next chunk.
        if cfg.overlap_chars > 0 and len(text) > cfg.overlap_chars:
            tail = text[-cfg.overlap_chars:]
            buffer, buf_len, char_offset = [tail], len(tail), char_offset + len(text) - len(tail)
        else:
            buffer, buf_len, char_offset = [], 0, char_offset + len(text)

    for el in elements:
        if not el.text:
            continue
        page_for_buffer = el.page if hasattr(el, "page") else page_for_buffer
        if hasattr(el, "section_path") and el.section_path:
            section_for_buffer = list(el.section_path)
        # Split the element text on whitespace so we don't carry massive
        # single-string buffers when an element is huge. Each word is
        # appended; flush when we cross target_chars.
        for piece in el.text.split():
            if buf_len + len(piece) + 1 > cfg.target_chars and buffer:
                _flush()
            buffer.append(piece)
            buf_len += len(piece) + 1
        # Add a separator between elements so words don't run together.
        if buffer:
            buffer.append(" ")
            buf_len += 1
    _flush()
    return chunks


def _chunk_sentence_bounded(elements: list[ParsedElement], doc_id: str,
                             file_name: str, cfg: ChunkConfig) -> list[ChunkRecord]:
    """Strategy B — sentence-bounded. Uses the existing
    `_split_sentences` helper (handles CJK and Latin scripts) to break
    each element into sentences, then greedily packs sentences into
    chunks up to `target_chars`. No sentence is ever split.

    Use this for narrative prose where sentence boundaries carry
    semantic weight (academic papers, articles, transcripts). Slower
    than baseline because of the per-element regex split, but
    produces cleaner retrieval units.
    """
    chunks: list[ChunkRecord] = []
    buf: list[str] = []
    buf_len = 0
    char_offset = 0
    page_for_buf: Optional[int] = None
    section_for_buf: list[str] = []

    def _flush() -> None:
        nonlocal buf, buf_len, char_offset
        if not buf:
            return
        text = " ".join(s for s in buf if s).strip()
        if not text:
            buf, buf_len, char_offset = [], 0, 0
            return
        source = SourceMetadata(
            file_name=file_name,
            page_number=page_for_buf if page_for_buf is not None else 0,
            bbox=None,
            section_path=list(section_for_buf),
            page_numbers=[page_for_buf] if page_for_buf is not None else [],
        )
        span = (char_offset, char_offset + len(text))
        chunks.append(ChunkRecord(
            doc_id=doc_id,
            chunk_id=str(uuid.uuid4()),
            text=text,
            embed_text=_with_context(text, section_for_buf),
            element_type="paragraph",
            source=source,
            char_span=span,
            content_hash=_content_hash(text, source),
        ))
        if cfg.overlap_chars > 0 and len(text) > cfg.overlap_chars:
            tail = text[-cfg.overlap_chars:]
            buf, buf_len, char_offset = [tail], len(tail), char_offset + len(text) - len(tail)
        else:
            buf, buf_len, char_offset = [], 0, char_offset + len(text)

    for el in elements:
        if not el.text:
            continue
        page_for_buf = el.page if hasattr(el, "page") else page_for_buf
        if hasattr(el, "section_path") and el.section_path:
            section_for_buf = list(el.section_path)
        for sentence in _split_sentences(el.text):
            s = sentence.strip()
            if not s:
                continue
            if buf_len + len(s) + 1 > cfg.target_chars and buf:
                _flush()
            buf.append(s)
            buf_len += len(s) + 1
    _flush()
    return chunks


def _chunk_recursive_token(elements: list[ParsedElement], doc_id: str,
                            file_name: str, cfg: ChunkConfig) -> list[ChunkRecord]:
    """Strategy C — langchain-style recursive split. Tries the biggest
    separator first (`\\n\\n` → paragraph), then `\\n` → line, then
    sentence, then word. Greedily packs each level into chunks up to
    `target_chars`; if a single unit at any level still exceeds
    target_chars, recurses into the next-finer separator.

    Use this for generic structured documents where you don't know the
    shape in advance. Slower than baseline and sentence-bounded
    because of the recursive descent, but more robust to mixed-shape
    inputs.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_idx: int) -> list[str]:
        if len(text) <= cfg.target_chars or sep_idx >= len(separators):
            return [text]
        parts = text.split(separators[sep_idx])
        out: list[str] = []
        for p in parts:
            if not p:
                continue
            if len(p) > cfg.target_chars:
                out.extend(_split(p, sep_idx + 1))
            else:
                out.append(p)
        return out

    chunks: list[ChunkRecord] = []
    buf: list[str] = []
    buf_len = 0
    char_offset = 0
    page_for_buf: Optional[int] = None
    section_for_buf: list[str] = []

    def _flush() -> None:
        nonlocal buf, buf_len, char_offset
        if not buf:
            return
        text = " ".join(buf).strip()
        if not text:
            buf, buf_len, char_offset = [], 0, 0
            return
        source = SourceMetadata(
            file_name=file_name,
            page_number=page_for_buf if page_for_buf is not None else 0,
            bbox=None,
            section_path=list(section_for_buf),
            page_numbers=[page_for_buf] if page_for_buf is not None else [],
        )
        span = (char_offset, char_offset + len(text))
        chunks.append(ChunkRecord(
            doc_id=doc_id,
            chunk_id=str(uuid.uuid4()),
            text=text,
            embed_text=_with_context(text, section_for_buf),
            element_type="paragraph",
            source=source,
            char_span=span,
            content_hash=_content_hash(text, source),
        ))
        buf, buf_len, char_offset = [], 0, char_offset + len(text)

    for el in elements:
        if not el.text:
            continue
        page_for_buf = el.page if hasattr(el, "page") else page_for_buf
        if hasattr(el, "section_path") and el.section_path:
            section_for_buf = list(el.section_path)
        for unit in _split(el.text, 0):
            if buf_len + len(unit) + 1 > cfg.target_chars and buf:
                _flush()
            buf.append(unit)
            buf_len += len(unit) + 1
    _flush()
    return chunks


def _chunk_structure_aware(elements: list[ParsedElement], doc_id: str,
                            file_name: str, cfg: ChunkConfig) -> list[ChunkRecord]:
    """Strategy D — the existing `chunk_elements`. This wrapper exists
    so the dispatcher can pick it by name without the rest of the
    code knowing it's a special case. Behavior is byte-equivalent to
    the pre-Strategy-dispatch code path; no internals changed."""
    return chunk_elements(
        elements, doc_id, file_name,
        target_chars=cfg.target_chars,
        overlap_chars=cfg.overlap_chars,
    )


def _chunk_parent_child(elements: list[ParsedElement], doc_id: str,
                         file_name: str, cfg: ChunkConfig) -> list[ChunkRecord]:
    """Strategy E — parent-child. Generates "parent" chunks via the
    existing `chunk_elements` at `parent_target_chars`, then for each
    parent generates "child" chunks at `target_chars` with
    `overlap_chars`. Each child's `parent_id` points at the parent;
    each parent's `child_ids` lists the children.

    The lab's retrieval layer (and `retrieve_and_rerank` in
    `atrium/retrieval.py`) reads these fields to do parent resolution:
    rank by child similarity, but expose the parent text in the
    response. This is the "context_rich" preset.

    Returns ONLY the children plus the parents (both as ChunkRecord
    entries). Parents are returned at the tail of the list so the
    embedding matrix the lab builds has children first, parents
    after — child embeddings are what cosine-sim queries against.
    """
    # 1) Generate parents using the existing structure-aware chunker
    # at the larger parent_target_chars. The structure-aware logic
    # is preserved end-to-end — we only override the size knobs.
    parents = chunk_elements(
        elements, doc_id, file_name,
        target_chars=cfg.parent_target_chars,
        overlap_chars=cfg.parent_overlap_chars,
    )
    # 2) For each parent, generate children by re-running chunk_elements
    # over just that parent's text. This is N times the work of a
    # single chunk_elements call (N = parent count) but is correct:
    # the structure-aware rules apply to the parent text the same
    # way they apply to a whole document. Faster approaches exist
    # (sliding window over the parent's char_span, etc.) but
    # correctness is what matters for the context_rich preset.
    all_records: list[ChunkRecord] = []
    for parent in parents:
        # Re-parse the parent's text into a minimal ParsedElement so
        # the existing chunker can run on it. The structure-aware
        # rules (heading detection, atomic handling) will still fire
        # but in practice a parent is usually pure prose.
        from atrium.universal_parser import ParsedElement
        pseudo_el = ParsedElement(
            text=parent.text,
            type="paragraph",
            page=parent.source.page_number,
            section_path=list(parent.source.section_path),
            bbox=None,
        )
        children = chunk_elements(
            [pseudo_el], doc_id, file_name,
            target_chars=cfg.target_chars,
            overlap_chars=cfg.overlap_chars,
        )
        # Wire the parent-child links. Children's chunk_ids and the
        # parent's child_ids are populated in place.
        child_ids: list[str] = []
        for child in children:
            child.parent_id = parent.chunk_id
            child_ids.append(child.chunk_id)
        parent.child_ids = child_ids
        all_records.extend(children)
    # Append parents at the end. The lab's cosine-sim search over the
    # embedding matrix therefore only sees children (the parent
    # embeddings are not part of the searchable index). Retrieval
    # resolves to parent on hit.
    all_records.extend(parents)
    return all_records


# Dispatch table. Adding a new strategy = one function + one entry here.
# Nothing else in the chunker changes. The OSS contract is: 5
# strategies, 3 presets, this dict is the source of truth.
STRATEGY_DISPATCH: dict = {
    STRATEGY_BASELINE:         _chunk_baseline,
    STRATEGY_SENTENCE_BOUNDED: _chunk_sentence_bounded,
    STRATEGY_RECURSIVE_TOKEN:  _chunk_recursive_token,
    STRATEGY_STRUCTURE_AWARE:  _chunk_structure_aware,
    STRATEGY_PARENT_CHILD:     _chunk_parent_child,
}


def _resolve_chunk_config(mode: str, config: Optional[dict]) -> ChunkConfig:
    """Resolve a ChunkConfig from a mode name + optional overrides.

    Resolution order:
      1. If `mode` is a preset name (balanced / precise / context_rich),
         start from PRESETS[mode].
      2. If `mode` is 'auto', start from PRESETS['precise']. (2026-10-17
         fix: was PRESETS['balanced'], which silently downgraded every
         /ingest + /retrieval_lab/prepare + harness call to Strategy A
         baseline. The published 0.8717 R@5 baseline was measured against
         D-precise, not the baseline the dispatcher was actually serving.)
      3. If `mode` is 'custom', require a non-None `config` dict;
         start from PRESETS['precise'] and overlay config.
      4. Apply `config` overrides on top (any ChunkConfig field the
         caller supplied wins). Unknown keys raise so callers can't
         silently misspell a field name.
    """
    if mode in PRESETS:
        base = PRESETS[mode]
    elif mode == "auto":
        # 2026-10-17: was PRESETS["balanced"]. See Resolution order above.
        base = replace(PRESETS["precise"], mode="auto")
    elif mode == "custom":
        if not config:
            raise ValueError(
                "chunking_mode='custom' requires a non-empty chunking_config dict"
            )
        base = replace(PRESETS["precise"], mode="custom")
    else:
        raise ValueError(
            f"chunking_mode must be one of {tuple(PRESETS.keys())} + 'auto'/'custom', got {mode!r}"
        )
    if config:
        valid_fields = {f for f in ChunkConfig.__dataclass_fields__}
        unknown = set(config) - valid_fields
        if unknown:
            raise ValueError(
                f"chunking_config has unknown keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_fields)}"
            )
        # frozen dataclass → use replace() to overlay
        return replace(base, **config)
    return base


def resolve_chunk_config(
    mode: Optional[str] = "auto", config: Optional[dict] = None,
) -> ChunkConfig:
    """Public: resolve a mode + override dict into a `ChunkConfig`.

    `chunk_with_mode` does this internally, but a caller that wants to
    *validate* a request before doing any work — the HTTP ingest
    endpoints, which must reject a bad `chunking_mode` with 422 rather
    than failing inside a background job — needs to resolve the config
    itself. Doing it through one function keeps the validation rules
    (unknown keys, missing config for `custom`, unknown mode names) in
    exactly one place.

    Raises `ValueError` with a human-readable message on any invalid
    input.
    """
    return _resolve_chunk_config(_validate_chunking_mode(mode), config)


def _stamp_strategy(chunks: list, strategy: str) -> list:
    """Record the producing strategy on every chunk it returns.

    One place, so the strategy is recorded by the thing that actually
    ran it — a consumer never has to re-derive it from the mode name,
    and a mode that resolves to a different strategy than its name
    suggests still records what really happened.
    """
    for c in chunks:
        try:
            c.chunking_strategy = strategy
        except AttributeError:
            # A strategy function returned something that isn't a
            # ChunkRecord. Leave it alone rather than crash the
            # chunker over an audit field.
            pass
    return chunks


def chunk_with_mode(elements, doc_id: str, file_name: str,
                    mode: str = "auto", config: Optional[dict] = None) -> list:
    """Mode-aware chunking entry point. The single dispatch the rest
    of the system (FastAPI endpoints, the benchmark harness) calls.

    Resolves a ChunkConfig from `mode` + optional `config` override
    and dispatches to the strategy function named in the config. The
    `structure_aware` strategy is the existing `chunk_elements`
    behavior, preserved end-to-end. Other strategies are add-on code
    that does not touch `chunk_elements` or its helpers.

    Speed/memory contract:
      - 'auto' / 'balanced' / 'precise' default to either baseline
        (the fastest path) or structure_aware (the existing behavior).
        Both run in the same O(N) cost profile as the pre-Strategy
        code.
      - 'context_rich' (parent_child) is intentionally more expensive
        — it runs the chunker N+1 times (1 for parents, N for child
        splits). Opt in when you need parent context, not as a default.
    """
    mode_v = _validate_chunking_mode(mode)
    cfg = _resolve_chunk_config(mode_v, config)
    strategy_fn = STRATEGY_DISPATCH.get(cfg.strategy)
    if strategy_fn is None:
        # Defensive: shouldn't happen because ChunkConfig defaults
        # to a known strategy and PRESETS are typed. But if someone
        # hand-builds a ChunkConfig with a bad name, fall back to
        # structure_aware so we don't crash on real input.
        log.warning("Unknown strategy %r, falling back to structure_aware", cfg.strategy)
        strategy_fn = _chunk_structure_aware
    return _stamp_strategy(
        strategy_fn(elements, doc_id, file_name, cfg), cfg.strategy,
    )


def chunk_default(elements, doc_id: str, file_name: str) -> list:
    """Back-compat entry point bound to `DEFAULT_CHUNK_CONFIG`
    (PRESETS["precise"] — structure_aware at 700/100 chars).

    Kept for call sites that predate the mode dispatcher. The signature
    matches `chunk_elements` so it's a drop-in replacement. New code
    should prefer `chunk_with_mode`, which can reach the other presets.
    """
    return _stamp_strategy(
        _chunk_structure_aware(elements, doc_id, file_name, DEFAULT_CHUNK_CONFIG),
        DEFAULT_CHUNK_CONFIG.strategy,
    )
