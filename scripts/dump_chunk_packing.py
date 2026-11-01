#!/usr/bin/env python3
"""
dump_chunk_packing.py

Regenerate the chunk-packing data the website shows.

    python scripts/dump_chunk_packing.py \
        --document tests/parsing_fixtures/26_gov_docx/source.docx \
        --out ../atrium-web/src/lib/chunk-packings.json

The page shows a document, the elements the parser made of it, and
what each of the five chunking strategies did with those elements. The
numbers on that page are a drawing of what the shipped code does, so
they come from the shipped code: this script parses the document,
chunks it five ways, and writes the result.

Nothing here is typed by hand, which is the point — the page says
"regenerate with scripts/dump_chunk_packing.py", and this is that
script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

STRATEGIES = (
    "baseline",
    "sentence_bounded",
    "recursive_token",
    "structure_aware",
    "parent_child",
)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _page_of(chunk: Any) -> Optional[int]:
    source = getattr(chunk, "source", None)
    return getattr(source, "page_number", None) if source is not None else None


def _section_path(chunk: Any) -> list:
    source = getattr(chunk, "source", None)
    return list(getattr(source, "section_path", []) or []) if source is not None else []


_SENTENCE_END = ".!?\u2026"
# A sentence can end with a closing quote or bracket after the full stop.
_TRAILING = "\"')]}\u201d\u2019"


def _mid_sentence_cut(text: str, next_text: str) -> bool:
    """Did a sentence run across this chunk's end?

    The chunk stops without finishing a sentence, *and* the next chunk
    picks that sentence up mid-way — which is what a cut looks like. A
    chunk that stops at a heading, a list item or a table row has not
    cut a sentence; the next chunk simply starts something else.
    """
    ending = (text or "").rstrip().rstrip(_TRAILING)
    if not ending or ending[-1] in _SENTENCE_END:
        return False
    following = (next_text or "").lstrip()
    return bool(following) and following[0].islower()


def _preview(text: str, width: int = 48) -> str:
    return (text or "")[:width]


def build(document: Path, target_chars: int, overlap_chars: int) -> dict:
    from atrium.smart_chunker import STRATEGY_DISPATCH, ChunkConfig
    from atrium.universal_parser import UniversalParser

    raw = document.read_bytes()
    elements = UniversalParser().parse(raw, document.name, DOCX_MIME)

    payload: dict[str, Any] = {
        "document": {
            "file_name": document.name,
            "mime": DOCX_MIME,
            "elements": len(elements),
            "chars": sum(len(e.text) for e in elements),
        },
        "target_chars": target_chars,
        "overlap_chars": overlap_chars,
        "elements": [
            {
                "type": element.type,
                "chars": len(element.text),
                "page": element.page,
                "heading_level": element.heading_level,
                "section_path": list(element.section_path or []),
            }
            for element in elements
        ],
        "strategies": {},
    }

    for name in STRATEGIES:
        config = ChunkConfig(
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            strategy=name,
            mode="custom",
        )
        chunks = STRATEGY_DISPATCH[name](elements, "doc", document.name, config)
        rows = []
        for position, chunk in enumerate(chunks):
            text = getattr(chunk, "text", "") or ""
            rows.append(
                {
                    "type": getattr(chunk, "element_type", ""),
                    "chars": len(text),
                    "span": list(getattr(chunk, "char_span", (0, 0)) or (0, 0)),
                    "page": _page_of(chunk),
                    "section_path": _section_path(chunk),
                    "is_parent": bool(getattr(chunk, "is_parent", False)),
                    "children": len(getattr(chunk, "child_ids", []) or []),
                    "mid_sentence_cut": _mid_sentence_cut(
                        text,
                        getattr(chunks[position + 1], "text", "")
                        if position + 1 < len(chunks)
                        else "",
                    ),
                    "preview": _preview(text),
                }
            )
        payload["strategies"][name] = {
            "chunks": len(chunks),
            "chars": sum(row["chars"] for row in rows),
            "mid_sentence_cuts": sum(1 for row in rows if row["mid_sentence_cut"]),
            "rows": rows,
        }

    return payload


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the chunk-packing data.")
    parser.add_argument(
        "--document",
        default="tests/parsing_fixtures/26_gov_docx/source.docx",
        help="the document the page shows",
    )
    parser.add_argument("--target-chars", type=int, default=700)
    parser.add_argument("--overlap-chars", type=int, default=100)
    parser.add_argument("--out", default=None, help="where to write the JSON")
    args = parser.parse_args(argv)

    document = Path(args.document).expanduser()
    if not document.exists():
        print(f"no document at {document}", file=__import__("sys").stderr)
        return 1

    payload = build(document, args.target_chars, args.overlap_chars)
    text = json.dumps(payload, indent=2) + "\n"

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)

    for name, data in payload["strategies"].items():
        print(
            f"  {name:<18} {data['chunks']:>3} chunks · "
            f"{data['mid_sentence_cuts']} cut mid-sentence",
            file=__import__("sys").stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
