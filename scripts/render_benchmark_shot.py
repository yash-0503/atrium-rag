#!/usr/bin/env python3
"""
render_benchmark_shot.py

Render the published benchmark run as a terminal-style PNG for the
README.

    python scripts/render_benchmark_shot.py

Reads `docs/benchmark/headline.json` — the summary the evaluation
harness extracts from a run and draws it the way the terminal prints it: the prompt
line, the per-fixture table, the pass count, the headline numbers.

Every number in the image comes out of that JSON — nothing is typed by
hand. If the run data is replaced, the image changes with it, so it
cannot drift into being a pretty lie.

The output goes to `docs/images/benchmark.png` (override with
`--out`). The image is regenerated on demand rather than on every
commit: it is a rendering of a specific dated run, and the date is
printed on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent

# Terminal palette (close to macOS Terminal's "Basic" dark theme).
BG = (30, 31, 36)
TITLEBAR = (45, 46, 52)
FG = (214, 218, 222)
DIM = (128, 134, 142)
GREEN = (126, 200, 126)
YELLOW = (222, 184, 108)
BLUE = (122, 168, 232)
RED = (232, 122, 122)
WHITE = (245, 247, 250)

FONT_CANDIDATES = [
    ("/System/Library/Fonts/Menlo.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
    ("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 0),
    ("C:/Windows/Fonts/consola.ttf", 0),
]

PAD = 22
LINE_H = 21
FONT_SIZE = 15
TITLE_H = 30


def _load_font(size: int = FONT_SIZE):
    for path, index in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size, index=index)
            except Exception:
                continue
    return ImageFont.load_default()


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def build_lines(headline: dict) -> list:
    """Turn the headline JSON into (text, colour) lines."""
    lines = []
    config = headline.get("configuration", {})

    lines.append((
        f"atrium benchmark — {headline.get('documents', '?')} documents · "
        f"{headline.get('questions', '?')} questions",
        BLUE,
    ))
    lines.append(("", FG))
    lines.append((
        f"parse {config.get('pdf_backend', '?')} · "
        f"chunk {config.get('chunker', '?')}",
        DIM,
    ))
    lines.append((
        f"retrieve {config.get('retrieval', '?')} · "
        f"reranker {config.get('reranker', '?')} · "
        f"embed {config.get('embedder', '?')}",
        DIM,
    ))
    lines.append(("", FG))

    name_w = 16
    header = (
        f"{'SYSTEM':<{name_w}} {'R@5':>6} {'95% CI':>17} {'R@10':>6} "
        f"{'MRR':>6} {'PARSE':>8} {'PEAK MB':>8}"
    )
    lines.append((header, DIM))
    lines.append(("─" * len(header), DIM))

    for system in headline.get("systems", []):
        low, high = (system.get("recall_at_5_ci") or [0.0, 0.0])[:2]
        colour = GREEN if system.get("name") == "atrium" else FG
        lines.append((
            f"{system['name']:<{name_w}} {system['recall_at_5']:>6.3f} "
            f"[{low:>6.3f},{high:>6.3f}] {system['recall_at_10']:>6.3f} "
            f"{system['mrr']:>6.3f} {system['parse_seconds']:>7.1f}s "
            f"{system['peak_rss_mb']:>8.0f}",
            colour,
        ))

    lines.append(("─" * len(header), DIM))
    lines.append((
        "each system uses its own parser and chunker · one embedding model",
        DIM,
    ))
    unreviewed = headline.get("unreviewed_gated_questions", 0)
    if unreviewed:
        lines.append((
            f"provisional: {unreviewed} of {headline.get('questions', '?')} "
            f"questions in the gated tiers are not human-reviewed",
            YELLOW,
        ))
    lines.append((
        f"run {str(headline.get('generated', '?'))[:10]} · "
        f"intervals bootstrapped by document",
        DIM,
    ))
    return lines


def render(lines, out_path: Path, title: str = "atrium — benchmark") -> Path:
    font = _load_font()

    # Measure the widest line.
    probe = Image.new("RGB", (10, 10))
    d = ImageDraw.Draw(probe)
    width = max((d.textlength(text, font=font) for text, _ in lines), default=400)
    img_w = int(max(880, width + PAD * 2))
    img_h = TITLE_H + PAD * 2 + len(lines) * LINE_H

    img = Image.new("RGB", (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)

    # Title bar with the three traffic lights.
    draw.rectangle([0, 0, img_w, TITLE_H], fill=TITLEBAR)
    for i, colour in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        cx = 18 + i * 18
        draw.ellipse([cx - 5, TITLE_H // 2 - 5, cx + 5, TITLE_H // 2 + 5], fill=colour)
    tw = draw.textlength(title, font=font)
    draw.text(((img_w - tw) / 2, TITLE_H / 2 - FONT_SIZE / 2 - 1), title,
              font=font, fill=DIM)

    y = TITLE_H + PAD
    for text, colour in lines:
        draw.text((PAD, y), text, font=font, fill=colour)
        y += LINE_H

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--headline",
        default=str(REPO / "docs" / "benchmark" / "headline.json"),
        help="Path to the headline JSON.",
    )
    ap.add_argument(
        "--out",
        default=str(REPO / "docs" / "images" / "benchmark.png"),
        help="Where to write the PNG.",
    )
    args = ap.parse_args(argv)

    headline_path = Path(args.headline)
    if not headline_path.exists():
        print(f"no such headline: {headline_path}", file=sys.stderr)
        return 1
    headline = json.loads(headline_path.read_text())

    out = render(build_lines(headline), Path(args.out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
