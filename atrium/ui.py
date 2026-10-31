"""
atrium/ui.py

Terminal presentation for Atrium.

Design language: deep space, restrained. Slate and indigo with one cool
gradient, thin glyphs, and motion that has a purpose:

  * the ATRIUM wordmark warps in behind a leading edge of light, then a
    diagonal sheen passes over it once; a faint three-layer starfield
    drifts past (stars slide *behind* the letters);
  * prompts are arrow-key driven: select, multiselect, review card;
  * a quiet orbit spinner for work in progress;
  * checks and the review card reveal row by row.

All motion switches itself off when output is not a terminal (pipes,
CI, --json), when NO_ANIM or NO_BANNER is set, or when TERM=dumb.
Scripts never see escape codes.

Keyboard notes: prompts use cbreak mode (not raw) so newlines behave,
and read bytes with os.read so an arrow key (ESC [ A) is never mistaken
for a lone Esc.
"""

from __future__ import annotations

import contextlib
import os
import random
import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

ACCENT = "#818CF8"      # indigo
ACCENT_L = "#A5B4FC"    # light indigo
SKY = "#7DD3FC"
OK = "#34D399"
BAD = "#F87171"
WARN = "#FBBF24"
SLATE = "#64748B"
SLATE_D = "#334155"

AURORA = ["#7DD3FC", "#818CF8", "#A78BFA", "#C084FC"]  # sky -> indigo -> violet

THEME = Theme(
    {
        "ok": f"bold {OK}",
        "warn": f"bold {WARN}",
        "bad": f"bold {BAD}",
        "key": f"bold {SKY}",
        "dim": SLATE,
        "accent": f"bold {ACCENT_L}",
    }
)


def _rgb(h: str) -> Tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _mix(a: str, b: str, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    ra, rb = _rgb(a), _rgb(b)
    return "#{:02X}{:02X}{:02X}".format(*tuple(round(ra[i] + (rb[i] - ra[i]) * t) for i in range(3)))


def _blend(stops: Sequence[str], t: float) -> str:
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    return _mix(stops[i], stops[i + 1], t - i)


def gradient_text(s: str, stops: Sequence[str] = AURORA, bold: bool = False) -> Text:
    out = Text()
    n = max(len(s) - 1, 1)
    for i, ch in enumerate(s):
        out.append(ch, style=("bold " if bold else "") + _blend(stops, i / n))
    return out


def _animations_ok(console: Console) -> bool:
    if not console.is_terminal:
        return False
    if os.environ.get("NO_ANIM") or os.environ.get("NO_BANNER") or os.environ.get("CI"):
        return False
    return os.environ.get("TERM", "") != "dumb"


@contextlib.contextmanager
def _quiet_input():
    """Hide keystrokes typed during an animation (no ^[[B junk on screen)."""
    if os.name == "nt" or not sys.stdin.isatty():
        yield
        return
    import termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~termios.ECHO
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Wordmark and sky
# ---------------------------------------------------------------------------

_GLYPHS = {
    "A": [" ███ ", "█   █", "█████", "█   █", "█   █"],
    "T": ["█████", "  █  ", "  █  ", "  █  ", "  █  "],
    "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
    "I": ["█████", "  █  ", "  █  ", "  █  ", "█████"],
    "U": ["█   █", "█   █", "█   █", "█   █", " ███ "],
    "M": ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
}

# (cells per tick, glyph, colour): far, mid, near. Slower = further away.
_LAYERS = [
    (0.10, "·", "#334155"),
    (0.25, "·", "#64748B"),
    (0.45, "•", "#94A3B8"),
]

_LEFT, _RIGHT, _TOP, _BOTTOM = 3, 5, 2, 2


def _wordmark_rows(word: str, gap: int = 1) -> List[str]:
    rows = []
    for r in range(5):
        row = (" " * gap).join(_GLYPHS[ch][r] for ch in word)
        rows.append("".join(c * 2 for c in row))  # double width: square pixels
    return rows


class _Sky:
    """The wordmark, its shadow, and a drifting starfield."""

    def __init__(self, word: str = "ATRIUM", seed: int = 5) -> None:
        self.rows = _wordmark_rows(word)
        self.w = len(self.rows[0])
        self.cw = self.w + _LEFT + _RIGHT
        self.ch = _TOP + len(self.rows) + _BOTTOM
        self.solid = {
            (y, x) for y, row in enumerate(self.rows) for x, c in enumerate(row) if c == "█"
        }
        self.shadow = {(y + 1, x + 2) for (y, x) in self.solid} - self.solid
        self.base = {
            (y, x): _mix(_blend(AURORA, x / max(self.w - 1, 1)), "#FFFFFF", 0.14 * (1 - y / 4))
            for (y, x) in self.solid
        }
        rng = random.Random(seed)
        self.stars = [
            (
                rng.randrange(self.ch),
                rng.randrange(self.cw),
                rng.choices([0, 1, 2], weights=[5, 3, 2])[0],
            )
            for _ in range(int(self.cw * self.ch * 0.04))
        ]

    def frame(self, progress: float, tick: int, sheen: Optional[float] = None) -> Group:
        """progress 0..1: how far the wordmark has warped in.
        sheen: position of the diagonal light sweep, or None."""
        reveal = progress * (self.w + 8)
        grid: List[List[Optional[Tuple[str, str]]]] = [
            [None] * self.cw for _ in range(self.ch)
        ]

        for (ly, lx) in self.shadow:
            if lx - 2 < reveal - 6:
                grid[ly + _TOP][lx + _LEFT] = ("▒", "#3730A3")

        for (ly, lx) in self.solid:
            if lx >= reveal:
                continue
            colour = self.base[(ly, lx)]
            edge = reveal - lx
            if progress < 1.0 and edge < 6:
                colour = _mix(colour, "#E0F2FE", (1 - edge / 6) * 0.85)
            if sheen is not None:
                d = abs(lx + ly * 2 - sheen)
                if d < 7:
                    colour = _mix(colour, "#FFFFFF", (1 - d / 7) ** 1.4 * 0.85)
            grid[ly + _TOP][lx + _LEFT] = ("█", colour)

        for (y, x0, layer) in self.stars:
            speed, glyph, colour = _LAYERS[layer]
            x = (x0 - int(tick * speed)) % self.cw
            if grid[y][x] is None:  # stars pass behind the letters
                grid[y][x] = (glyph, colour)

        lines = []
        for row in grid:
            t = Text(no_wrap=True, overflow="crop")
            for cell in row:
                if cell is None:
                    t.append(" ")
                else:
                    t.append(cell[0], style=cell[1])
            lines.append(t)
        return Group(*lines)


_TAGLINE = "Universal RAG ingestion"


def brand(
    console: Console,
    subtitle: str = "",
    version: str = "",
    animate: bool = False,
) -> None:
    """Print the ATRIUM banner (animated when animate=True and allowed)."""
    if not console.is_terminal or os.environ.get("NO_BANNER"):
        return

    sky = _Sky()
    if console.width < sky.cw:
        console.print(gradient_text("ATRIUM", bold=True))
    elif animate and _animations_ok(console):
        total, reveal_frames = 34, 20
        with _quiet_input(), Live(
            sky.frame(0.0, 0), console=console, auto_refresh=False, transient=False
        ) as live:
            for i in range(total + 1):
                p = min(1.0, i / reveal_frames)
                p = 1 - (1 - p) ** 3  # fast launch, soft landing
                sheen = None
                if i >= 16:
                    s = (i - 16) / (total - 16)
                    sheen = -12 + s * (sky.w + 30)
                live.update(sky.frame(p, i, sheen), refresh=True)
                time.sleep(0.03)
    else:
        console.print(sky.frame(1.0, 0))

    line = Text(" " * _LEFT)
    line.append_text(gradient_text(_TAGLINE))
    meta = "  ·  ".join(p for p in (f"v{version}" if version else "", subtitle) if p)
    if meta:
        line.append("   " + meta, style="dim")
    console.print(line)
    console.print()


# ---------------------------------------------------------------------------
# Small motion helpers
# ---------------------------------------------------------------------------


def launch_ready(console: Console, message: str) -> None:
    """The success line, with a light sweep passing over it once."""
    label = f"✓ {message}"
    if not _animations_ok(console):
        console.print(Text(label, style=f"bold {OK}"))
        return
    with _quiet_input(), Live(console=console, auto_refresh=False, transient=False) as live:
        for step in range(len(label) + 14):
            t = Text()
            for i, ch in enumerate(label):
                d = abs(i - (step - 6))
                colour = OK if d >= 5 else _mix(OK, "#FFFFFF", (1 - d / 5) * 0.9)
                t.append(ch, style=f"bold {colour}")
            live.update(t, refresh=True)
            time.sleep(0.02)


def reveal_checks(console: Console, checks: Iterable, animate: bool = True) -> None:
    """Print check results one by one (name, ok, detail attributes)."""
    pace = 0.09 if animate and _animations_ok(console) else 0.0
    for c in checks:
        line = Text("  ")
        line.append("✓ " if c.ok else "✕ ", style=f"bold {OK if c.ok else BAD}")
        line.append(str(c.name).ljust(22))
        line.append(str(c.detail), style="dim")
        console.print(line)
        if pace:
            time.sleep(pace)


def _register_spinners() -> None:
    try:
        from rich._spinners import SPINNERS

        SPINNERS["atrium_orbit"] = {
            "interval": 100,
            "frames": ["◜", "◠", "◝", "◞", "◡", "◟"],
        }
    except Exception:  # private API moved: the default spinner is fine
        pass


_register_spinners()


def space_status(console: Console, message: str):
    """`with space_status(console, "Ingesting…"):` — a quiet orbit spinner."""
    try:
        return console.status(message, spinner="atrium_orbit", spinner_style=ACCENT)
    except Exception:
        return console.status(message)


# ---------------------------------------------------------------------------
# Review card
# ---------------------------------------------------------------------------

Section = Tuple[str, List[Tuple[str, str]]]  # (heading, [(label, value)])


def review_card(
    console: Console,
    sections: Sequence[Section],
    title: str = "Review",
    animate: bool = False,
) -> None:
    """A bordered summary. Values must already be masked by the caller."""
    label_w = max((len(l) for _, rows in sections for l, _ in rows), default=8) + 2
    width = max(40, min(console.width - 2, 84))

    def build(count: int) -> Panel:
        blocks: List[Text] = []
        for si, (heading, rows) in enumerate(sections[:count]):
            if si:
                blocks.append(Text(""))
            blocks.append(Text(heading, style=f"bold {ACCENT_L}"))
            for label, value in rows:
                row = Text("  ")
                row.append(label.ljust(label_w), style="dim")
                row.append(str(value))
                blocks.append(row)
        return Panel(
            Group(*blocks),
            title=Text(f" {title} ", style=f"bold {ACCENT_L}"),
            title_align="left",
            border_style=SLATE_D,
            padding=(1, 2),
            width=width,
        )

    if animate and _animations_ok(console):
        with _quiet_input(), Live(
            build(0), console=console, auto_refresh=False, transient=False
        ) as live:
            for n in range(1, len(sections) + 1):
                live.update(build(n), refresh=True)
                time.sleep(0.12)
    else:
        console.print(build(len(sections)))


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _keys():
    """cbreak mode for the whole prompt: single keys, no echo."""
    if os.name == "nt":
        yield
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key() -> str:
    """One of: up, down, enter, esc, ctrl-c, or the character typed."""
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return {"H": "up", "P": "down"}.get(msvcrt.getwch(), "")
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "ctrl-c"
        if ch == "\x1b":
            return "esc"
        return ch

    import select as _select

    fd = sys.stdin.fileno()
    ch = os.read(fd, 1)  # raw read: bypasses Python's input buffer
    if ch == b"\x1b":
        if _select.select([fd], [], [], 0.05)[0]:
            seq = os.read(fd, 2)
            return {
                b"[A": "up", b"[B": "down",
                b"OA": "up", b"OB": "down",
                b"[C": "down", b"[D": "up",
            }.get(seq, "")
        return "esc"
    if ch in (b"\r", b"\n"):
        return "enter"
    if ch in (b"\x03", b"\x04"):
        return "ctrl-c"
    return ch.decode("utf-8", "ignore")


# ---------------------------------------------------------------------------
# select / multiselect
# ---------------------------------------------------------------------------

Option = Tuple[str, str]  # (label, hint)


def _window(n: int, index: int, rows: int) -> Tuple[int, int]:
    if n <= rows:
        return 0, n
    start = min(max(index - rows // 2, 0), n - rows)
    return start, start + rows


def _render(
    title: str,
    options: Sequence[Option],
    index: int,
    console: Console,
    checked: Optional[set] = None,
) -> Group:
    width = console.width
    rows = max(4, min(12, console.height - 6))
    label_w = max(len(label) for label, _ in options)

    def line(*parts: Tuple[str, str]) -> Text:
        t = Text(no_wrap=True, overflow="crop")
        for s, style in parts:
            t.append(s, style=style)
        return t

    lines = [line(("◆ ", f"bold {ACCENT}"), (title, "bold"))]
    start, end = _window(len(options), index, rows)
    if start > 0:
        lines.append(line(("    ↑ more", "dim")))
    for i in range(start, end):
        label, hint = options[i]
        active = i == index
        parts = [("  ", ""), ("❯ " if active else "  ", f"bold {ACCENT}")]
        used = 4
        if checked is not None:
            on = i in checked
            parts.append(("◼ " if on else "◻ ", ACCENT_L if on else SLATE))
            used += 2
        parts.append((label.ljust(label_w), f"bold {ACCENT_L}" if active else ""))
        used += label_w
        if hint:
            room = max(0, width - used - 3)
            parts.append(("  " + hint[:room], "dim"))
        lines.append(line(*parts))
    if end < len(options):
        lines.append(line(("    ↓ more", "dim")))
    footer = (
        "↑↓ navigate · space toggle · enter confirm · esc back"
        if checked is not None
        else "↑↓ navigate · enter select · esc cancel"
    )
    lines.append(line(("  " + footer, "dim")))
    return Group(*lines)


def _interactive(console: Console) -> bool:
    return sys.stdin.isatty() and console.is_terminal


def select(
    console: Console,
    title: str,
    options: Sequence[Option],
    default: int = 0,
) -> int:
    """Pick one option; returns its index. Esc / Ctrl-C raise KeyboardInterrupt."""
    if not options:
        raise ValueError("select() needs at least one option")
    index = min(max(default, 0), len(options) - 1)

    if not _interactive(console):
        return _select_numbered(console, title, options, index)

    with _keys(), Live(
        _render(title, options, index, console),
        console=console,
        auto_refresh=False,
        transient=False,
    ) as live:
        while True:
            key = _read_key()
            if key == "up":
                index = (index - 1) % len(options)
            elif key == "down":
                index = (index + 1) % len(options)
            elif key == "enter":
                break
            elif key in ("ctrl-c", "esc"):
                raise KeyboardInterrupt
            elif key.isdigit() and 1 <= int(key) <= len(options):
                index = int(key) - 1
            live.update(_render(title, options, index, console), refresh=True)
        done = Text(no_wrap=True, overflow="crop")
        done.append("◇ ", style=SLATE)
        done.append(title + "  ", style="bold")
        done.append(options[index][0], style=f"bold {ACCENT_L}")
        live.update(done, refresh=True)
    return index


def multiselect(
    console: Console,
    title: str,
    options: Sequence[Option],
    default: Sequence[int] = (),
) -> List[int]:
    """Tick any number of options with space. Returns the chosen indices."""
    if not options:
        raise ValueError("multiselect() needs at least one option")
    checked = set(default)
    index = 0

    if not _interactive(console):
        return _multiselect_numbered(console, title, options)

    with _keys(), Live(
        _render(title, options, index, console, checked),
        console=console,
        auto_refresh=False,
        transient=False,
    ) as live:
        while True:
            key = _read_key()
            if key == "up":
                index = (index - 1) % len(options)
            elif key == "down":
                index = (index + 1) % len(options)
            elif key == " ":
                checked ^= {index}
            elif key == "enter":
                break
            elif key in ("ctrl-c", "esc"):
                raise KeyboardInterrupt
            live.update(_render(title, options, index, console, checked), refresh=True)
        names = ", ".join(options[i][0] for i in sorted(checked)) or "nothing"
        done = Text(no_wrap=True, overflow="crop")
        done.append("◇ ", style=SLATE)
        done.append(title + "  ", style="bold")
        done.append(names, style=f"bold {ACCENT_L}")
        live.update(done, refresh=True)
    return sorted(checked)


def _select_numbered(
    console: Console, title: str, options: Sequence[Option], default: int
) -> int:
    """Plain fallback for pipes, CI and dumb terminals."""
    console.print(title)
    for i, (label, hint) in enumerate(options, 1):
        console.print(f"  {i}) {label}" + (f"  — {hint}" if hint else ""))
    while True:
        raw = input(f"Choose 1-{len(options)} [{default + 1}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        console.print("Please enter a number from the list.")


def _multiselect_numbered(
    console: Console, title: str, options: Sequence[Option]
) -> List[int]:
    console.print(title)
    for i, (label, hint) in enumerate(options, 1):
        console.print(f"  {i}) {label}" + (f"  — {hint}" if hint else ""))
    raw = input("Numbers, comma separated (blank = none): ").strip()
    picked = set()
    for part in raw.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(options):
            picked.add(int(part) - 1)
    return sorted(picked)