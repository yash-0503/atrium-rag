#!/usr/bin/env python3
"""
record_visualizer_gif.py

Regenerate the GIF the README shows: the chunk visualiser, parsing a
real document.

    # 1. a server, on a port of your choosing
    atrium serve --port 8077

    # 2. the recording
    python scripts/record_visualizer_gif.py \
        --document ../atrium-benchmarks/tests/benchmarks/fixtures/24_nist_incident_handling/source.pdf \
        --out docs/images/visualize.gif

    # 3. look at it
    open docs/images/visualize.gif

Why this exists: the clip is the parser's own output, on a real
document, and a clip that nobody can reproduce is a picture of
somebody's parser. This script drives a real Chrome over the DevTools
protocol, drops the document into the real `/visualize` page, and
records what the page shows. There is no mock and no re-implementation.

The document is a NIST special publication because it parses cleanly
— two-column papers do not, and a hero clip should not display the
known weak spot.

Needs `pip install pillow websockets` (both are in the dev extra) and
a Chrome. If Chrome is installed in the usual places it is found
automatically; otherwise pass --chrome /path/to/chrome.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]


def find_chrome(explicit: str | None) -> str:
    """The Chrome to drive. An explicit path wins; then the usual
    installs; then the puppeteer cache, which is where a machine that
    has run a headless browser before keeps its build."""
    if explicit:
        if not Path(explicit).exists():
            raise SystemExit(f"no Chrome at {explicit}")
        return explicit
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    cache = os.path.expanduser(
        "~/.cache/puppeteer/chrome/*/chrome-mac*/Google Chrome for Testing.app/"
        "Contents/MacOS/Google Chrome for Testing"
    )
    found = sorted(glob.glob(cache))
    if found:
        return found[-1]
    raise SystemExit(
        "no Chrome found. Install one, or pass --chrome /path/to/chrome."
    )


async def record(
    chrome: str,
    url: str,
    document: str,
    seconds: float,
    fps: int,
    width: int,
    port: int,
) -> tuple[list[tuple[float, bytes]], float]:
    """Drive the page and collect (seconds, jpeg) frames."""
    import websockets

    profile = f"/tmp/atrium-gif-profile-{port}"
    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            f"--remote-debugging-port={port}",
            "--window-size=1600,1000",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--no-first-run",
            "--disable-gpu",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    target = None
    for _ in range(100):
        time.sleep(0.2)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/list", timeout=2
            ) as response:
                targets = json.load(response)
        except Exception:
            continue
        pages = [t for t in targets if t.get("type") == "page"]
        if pages:
            target = pages[0]
            break
    if not target:
        proc.terminate()
        raise SystemExit("Chrome did not start")

    frames: list[tuple[float, bytes]] = []

    try:
        async with websockets.connect(
            target["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024
        ) as ws:
            replies: dict[int, dict] = {}
            state = {"filled_at": None}

            async def reader() -> None:
                """One reader for the socket: frames go to `frames`,
                command replies to `replies`."""
                while True:
                    data = json.loads(await ws.recv())
                    if data.get("method") == "Page.screencastFrame":
                        frames.append(
                            (
                                time.time(),
                                base64.b64decode(data["params"]["data"]),
                            )
                        )
                        await ws.send(
                            json.dumps(
                                {
                                    "id": 0,
                                    "method": "Page.screencastFrameAck",
                                    "params": {
                                        "sessionId": data["params"]["sessionId"]
                                    },
                                }
                            )
                        )
                        continue
                    if "id" in data:
                        replies[data["id"]] = data

            mid = 0

            async def call(method: str, params: dict | None = None) -> dict:
                nonlocal mid
                mid += 1
                mine = mid
                await ws.send(
                    json.dumps(
                        {"id": mine, "method": method, "params": params or {}}
                    )
                )
                while mine not in replies:
                    await asyncio.sleep(0.01)
                return replies.pop(mine)

            reader_task = asyncio.create_task(reader())
            await call("Page.enable")
            await call("DOM.enable")
            await call("Runtime.enable")
            await call("Page.navigate", {"url": url})
            await asyncio.sleep(6)  # the Tailwind CDN

            # Start recording before the file is chosen, so the parse
            # is part of the clip.
            await call(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": 88,
                    "maxWidth": 1280,
                    "maxHeight": 800,
                    "everyNthFrame": 1,
                },
            )
            await asyncio.sleep(1.5)

            doc = await call("DOM.getDocument", {"depth": 0})
            node = await call(
                "DOM.querySelector",
                {
                    "nodeId": doc["result"]["root"]["nodeId"],
                    "selector": "#fileInput",
                },
            )
            await call(
                "DOM.setFileInputFiles",
                {"files": [document], "nodeId": node["result"]["nodeId"]},
            )
            await call(
                "Runtime.evaluate",
                {
                    "expression": (
                        "document.getElementById('fileInput').dispatchEvent("
                        "new Event('change',{bubbles:true}));'x'"
                    ),
                    "awaitPromise": True,
                },
            )

            # Wait for the parse to land before the paging starts.
            for _ in range(240):
                await asyncio.sleep(0.25)
                result = await call(
                    "Runtime.evaluate",
                    {
                        "expression": (
                            "document.querySelectorAll('#grid > div').length"
                        ),
                        "returnByValue": True,
                    },
                )
                if result["result"]["result"].get("value", 0):
                    state["filled_at"] = time.time()
                    break
            if state["filled_at"] is None:
                raise SystemExit(
                    "the visualiser never rendered a chunk — is the server "
                    "running, and is the document one it can parse?"
                )
            await asyncio.sleep(1.6)

            # Page into the body of the document: the table of contents
            # is not the best advertisement for a chunker.
            y = 2400
            for _ in range(6):
                await call(
                    "Runtime.evaluate",
                    {
                        "expression": f"window.scrollTo(0, {y}); 'y'",
                        "returnByValue": True,
                    },
                )
                y += 1500
                await asyncio.sleep(1.3)

            await call("Page.stopScreencast")
            reader_task.cancel()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not frames:
        raise SystemExit("no frames were recorded")
    return frames, state["filled_at"] or frames[-1][0]


def build_gif(
    frames: list[tuple[float, bytes]],
    filled_at: float,
    out: Path,
    width: int,
    fps: int,
    seconds: float,
) -> int:
    """Assemble the GIF.

    Three parts: the empty page before the file is chosen, the moment
    the grid lands, and everything after it. The spinner frames in
    between are dropped — they are the least interesting thing on the
    screen, and there are a lot of them."""
    from PIL import Image

    t0 = frames[0][0]
    selected = [(stamp - t0, data) for stamp, data in frames if stamp - t0 <= 1.6]
    transition = filled_at - t0
    # One frame of the parse, and the frame the grid lands on. A whole
    # second of spinner is a second of nothing happening.
    before = [f for f in frames if f[0] - t0 <= transition]
    # The screencast lags the DOM by a frame or two, so the first
    # captured frame after the parse is still the spinner. Wait for
    # the compositor to catch up before the grid is shown.
    after = [f for f in frames if f[0] - t0 > transition + 0.4]
    if before:
        selected.append((before[-1][0] - t0, before[-1][1]))
    selected += [(stamp - t0, data) for stamp, data in after]
    selected = [(t, d) for t, d in selected if t <= seconds + 6]
    if not selected:
        selected = [(stamp - t0, data) for stamp, data in frames]

    # Resample to the target rate: the screencast runs faster than the
    # clip should, and the spinner alone would fill it with frames.
    minimum_gap = 1.0 / max(fps, 1)
    thinned: list[tuple[float, bytes]] = []
    last = -1e9
    for stamp, data in selected:
        if stamp - last >= minimum_gap:
            thinned.append((stamp, data))
            last = stamp
    selected = thinned

    images = [Image.open(io.BytesIO(data)).convert("RGB") for _, data in selected]
    palette = images[0].quantize(
        colors=128, method=Image.MEDIANCUT, dither=Image.Dither.NONE
    )
    frames_out = [
        image.resize(
            (width, round(image.height * width / image.width)), Image.LANCZOS
        ).quantize(palette=palette, dither=Image.Dither.NONE)
        for image in images
    ]
    durations = []
    for index in range(len(selected)):
        if index + 1 < len(selected):
            gap = int((selected[index + 1][0] - selected[index][0]) * 1000)
        else:
            gap = 1200
        durations.append(max(120, min(gap, 1600)))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_out[0].save(
        out,
        save_all=True,
        append_images=frames_out[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return len(frames_out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", required=True, help="the PDF to show")
    parser.add_argument("--out", required=True, help="the GIF to write")
    parser.add_argument("--url", default="http://127.0.0.1:8077/visualize")
    parser.add_argument("--chrome", default=None)
    parser.add_argument("--seconds", type=float, default=26.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--port", type=int, default=9224)
    args = parser.parse_args()

    chrome = find_chrome(args.chrome)
    document = str(Path(args.document).expanduser().resolve())
    if not Path(document).exists():
        raise SystemExit(f"no document at {document}")

    frames, filled_at = asyncio.run(
        record(
            chrome=chrome,
            url=args.url,
            document=document,
            seconds=args.seconds,
            fps=args.fps,
            width=args.width,
            port=args.port,
        )
    )
    count = build_gif(
        frames,
        filled_at,
        Path(args.out).expanduser(),
        args.width,
        args.fps,
        args.seconds,
    )
    size = Path(args.out).expanduser().stat().st_size // 1024
    print(f"wrote {args.out}: {count} frames, {size} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
