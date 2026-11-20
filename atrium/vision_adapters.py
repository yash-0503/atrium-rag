"""
vision_adapters.py

Concrete VisionCaptionAdapter implementations. Any of these can be
passed to UniversalParser(vision_adapter=...) — the parser only ever
calls describe_image(image_bytes, mimetype, context, max_tokens), so
swapping providers never touches parsing or chunking logic.

max_tokens is optional and per-call: universal_parser.py passes a small
cap for embedded-diagram captions (which should stay to 2-3 sentences)
and leaves it unset for full-page scan transcriptions and standalone
images (where the "caption" is the actual document content and needs
room). This is a real generation-length limit at the model level, not
just a prompt asking nicely — small local vision models in particular
don't always reliably follow "be brief" instructions on their own.
"""

from __future__ import annotations

import base64
import re
from typing import Optional
import logging


DEFAULT_MAX_TOKENS = 1000   # generous default for full transcriptions/standalone images
log = logging.getLogger("ingest.vision")


class VisionError(RuntimeError):
    """The model did not produce a usable caption."""
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)
_OPEN_RE = re.compile(r"<think(?:ing)?>", re.IGNORECASE)

# Untagged reasoning openers. Used ONLY to log a warning, never to edit text.
_REASONING_HINT_RE = re.compile(
    r"^\s*(okay|ok,|let me|let's|first,|i need to|i should|hmm|so,|the user)",
    re.IGNORECASE,
)

_SYSTEM = ("You are an image captioning component. Output ONLY the final "
           "caption or transcription. No reasoning, no preamble, no commentary.")


def _clean_model_output(text: str) -> str:
    """Strip tagged reasoning. Raises instead of returning "" so a failure
    can never be mistaken for an empty-but-valid caption."""
    raw = (text or "").strip()
    text = _THINK_RE.sub("", raw)
    closes = list(_CLOSE_RE.finditer(text))
    if closes:
        text = text[closes[-1].end():]
    if _OPEN_RE.search(text):
        raise VisionError("output was only reasoning (truncated before the answer)")
    text = text.strip()
    if not text:
        raise VisionError("model returned an empty caption")
    if _REASONING_HINT_RE.match(text):
        # Can't be fixed by parsing; surface it so the trigger can be found.
        log.warning("vision output looks like untagged reasoning: %.200r", text)
    return text


class OllamaVisionAdapter:
    """Local vision captioning via Ollama's /api/chat."""

    def __init__(self, model: str = "llava", base_url: str = "http://localhost:11434",
                 timeout_seconds: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _chat(self, b64: str, prompt: str, num_predict: Optional[int]) -> dict:
        import httpx  # deferred; see the module header.

        options = {"temperature": 0}
        if num_predict:
            options["num_predict"] = num_predict
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "think": False,
                "options": options,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt, "images": [b64]},
                ],
            },
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()

    def describe_image(self, image_bytes: bytes, mimetype: str,
                       context: Optional[str] = None,
                       max_tokens: Optional[int] = None) -> str:
        prompt = context or "Describe this image in detail, including any readable text."
        b64 = base64.b64encode(image_bytes).decode("ascii")
        budget = max_tokens

        for attempt in (1, 2):
            data = self._chat(b64, prompt, budget)
            content = (data.get("message") or {}).get("content") or ""  # never "thinking"
            try:
                return _clean_model_output(content)
            except VisionError:
                # Truncated with a cap set: retry once with more room.
                if attempt == 1 and budget and data.get("done_reason") == "length":
                    budget *= 4
                    continue
                raise

    def health(self) -> bool:
        try:
            import httpx  # deferred; see the module header.

            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


class OpenAICompatibleVisionAdapter:
    """Works with OpenAI, Azure OpenAI, or any OpenAI-compatible vision
    endpoint (gpt-4o-mini, gemini via an OpenAI-compatible proxy, etc.)."""

    def __init__(self, model: str, api_key: str,
                 base_url: str = "https://api.openai.com/v1", timeout_seconds: float = 60.0,
                 default_max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_max_tokens = default_max_tokens

    def describe_image(self, image_bytes: bytes, mimetype: str,
                        context: Optional[str] = None,
                        max_tokens: Optional[int] = None) -> str:
        prompt = context or "Describe this image in detail, including any readable text."
        b64 = base64.b64encode(image_bytes).decode("ascii")
        import httpx  # deferred; see the module header.

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mimetype};base64,{b64}"}},
                    ],
                }],
                "max_tokens": max_tokens or self.default_max_tokens,
            },
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return _clean_model_output(resp.json()["choices"][0]["message"]["content"])

    def health(self) -> bool:
        try:
            import httpx  # deferred; see the module header.

            resp = httpx.get(f"{self.base_url}/models",
                              headers={"Authorization": f"Bearer {self.api_key}"}, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False