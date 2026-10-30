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
from typing import Optional

# `httpx` is imported lazily, inside the methods that make a request.
# It costs ~48 ms to import — more than this whole module — and a
# deployment that never captions an image (no VISION_ENABLED, no
# vision-capable document) should not pay for it at boot.

DEFAULT_MAX_TOKENS = 1000   # generous default for full transcriptions/standalone images


class OllamaVisionAdapter:
    """Local, offline vision captioning via Ollama (e.g. llava, qwen2-vl,
    minicpm-v). No API key, no external network call."""

    def __init__(self, model: str = "llava", base_url: str = "http://localhost:11434",
                 timeout_seconds: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def describe_image(self, image_bytes: bytes, mimetype: str,
                        context: Optional[str] = None,
                        max_tokens: Optional[int] = None) -> str:
        prompt = context or "Describe this image in detail, including any readable text."
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {"model": self.model, "prompt": prompt, "images": [b64], "stream": False}
        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}
        import httpx  # deferred; see the module header.

        resp = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

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
        return resp.json()["choices"][0]["message"]["content"].strip()

    def health(self) -> bool:
        try:
            import httpx  # deferred; see the module header.

            resp = httpx.get(f"{self.base_url}/models",
                              headers={"Authorization": f"Bearer {self.api_key}"}, timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False