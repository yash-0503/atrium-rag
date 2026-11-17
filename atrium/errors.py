"""Errors that carry a way out.

A missing service or an unset variable is a setup problem, not a bug,
and it must never reach the user as a stack trace. `AtriumSetupError`
carries the message and the ways out; the CLI prints it and exits
non-zero, and the API turns it into a clean startup failure.

The rule the whole codebase follows: if the user's next action is
obvious, the error says it.
"""
from __future__ import annotations

from typing import Dict, Iterable

__all__ = ["AtriumSetupError", "NoReadableContentError"]


class AtriumSetupError(Exception):
    """Something is not set up, and here is how to set it up.

    `hints` are the ways out — usually two: start the thing, or use the
    alternative that needs nothing.
    """

    def __init__(self, message: str, hints: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.hints = tuple(h for h in hints if h)

    def __str__(self) -> str:
        return self.message

    def render(self) -> str:
        """The message and its ways out, as one block of text."""
        if not self.hints:
            return self.message
        lines = [self.message, ""]
        lines.extend(f"  {hint}" for hint in self.hints)
        return "\n".join(lines)


class NoReadableContentError(ValueError):
    """The document parsed, and none of it was readable.

    A scanned PDF with no OCR and no vision configured is the usual
    cause: every page is a picture, nothing can read it, and the
    document has no text at all. The ingest fails rather than
    reporting a clean success with nothing in it — a document that
    produced zero chunks must not look like an ingested document.

    It is a `ValueError` because that is what the API maps to 422,
    "the document could not be parsed". It is *not* an
    `AtriumSetupError`, which maps to 503: the service is fine, the
    document is an unreadable scan, and telling the client to retry
    would be advice that never helps.

    The counts travel on the exception so the job row can record why
    the document was empty without reading the parser's state again
    — the parser is shared, and its state belongs to whichever parse
    ran last.
    """

    def __init__(
        self,
        message: str,
        skipped_elements: int = 0,
        skipped_reasons: Dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.skipped_elements = int(skipped_elements)
        self.skipped_reasons: Dict[str, int] = dict(skipped_reasons or {})
