"""Errors that carry a way out.

A missing service or an unset variable is a setup problem, not a bug,
and it must never reach the user as a stack trace. `AtriumSetupError`
carries the message and the ways out; the CLI prints it and exits
non-zero, and the API turns it into a clean startup failure.

The rule the whole codebase follows: if the user's next action is
obvious, the error says it.
"""
from __future__ import annotations

from typing import Iterable

__all__ = ["AtriumSetupError"]


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
