"""
atrium/healthcheck.py

Container health probe: `python -m atrium.healthcheck`.

Answers one question — is the API process serving? — without pulling
in the whole application. It talks to `/health` over HTTP and reports:

* exit 0 — the API answered. A 503 counts: that status means "the API
  is up, the vector DB is not", and restarting the API would not fix
  that. Only a refused connection means the process itself is gone.
* exit 1 — nothing answered, or the URL is unreachable.

Kept as a module (rather than an inline `python -c` in the Dockerfile)
so the same probe works locally, in Compose, and in Kubernetes, and so
it stays readable and testable.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000/health"
TIMEOUT_SECONDS = 3.0


def check(url: str = DEFAULT_URL, timeout: float = TIMEOUT_SECONDS) -> bool:
    """Return True when the server answers at all."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except urllib.error.HTTPError:
        # The server answered — with a status code we don't like. It is
        # alive; something behind it is not.
        return True
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    url = args[0] if args else DEFAULT_URL
    ok = check(url)
    print("ok" if ok else "unreachable", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
