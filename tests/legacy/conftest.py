"""Ignore everything in this directory.

`tests/legacy/` holds the orphaned token-based chunker generation — see
the README here. The tests describe an API that was never committed, so
collecting them would fail the suite for reasons that have nothing to do
with the shipped code.

`collect_ignore_glob` is directory-scoped: pytest applies it to the
files in this directory only.
"""

collect_ignore_glob = ["*"]
