# Contributing

Thanks for taking a look. This is a small library, so the bar is mostly
"does it work, and does the test prove it".

## Set up

```bash
git clone https://github.com/ItsMeYash777/atrium-rag
cd atrium-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
pytest tests/ -q          # the unit suite
ruff check atrium tests scripts
mypy atrium               # advisory: it reports, it does not fail
make build                # wheel + sdist, then twine check
```

`make build` must pass. `twine check` on both artifacts is the same
check CI runs, so it is worth running before you push.

## What a change should look like

- **A behaviour change comes with a test.** The suite is fast and needs no
  services: no Qdrant, no Ollama, no network. If your change needs one of
  those to be tested, it belongs behind a skip.
- **A user-visible change comes with a README edit.** The configuration
  table, the API table and the CLI table are all maintained by hand, and
  they are the first thing a reader checks.
- **Keep the parse path free of model calls.** The parser is pure Python
  and heuristic on purpose; it is the reason the product is fast and the
  reason it is MIT-licensed. A pull request that adds a model to the
  parse path needs a very good argument.
- **One thing per pull request.** A refactor and a behaviour change in the
  same diff are two pull requests.

## The parser

The parsing heuristics live in `atrium/universal_parser.py`, the chunking
strategies in `atrium/smart_chunker.py`. Both are the parts of the system
that most need care, because both are tuned against documents rather than
against tests: a change that looks right on one PDF is often wrong on the
next one. If you touch either, say which documents you checked it against.

## The benchmark

The evaluation harness, the corpus and the results live in a separate
repository, `atrium-benchmarks`, so that the numbers can be reproduced
independently of the library. A change to the parser or the chunker
invalidates the published numbers; say so in the pull request, and the
harness has to be re-run before they are re-published.
