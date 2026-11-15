# Contributing

Thanks for taking a look. This is a small library, so the bar is mostly
"does it work, and does the test prove it".

## Set up

```bash
git clone https://github.com/yash-0503/atrium-rag
cd atrium-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

If you do not have write access, fork the repository first and clone your
fork instead, then open the pull request from a branch on your fork.

## Before you open a pull request

Run the local CI script from the repository root:

```bash
scripts/ci_local.sh
scripts/ci_local.sh --with-docker #for container image
```

It runs the same checks CI runs, in order:

```bash
pytest tests/ -q          # the unit suite
ruff check atrium tests scripts
mypy atrium               # advisory: it reports, it does not fail
make build                # wheel + sdist, then twine check
```

`pytest`, `ruff` and `make build` must pass. `mypy` is advisory: it reports,
it does not fail. `twine check` on both artifacts is the same check CI runs,
so it is worth running before you push.

### Proof for your pull request

Every pull request must include a **screenshot of the terminal after
`scripts/ci_local.sh` finishes**. Paste it into the pull request description.

The screenshot must show:

1. The command you ran (`scripts/ci_local.sh`).
2. The final summary lines of every step: the `pytest` summary
   (for example `N passed in X.XXs`), the `ruff` result, and the
   `make build` / `twine check` result.
3. No cut-off output. If the output is longer than one screen, attach more
   than one screenshot, or also paste the text in a collapsed `<details>`
   block.

A pull request without this screenshot will not be reviewed. A passing run
on your machine is the minimum evidence that the change works.

For a bug fix, also show that the new test **fails without your change**:
stash the fix, run the test, and paste that failing output too. That is
what "does the test prove it" means.

## What a change should look like

* A behaviour change comes with a test. The suite is fast and needs no
  services: no Qdrant, no Ollama, no network. If your change needs one of
  those to be tested, it belongs behind a skip.
* A user-visible change comes with a README edit. The configuration
  table, the API table and the CLI table are all maintained by hand, and
  they are the first thing a reader checks. A new environment variable
  goes in the configuration table.
* Keep the parse path free of model calls. The parser is pure Python
  and heuristic on purpose; it is the reason the product is fast and the
  reason it is MIT-licensed. A pull request that adds a model to the
  parse path needs a very good argument.
* One thing per pull request. A refactor and a behaviour change in the
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