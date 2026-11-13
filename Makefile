# Development tasks for atrium.
#
#   make install     install the package + dev extras
#   make test        run the unit suite
#   make lint        ruff
#   make typecheck   mypy (advisory — see pyproject.toml)
#   make build       build the wheel + sdist
#   make shot        regenerate the README benchmark image
#   make docker      build the container image

PYTHON ?= python3
VENV   ?= venv
PIP    := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF   := $(VENV)/bin/ruff
MYPY   := $(VENV)/bin/mypy

.PHONY: install test lint format typecheck build shot smoke docker clean

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) tests/ -q

lint:
	$(RUFF) check atrium tests scripts

format:
	$(RUFF) check atrium tests scripts --fix

typecheck:
	$(MYPY) atrium

build:
	$(VENV)/bin/python -m build
	$(VENV)/bin/twine check dist/*

# Install the built wheel into a throwaway environment and run the
# smoke test against it — the artifact a user gets, not the source
# tree. This is what CI's "Install the built artifact" job does.
smoke:
	$(VENV)/bin/python -m build --wheel
	rm -rf /tmp/atrium-smoke
	$(PYTHON) -m venv /tmp/atrium-smoke
	/tmp/atrium-smoke/bin/pip install --quiet dist/*.whl
	cd /tmp && /tmp/atrium-smoke/bin/python $(CURDIR)/scripts/smoke_test.py

# Regenerate docs/images/benchmark.png from the published run's raw
# JSON. The README image must stay in step with the data behind it.

docker:
	docker build -t atrium-rag .

clean:
	rm -rf build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
