# Convenience entry points for slack-certify-mapf.
# All targets are PHONY; nothing here builds an actual file.

.PHONY: install install-dev fmt lint type test smoke repro docs clean figures tables help

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip

help:
	@echo "Available targets:"
	@echo "  install      Install the package (runtime deps only)"
	@echo "  install-dev  Install the package with dev/docs/viz/ilp extras"
	@echo "  fmt          Auto-format with black and ruff"
	@echo "  lint         Lint with ruff and check formatting"
	@echo "  type         Type-check with mypy --strict"
	@echo "  test         Run the pytest suite"
	@echo "  smoke        Run the reproducibility smoke test"
	@echo "  repro        Reproduce all paper figures and tables"
	@echo "  docs         Build the mkdocs documentation site"
	@echo "  figures      Regenerate paper figures from cached results"
	@echo "  tables       Regenerate paper tables from cached results"
	@echo "  clean        Remove build, cache, and coverage artifacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,docs,viz,ilp]" && pre-commit install

fmt:
	black . && ruff check --fix .

lint:
	ruff check . && ruff format --check . && black --check .

type:
	mypy --strict src/slackcertify

test:
	pytest

smoke:
	bash scripts/repro_smoke.sh

repro:
	bash scripts/repro_all.sh

docs:
	mkdocs build --strict

figures:
	$(PYTHON) -m slackcertify.analysis.figures

tables:
	$(PYTHON) -m slackcertify.analysis.tables

clean:
	rm -rf build dist *.egg-info src/*.egg-info \
	       .coverage htmlcov coverage.xml \
	       .mypy_cache .ruff_cache .pytest_cache \
	       docs/_build site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
