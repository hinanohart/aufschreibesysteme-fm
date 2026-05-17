.PHONY: help install dev test lint format audit-licenses fetch teaser space-deploy clean

PYTHON ?= python3

help:
	@echo "Common targets:"
	@echo "  install         pip install -e .[dev]"
	@echo "  test            pytest -ra"
	@echo "  lint            ruff + mypy"
	@echo "  format          ruff format"
	@echo "  audit-licenses  pip-licenses --fail-on GPL,AGPL"
	@echo "  audit-audio     scripts/audit_audio.py"
	@echo "  fetch           fetch all regime measurements"
	@echo "  teaser          render the 7-regime teaser image"
	@echo "  clean           remove caches / __pycache__"

install:
	$(PYTHON) -m pip install -e ".[dev]"

dev: install
	$(PYTHON) -m pip install -e ".[dev,paper]"

test:
	$(PYTHON) -m pytest -ra tests

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy afm

format:
	$(PYTHON) -m ruff format .

audit-licenses:
	$(PYTHON) -m pip_licenses --fail-on=GPL,AGPL

audit-audio:
	$(PYTHON) scripts/audit_audio.py

fetch:
	$(PYTHON) scripts/fetch_measurements.py --regime all

teaser:
	$(PYTHON) space/teaser_gen.py --config configs/mvp.yaml

space-deploy:
	afm space-deploy --config configs/mvp.yaml

clean:
	find . -type d \( -name __pycache__ -o -name '.pytest_cache' -o -name '.ruff_cache' -o -name '.mypy_cache' \) -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
