# Edible project — convenience targets
# Run all commands via: make <target>

.PHONY: test lint fmt check install install-dev scrape clean

# PYTHONPATH must be cleared to avoid Python 3.9 system package pollution
PYTHON_ENV = unset PYTHONPATH && export PATH="$(HOME)/.local/bin:$(PATH)"

install:
	$(PYTHON_ENV) && uv sync

install-dev:
	$(PYTHON_ENV) && uv sync --extra dev

test:
	$(PYTHON_ENV) && uv run pytest

test-v:
	$(PYTHON_ENV) && uv run pytest -v

lint:
	$(PYTHON_ENV) && uv run ruff check src/ tests/

fmt:
	$(PYTHON_ENV) && uv run ruff format src/ tests/

scrape:
	$(PYTHON_ENV) && uv run python scripts/scrape.py $(ARGS)

check: lint test

clean:
	rm -rf .venv __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
