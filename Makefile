.PHONY: setup chat ingest eval test lint api mcp clean

setup:
	./scripts/setup.sh

chat:
	arion chat

ingest:
	arion ingest --path data/docs

eval:
	arion eval

test:
	pytest -v

lint:
	ruff check src tests
	ruff format --check src tests

fix:
	ruff check --fix src tests
	ruff format src tests

api:
	uvicorn arion.api:app --reload --port 8000

mcp:
	python -m arion.mcp.server

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
