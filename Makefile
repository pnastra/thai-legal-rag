.PHONY: data splits lint test check

data:
	uv run python -m thairag.data

splits: data
	uv run python -m thairag.splits

lint:
	uv run ruff check .

test:
	uv run pytest -q

check: lint test
