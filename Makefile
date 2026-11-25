.PHONY: data splits index lint test check

data:
	uv run python -m thairag.data

splits: data
	uv run python -m thairag.splits

index: data
	uv run python -m thairag.index

lint:
	uv run ruff check .

test:
	uv run pytest -q

check: lint test
