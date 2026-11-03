.PHONY: lint test check

lint:
	uv run ruff check .

test:
	uv run pytest -q

check: lint test
