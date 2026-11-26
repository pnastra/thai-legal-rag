.PHONY: data splits index eval-retrieval lint test check

data:
	uv run python -m thairag.data

splits: data
	uv run python -m thairag.splits

index: data
	uv run python -m thairag.index

data/index/section/embeddings.npy:
	$(MAKE) index

eval-retrieval: data/index/section/embeddings.npy
	uv run python -m thairag.eval_retrieval

lint:
	uv run ruff check .

test:
	uv run pytest -q

check: lint test
