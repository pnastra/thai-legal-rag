"""Embed the corpus once and save the vectors and metadata to data/index/<name>/."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from thairag.chunk import make_chunks
from thairag.data import load_corpus

INDEX = Path("data/index")
MODEL = "BAAI/bge-m3"
MODEL_SHA = "5617a9f61b028005a4858fdac845db406aefb181"
VARIANTS = {"section": None, "split512": 512}  # name -> max tokens per chunk


def load_model():
    from sentence_transformers import (
        SentenceTransformer,  # slow import, only when needed
    )

    return SentenceTransformer(MODEL, revision=MODEL_SHA)


def embed_unique(texts: list[str], encode) -> np.ndarray:
    """Encode each distinct text once, then return one row per input text."""
    unique = list(dict.fromkeys(texts))
    vectors = encode(unique)
    row = {t: i for i, t in enumerate(unique)}
    return vectors[[row[t] for t in texts]]


def save_index(name: str, chunks: pd.DataFrame, vectors: np.ndarray, meta: dict, out: Path = INDEX) -> None:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "embeddings.npy", vectors.astype(np.float32))
    chunks.to_csv(folder / "chunks.csv", index=False)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def load_index(name: str, out: Path = INDEX) -> tuple[pd.DataFrame, np.ndarray]:
    chunks = pd.read_csv(out / name / "chunks.csv", dtype=str, keep_default_na=False)
    return chunks, np.load(out / name / "embeddings.npy")


def build() -> None:
    model = load_model()

    def count(text: str) -> int:
        return len(model.tokenizer(text)["input_ids"])

    corpus = load_corpus()
    variants = {name: make_chunks(corpus, n, count) for name, n in VARIANTS.items()}
    texts = [t for c in variants.values() for t in c["text"]]

    start = time.perf_counter()
    vectors = embed_unique(
        texts, lambda t: model.encode(t, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
    )
    seconds = round(time.perf_counter() - start, 1)

    at = 0
    for name, chunks in variants.items():
        part = vectors[at : at + len(chunks)]
        at += len(chunks)
        meta = {"model": MODEL, "revision": MODEL_SHA, "max_tokens": VARIANTS[name], "chunks": len(chunks)}
        save_index(name, chunks, part, meta)
        print(f"{name}: {len(chunks)} chunks, {part.nbytes / 1e6:.1f} MB of vectors")
    print(f"embedded {len(set(texts))} distinct texts in {seconds} s on {model.device}")


if __name__ == "__main__":
    build()
