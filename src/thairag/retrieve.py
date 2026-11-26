"""Retrievers: BM25 (words), dense (BGE-M3 vectors) and hybrid (reciprocal rank fusion).

Every retriever returns, per question, a ranked list of section IDs (best first).
"""

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from thairag.tokenize_th import clean, words

TOP_N = 100  # how many sections each retriever returns; enough for fusion and @10 metrics
RRF_K = 60  # standard constant from the RRF paper (Cormack et al., 2009); not tuned


def best_part_per_section(chunk_section_ids, scores: np.ndarray, top_n: int = TOP_N) -> list[str]:
    """Rank sections by their best chunk's score; each section appears once."""
    s = pd.Series(scores).groupby(np.asarray(chunk_section_ids)).max()
    s = s.sort_values(ascending=False, kind="stable")
    return s.index[:top_n].tolist()


class BM25:
    """Okapi BM25 over pythainlp words (k1=1.5, b=0.75: library defaults, not tuned)."""

    def __init__(self, chunks: pd.DataFrame):
        self.section_ids = chunks["section_id"].to_numpy()
        self.model = BM25Okapi([words(t) for t in chunks["text"]])

    def search(self, question: str, top_n: int = TOP_N) -> list[str]:
        scores = self.model.get_scores(words(question))
        keep = scores > 0  # no shared word: not a match at all, not a low-ranked match
        return best_part_per_section(self.section_ids[keep], scores[keep], top_n)


class Dense:
    """Cosine similarity between the question's BGE-M3 vector and every chunk vector."""

    def __init__(self, chunks: pd.DataFrame, vectors: np.ndarray, encode):
        self.section_ids = chunks["section_id"].to_numpy()
        self.vectors = vectors
        self.encode = encode  # list[str] -> unit-length vectors

    def search_many(self, questions: list[str], top_n: int = TOP_N) -> list[list[str]]:
        q = self.encode([clean(x) for x in questions])
        scores = q @ self.vectors.T  # vectors have length 1, so dot product = cosine
        return [best_part_per_section(self.section_ids, row, top_n) for row in scores]


def rrf(*rankings: list[str], k: int = RRF_K, top_n: int = TOP_N) -> list[str]:
    """Reciprocal rank fusion: score = sum over rankings of 1 / (k + rank)."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, section in enumerate(ranking, start=1):
            score[section] = score.get(section, 0.0) + 1 / (k + rank)
    return sorted(score, key=lambda s: -score[s])[:top_n]  # sorted() is stable: ties keep first-seen order
