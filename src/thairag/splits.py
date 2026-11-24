"""Choose the fixed dev and test questions and save their IDs to data/splits/.

Test comes from the official test split; dev comes from train, filtered to look like test.
Both follow the law mix of the official test split. Decisions: PLAN.md §12 (2026-11-24).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from thairag.data import RAW, load_questions

OUT = Path("data/splits")
SEED = 42
N_DEV, N_TEST = 500, 1000  # retrieval metrics (free to run)
N_DEV_LLM, N_TEST_LLM = 50, 300  # subsets that also get LLM answers (slow or limited)
NEAR_DUP = 0.8  # character n-gram cosine above this = the same question reworded
SEEN_TEST_IDS = {"test-00000"}  # printed while exploring the raw data in M1


def normalise(questions: pd.Series) -> pd.Series:
    """Remove all whitespace, so spacing differences don't hide duplicate questions."""
    return questions.str.replace(r"\s+", "", regex=True)


def allocate(counts: pd.Series, n: int) -> pd.Series:
    """Share n slots in proportion to counts (largest-remainder rounding, sums to exactly n)."""
    exact = counts / counts.sum() * n
    slots = exact.astype(int)
    leftover = n - slots.sum()
    # Sort by name first so that ties are always broken the same way.
    order = (exact - slots).sort_index().sort_values(ascending=False, kind="stable").index
    slots[order[:leftover]] += 1
    return slots


def stratified_sample(pool: pd.DataFrame, counts: pd.Series, n: int, seed: int = SEED) -> pd.DataFrame:
    """Draw n rows from pool with the law mix given by counts (a count per law_name)."""
    parts = []
    for law, k in allocate(counts, n).items():
        rows = pool[pool["law_name"] == law]
        if len(rows) < k:
            raise ValueError(f"need {k} questions from {law}, pool has {len(rows)}")
        parts.append(rows.sample(k, random_state=seed))
    return pd.concat(parts).sort_values("question_id")


def max_similarity(a: pd.Series, b: pd.Series) -> pd.Series:
    """For each question in a, its highest character n-gram cosine similarity to any in b."""
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32).fit(pd.concat([a, b]))
    x, y = vec.transform(a), vec.transform(b)
    best = []
    for i in range(0, x.shape[0], 500):  # in blocks, to keep memory small
        best.extend((x[i : i + 500] @ y.T).max(axis=1).toarray().ravel())
    return pd.Series(best, index=a.index)


def eligible_test(test: pd.DataFrame, official_questions: pd.Series) -> pd.DataFrame:
    """Test questions in NitiBench-CCL that appear only once and that we haven't looked at."""
    key = normalise(test["question"])
    keep = (
        key.isin(set(normalise(official_questions)))
        & ~key.duplicated(keep=False)
        & ~test["question_id"].isin(SEEN_TEST_IDS)
    )
    return test[keep]


def dev_pool(train: pd.DataFrame, all_test: pd.DataFrame, test_sample: pd.DataFrame) -> pd.DataFrame:
    """Train questions shaped like test: one correct section, unique, not a reworded test
    question, and not about a section used by the test sample."""
    key = normalise(train["question"])
    test_sections = {s for ids in test_sample["positive_ids"] for s in ids}
    keep = (
        train["positive_ids"].map(len).eq(1)
        & ~key.duplicated(keep=False)
        & ~train["positive_ids"].map(lambda ids: bool(test_sections & set(ids)))
        & (max_similarity(key, normalise(all_test["question"])) <= NEAR_DUP)
    )
    return train[keep]


def make_splits(train: pd.DataFrame, test: pd.DataFrame, official_questions: pd.Series):
    """Return (dev, test) with columns question_id, section_id, law_name, llm_subset."""
    pool_test = eligible_test(test, official_questions)
    mix = pool_test["law_name"].value_counts()
    test_s = stratified_sample(pool_test, mix, N_TEST)
    dev_s = stratified_sample(dev_pool(train, test, test_s), mix, N_DEV)

    out = []
    for sample, n_llm in ((dev_s, N_DEV_LLM), (test_s, N_TEST_LLM)):
        llm = stratified_sample(sample, mix, n_llm)
        out.append(
            pd.DataFrame(
                {
                    "question_id": sample["question_id"],
                    "section_id": sample["positive_ids"].str[0],
                    "law_name": sample["law_name"],
                    "llm_subset": sample["question_id"].isin(llm["question_id"]),
                }
            )
        )
    return tuple(out)


def load_split(name: str, out: Path = OUT) -> pd.DataFrame:
    """Read the committed 'dev' or 'test' split."""
    return pd.read_csv(out / f"{name}.csv")


if __name__ == "__main__":
    official = pd.read_parquet(RAW / "nitibench_ccl" / "ccl.parquet")["question"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in zip(("dev", "test"), make_splits(load_questions("train"), load_questions("test"), official)):
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} questions, {df['llm_subset'].sum()} in the LLM subset, {df['law_name'].nunique()} laws")
