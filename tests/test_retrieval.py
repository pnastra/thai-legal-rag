import numpy as np
import pandas as pd

from thairag.eval_retrieval import bootstrap, metrics, paired, rank_of
from thairag.retrieve import BM25, Dense, best_part_per_section, rrf


def test_worked_example_from_the_lesson():
    ranks = np.array([1, 3, np.inf])  # found first, found third, not found
    m = metrics(ranks)
    assert m["hit@1"] == 1 / 3 and m["hit@5"] == 2 / 3
    assert np.isclose(m["mrr@10"], (1 + 1 / 3 + 0) / 3)


def test_mrr_at_k_ignores_ranks_beyond_k():
    assert metrics(np.array([7.0]))["mrr@5"] == 0 and np.isclose(metrics(np.array([7.0]))["mrr@10"], 1 / 7)


def test_rank_of():
    assert rank_of("b", ["a", "b"]) == 2 and rank_of("z", ["a", "b"]) == np.inf


def test_best_part_per_section_lists_each_section_once():
    assert best_part_per_section(["A", "B", "A"], np.array([0.2, 0.5, 0.9])) == ["A", "B"]


def test_rrf_rewards_agreement():
    # "c" is 2nd in both lists, so it beats "a" and "b", which are 1st in one list and missing from the other
    assert rrf(["a", "c"], ["b", "c"])[0] == "c"


def test_bm25_finds_the_section_sharing_rare_words_and_skips_no_overlap():
    chunks = pd.DataFrame(
        {
            "section_id": ["s1", "s2", "s3"],
            "text": ["ผู้ค้ำประกันต้องชำระหนี้", "บริษัทต้องจัดทำบัญชี", "สมาคมการค้าต้องจดทะเบียน"],
        }
    )
    result = BM25(chunks).search("ผู้ค้ำประกันชำระหนี้เมื่อใด")
    assert result[0] == "s1" and "s3" not in result


def test_dense_ranks_by_cosine():
    chunks = pd.DataFrame({"section_id": ["x", "y"]})
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    dense = Dense(chunks, vectors, lambda qs: np.array([[0.6, 0.8] for _ in qs]))
    assert dense.search_many(["anything"]) == [["y", "x"]]


def test_paired_difference_of_identical_retrievers_is_zero():
    r = np.array([1, 2, np.inf, 4])
    assert paired(r, r)["hit@5"] == (0.0, 0.0, 0.0)
    lo, hi = bootstrap(np.array([1.0, 0.0, 1.0, 1.0]))
    assert 0 <= lo <= 0.75 <= hi <= 1
