import pandas as pd
import pytest

from thairag.splits import (
    allocate,
    dev_pool,
    eligible_test,
    load_split,
    stratified_sample,
)


@pytest.fixture(scope="module")
def dev():
    return load_split("dev")


@pytest.fixture(scope="module")
def held_out():
    return load_split("test")


def test_no_question_or_section_in_both_dev_and_test(dev, held_out):
    assert set(dev["question_id"]).isdisjoint(held_out["question_id"])
    assert set(dev["section_id"]).isdisjoint(held_out["section_id"])


def test_split_sizes_and_sources(dev, held_out):
    assert (len(dev), dev["llm_subset"].sum()) == (500, 50)
    assert (len(held_out), held_out["llm_subset"].sum()) == (1000, 300)
    assert dev["question_id"].is_unique and held_out["question_id"].is_unique
    assert dev["question_id"].str.startswith("train-").all()
    assert held_out["question_id"].str.startswith("test-").all()


def test_dev_has_the_same_law_mix_as_test(dev, held_out):
    mix = pd.concat([d["law_name"].value_counts(normalize=True) for d in (dev, held_out)], axis=1).fillna(0)
    assert (mix.iloc[:, 0] - mix.iloc[:, 1]).abs().max() < 0.005


def test_allocate_sums_to_n_and_is_proportional():
    assert allocate(pd.Series({"a": 60, "b": 30, "c": 10}), 10).to_dict() == {"a": 6, "b": 3, "c": 1}
    assert allocate(pd.Series({"a": 1, "b": 1, "c": 1}), 10).sum() == 10


def test_stratified_sample_refuses_a_law_with_too_few_questions():
    pool = pd.DataFrame({"question_id": ["q1"], "law_name": ["a"]})
    with pytest.raises(ValueError, match="pool has 1"):
        stratified_sample(pool, pd.Series({"a": 5}), 2)


def questions(prefix, rows):
    df = pd.DataFrame(rows, columns=["question", "positive_ids"])
    df["question_id"] = [f"{prefix}-{i:05d}" for i in range(len(df))]
    return df


def test_eligible_test_drops_seen_duplicated_and_unofficial_questions():
    test = questions("test", [("seen", ["s0"]), ("good", ["s1"]), ("twice", ["s2"]), ("tw ice", ["s3"]), ("extra", ["s4"])])
    kept = eligible_test(test, pd.Series(["seen", "good", "twice"]))
    assert kept["question"].tolist() == ["good"]


def test_dev_pool_drops_multi_label_repeated_reworded_and_test_section_questions():
    test = questions("test", [("ผู้ซื้อต้องชำระราคาภายในกำหนดเวลาใด", ["t1"])])
    train = questions(
        "train",
        [
            ("ใครเป็นผู้มีอำนาจแต่งตั้งคณะกรรมการ", ["a"]),
            ("บริษัทต้องจัดทำบัญชีอย่างไร", ["b", "c"]),
            ("โทษปรับเท่าใด", ["d"]),
            ("โทษ ปรับเท่าใด", ["e"]),
            ("ผู้ซื้อต้องชำระราคาภายในกำหนดเวลาใดบ้าง", ["f"]),
            ("สัญญาเช่าสิ้นสุดเมื่อไร", ["t1"]),
            ("ไม่มีมาตราที่เกี่ยวข้อง", []),
        ],
    )
    assert dev_pool(train, test, test)["question_id"].tolist() == ["train-00000"]
