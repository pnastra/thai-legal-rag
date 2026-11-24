import json

import pandas as pd

from thairag.data import load_corpus, load_questions, make_section_id


def test_section_id_matches_label_format():
    assert make_section_id("ป0003-1D-0002", "456") == "ป0003-1D-0002-456"
    assert make_section_id("ป0008-1D-0001", " 8 ทวิ ") == "ป0008-1D-0001-8 ทวิ"


def test_stamp_duty_items_get_one_id_whichever_source_named_them():
    in_law_file = make_section_id("ป0008-1D-0001", "บัญชีอัตราอากรแสตมป์ 4")
    in_labels = make_section_id("ป0008-1D-0001", "ลักษณะแห่งตราสาร 4")
    assert in_law_file == in_labels == "ป0008-1D-0001-s4"


def test_labels_join_to_corpus(tmp_path):
    (tmp_path / "nitibench_laws").mkdir()
    (tmp_path / "wangchanx").mkdir()
    law = [{"law_name": "กฎหมาย ก", "section_num": "1", "section_content": "ข้อความ", "reference": []}]
    (tmp_path / "nitibench_laws" / "A-1.json").write_text(json.dumps(law, ensure_ascii=False))
    ctx = {"context": "ข้อความ", "metadata": {"law_code": "A-1", "law_title": "กฎหมาย ก", "section": "1"}}
    pd.DataFrame(
        {
            "question": ["ถาม 1", "ถาม 2"],
            "positive_contexts": [[ctx], []],
            "hard_negative_contexts": [[], []],
            "positive_answer": ["ตอบ", "ตอบ"],
        }
    ).to_parquet(tmp_path / "wangchanx" / "train.parquet")

    corpus, q = load_corpus(tmp_path), load_questions("train", tmp_path)

    assert q["question_id"].tolist() == ["train-00000", "train-00001"]
    assert set(q["positive_ids"][0]) <= set(corpus["section_id"])
    assert q["positive_ids"][1] == [] and pd.isna(q["law_name"][1])
