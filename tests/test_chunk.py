import pandas as pd
import pytest

from thairag.chunk import citation, make_chunks, split_text

CORPUS = pd.DataFrame(
    {
        "section_id": ["A-1", "A-2"],
        "law_name": ["กฎหมาย ก", "กฎหมาย ก"],
        "section_num": ["1", "2"],
        "text": ["กฎหมาย ก มาตรา 1 สั้น", "กฎหมาย ก มาตรา 2 นำ\n(1) หนึ่ง\n(2) สอง\n(3) สาม\n(4) สี่"],
    }
)


def test_one_chunk_per_section_keeps_text_unchanged():
    chunks = make_chunks(CORPUS)
    assert chunks["text"].tolist() == CORPUS["text"].tolist()
    assert chunks["section_id"].tolist() == ["A-1", "A-2"]


@pytest.mark.parametrize("max_tokens", [None, 40])
def test_every_chunk_has_metadata_and_text(max_tokens):
    chunks = make_chunks(CORPUS, max_tokens)
    for col in ("chunk_id", "section_id", "law_name", "section_num", "text"):
        assert chunks[col].str.strip().ne("").all()
    assert chunks["chunk_id"].is_unique
    assert set(chunks["section_id"]) == set(CORPUS["section_id"])


def test_split_respects_limit_repeats_header_and_loses_nothing():
    chunks = make_chunks(CORPUS, 40)
    parts = chunks.loc[chunks["section_id"] == "A-2", "text"].tolist()
    assert len(parts) > 1
    assert all(len(p) <= 40 for p in parts)
    assert all(p.startswith("กฎหมาย ก มาตรา 2") for p in parts)
    body = [parts[0]] + [p.split("\n", 1)[1] for p in parts[1:]]
    assert "\n".join(body) == CORPUS["text"][1]


def test_one_overlong_line_is_cut_at_spaces():
    parts = split_text("a " * 30 + "a", "H", 20, len)
    assert all(len(p) <= 20 for p in parts)
    assert sum(p.count("a") for p in parts) == 31


def test_stamp_duty_header_has_no_matra():
    assert citation("ประมวลรัษฎากร", "40 ทวิ") == "ประมวลรัษฎากร มาตรา 40 ทวิ"
    assert "มาตรา" not in citation("ประมวลรัษฎากร", "บัญชีอัตราอากรแสตมป์ 4")
