import numpy as np
import pandas as pd
import pytest

from thairag.chunk import make_chunks
from thairag.index import embed_unique, load_index, save_index
from thairag.tokenize_th import clean, words


def test_embed_unique_encodes_each_text_once_and_keeps_order():
    seen = []

    def fake(texts):
        seen.extend(texts)
        return np.array([[len(t)] for t in texts])

    assert embed_unique(["a", "bbb", "a"], fake).ravel().tolist() == [1, 3, 1]
    assert seen == ["a", "bbb"]


def test_index_round_trip(tmp_path):
    corpus = pd.DataFrame(
        {"section_id": ["A-1"], "law_name": ["ก"], "section_num": ["1"], "text": ["ก มาตรา 1\nข้อความ"]}
    )
    chunks = make_chunks(corpus)
    save_index("x", chunks, np.ones((1, 4)), {}, tmp_path)
    loaded, vectors = load_index("x", tmp_path)
    assert loaded.equals(chunks.astype(loaded.dtypes)) and vectors.shape == (1, 4)
    with pytest.raises(ValueError):
        save_index("x", chunks, np.ones((2, 4)), {}, tmp_path)


def test_thai_words_ignore_invisible_characters():
    assert clean("ณ \u200bที่ใด") == "ณ ที่ใด"
    assert words("ผู้ซื้อ\u200bชำระราคา") == words("ผู้ซื้อชำระราคา")
    assert len(words("ผู้ซื้อต้องชำระราคา")) >= 3
    assert all(w.strip() for w in words("ก  ข\nค"))
