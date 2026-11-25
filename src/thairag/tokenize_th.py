"""Clean Thai text and split it into words (Thai writes no spaces between words)."""

import re

from pythainlp.tokenize import word_tokenize

INVISIBLE = re.compile("[\u200b‌‍﻿]")  # zero-width characters


def clean(text: str) -> str:
    """Remove invisible characters and collapse runs of whitespace."""
    return " ".join(INVISIBLE.sub("", text).split())


def words(text: str) -> list[str]:
    """Split into words with pythainlp's dictionary tokeniser; whitespace is dropped."""
    return [w for w in word_tokenize(clean(text).lower(), engine="newmm", keep_whitespace=False) if w.strip()]
