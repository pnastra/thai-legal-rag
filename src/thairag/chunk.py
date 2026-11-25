"""Cut law sections into chunks (the pieces of text that each get one vector)."""

from collections.abc import Callable

import pandas as pd

COLUMNS = ["chunk_id", "section_id", "law_name", "section_num", "text"]


def citation(law_name: str, section_num: str) -> str:
    """Header for the later parts of a split section, e.g. 'ประมวลรัษฎากร มาตรา 40'."""
    if section_num[:1].isdigit():
        return f"{law_name} มาตรา {section_num}"
    return f"{law_name} {section_num}"  # stamp-duty schedule items are not มาตรา


def split_text(text: str, header: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Pack whole lines into parts of at most max_tokens; later parts start with the header.

    A single line that is too long on its own is cut at spaces (phrase boundaries in Thai).
    """
    if count(text) <= max_tokens:
        return [text]
    parts, current = [], ""

    def add(piece: str, sep: str) -> None:
        nonlocal current
        longer = f"{current}{sep}{piece}" if current else piece
        if current and count(longer) > max_tokens:
            parts.append(current)
            current = f"{header}\n{piece}"
        else:
            current = longer

    for line in (ln for ln in text.split("\n") if ln.strip()):
        if count(f"{header}\n{line}") <= max_tokens:
            add(line, "\n")
        else:
            for i, phrase in enumerate(line.split(" ")):
                add(phrase, "\n" if i == 0 else " ")
    parts.append(current)
    return parts


def make_chunks(
    corpus: pd.DataFrame, max_tokens: int | None = None, count: Callable[[str], int] = len
) -> pd.DataFrame:
    """One chunk per section (max_tokens=None), or split sections longer than max_tokens."""
    rows = []
    for s in corpus.itertuples():
        parts = [s.text]
        if max_tokens is not None:
            parts = split_text(s.text, citation(s.law_name, s.section_num), max_tokens, count)
        for i, part in enumerate(parts):
            rows.append((f"{s.section_id}#{i}", s.section_id, s.law_name, s.section_num, part))
    return pd.DataFrame(rows, columns=COLUMNS)
