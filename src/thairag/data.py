"""Download the benchmark (pinned versions), build the corpus and load the questions."""

import json
import re
from pathlib import Path
from urllib.parse import quote

import httpx
import pandas as pd

RAW = Path("data/raw")

# Pinned so that row numbers (our question IDs) and section texts never change under us.
WANGCHANX_SHA = "654e1a10afc44ecaee36fd739bc90078dcc62388"
NITIBENCH_SHA = "f39657603113ad77f7f8894996ff33d407a5fd01"
WANGCHANX_URL = (
    "https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG"
    f"/resolve/{WANGCHANX_SHA}/data/{{split}}-00000-of-00001.parquet"
)
CCL_SHA = "9f75697075e0f5691e3b821037b908c05f1821a7"
CCL_URL = (
    "https://huggingface.co/datasets/VISAI-AI/nitibench"
    f"/resolve/{CCL_SHA}/data/ccl-00000-of-00001.parquet"
)
LAW_TREE_URL = f"https://api.github.com/repos/vistec-ai/nitibench/git/trees/{NITIBENCH_SHA}?recursive=1"
LAW_RAW_URL = f"https://raw.githubusercontent.com/vistec-ai/nitibench/{NITIBENCH_SHA}/"

STAMP_DUTY = re.compile(r"^(?:บัญชีอัตราอากรแสตมป์|ลักษณะแห่งตราสาร)\s*(\d+)$")


def _download(client: httpx.Client, url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    r = client.get(url)
    r.raise_for_status()
    # Write then rename, so an interrupted download is never mistaken for a complete file.
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(r.content)
    tmp.rename(path)


def download(raw: Path = RAW) -> None:
    """Fetch WangchanX (train/test), the NitiBench-CCL test list and the law files. Skips files already on disk."""
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for split in ("train", "test"):
            _download(client, WANGCHANX_URL.format(split=split), raw / "wangchanx" / f"{split}.parquet")
        _download(client, CCL_URL, raw / "nitibench_ccl" / "ccl.parquet")
        tree = raw / "nitibench_tree.json"
        _download(client, LAW_TREE_URL, tree)  # saved, so later runs don't need GitHub's API
        for item in json.loads(tree.read_text())["tree"]:
            p = item["path"]
            if p.startswith("test_data/laws/") and p.endswith(".json"):
                _download(client, LAW_RAW_URL + quote(p), raw / "nitibench_laws" / Path(p).name)


def make_section_id(law_code: str, section_num: str) -> str:
    """Build the section ID shared by corpus and labels, e.g. 'ป0003-1D-0002-456'.

    We don't use the labels' own `unique_key`: it spells suffixes differently from the
    law files ('12_2' for '12 ทวิ'), so 4% of test labels would not join. Stamp-duty
    items have different names in the two sources, so both become 's<N>'.
    """
    section_num = " ".join(section_num.split())
    m = STAMP_DUTY.match(section_num)
    return f"{law_code}-s{m.group(1)}" if m else f"{law_code}-{section_num}"


def _context_ids(contexts) -> list[str]:
    return [make_section_id(c["metadata"]["law_code"], c["metadata"]["section"]) for c in contexts]


def load_corpus(raw: Path = RAW) -> pd.DataFrame:
    """One row per law section: section_id, law_code, law_name, section_num, text."""
    rows = []
    for path in sorted((raw / "nitibench_laws").glob("*.json")):
        for s in json.loads(path.read_text()):
            rows.append(
                {
                    "section_id": make_section_id(path.stem, s["section_num"]),
                    "law_code": path.stem,
                    "law_name": s["law_name"],
                    "section_num": s["section_num"].strip(),
                    "text": s["section_content"],
                }
            )
    return pd.DataFrame(rows)


def load_questions(split: str, raw: Path = RAW) -> pd.DataFrame:
    """One row per question; question_id is '<split>-<row number>' at the pinned revision."""
    df = pd.read_parquet(raw / "wangchanx" / f"{split}.parquet")
    return pd.DataFrame(
        {
            "question_id": [f"{split}-{i:05d}" for i in range(len(df))],
            "question": df["question"],
            "answer": df["positive_answer"],
            "positive_ids": [_context_ids(cs) for cs in df["positive_contexts"]],
            "hard_negative_ids": [_context_ids(cs) for cs in df["hard_negative_contexts"]],
            "law_name": [cs[0]["metadata"]["law_title"] if len(cs) else None for cs in df["positive_contexts"]],
        }
    )


if __name__ == "__main__":
    download()
    print(f"corpus: {len(load_corpus())} sections")
    for s in ("train", "test"):
        print(f"{s}: {len(load_questions(s))} questions")
