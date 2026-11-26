"""Evaluate retrievers on the dev split: hit@k, MRR@k, bootstrap intervals, paired comparisons."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from thairag.data import load_questions
from thairag.splits import load_split

KS = (1, 5, 10)
SEED = 42
N_BOOT = 2000
REPORTS = Path("reports")
CCC = "ประมวลกฎหมายแพ่งและพาณิชย์"
TEST_LIKE_ROWS = 837  # train rows 0-836 look like test (one label, not BGE-M3-selected); see PLAN §12


def rank_of(correct: str, ranking: list[str]) -> float:
    """1-based rank of the correct section, or inf if it was not retrieved."""
    return ranking.index(correct) + 1 if correct in ranking else np.inf


def metrics(ranks: np.ndarray) -> dict[str, float]:
    out = {f"hit@{k}": float(np.mean(ranks <= k)) for k in KS}
    for k in (5, 10):
        out[f"mrr@{k}"] = float(np.mean(np.where(ranks <= k, 1 / ranks, 0)))
    return out


def bootstrap(values: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling questions with replacement."""
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_question(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """Per-question scores whose mean is the metric (hit@5 is 0/1, MRR@10 is 1/rank or 0)."""
    return {"hit@5": (ranks <= 5).astype(float), "mrr@10": np.where(ranks <= 10, 1 / ranks, 0)}


def paired(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compare two retrievers on the same questions: mean difference a - b with a 95% interval."""
    out = {}
    for name in ("hit@5", "mrr@10"):
        d = per_question(a)[name] - per_question(b)[name]
        lo, hi = bootstrap(d)
        out[name] = (float(d.mean()), lo, hi)
    return out


def load_dev() -> pd.DataFrame:
    dev = load_split("dev").merge(load_questions("train")[["question_id", "question"]], on="question_id")
    dev["group_law"] = np.where(dev["law_name"] == CCC, "Civil and Commercial Code", "other laws")
    dev["group_rows"] = np.where(dev["question_id"].str[-5:].astype(int) < TEST_LIKE_ROWS, "test-like", "other")
    return dev


def run() -> None:
    import mlflow

    from thairag.index import MODEL, MODEL_SHA, load_index, load_model
    from thairag.retrieve import BM25, RRF_K, Dense, rrf

    dev = load_dev()
    questions = dev["question"].tolist()
    model = load_model()

    def encode(texts):
        return model.encode(texts, normalize_embeddings=True, batch_size=32)

    rankings, seconds = {}, {}
    chunks, _ = load_index("section")
    t = time.perf_counter()
    bm25 = BM25(chunks)
    build_s = time.perf_counter() - t
    t = time.perf_counter()
    rankings["bm25"] = [bm25.search(q) for q in questions]
    seconds["bm25"] = time.perf_counter() - t
    for name in ("section", "split512"):
        c, v = load_index(name)
        t = time.perf_counter()
        rankings[f"dense_{name}"] = Dense(c, v, encode).search_many(questions)
        seconds[f"dense_{name}"] = time.perf_counter() - t
    rankings["hybrid_rrf"] = [rrf(a, b) for a, b in zip(rankings["bm25"], rankings["dense_section"])]
    seconds["hybrid_rrf"] = seconds["bm25"] + seconds["dense_section"]

    params = {
        "bm25": {"tokeniser": "pythainlp newmm", "k1": 1.5, "b": 0.75, "index": "section"},
        "dense_section": {"model": MODEL, "revision": MODEL_SHA, "index": "section"},
        "dense_split512": {"model": MODEL, "revision": MODEL_SHA, "index": "split512", "section_score": "best part"},
        "hybrid_rrf": {"fuse": "bm25 + dense_section", "rrf_k": RRF_K},
    }

    ranks = {n: np.array([rank_of(c, r) for c, r in zip(dev["section_id"], rk)]) for n, rk in rankings.items()}
    REPORTS.mkdir(exist_ok=True)
    rows = []
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("m4-retrieval-dev")
    for name, r in ranks.items():
        m = metrics(r)
        ci = {}
        for k, v in per_question(r).items():
            ci[f"{k}_lo"], ci[f"{k}_hi"] = bootstrap(v)
        groups = {
            f"{col}={g}": metrics(r[(dev[col] == g).to_numpy()])
            for col in ("group_law", "group_rows")
            for g in sorted(dev[col].unique())
        }
        rows.append({"retriever": name, **m, **ci, "ms_per_question": 1000 * seconds[name] / len(dev)})
        with mlflow.start_run(run_name=name):
            mlflow.log_params({**params[name], "split": "dev", "n_questions": len(dev)})
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in m.items()})
            mlflow.log_metric("ms_per_question", rows[-1]["ms_per_question"])
            if name == "bm25":
                mlflow.log_metric("index_build_s", build_s)
            for g, gm in groups.items():
                gname = g.replace("=", "_").replace(" ", "_").replace("-", "_")
                mlflow.log_metrics({f"{gname}__{k.replace('@', '_at_')}": v for k, v in gm.items()})
            per_q = dev[["question_id", "section_id"]].assign(rank=r, top10=[" ".join(x[:10]) for x in rankings[name]])
            mlflow.log_table(per_q, artifact_file="ranks.json")

    table = pd.DataFrame(rows).set_index("retriever")
    table.to_csv(REPORTS / "dev_retrieval.csv", float_format="%.4f")
    groups = pd.DataFrame(
        {
            (n, f"{col}={g}"): {"n": int((dev[col] == g).sum()), **metrics(r[(dev[col] == g).to_numpy()])}
            for n, r in ranks.items()
            for col in ("group_law", "group_rows")
            for g in sorted(dev[col].unique())
        }
    ).T
    groups.to_csv(REPORTS / "dev_retrieval_groups.csv", float_format="%.4f")
    pairs = {
        ("hybrid_rrf", "dense_section"),
        ("hybrid_rrf", "bm25"),
        ("dense_section", "bm25"),
        ("dense_split512", "dense_section"),
    }
    comp = pd.DataFrame(
        [{"a": a, "b": b, "metric": k, "diff": d, "lo": lo, "hi": hi}
         for a, b in sorted(pairs) for k, (d, lo, hi) in paired(ranks[a], ranks[b]).items()]
    )
    comp.to_csv(REPORTS / "dev_retrieval_paired.csv", index=False, float_format="%.4f")
    per_q = dev[["question_id", "section_id", "question"]].copy()
    for n, r in ranks.items():
        per_q[n] = r
        per_q[f"{n}_top10"] = [" | ".join(x[:10]) for x in rankings[n]]
    per_q.to_csv(REPORTS / "dev_ranks.csv", index=False)  # reports/ is gitignored: it holds question text
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3), "\n", groups.round(3), "\n", comp.round(3))


if __name__ == "__main__":
    run()
