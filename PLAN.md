# PLAN — thai-legal-rag

A 2.5-week portfolio project for a **GenAI beginner**: build a **retrieval-augmented generation (RAG)** system that answers Thai financial/commercial law questions with **citations to the exact law sections**, and **measure** how each design choice (retriever, chunking, reranker, LLM, prompt) changes the results, using the expert-labelled **NitiBench / WangchanX-Legal-ThaiCCL-RAG** benchmark.

**Audience:** hiring managers for GenAI / LLM / data scientist IC roles in Thailand (banks, insurers, enterprises, consultancies).
**What it should prove:** I can build a Thai-language RAG pipeline end to end **and evaluate it like a data scientist**: baselines, ablations, retrieval metrics, LLM-as-judge with a human check, honest failure analysis. Plus the production basics from my earlier projects: API, Docker, Cloud Run, CI.
**Builds on projects 1–3:** same conventions (`uv`, `src/`, Makefile, MLflow, tests, teaching mode, 🧠 brainstorms). New here: LLMs, prompts, embeddings, vector search, BM25, rerankers, LLM-as-judge, Thai NLP.

**I am new to GenAI.** This plan goes slower at the start, builds concepts by hand before using frameworks, and explains every new idea before using it (see "Learning mode").

**Timebox:** about 46 hours over 2.5 weeks (start 17 Oct 2026, ship 5 Nov 2026; confirmed in M0). Week 1 = learn + retrieval, week 2 = generation + evaluation, last half-week = ship. No new scope after M8.

---

## 1. Working rules for Claude Code

**General working rules** (session workflow, the 🧠 brainstorm protocol, teaching mode, "always ask before", code conventions, evaluation honesty, cloud defaults) **are in `~/Projects/CLAUDE.md`**, which Claude Code loads automatically. This section holds only the rules specific to this project; where the two differ, this file wins.

### Project-specific rules
- Decisions from 🧠 brainstorms go in **§12 Decisions log**; session notes go in **§13 Progress log**.
- **🧠 topics** are listed under each marked milestone. If a decision comes up that isn't listed, stop and brainstorm it too. Because I'm new to GenAI, add one line per option explaining *what it is*.
- **Evaluation discipline:** tune on the **dev split** (§4); report final numbers on the **held-out test split** once, at M7. Never tune prompts or thresholds on test questions.
- **No frameworks before fundamentals.** Build retrieval and the RAG loop in plain Python first (M2–M6). LangChain / LlamaIndex only with my OK, and only after M6.
- **Cost guard:** prefer local models (Ollama) and free tiers. Before any run that calls a hosted API more than ~100 times, tell me the estimated number of calls and wait for my OK.
- Never commit vector indexes, model weights (`.gguf`, `.safetensors`) or downloaded datasets.
- Ask before downloading any model over 5 GB.

### Learning mode (I'm new to GenAI)
- **Before a new concept is used for the first time** (token, embedding, cosine similarity, vector index, BM25, chunk, reranker, context window, temperature, prompt template, hallucination, LLM-as-judge…): explain it in 3–5 plain sentences, with an analogy to something I know (statistics, SQL, search engines, classification), then show the smallest possible code example.
- **Hands-on before abstraction:** the first time we do something (embed, search, prompt), do it in ~20 lines of plain Python I can read, *then* refactor into `src/`.
- **Check my understanding:** at the end of each milestone, ask me 2 short questions about the new concepts. If I get one wrong, re-explain it differently.
- Keep a **glossary** in `docs/glossary.md`: add each new term with a one-line definition when it first appears.

### Go deeper on (this project)
Embeddings and vector search, BM25 and Thai tokenisation, rerankers, prompting for grounded answers and citations, LLM-as-judge and its biases, Ollama, Docker for model serving. Keep it brief for FastAPI, Docker basics, Cloud Run and GitHub Actions (learned in earlier projects).

---

## 2. Environment

| Item | Status |
|---|---|
| Mac (Apple silicon), Git, VS Code, Claude Code, `uv`, Docker Desktop | ✅ (from earlier projects) |
| GitHub account (github.com/pnastra) | ✅ |
| Mac RAM (decides which local LLM size fits) | ✅ M1 Pro, 16 GB → 7–9B at 4-bit |
| **Ollama** for local LLMs | ✅ 0.40.5 via `/opt/homebrew` (native arm64), runs as a brew service; Typhoon2-8B 100 % GPU |
| Hosted LLM API key for the deployed demo (Gemini, free tier) | ✅ in `.env`; key in a separate Google project **without billing** (stays free tier) |
| Hosted LLM API key for the judge (Anthropic, Claude Haiku 5.5, budget $3–5) | ⬜ M0 (needed by M7) |
| Google Cloud project with **$1 budget alert** | ✅ from fruit project (reuse) |
| Kaggle/Colab GPU | Only if embedding the corpus on CPU/MPS is too slow (M3) |

**Python:** 3.12. Use `uv python pin 3.12`.

---

## 3. Dependencies (managed by `uv`)

- **Core:** numpy, pandas, pyarrow, datasets (Hugging Face), scikit-learn, mlflow, matplotlib
- **Thai text and retrieval:** pythainlp (Thai word tokenisation for BM25), rank-bm25, sentence-transformers (or FlagEmbedding) for BGE-M3 and a reranker
- **LLM access:** ollama (Python client), httpx; one hosted-provider SDK chosen in M0
- **Vector store:** start with NumPy (M2–M4); add chromadb or qdrant-client only after the 🧠 brainstorm in M3
- **Serving:** fastapi, uvicorn, pydantic
- **Dev:** pytest, ruff, jupyter / ipykernel
- **Ask first:** langchain, llama-index, ragas, any GPU-only package, any fine-tuning library

---

## 4. Data

### 4.1 Benchmark
- **WangchanX-Legal-ThaiCCL-RAG** (Hugging Face: `airesearch/WangchanX-Legal-ThaiCCL-RAG`), VISTEC-depa AI Research Institute. **License: MIT.**
  - ~12k rows: **train ~8.2k, test ~3.7k**.
  - Fields: `question`, `positive_contexts` (relevant law sections), `hard_negative_contexts`, `positive_answer` (expert-reviewed), `hard_negative_answer`.
  - Covers 35 Thai financial/commercial laws (e.g. Civil and Commercial Code, Securities and Exchange Act, Revenue Code). Test questions were written by legal experts and have **one correct section each**.
- **NitiBench** (Hugging Face: `VISAI-AI/nitibench`; code: github.com/vistec-ai/nitibench), paper: *NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai Legal Question Answering* (arXiv 2502.10868).
  - **NitiBench-CCL:** 3,730 test questions (the WangchanX test set).
  - **NitiBench-Tax:** 50 real Revenue Department tax cases, multi-section, much harder. Stretch only.
  - **License:** not stated in the paper; check the dataset card in M1 and record it in the README.
  - Paper reference points to compare against: the best retriever (human-fine-tuned BGE-M3) reached **MRR@5 ≈ 0.80** on CCL; section-based chunking beat naive line chunking; RAG beat putting all laws into a long-context LLM. **Use these as context for my results, not as a target to chase.**

### 4.2 Corpus
- The law sections the questions refer to. Confirm in M1 whether the NitiBench repo/dataset ships the full corpus (all sections of the 35 laws) or only the contexts inside the QA rows. If only the latter, build the corpus as the de-duplicated union of all `positive_contexts` + `hard_negative_contexts` and note this limitation in the README.

### 4.3 Splits (decided in M1 🧠)
- **dev:** a fixed random sample from the WangchanX **train** split (e.g. 300 questions) for all tuning.
- **test:** a fixed random sample from the **test** split (e.g. 500 questions) used **once** for final numbers in M7.
- Fixed seeds; save question IDs to `data/splits/` (small, committed).

### 4.4 Disclaimer
- This is legal *information retrieval*, not legal advice. The README and API responses must say so.

---

## 5. Repo structure

```
thai-legal-rag/
├── PLAN.md
├── CLAUDE.md               # one line: @PLAN.md
├── README.md               # demo link, results tables, architecture, model card, credits
├── LICENSE                 # MIT (code)
├── .gitignore
├── .env.example            # names of env vars, never values
├── pyproject.toml / uv.lock
├── Makefile                # data, index, eval-retrieval, eval-e2e, serve, docker-build, docker-run, deploy
├── Dockerfile
├── .github/workflows/
│   ├── ci.yml              # ruff + pytest on every push/PR
│   └── deploy.yml          # build → Artifact Registry → Cloud Run on push to main (reuse fruit project setup)
├── docs/
│   ├── glossary.md         # GenAI terms, added as we go
│   └── learning-notes.md   # my notes / answers to the end-of-milestone questions
├── data/raw/               # gitignored
├── data/splits/            # committed: dev/test question IDs
├── notebooks/
│   ├── 00_hello_llm.ipynb
│   ├── 01_explore.ipynb
│   └── 02_embeddings_by_hand.ipynb
├── src/thairag/
│   ├── data.py             # load benchmark + build corpus
│   ├── chunk.py            # chunking strategies
│   ├── tokenize_th.py      # Thai word tokenisation for BM25
│   ├── retrieve.py         # BM25, dense (BGE-M3), hybrid
│   ├── rerank.py           # cross-encoder reranker
│   ├── llm.py              # one interface for Ollama + hosted provider
│   ├── prompts.py          # prompt templates (versioned)
│   ├── generate.py         # RAG: retrieve → prompt → answer with citations
│   ├── eval_retrieval.py   # hit@k, recall@k, MRR
│   ├── eval_e2e.py         # LLM-as-judge: coverage, contradiction, citation P/R
│   └── tracing.py          # log each query (retrieved IDs, prompt version, latency, tokens) to MLflow
├── api/main.py             # FastAPI: /health, /ask
└── tests/
```

---

## 6. Milestones

Twelve milestones (M0–M11): about 18 h in week 1, 18 h in week 2, 10 h in the last half-week. Hours include learning and brainstorm time; if one runs 50% over, stop and note it in §13.

### Week 1 — Learn the basics, build retrieval

#### M0 — Setup and "hello LLM" (≈3 h) 🧠
- [x] Create the public GitHub repo `thai-legal-rag` (MIT); `uv init`, `uv python pin 3.12`, core deps, `.gitignore` (§9), `CLAUDE.md`, `.env.example`, `docs/glossary.md`, one trivial passing test
- [x] **🧠 Brainstorm — LLMs:** which local model fits my Mac's RAM (e.g. a Thai-tuned model such as Typhoon vs a strong multilingual open model, at 7–9B and 4-bit); which hosted provider for the deployed demo and the judge (free-tier limits, Thai quality, data-use terms); confirm the start/ship dates
- [x] Install Ollama, pull the chosen model; `notebooks/00_hello_llm.ipynb`: send one Thai and one English prompt to the local model and the hosted API
- [x] **Learn:** tokens and context window, temperature, system vs user prompt, why LLMs hallucinate. Try the same question at temperature 0 and 1.

**Done when:** both LLMs answer from the notebook, the API key lives only in `.env`, the first commit is pushed, and the glossary has its first 6–8 terms.

#### M1 — Data and evaluation splits (≈2.5 h) 🧠
- [ ] `data.py`: load the WangchanX dataset; check the NitiBench repo/dataset for the full corpus and license; record both in §13
- [ ] `notebooks/01_explore.ipynb`: questions per law, question and section lengths (in characters and tokens), 5 example rows read together with me
- [ ] **🧠 Brainstorm — evaluation design:** dev/test sizes (cost vs statistical precision), whether to stratify by law, what to do if the corpus is incomplete, which metrics will be the headline numbers
- [ ] Save fixed dev/test question IDs to `data/splits/`

**Done when:** the corpus size, license status and splits are recorded, and a test checks that no question ID is in both dev and test.

#### M2 — Embeddings and search by hand (≈3 h)
- [ ] **Learn:** what an embedding is (a vector where similar meanings are close), cosine similarity, why "nearest neighbour" = search
- [ ] `notebooks/02_embeddings_by_hand.ipynb`: embed ~200 law sections and 10 dev questions with BGE-M3; compute cosine similarity with NumPy; show the top-3 sections per question; check whether the correct section appears
- [ ] Compare with keyword search on the same 10 questions using a naive split-on-spaces approach, and see why it fails for Thai (no spaces between words)

**Done when:** I can explain in my own words (in `docs/learning-notes.md`) what the notebook did and why Thai needs a word tokeniser.

#### M3 — Chunking and indexing the corpus (≈3.5 h) 🧠
- [ ] **🧠 Brainstorm — chunking:** one chunk per law section (structure-aware) vs fixed-size chunks with overlap vs splitting long sections; how to keep metadata (law name, section number) for citations
- [ ] **🧠 Brainstorm — vector store:** NumPy matrix + brute force (fine at this size?) vs Chroma vs Qdrant; what each teaches and when you'd need a real vector DB
- [ ] `chunk.py`, `tokenize_th.py` (pythainlp), indexing script: embed the full corpus once, save embeddings + metadata to `data/index/` (gitignored)
- [ ] Test: every chunk has a law name and section ID; no empty chunks

**Done when:** `make index` builds the index reproducibly and its size/time are recorded in §13.

#### M4 — Retrieval baselines and evaluation (≈4.5 h) 🧠
- [ ] **Learn:** BM25 (TF-IDF's smarter cousin), dense vs sparse retrieval, hybrid search, and the metrics hit@k, recall@k, MRR (explained with a small worked example)
- [ ] **🧠 Brainstorm — retrievers to compare:** BM25 (with pythainlp tokens) vs dense BGE-M3 vs hybrid (how to combine scores: reciprocal rank fusion vs weighted sum); which k values to report
- [ ] `retrieve.py`, `eval_retrieval.py`; run all retrievers on **dev**; log each as an MLflow run
- [ ] Error analysis: read 10 dev questions where the correct section is not in the top 10. Why? (paraphrase, cross-references, very short sections…)

**Done when:** a dev results table (retriever × hit@1/5/10, MRR@10) is in MLflow and §13, and the error notes are in `docs/learning-notes.md`.

### Week 2 — Generation and evaluation

#### M5 — Reranking (≈3 h) 🧠
- [ ] **Learn:** bi-encoder (embed separately) vs cross-encoder (read question + section together); why rerankers are slower but more accurate
- [ ] **🧠 Brainstorm — reranker:** which model (a multilingual BGE reranker vs none), how many candidates to rerank (20? 50?), latency budget per query
- [ ] `rerank.py`; evaluate best retriever + reranker on dev; measure latency per query

**Done when:** the dev table has a reranker row with latency, and the retrieval pipeline for the rest of the project is chosen and recorded in §12.

#### M6 — Answer generation with citations (≈4 h) 🧠
- [ ] **Learn:** prompt templates, grounding ("answer only from these sections"), asking for citations in a fixed format, and what happens when the answer isn't in the context
- [ ] **🧠 Brainstorm — prompting:** answer language (Thai), citation format (e.g. `[ประมวลกฎหมายแพ่งและพาณิชย์ มาตรา 456]`), how many sections to include, when to say "not found in the provided sections", output as plain text vs JSON
- [ ] `llm.py` (one interface for Ollama + hosted), `prompts.py` (versioned templates), `generate.py`
- [ ] Try 3 prompt versions on 20 dev questions; read the answers together with me; pick one

**Done when:** `make ask Q="..."` returns a Thai answer with citations to real section IDs, and a test checks that every cited ID exists in the retrieved set.

#### M7 — End-to-end evaluation (≈5 h) 🧠 — **cost guard applies**
- [ ] **Learn:** LLM-as-judge — how it works, its biases (prefers long answers, its own style), and why we validate it against human judgement
- [ ] **🧠 Brainstorm — judging:** which judge model (must differ from the answering model if possible), scoring rubric (coverage 0/50/100 and contradiction 0/1 as in NitiBench, plus citation precision/recall computed exactly from section IDs), how many answers I check by hand (target 30)
- [ ] `eval_e2e.py`; run on dev; **I hand-score 30 answers**; compute judge–human agreement and record it
- [ ] Run the **final comparison once on test**: (a) LLM with no retrieval, (b) BM25 RAG, (c) best RAG, (d) best RAG with the local vs hosted LLM, (e) golden context (correct section given) as the upper bound
- [ ] Failure analysis: 15 wrong answers, grouped by cause (retrieval miss, wrong section cited, misread law, Thai language issue)

**Done when:** the final test table, judge–human agreement and the failure-analysis notes are saved for the README.

#### M8 — API, guardrails and tracing (≈3.5 h)
- [ ] `api/main.py`: `GET /health`, `POST /ask` → answer, cited sections (law, section number, text snippet), latency, disclaimer
- [ ] Guardrails: empty/too-long questions → 422; "not found in the provided sections" path; no answer without at least one citation
- [ ] `tracing.py`: each request logged as one JSON line (timestamp, question hash, retrieved IDs, prompt version, model, latency, token counts); no raw personal data
- [ ] `TestClient` tests for normal, not-found and invalid-input cases

**Done when:** `make serve` works locally and the API tests pass.

### Last half-week — Ship

#### M9 — Docker and Cloud Run (≈4 h) 🧠 — **ask me before each cloud step**
- [ ] **🧠 Brainstorm — deployment:** where the LLM runs in the cloud (hosted API from the container vs a small local model inside Cloud Run — memory/cold-start/cost trade-offs); where the index lives (baked into the image vs Cloud Storage); request limits to protect the free tier
- [ ] `Dockerfile` (python:3.12-slim, serving deps only, `--platform linux/amd64`); API key passed as a Cloud Run secret, never in the image
- [ ] Deploy (min 0, max 1 instance, unauthenticated, simple rate limit); check `/health` and `/docs` from a cold start

**Done when:** the public `/docs` URL answers a Thai question with citations, and the URL is in §13.

#### M10 — CI/CD (≈2 h)
- [ ] `ci.yml` (ruff + pytest; tests must not call real LLMs — use a fake LLM in tests) and `deploy.yml` (reuse Workload Identity Federation from the fruit project)
- [ ] CI badge in the README; a small visible change redeploys automatically

**Done when:** a push to `main` runs CI and updates the live service with no manual steps.

#### M11 — README, results and model card (≈4 h)
- [ ] Top: live `/docs` link, one example Q&A (Thai + English gloss), one-sentence summary, CI badge
- [ ] Results: retrieval table (dev), final test table (no-RAG vs RAG variants vs golden context), judge–human agreement, cost/latency per query
- [ ] Architecture diagram: question → Thai tokenise / embed → hybrid retrieve → rerank → prompt → LLM → answer + citations → trace log
- [ ] Model card: intended use, limits (financial/commercial law only, 35 laws, laws change over time, LLM-judge bias, not legal advice), data sources and licenses
- [ ] "What I learned" section (short, honest; good for interviews)

**Done when:** a stranger could understand and rerun the project from the README alone.

### Stretch (only after M11, or as version 2)
- [ ] NitiBench-Tax (50 hard multi-section cases) as a stress test
- [ ] Query rewriting / multi-query retrieval
- [ ] Agentic version: the LLM decides when to search again, or follows cross-references between sections (tool calling)
- [ ] Simple chat UI (Streamlit) calling the Cloud Run API
- [ ] Fine-tune the embedding model on the train split (compare with the published fine-tuned retriever)

---

## 7. Out of scope

- Fine-tuning any LLM or embedding model (stretch only)
- LangChain / LlamaIndex before M6, and only with my OK after that
- Multi-turn chat memory, user accounts, a front-end UI (FastAPI `/docs` is the demo)
- Paid GPU hosting, Kubernetes, Terraform, any paid service beyond free tiers and the $1-alert GCP project
- NitiBench-Tax in the main results (stretch)
- Giving legal advice; the system retrieves and summarises law text only

---

## 8. Definition of done

- [ ] Public GitHub repo with README, MIT license (code), passing CI badge
- [ ] A live Cloud Run URL that answers Thai legal questions with citations
- [ ] Retrieval ablation (BM25 / dense / hybrid / reranker) on dev, final comparison on test run **once**
- [ ] LLM-as-judge validated against my 30 hand-scored answers, agreement reported
- [ ] Every 🧠 decision recorded in §12; glossary and learning notes in `docs/`
- [ ] No API keys, datasets, indexes or model weights in the git history
- [ ] The legal-information disclaimer appears in the README and in API responses

---

## 9. `.gitignore`

```
# Python / uv
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/

# Data, indexes, models
data/raw/
data/index/
*.parquet
*.npy
*.faiss
chroma/
qdrant_storage/
*.gguf
*.safetensors
*.bin

# MLflow
mlruns/
mlflow.db
mlartifacts/

# Secrets
.env
*.json.key
gcp-*.json
*.token

# macOS / editors
.DS_Store
.vscode/
```

(`data/splits/*.csv` stays committed — keep split files as CSV, not parquet.)

---

## 10. RAG in one page (for me, a beginner)

1. **Index (once):** split the law text into chunks → turn each chunk into a vector (embedding) → store vectors + metadata.
2. **Retrieve (per question):** embed the question → find the closest chunks (dense), and/or match words (BM25) → optionally rerank the top candidates with a slower, more accurate model.
3. **Generate:** put the question + top chunks into a prompt that says "answer only from these sections and cite them" → the LLM writes the answer.
4. **Evaluate:** retrieval metrics check step 2 against the known correct section; the judge (plus my hand check) scores step 3's answer.
5. **Why it beats a plain LLM:** the model reads the actual law at answer time, so it can cite sources and is less likely to invent rules. **Why it still fails:** if retrieval misses the right section, the answer will be wrong no matter how good the LLM is. That's why most of the work is in retrieval and evaluation.

---

## 11. Risks

| Risk | Fallback |
|---|---|
| The full law corpus isn't published | Use the union of all contexts in the dataset as the corpus; state the limitation clearly |
| Local LLM too slow or too weak in Thai on my Mac | Use a smaller model for development and the hosted API for final runs (cost guard) |
| Hosted free tier limits block the M7 test run | Reduce the test sample (e.g. 300); run over two days; or use the local model as the answering model |
| Embedding the corpus is slow on CPU/MPS | Embed once in Colab/Kaggle, download the vectors |
| LLM-as-judge disagrees with my hand scores | Report it honestly; lean on the exact citation metrics, which need no judge |
| I fall behind because the concepts are new | Cut M5 (reranker) to one config and M10 (CD) to CI-only; don't cut M7 (evaluation is the point) |
| Cloud Run + hosted LLM costs | Min 0 / max 1 instance, rate limit, budget alert, API spending cap |

---

## 12. Decisions log (from 🧠 brainstorms)

| Date | Milestone | Decision | Options considered | Reason |
|---|---|---|---|---|
| 2026-10-16 | M0 | Local LLM: Typhoon (Thai-tuned, ~8B, 4-bit) via Ollama; fallback Qwen ~8B if no ≤5 GB Typhoon build | Typhoon, Qwen, Gemma 4B/12B, small 3–4B | Fits 16 GB; Thai-tuned is a good story for Thai employers; Qwen is an easy fallback |
| 2026-10-16 | M0 | Hosted answering LLM (demo): Gemini, free tier | Gemini, Typhoon API, Claude Haiku | Free, good Thai, same Google ecosystem as Cloud Run; law text is public, so free-tier data use is acceptable |
| 2026-10-16 | M0 | Judge LLM: Claude Haiku 5.5 (`claude-haiku-5-5`), budget $3–5 | Gemini (other size), Typhoon API, Claude Haiku | A different provider from both answering models avoids self/same-family bias; est. ~$1–3 for the whole project |
| 2026-10-16 | M0 | Dates: start 17 Oct 2026, ship 5 Nov 2026 | Start 19 Oct vs 17 Oct | Starting earlier with the same ship date = 2 days of buffer |
| 2026-10-16 | M0 | Install Ollama via Homebrew | Official app vs brew | User preference |
| 2026-11-03 | M0 | Local model: `scb10x/llama3.1-typhoon2-8b-instruct` (official SCB 10X Ollama build, 4.9 GB) | Typhoon2.1-Gemma3-4B/12B, Typhoon2.5-Qwen3-4B, Qwen fallback | Matches the ~8B / ≤5 GB decision; no fallback needed |
| 2026-11-03 | M0 | Gemini model: `gemini-3.5-flash` (set via `GEMINI_MODEL`) | gemini-3.8-flash, gemini-2.5-flash | 3.8-flash is listed as free tier but timed out (>60 s) on generate; 3.5-flash answers in ~2.5 s |
| 2026-11-03 | M0 | Gemini key lives in its own Google project without billing, not in `fruit-ripeness-hsi` | Key in the billed fruit project | Billing-enabled project = paid tier; separate project keeps it free |

---

## 13. Progress log

Claude Code adds 2–4 lines at the end of every session: date, milestone, what was done, decisions, what's next.

| Date | Milestone | Notes |
|---|---|---|
| 2026-11-03 | M0 | Scaffolded repo (uv, Py 3.12, src/thairag, Makefile, tests, .env loader); installed Ollama natively (Intel Homebrew in /usr/local compiled from source → switched to /opt/homebrew); pulled Typhoon2-8B; Gemini key in .env (first key was pasted in chat → rotated). `00_hello_llm.ipynb` runs both LLMs: Thai ~1.5× tokens vs English, both models hallucinate a non-existent CCC section. Glossary: 10 terms. |
| 2026-11-03 | M0 | Note: M0 spanned sessions 16 Oct → 3 Nov; ship date 5 Nov is no longer realistic — new dates to be agreed before M1. Next: M1 (data + evaluation splits, 🧠). |
