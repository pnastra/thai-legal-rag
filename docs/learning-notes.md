# Learning notes

My own notes and answers to the end-of-milestone questions.

## M0 — Setup and "hello LLM"

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_hello_llm.ipynb` (needs Ollama running and `GEMINI_API_KEY` in `.env`).

- **Tokens:** Thai ≈ 3.1 chars/token vs English ≈ 4.6 (Gemini tokenizer): but that isn't "same content": the same sentence took 14 Thai vs 12 English tokens (1.17×, one sentence only). *(Corrected 24 Nov; I first wrote 1.5×.)* Typhoon's model file has a ~200-token default system prompt, used only when we don't send our own.
- **Context window:** Typhoon supports 128K, but this model's own Ollama file sets `num_ctx` to 8,192; longer prompts may be silently truncated (from the docs, not tested here) unless `num_ctx` is raised.
- **Temperature:** 0 = argmax = repeatable (use for evaluation); 1 = sampling = varied.
- **Hallucination:** both models invented text for a non-existent section (CCC มาตรา 1799) and Typhoon got a real one (456) wrong. Gemini's fake answer was the most convincing. LLMs generate *plausible* text, not *verified* text → RAG + citations + evaluation.
- **Thinking tokens:** Gemini 3.x reasons in hidden tokens before answering (68 tokens to say "OK").
- **Ops lessons:** two Homebrews on this Mac (Intel `/usr/local` builds from source on Apple silicon → use `/opt/homebrew`); a listed free-tier model (`gemini-3.8-flash`) can still time out — test with one tiny call first.

### End-of-M0 questions
1. *Same answer 3× at temperature 0 — is it correct?* No: temperature 0 makes the answer **repeatable**, not **right** (Typhoon repeated the same wrong text for มาตรา 456). Only evaluation against the true section tells correctness.
2. *System prompt + 8 sections × ~600 Thai chars to Typhoon with default Ollama settings?* ~600/3.1 ≈ 194 tokens per section → ~1,550 for 8, plus ~100 instructions (these *replace* the ~200-token default prompt) + question + ~300 for the answer ≈ **2,000 tokens**: fits in 8,192. It breaks with more/longer sections: Ollama may **silently truncate** (no error; not tested by us), possibly dropping the instructions. Fix: `options={"num_ctx": 16384}` (more RAM) or fewer, better sections (reranker, M5). Measure tokens with the model's own tokenizer — 3.1 chars/token was Gemini's.
   - Mental model: the context window is a desk of fixed size; prompt **and** answer must fit on it at once.

## M1 — Data and evaluation splits

Run: `make data`, `make splits`, `make check`.

- **Look at the labels before trusting them.** Train and test come from the same dataset but are made differently: train has several labels per question (or none), test has exactly one written by experts.
- **Labels can favour one system.** Train labels were picked from BGE-M3's search results, so BGE-M3 may look better on dev than it is.
- **IDs must join.** The dataset's own key spells `12 ทวิ` as `12_2`; using it would have counted correct retrievals as wrong.
- **Grouped split:** dev and test share no question, no near-copy and no law section.
- **Sample size:** margin of error shrinks with √n, so 4× the questions halves it.

### End-of-M1 questions
1. *Why must dev and test share no law section, as well as no question?* So that test stays unseen and unbiased. Two differently worded questions can have the same answer section, so anything tuned on that section in dev (a prompt, a chunk, a threshold) would already fit the test question.
2. *BGE-M3 beats BM25 on dev; why might the gap shrink on test?* Most dev labels were picked from BGE-M3's own search results, which favours BGE-M3. Test labels come from legal experts.

## M2 — Embeddings and search by hand

Run: `make data`, then `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_embeddings_by_hand.ipynb`

### In my own words
- **What the notebook did:** Load then sampling 10 questions from train. Draws 200 distractors from the same laws with the samples. Embed them to vector space via BGE-M3 then sort each possible pair by cosine similarity. *(Correction: 200 sections in total = the 10 correct ones + 190 distractors; and the sorting is done separately for each question, not over all pairs together. The questions are the dev split, which comes from train.)*
- **Why Thai needs a word tokeniser:** Because Thai text does not split by space like English. It need vocabulary from Thai dictionary.

### Results of the toy search (10 questions, 200 sections; not a benchmark)
Correct section ranked first: split on spaces 1/10, Thai tokeniser 5/10, BGE-M3 10/10.
