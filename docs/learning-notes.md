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

### End-of-M2 questions
1. *"Show a section only if its similarity is above 0.6": what's the problem?* My answer: without the law and/or text, a wrong answer may be shown and can't be rechecked. *(That is a good point about citations, but not the main problem here: scores are only comparable within one question. In the notebook a correct section scored 0.58 and would be hidden, while a wrong one scored 0.68 and would be shown. Use the top k; tune any cutoff on dev.)*
2. *Why did keywords rank the correct section 74th and embeddings 1st for ใครที่สามารถทำอาชีพผู้ทำบัญชีได้?* Explained by Claude: the law says ผู้ใด and วิชาชีพ, the question says ใคร and อาชีพ. Keywords need the same words; embeddings can match similar meanings.

## M3 — Chunking and indexing

Run: `make index` (about 4 minutes, builds `data/index/`).

- **A long chunk blurs:** one vector must summarise everything in it. **A short chunk loses context:** "(3) ten years" means little without its opening sentence.
- **Law sections are natural chunks:** median 90 tokens, and each is exactly what we cite.
- **Estimate the possible gain first:** only 14 of 500 dev questions have a split section as the answer, so splitting can add at most 2.8 points to hit@5.

### End-of-M3 questions
1. *Why might one vector represent a 1,685-token definitions section poorly?* My answer: it has to be embedded to a vector smaller than its tokens (1,685 → 1,024), plus it carries many unrelated definitions. *(The second reason is right. The first is not: 1,024 is the number of dimensions, not a token limit; a 15-token and a 1,685-token text both get 1,024 numbers. The problem is one point standing for many topics.)*
2. *Why is a 4-part section with 4 chances a problem, and does best-part scoring solve it?* My answer: one true section could appear up to 4 times; taking the best score gives a distinct predicted section, whether it is correct or not. *(Right about duplicates. Not complete: the maximum of 4 scores tends to be higher than 1 score, so long sections still have an advantage.)*

## M4 — Retrieval baselines on dev

Run: `make eval-retrieval` (≈30 s once the index exists). Runs: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`, experiment `m4-retrieval-dev`.

- **BM25** scores word overlap (rare words count more, repeats saturate, long sections are penalised); **dense** compares meaning vectors; **hybrid RRF** adds 1/(60 + rank) from each list.
- **Dev (500), hit@5 / MRR@10:** BM25 0.652 / 0.534, dense 0.806 / 0.684, split512 0.804 / 0.682, RRF 0.774 / 0.668.
- **Paired comparisons** (same questions, bootstrap the differences) are much tighter than comparing two separate intervals: split512 − section = −0.002 (−0.010 to 0.004).
- **Hybrid lost to dense** (−3.2 pts hit@5): dense alone had the answer in its top 5 and RRF did not for 37 questions, against 21 the other way.

### Error analysis: 10 random dev questions where dense missed the top 10 (seed 42)

| Question | Rank dense / BM25 / RRF | Cause |
|---|---|---|
| train-07347 foundation dissolution | >100 / >100 / >100 | **Label wrong** (checked by me): label 93 is about association meetings; the answer's content is today's 131, which dense found first |
| train-08157 owner of ส่วนควบ | >100 / >100 / >100 | **Label wrong** (checked): label 107; answer's content is today's 144, dense's #1 |
| train-07522 who can ask for the absentee property report | 22 / >100 / 51 | **Label wrong** (checked): label 60 only says "apply agency rules"; answer's content is today's 56, dense's #1 |
| train-07803 who cares for an incompetent person | >100 / 33 / 63 | Label doubtful (30 is about acts of an insane person); dense's #1 (430, tort liability) is also wrong |
| train-00453 specific business tax reports | >100 / 6 / 31 | Cross-reference: label 91/21 only lists sections to apply; dense found the content (87/3) |
| train-00018 shareholder over 15% | >100 / 86 / >100 | Cross-reference: Banking Act 22 points to 16–19; law not named |
| train-04712 liquidator, บริษัทจำกัด | 51 / 4 / 11 | Similar law: dense picked the Public Companies Act twin rule; BM25 matched the exact words |
| train-03240 secret-ballot exceptions | 21 / 1 / 6 | Law not named: dense picked the CCC rule, the label is the Public Companies Act |
| train-00171 who is the auditor | 19 / 71 / 17 | Law not named; term in several laws |
| train-07315 when is tax payment complete | 12 / >100 / 50 | Short, vague question |

**Takeaways**
- **3 of 10 misses are label errors**, maybe 4. In each, the stored answer quotes the right content but an old section number. Hypothesis (agreed, not verified): some train rows cite the Civil and Commercial Code's numbering before Book 1 was revised in 1992, and the label was matched by number to today's law. So dev scores are slightly pessimistic. Labels are not edited; test labels are expert-written.
- **Cross-references** (2): the labelled section only points to others. Following references is a stretch item.
- **Similar law / law not named** (3): this is where BM25 sometimes wins. RRF didn't help on average, but this is its niche.

### End-of-M4 questions
1. *Why can MRR@10 differ between two retrievers with the same hit@10?* My answer: hit@k only checks rank ≤ k; MRR also accounts for the rank. ✅
2. *Hybrid is usually better; why might it lose to dense here?* My answer: BGE-M3 is pre-trained and better than typical dense embedders. *(Only part of it. RRF is an equal-weight vote: when one voter (BM25, 0.65) is much weaker than the other (dense, 0.81), it pulls more right answers down than it lifts. Hybrid helps when both parts are about equally good and make different mistakes. Dev labels partly picked with BGE-M3 also tilt dev towards dense.)*
3. *Re-check: if BM25 also scored ≈0.80 hit@5 and missed different questions from dense, would RRF beat dense?* My answer: no. *(Expected: yes. It is the ensemble idea from classical ML: equally good models with uncorrelated errors, averaged, beat each one (like a random forest). Example: dense ranks the correct section 15th, BM25 1st; dense's wrong #1 is BM25's 50th → correct 1/75 + 1/61 = 0.0297 beats wrong 1/61 + 1/110 = 0.0255. Our RRF lost because BM25 is much weaker, not because fusion can't work.)*
