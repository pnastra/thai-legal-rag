# Glossary

GenAI terms, added the first time they appear in the project.

| Term | One-line definition | First seen |
|---|---|---|
| **LLM (large language model)** | A neural network trained to predict the next token of text; chat models are LLMs further tuned to follow instructions. | M0 |
| **Token** | The unit an LLM reads and writes: a word, part of a word or a few characters. Cost, speed and limits are all counted in tokens. Thai usually needs more tokens per sentence than English. | M0 |
| **Context window** | The maximum number of tokens (prompt + answer) the model can see at once; anything beyond it is cut off. Typhoon2-8B: 128K, but its Ollama model file sets 8,192. | M0 |
| **Temperature** | A sampling knob: 0 = always pick the most likely next token (repeatable), higher = sample more randomly (more varied, more creative, more errors). | M0 |
| **System prompt** | Instructions that set the model's role and rules for the whole conversation ("You are a Thai legal assistant; cite sections"). | M0 |
| **User prompt** | The actual message or question for this turn. | M0 |
| **Hallucination** | A fluent, confident answer that is not true or not supported by any source, because the model generates *plausible* text, not *verified* text. | M0 |
| **Quantisation (e.g. 4-bit)** | Storing model weights with fewer bits (4 instead of 16) so an 8B model fits in ~5 GB of RAM, at a small cost in quality. | M0 |
| **RAG (retrieval-augmented generation)** | Retrieve relevant documents first, put them in the prompt, then let the LLM answer *from them* and cite them. | M0 |
| **Thinking tokens** | Hidden reasoning tokens some models (e.g. Gemini 3.x) generate before the visible answer; they add latency and count toward limits/cost. Seen in M0: 68 thinking tokens to answer "OK". | M0 |
| **Corpus** | The full set of documents the system can search: here 5,127 law sections from 36 laws. | M1 |
| **Positive / hard negative** | Positive: a section labelled correct for a question. Hard negative: a section that looks relevant but is labelled wrong. | M1 |
| **Near-duplicate** | A question that is almost the same text as another one; kept out of dev if it resembles a test question. | M1 |
| **hit@k** | Share of questions whose correct section appears in the top k results (worked example in M4). | M1 |
| **MRR (mean reciprocal rank)** | Average of 1/rank of the correct section: 1 if first, 0.5 if second, 0 if not found (worked example in M4). | M1 |
| **Embedding** | A list of numbers (a vector) representing a text, so that texts with similar meaning get similar vectors. BGE-M3 gives 1,024 numbers. | M2 |
| **Cosine similarity** | How closely two vectors point the same way (1 = same direction, 0 = unrelated); for vectors of length 1 it is the dot product. | M2 |
| **Nearest-neighbour search** | Rank all stored vectors by similarity to the question's vector and return the top k. | M2 |
| **BGE-M3** | The multilingual open embedding model we use (MIT license, reads up to 8,192 tokens). | M2 |
| **Word tokeniser (Thai)** | Splits Thai text into words, which Thai writing does not separate with spaces; we use pythainlp (`newmm`). Not the same as an LLM's tokens. | M2 |
| **Chunk** | The piece of text that gets one vector and is returned by search; here one law section (or part of a long one). | M3 |
| **Vector index / vector store** | The saved vectors plus metadata that search runs on. Ours is a NumPy file; a vector database adds fast approximate search, updates and filters at large scale. | M3 |
| **Brute-force search** | Compare the question with every vector (exact); fine for thousands, slow for many millions. | M3 |
| **BM25** | Keyword ranking: rewards question words that are rare in the corpus (IDF), lets repeats saturate (k1) and penalises long texts (b). TF-IDF's smarter cousin. | M4 |
| **Sparse vs dense retrieval** | Sparse matches words (BM25; a vector with one slot per word, mostly zeros); dense matches meaning (embeddings, every slot used). | M4 |
| **Hybrid search / RRF** | Combine several rankings. Reciprocal rank fusion adds 1/(60 + rank) from each list, so only ranks matter, not score scales. | M4 |
| **hit@k / MRR@k** | hit@k: share of questions with the correct section in the top k. MRR@k: mean of 1/rank (0 if below k), so higher placement counts more. | M4 |
| **Bootstrap interval / paired comparison** | Resample questions with replacement to get a 95% interval; for two retrievers, resample the per-question differences, which gives a much tighter interval. | M4 |
