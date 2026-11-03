# Glossary

GenAI terms, added the first time they appear in the project.

| Term | One-line definition | First seen |
|---|---|---|
| **LLM (large language model)** | A neural network trained to predict the next token of text; chat models are LLMs further tuned to follow instructions. | M0 |
| **Token** | The unit an LLM reads and writes: a word, part of a word or a few characters. Cost, speed and limits are all counted in tokens. Thai usually needs more tokens per sentence than English. | M0 |
| **Context window** | The maximum number of tokens (prompt + answer) the model can see at once; anything beyond it is cut off. Typhoon2-8B: 128K. | M0 |
| **Temperature** | A sampling knob: 0 = always pick the most likely next token (repeatable), higher = sample more randomly (more varied, more creative, more errors). | M0 |
| **System prompt** | Instructions that set the model's role and rules for the whole conversation ("You are a Thai legal assistant; cite sections"). | M0 |
| **User prompt** | The actual message or question for this turn. | M0 |
| **Hallucination** | A fluent, confident answer that is not true or not supported by any source, because the model generates *plausible* text, not *verified* text. | M0 |
| **Quantisation (e.g. 4-bit)** | Storing model weights with fewer bits (4 instead of 16) so an 8B model fits in ~5 GB of RAM, at a small cost in quality. | M0 |
| **RAG (retrieval-augmented generation)** | Retrieve relevant documents first, put them in the prompt, then let the LLM answer *from them* and cite them. | M0 |
| **Thinking tokens** | Hidden reasoning tokens some models (e.g. Gemini 3.x) generate before the visible answer; they add latency and count toward limits/cost. Seen in M0: 68 thinking tokens to answer "OK". | M0 |
