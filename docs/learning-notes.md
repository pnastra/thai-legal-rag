# Learning notes

My own notes and answers to the end-of-milestone questions.

## M0 — Setup and "hello LLM"

Run: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/00_hello_llm.ipynb` (needs Ollama running and `GEMINI_API_KEY` in `.env`).

- **Tokens:** Thai ≈ 3.1 chars/token vs English ≈ 4.6 (Gemini tokenizer): Thai prompts cost ~1.5× more. Chat templates add hidden tokens (~200 for Typhoon's default system prompt).
- **Context window:** Typhoon supports 128K, but Ollama loads it with 8,192 by default; longer prompts are silently truncated unless `num_ctx` is raised.
- **Temperature:** 0 = argmax = repeatable (use for evaluation); 1 = sampling = varied.
- **Hallucination:** both models invented text for a non-existent section (CCC มาตรา 1799) and Typhoon got a real one (456) wrong. Gemini's fake answer was the most convincing. LLMs generate *plausible* text, not *verified* text → RAG + citations + evaluation.
- **Thinking tokens:** Gemini 3.x reasons in hidden tokens before answering (68 tokens to say "OK").
- **Ops lessons:** two Homebrews on this Mac (Intel `/usr/local` builds from source on Apple silicon → use `/opt/homebrew`); a listed free-tier model (`gemini-3.8-flash`) can still time out — test with one tiny call first.
