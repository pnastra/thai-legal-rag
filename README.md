# thai-legal-rag

Retrieval-augmented generation (RAG) for Thai financial and commercial law, with answers that cite the exact law sections, evaluated on the NitiBench / WangchanX-Legal-ThaiCCL-RAG benchmark.

🚧 Work in progress (started Oct 2026).

> **Disclaimer:** this project retrieves and summarises law text for information only. It is not legal advice.

## Data and credits

- Questions and labels: [WangchanX-Legal-ThaiCCL-RAG](https://huggingface.co/datasets/airesearch/WangchanX-Legal-ThaiCCL-RAG) (MIT) and [NitiBench](https://huggingface.co/datasets/VISAI-AI/nitibench) (MIT).
- Law text: 36 laws, 5,127 sections, downloaded from the [NitiBench repository](https://github.com/vistec-ai/nitibench), which states no license; the files are not redistributed here.
- `make data` downloads pinned versions; `make splits` rebuilds the dev (500) and test (1,000) question lists in `data/splits/`.
