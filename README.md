# RAG-Med

This repository contains a collection of utilities to build a Retrieval-Augmented
Generation (RAG) pipeline over veterinary medicine documents. The goal is to
store technical sheets, retrieve relevant passages using BM25 or FAISS and
answer questions with a language model served by [vLLM](https://github.com/vllm-project/vllm).

## Project structure

```
.
├── client_vllm.py            # Example of querying a running vLLM server
├── client_rag.py             # End-to-end RAG pipeline client (retrieval + generation)
├── launch_vllm_server.py     # Helper to start the vLLM OpenAI API server
├── run_with_gpu.py           # Utility to run another script on the least used GPU
├── data/                     # Sample data and resources
│   ├── info_models/          # Model configuration for Azure, Ollama and vLLM
│   ├── posteriori_resources/ # Generated indices (BM25, FAISS)
│   ├── priori_resources/     # Static resources such as stopwords
│   └── prompts/              # System and user prompt templates
├── resource_builder/         # Scripts to build indices from raw data
│   ├── scripts/              # Individual components (BM25/FAISS builders, parsers)
│   └── unified_cimavet_processor.py  # End‑to‑end processor for CIMAVet dataset
├── shared/                   # Reusable utilities
│   └── veterinary_utils/     # Embedding models and text preprocessing helpers
├── tools/                    # Core RAG pipeline and evaluation scripts
│   ├── arqa/                 # Automatic Retrieval Question Answering tools
│   ├── reader.py             # vLLM client helper to generate final answers
│   ├── retrievers.py         # Retrieval functions (BM25, FAISS, hybrid)
│   ├── rag_orchestrator.py   # Pipeline orchestrating retrieval and generation
│   ├── rag_pipeline.py       # Library exposing the `run_rag_pipeline` function
│   └── test_orchestrator.py  # Minimal tests ensuring the orchestrator loads
└── requirements.txt          # Python dependencies
```

## Setup

1. Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

2. Start the vLLM server (GPU recommended):

```bash
python launch_vllm_server.py
```

3. In a new terminal, run the end-to-end RAG client:

 ```bash
 python client_rag.py
 ```

This script combines document retrieval (BM25, FAISS or hybrid) with generation
using a vLLM-served model. It is the recommended way to run the system interactively.

## Running the RAG pipeline

The command-line entry point is `tools/rag_orchestrator.py`. Internally it uses
`tools/rag_pipeline.py` which exposes the `run_rag_pipeline` function.  This
function can also be imported in your own scripts to obtain an answer given a
question and a retrieval strategy.

Example usage:

```bash
python tools/rag_orchestrator.py \
    --question "¿Qué medicamentos se pueden utilizar en cerdos para tratar coccidiosis?" \
    --retriever late+fallback --max_docs 10 --bm25_k 30 --faiss_k 15
```

The helper `run_with_gpu.py` can execute any script on the GPU with the lowest
memory usage:

```bash
python run_with_gpu.py tools/rag_orchestrator.py --question "..."
```

## Results

Evaluation scripts and logs are available under `tools/arqa/evaluation`. Several
retriever architectures were tested:

- **BM25** – keyword based retrieval using a sparse index.
- **FAISS** – dense retrieval with embeddings and a FAISS index.
- **Hierarchical FAISS** – first retrieves documents then re-ranks their
  chunks.
- **Late fusion (BM25 + FAISS)** – independent BM25 and FAISS rankings are
  linearly combined.

Empirically the late fusion strategy offered the best trade-off between
precision and recall. It consistently achieved the highest MRR in our logs
(see `tools/arqa/evaluation/hybrid/logs/latefusion_k10.md`).

Example result files such as `bm25_k10.md` or `latefusion_k50.md` report
precision, recall and MRR scores for each configuration.

```
$ head tools/arqa/evaluation/bm25/logs/bm25_k10.md
```

shows global metrics such as mean precision@k and mean recall@k.

## Running tests

A very small test suite ensures that the orchestrator module can be imported.
Execute it with:

```bash
pytest -q
```

## License

This project is provided as-is for experimentation with RAG techniques over
veterinary medicine documents.
