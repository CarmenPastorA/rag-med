"""
Evaluate hierarchical FAISS retriever over synthetic QA pairs.
This uses the HierarchicalFaissSearch class to first retrieve relevant
SmPC documents from an index built on essential information and then
fetch the most relevant chunks from those documents.
"""

import os
import sys
import json
from collections import defaultdict
import numpy as np
import argparse

# Paths setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.append(PROJECT_ROOT)

from tools.arqa.faiss_search import HierarchicalFaissSearch
from shared.veterinary_utils.embedding_model import EmbeddingModel
from tools.arqa.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    mrr,
    normalized_precision_at_k,
    n_relevant_retrieved,
)
from shared.veterinary_utils.utils import normalize_doc_id

# Evaluation configuration
QUESTIONS_PATH = os.path.join(
    PROJECT_ROOT,
    "tools/arqa/evaluation/generated_datasets/structured_mistral_min3.jsonl",
)
LOG_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/hybrid/logs")
os.makedirs(LOG_DIR, exist_ok=True)

EMBEDDING_MODEL_PATH = "intfloat/multilingual-e5-large"
ESSENTIAL_INDEX_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_index.faiss"
)
ESSENTIAL_MAPPING_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_mapping.json"
)
ESSENTIAL_CACHE_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_cache.json"
)
CHUNKS_INDEX_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_index.faiss"
)
CHUNKS_MAPPING_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_mapping.json"
)
CHUNKS_CACHE_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_cache.json"
)

parser = argparse.ArgumentParser(
    description="Evaluate the hierarchical FAISS retriever"
)
parser.add_argument("--doc_top_n", type=int, default=10,
                    help="Number of documents retrieved from essential index")
parser.add_argument("--chunk_top_k", type=int, default=50,
                    help="Number of chunks retrieved from selected documents")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

run_name = (
    f"hierarchical_faiss_eval_doc{args.doc_top_n}_chunk{args.chunk_top_k}"
)
LOG_JSON = os.path.join(LOG_DIR, f"{run_name}.json")
LOG_MD = os.path.join(LOG_DIR, f"{run_name}.md")

# Load retriever
embedding_model = EmbeddingModel(EMBEDDING_MODEL_PATH, args.device, 512)
retriever = HierarchicalFaissSearch(embedding_model, verbose=False)
retriever.load_indices(
    ESSENTIAL_INDEX_PATH,
    ESSENTIAL_MAPPING_PATH,
    ESSENTIAL_CACHE_PATH,
    CHUNKS_INDEX_PATH,
    CHUNKS_MAPPING_PATH,
    CHUNKS_CACHE_PATH,
)

# Load questions
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

# Evaluate each question
results = []
for q in questions:
    qid = q["id"]
    question = q["question"]
    expected_ids = [normalize_doc_id(doc) for doc in q["relevant_doc_ids"]]

    retrieved_chunks = retriever.hierarchical_search(
        question, top_documents=args.doc_top_n, top_chunks=args.chunk_top_k
    )
    retrieved_ids = [
        normalize_doc_id(chunk["metadata"]["document_id"])
        for chunk in retrieved_chunks
    ]

    results.append(
        {
            "id": qid,
            "question": question,
            "relevant_doc_ids": expected_ids,
            "retrieved_doc_ids": retrieved_ids,
            "precision@k": precision_at_k(retrieved_ids, expected_ids),
            "recall@k": recall_at_k(retrieved_ids, expected_ids),
            "hit@k": hit_at_k(retrieved_ids, expected_ids),
            "mrr": mrr(retrieved_ids, expected_ids),
            "normalized_precision@k": normalized_precision_at_k(
                retrieved_ids, expected_ids, k=args.chunk_top_k
            ),
            "n_relevant_retrieved": n_relevant_retrieved(
                retrieved_ids, expected_ids
            ),
        }
    )

# Aggregate metrics
summary = defaultdict(list)
for r in results:
    for m in [
        "precision@k",
        "recall@k",
        "hit@k",
        "mrr",
        "normalized_precision@k",
        "n_relevant_retrieved",
    ]:
        summary[m].append(r[m])

global_metrics = {
    "mean_precision@k": round(np.mean(summary["precision@k"]), 4),
    "mean_normalized_precision@k": round(
        np.mean(summary["normalized_precision@k"]), 4
    ),
    "mean_recall@k": round(np.mean(summary["recall@k"]), 4),
    "mean_hit@k": round(np.mean(summary["hit@k"]), 4),
    "mean_mrr": round(np.mean(summary["mrr"]), 4),
    "mean_n_relevant_retrieved": round(
        np.mean(summary["n_relevant_retrieved"]), 2
    ),
    "questions_evaluated": len(results),
    "doc_top_n": args.doc_top_n,
    "chunk_top_k": args.chunk_top_k,
    "device": args.device,
}

# Save JSON log
with open(LOG_JSON, "w", encoding="utf-8") as f:
    json.dump({"summary": global_metrics, "results": results}, f, indent=2, ensure_ascii=False)

# Save Markdown summary
with open(LOG_MD, "w", encoding="utf-8") as f:
    f.write(f"# Hierarchical FAISS Evaluation - {run_name}\n\n")
    f.write("## Configuration\n")
    f.write(f"- Doc top-N: {args.doc_top_n}\n")
    f.write(f"- Chunk top-K: {args.chunk_top_k}\n")
    f.write(f"- Device: {args.device}\n\n")
    f.write("## Global Results\n\n")
    for k, v in global_metrics.items():
        f.write(f"- **{k}**: {v}\n")
    f.write("\n---\n## Sample Questions (first 5)\n\n")
    for r in results[:5]:
        f.write(f"- **Question:** {r['question']}\n")
        f.write(f"  - Relevant doc IDs: {r['relevant_doc_ids']}\n")
        f.write(f"  - Retrieved doc IDs: {r['retrieved_doc_ids']}\n")
        f.write(
            f"  - P@{args.chunk_top_k}: {r['precision@k']}, R@{args.chunk_top_k}: {r['recall@k']}, MRR: {r['mrr']}\n\n"
        )

print("\nEvaluation completed.")
for k, v in global_metrics.items():
    print(f"{k}: {v}")
print(f"\nLogs saved to:\n- JSON: {LOG_JSON}\n- Markdown: {LOG_MD}")
