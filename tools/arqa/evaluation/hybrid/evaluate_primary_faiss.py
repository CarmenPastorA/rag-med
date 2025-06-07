"""evaluate_primary_faiss.py
----------------------------

Evaluate only the **first** stage of the hierarchical FAISS retriever on a
set of synthetic question/answer pairs.

This script loads the essential-information FAISS index and measures how
well it retrieves relevant documents without executing the second stage of
chunk retrieval.  The goal is to assess the standalone performance of the
primary (document-level) index.
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

EMBEDDING_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models/multilingual-e5-large-local"
)
ESSENTIAL_INDEX_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_index.faiss"
)
ESSENTIAL_MAPPING_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_mapping.json"
)
ESSENTIAL_CACHE_PATH = os.path.join(
    PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_cache.json"
)
# Paths for the second-stage chunk index are kept for reference but not used in
# this evaluation.  They can be enabled if chunk retrieval needs to be tested
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
    description="Evaluate the primary (document-level) FAISS index"
)
parser.add_argument(
    "--doc_top_n",
    type=int,
    default=50,
    help="Number of documents retrieved from the essential index",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device used for the embedding model (cpu or cuda)",
)
args = parser.parse_args()

run_name = f"primary_faiss_k{args.doc_top_n}"
LOG_JSON = os.path.join(LOG_DIR, f"{run_name}.json")
LOG_MD = os.path.join(LOG_DIR, f"{run_name}.md")

# Load retriever
embedding_model = EmbeddingModel(EMBEDDING_MODEL_PATH, args.device, 512)
retriever = HierarchicalFaissSearch(embedding_model, verbose=False)

# Only load the essential-information index.  This evaluation measures how
# well the first retrieval stage works in isolation.
retriever.load_essential_index(
    ESSENTIAL_INDEX_PATH,
    ESSENTIAL_MAPPING_PATH,
    ESSENTIAL_CACHE_PATH,
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

    # Retrieve only document identifiers from the essential index
    retrieved_doc_ids = retriever.get_relevant_document_ids(
        question, top_k=args.doc_top_n
    )
    retrieved_ids = [normalize_doc_id(doc_id) for doc_id in retrieved_doc_ids]

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
                retrieved_ids, expected_ids, k=args.doc_top_n
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
    "device": args.device,
}

# Save JSON log
with open(LOG_JSON, "w", encoding="utf-8") as f:
    json.dump({"summary": global_metrics, "results": results}, f, indent=2, ensure_ascii=False)

# Save Markdown summary
with open(LOG_MD, "w", encoding="utf-8") as f:
    f.write(f"# Primary FAISS Evaluation - {run_name}\n\n")
    f.write("## Configuration\n")
    f.write(f"- Doc top-N: {args.doc_top_n}\n")
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
            f"  - P@{args.doc_top_n}: {r['precision@k']}, R@{args.doc_top_n}: {r['recall@k']}, MRR: {r['mrr']}\n\n"
        )

print("\nEvaluation completed.")
for k, v in global_metrics.items():
    print(f"{k}: {v}")
print(f"\nLogs saved to:\n- JSON: {LOG_JSON}\n- Markdown: {LOG_MD}")
