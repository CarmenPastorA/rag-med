# === evaluate_hybrid_retriever.py ===

"""
This script evaluates hybrid retrieval strategies (early or late fusion) over a set of synthetic QA pairs.

It supports two fusion modes:
- early: BM25 filters the documents and FAISS reranks the relevant chunks (slow but precise).
- late: BM25 and FAISS both retrieve independently from the full index and are merged by score (faster).

Expected arguments:
--mode [early|late]        : Fusion strategy to evaluate.
--bm25_top_n               : Number of top documents to retrieve from BM25.
--faiss_top_k              : Number of top chunks to retrieve from FAISS (used in late fusion).
--final_top_k              : Final number of results used for metric evaluation (i.e., Precision@k).
--alpha                    : Score weight (0 = pure FAISS, 1 = pure BM25).
--device                   : Device used for FAISS embeddings (default: cuda).

Logs and metrics are saved to:
  tools/arqa/evaluation/hybrid/logs/hybrid_eval_<mode>_k<topk>.json
  tools/arqa/evaluation/hybrid/logs/hybrid_eval_<mode>_k<topk>.md
"""

import os
import sys
import json
from collections import defaultdict
import numpy as np
import argparse

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.append(PROJECT_ROOT)

from tools.retrievers import get_early_fusion_results, get_late_fusion_results
from tools.arqa.evaluation.metrics import (
    precision_at_k, recall_at_k, hit_at_k, mrr,
    normalized_precision_at_k, n_relevant_retrieved
)
from shared.veterinary_utils.utils import normalize_doc_id

# === Configuration ===
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets/structured_mistral_min3.jsonl")
LOG_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/hybrid/logs")
os.makedirs(LOG_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description="Evaluate hybrid retriever")
parser.add_argument("--mode", type=str, choices=["early", "late"], required=True, help="Fusion strategy to evaluate")
parser.add_argument("--bm25_top_n", type=int, default=50)
parser.add_argument("--faiss_top_k", type=int, default=50)
parser.add_argument("--final_top_k", type=int, default=50)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

run_name = f"hybrid_eval_{args.mode}_k{args.final_top_k}_apha{args.alpha}"
LOG_JSON = os.path.join(LOG_DIR, f"{run_name}.json")
LOG_MD = os.path.join(LOG_DIR, f"{run_name}.md")

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

results = []
fallback_count = 0
fallback_questions = []

for q in questions:
    qid = q["id"]
    query = q["question"]
    expected_ids = [normalize_doc_id(doc_id) for doc_id in q["relevant_doc_ids"]]

    if args.mode == "early":
        retrieved_chunks, used_fallback = get_early_fusion_results(
            question=query,
            bm25_top_n=args.bm25_top_n,
            bm25_score_threshold=8.0,
            faiss_top_k=args.final_top_k,
            device=args.device
        )
        if used_fallback:
            fallback_count += 1
            fallback_questions.append(qid)
        retrieved_ids = [normalize_doc_id(doc["doc_id"]) for doc in retrieved_chunks]

    elif args.mode == "late":
        retrieved_docs = get_late_fusion_results(
            question=query,
            bm25_top_n=args.bm25_top_n,
            faiss_top_k=args.faiss_top_k,
            fusion_top_k=args.final_top_k,
            alpha=args.alpha,
            device=args.device
        )
        retrieved_ids = [normalize_doc_id(doc["doc_id"]) for doc in retrieved_docs]

    results.append({
        "id": qid,
        "question": query,
        "relevant_doc_ids": expected_ids,
        "retrieved_doc_ids": retrieved_ids,
        "precision@k": precision_at_k(retrieved_ids, expected_ids),
        "recall@k": recall_at_k(retrieved_ids, expected_ids),
        "hit@k": hit_at_k(retrieved_ids, expected_ids),
        "mrr": mrr(retrieved_ids, expected_ids),
        "normalized_precision@k": normalized_precision_at_k(retrieved_ids, expected_ids, k=args.final_top_k),
        "n_relevant_retrieved": n_relevant_retrieved(retrieved_ids, expected_ids)
    })

summary = defaultdict(list)
for r in results:
    for m in ["precision@k", "recall@k", "hit@k", "mrr", "normalized_precision@k", "n_relevant_retrieved"]:
        summary[m].append(r[m])

global_metrics = {
    "mean_precision@k": round(np.mean(summary["precision@k"]), 4),
    "mean_normalized_precision@k": round(np.mean(summary["normalized_precision@k"]), 4),
    "mean_recall@k": round(np.mean(summary["recall@k"]), 4),
    "mean_hit@k": round(np.mean(summary["hit@k"]), 4),
    "mean_mrr": round(np.mean(summary["mrr"]), 4),
    "mean_n_relevant_retrieved": round(np.mean(summary["n_relevant_retrieved"]), 2),
    "questions_evaluated": len(results),
    "mode": args.mode,
    "bm25_top_n": args.bm25_top_n,
    "faiss_top_k": args.faiss_top_k,
    "final_top_k": args.final_top_k,
    "alpha": args.alpha,
    "device": args.device,
    "fallback_count": fallback_count,
    "fallback_ratio": round(fallback_count / len(questions), 4),
    "fallback_questions": fallback_questions
}

# Save JSON log
with open(LOG_JSON, "w", encoding="utf-8") as f:
    json.dump({"summary": global_metrics, "results": results}, f, indent=2, ensure_ascii=False)

# Save Markdown summary log
with open(LOG_MD, "w", encoding="utf-8") as f:
    f.write(f"# Hybrid Evaluation Report ({args.mode} fusion)\n\n")
    f.write(f"## Configuration\n")
    f.write(f"- BM25 top-N: {args.bm25_top_n}\n")
    f.write(f"- FAISS top-K: {args.faiss_top_k}\n")
    f.write(f"- Final top-K (for metrics): {args.final_top_k}\n")
    f.write(f"- Alpha (fusion weight): {args.alpha}\n")
    f.write(f"- Device: {args.device}\n")
    f.write(f"\n## Global Metrics\n")
    for k, v in global_metrics.items():
        if k != "fallback_questions":
            f.write(f"- **{k}**: {v}\n")
    f.write("\n---\n## Sample Questions (First 5)\n\n")
    for r in results[:5]:
        f.write(f"- **Question:** {r['question']}\n")
        f.write(f"  - Relevant doc IDs: {r['relevant_doc_ids']}\n")
        f.write(f"  - Retrieved doc IDs: {r['retrieved_doc_ids']}\n")
        f.write(f"  - P@{args.final_top_k}: {r['precision@k']}, R@{args.final_top_k}: {r['recall@k']}, MRR: {r['mrr']}\n\n")

print("\nEvaluation Completed")
for k, v in global_metrics.items():
    if k != "fallback_questions":
        print(f"{k}: {v}")
print(f"\nLogs saved to:\n- JSON: {LOG_JSON}\n- Markdown: {LOG_MD}")


