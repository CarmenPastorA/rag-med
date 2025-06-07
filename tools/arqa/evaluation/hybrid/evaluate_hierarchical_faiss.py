"""Evaluation script for hierarchical FAISS retrieval.

This module evaluates a two-stage retriever based on
``HierarchicalFaissSearch``.  It loads the required indices and runs the search
for a set of questions.  Results are measured using common information
retrieval metrics and written both as JSON and Markdown reports.
"""

import os
import sys
import json
from collections import defaultdict
import argparse
import numpy as np

# === Setup paths ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.append(PROJECT_ROOT)

from shared.veterinary_utils.embedding_model import EmbeddingModel
from tools.arqa.faiss_search import HierarchicalFaissSearch
from tools.arqa.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    mrr,
    normalized_precision_at_k,
    n_relevant_retrieved,
)


def search_chunks(
    query: str,
    searcher: HierarchicalFaissSearch,
    top_documents: int,
    top_chunks: int,
    with_context: bool = False,
) -> list[dict]:
    """Retrieve ranked chunks using the hierarchical searcher.

    Args:
        query: Question to embed and search for.
        searcher: Initialized hierarchical search object with loaded indices.
        top_documents: Number of documents retrieved in the first stage.
        top_chunks: Number of chunks retrieved from those documents.
        with_context: Whether to include hierarchical context for each chunk.

    Returns:
        Ranked list of chunk dictionaries as returned by the searcher.
    """

    if with_context:
        return searcher.search_with_context(
            query=query,
            top_documents=top_documents,
            top_chunks=top_chunks,
        )
    return searcher.search(
        query=query,
        top_documents=top_documents,
        top_chunks=top_chunks,
    )


def unique_doc_ids(results: list[dict], limit: int) -> list[str]:
    """Get a list of unique document identifiers from ranked chunks.

    Args:
        results: List of chunk dictionaries returned by the searcher.
        limit: Maximum number of unique document ids to return.

    Returns:
        Ordered list of unique document ids found in ``results``.
    """

    seen = set()
    ordered: list[str] = []
    for r in results:
        doc_id = r["metadata"].get("document_id", r.get("chunk_id", "").split("@")[0])
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
        if len(ordered) >= limit:
            break
    return ordered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the hierarchical FAISS retriever",
    )
    # ``top_docs`` controls how many documents are retrieved in the first stage
    # of the hierarchical search.
    parser.add_argument(
        "--top_docs",
        type=int,
        default=10,
        help="Number of documents retrieved in the first stage",
    )
    # ``top_chunks`` is the number of ranked chunks returned from those
    # documents. Higher values may increase recall.
    parser.add_argument(
        "--top_chunks",
        type=int,
        default=50,
        help="Number of chunks retrieved from the selected documents",
    )
    # ``final_top_k`` is the number of unique document identifiers used to
    # compute metrics. Duplicated documents are collapsed before evaluation.
    parser.add_argument(
        "--final_top_k",
        type=int,
        default=50,
        help="Unique documents used for evaluation",
    )
    # ``device`` selects the hardware for the embedding model (cuda or cpu).
    parser.add_argument("--device", type=str, default="cuda")
    # ``with_context`` indicates whether ``search_with_context`` should be used
    # to enrich the retrieved chunks with hierarchical information.
    parser.add_argument(
        "--with_context",
        action="store_true",
        help="Return extra context for each retrieved chunk",
    )
    args = parser.parse_args()

    # Paths to all required resources. These locations are project specific and
    # should match those used when building the FAISS indices.
    #EMBEDDING_MODEL_PATH = "intfloat/multilingual-e5-large"
    EMBEDDING_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/multilingual-e5-large-local")
    ESSENTIAL_INDEX = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_index.faiss")
    ESSENTIAL_MAP = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_mapping.json")
    CHUNKS_INDEX = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_index.faiss")
    CHUNKS_MAP = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_mapping.json")
    CHUNKS_CACHE = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_cache.json")
    DOCS_DIR = os.path.join(PROJECT_ROOT, "data/posteriori_resources/processed_json")
    QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets/structured_mistral_min3.jsonl")

    run_name = f"hierarchical_faiss_eval_k{args.final_top_k}"
    if args.with_context:
        run_name += "_context"
    LOG_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/hybrid/logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_JSON = os.path.join(LOG_DIR, f"{run_name}.json")
    LOG_MD = os.path.join(LOG_DIR, f"{run_name}.md")

    # --- Load resources ---
    embedding_model = EmbeddingModel(EMBEDDING_MODEL_PATH, args.device, 512)

    searcher = HierarchicalFaissSearch(embedding_model)
    searcher.load_indices(
        essential_index_path=ESSENTIAL_INDEX,
        essential_mapping_path=ESSENTIAL_MAP,
        chunks_index_path=CHUNKS_INDEX,
        chunks_mapping_path=CHUNKS_MAP,
        chunks_cache_path=CHUNKS_CACHE,
        documents_directory=DOCS_DIR,
    )

    # --- Load evaluation questions ---
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    results = []
    for q in questions:
        qid = q["id"]
        query = q["question"]
        relevant_ids = q["relevant_doc_ids"]

        chunk_results = search_chunks(
            query,
            searcher,
            args.top_docs,
            args.top_chunks,
            with_context=args.with_context,
        )
        retrieved_ids = unique_doc_ids(chunk_results, args.final_top_k)

        results.append({
            "id": qid,
            "question": query,
            "relevant_doc_ids": relevant_ids,
            "retrieved_doc_ids": retrieved_ids,
            "precision@k": precision_at_k(retrieved_ids, relevant_ids),
            "recall@k": recall_at_k(retrieved_ids, relevant_ids),
            "hit@k": hit_at_k(retrieved_ids, relevant_ids),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "normalized_precision@k": normalized_precision_at_k(retrieved_ids, relevant_ids, k=args.final_top_k),
            "n_relevant_retrieved": n_relevant_retrieved(retrieved_ids, relevant_ids),
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
        "top_docs": args.top_docs,
        "top_chunks": args.top_chunks,
        "final_top_k": args.final_top_k,
        "device": args.device,
        "with_context": args.with_context,
    }

    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": global_metrics, "results": results}, f, indent=2, ensure_ascii=False)

    with open(LOG_MD, "w", encoding="utf-8") as f:
        f.write(f"# Hierarchical FAISS Evaluation - {run_name}\n\n")
        f.write("## Configuration\n")
        f.write(f"- Top documents: {args.top_docs}\n")
        f.write(f"- Top chunks: {args.top_chunks}\n")
        f.write(f"- Final top-K: {args.final_top_k}\n")
        f.write(f"- Device: {args.device}\n")
        f.write(f"- With context: {args.with_context}\n\n")
        f.write("## Global Metrics\n")
        for k, v in global_metrics.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n---\n## Sample Questions (First 5)\n\n")
        for r in results[:5]:
            f.write(f"- **Question:** {r['question']}\n")
            f.write(f"  - Relevant doc IDs: {r['relevant_doc_ids']}\n")
            f.write(f"  - Retrieved doc IDs: {r['retrieved_doc_ids']}\n")
            f.write(f"  - P@{args.final_top_k}: {r['precision@k']}, R@{args.final_top_k}: {r['recall@k']}, MRR: {r['mrr']}\n\n")

    print("\nEvaluation Completed")
    for k, v in global_metrics.items():
        print(f"{k}: {v}")

