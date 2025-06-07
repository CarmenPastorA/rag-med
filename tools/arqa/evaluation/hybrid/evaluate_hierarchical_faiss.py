import os
import sys
import json
from datetime import datetime
from collections import defaultdict
import argparse
import numpy as np
import faiss

# === Setup paths ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.append(PROJECT_ROOT)

from shared.veterinary_utils.embedding_model import EmbeddingModel
from tools.arqa.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    mrr,
    normalized_precision_at_k,
    n_relevant_retrieved,
)


def load_mapping(path):
    """Load a JSON mapping using integer keys."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def hierarchical_search(query, embedding_model, top_docs, top_chunks,
                        essential_index, essential_map, chunks_index,
                        chunks_map, chunks_cache):
    """Two stage FAISS search using essential info then subsections."""
    # Embed query
    query_emb = embedding_model.get_embeddings(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )

    # --- Stage 1: document retrieval ---
    doc_limit = min(top_docs, essential_index.ntotal)
    doc_scores, doc_indices = essential_index.search(query_emb, doc_limit)
    doc_ids = []
    for idx in doc_indices[0]:
        info = essential_map.get(int(idx))
        if info:
            doc_ids.append(info.get("document_id"))
    doc_set = set(doc_ids)

    if not doc_ids:
        return []

    # --- Stage 2: chunk retrieval filtered by document id ---
    search_k = min(top_chunks * len(doc_ids) * 2, chunks_index.ntotal)
    chunk_scores, chunk_indices = chunks_index.search(query_emb, search_k)

    results = []
    for score, idx in zip(chunk_scores[0], chunk_indices[0]):
        info = chunks_map.get(int(idx))
        if not info:
            continue
        doc_id = info["metadata"].get("document_id")
        if doc_id in doc_set:
            chunk_id = info.get("chunk_id")
            text = chunks_cache.get(chunk_id, {}).get("text", "")
            results.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": float(score),
                "text": text,
            })
            if len(results) >= top_chunks:
                break
    return results


def unique_doc_ids(results, limit):
    """Extract unique document ids from ranked chunk results."""
    seen = set()
    ordered = []
    for r in results:
        d = r["doc_id"]
        if d not in seen:
            seen.add(d)
            ordered.append(d)
        if len(ordered) >= limit:
            break
    return ordered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hierarchical FAISS retriever")
    parser.add_argument("--top_docs", type=int, default=10, help="Top documents from essential stage")
    parser.add_argument("--top_chunks", type=int, default=50, help="Top chunks from subsection stage")
    parser.add_argument("--final_top_k", type=int, default=50, help="Number of unique documents used for metrics")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    EMBEDDING_MODEL_PATH = "intfloat/multilingual-e5-large"
    ESSENTIAL_INDEX = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_index.faiss")
    ESSENTIAL_MAP = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_mapping.json")
    CHUNKS_INDEX = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_index.faiss")
    CHUNKS_MAP = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_mapping.json")
    CHUNKS_CACHE = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_cache.json")
    QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets/structured_mistral_min3.jsonl")

    run_name = f"hierarchical_faiss_eval_k{args.final_top_k}"
    LOG_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/hybrid/logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_JSON = os.path.join(LOG_DIR, f"{run_name}.json")
    LOG_MD = os.path.join(LOG_DIR, f"{run_name}.md")

    # --- Load resources ---
    embedding_model = EmbeddingModel(EMBEDDING_MODEL_PATH, args.device, 512)

    essential_index = faiss.read_index(ESSENTIAL_INDEX)
    essential_map = load_mapping(ESSENTIAL_MAP)

    chunks_index = faiss.read_index(CHUNKS_INDEX)
    chunks_map = load_mapping(CHUNKS_MAP)
    with open(CHUNKS_CACHE, "r", encoding="utf-8") as f:
        chunks_cache = json.load(f)

    # --- Load evaluation questions ---
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    results = []
    for q in questions:
        qid = q["id"]
        query = q["question"]
        relevant_ids = q["relevant_doc_ids"]

        chunk_results = hierarchical_search(
            query,
            embedding_model,
            args.top_docs,
            args.top_chunks,
            essential_index,
            essential_map,
            chunks_index,
            chunks_map,
            chunks_cache,
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
    }

    with open(LOG_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": global_metrics, "results": results}, f, indent=2, ensure_ascii=False)

    with open(LOG_MD, "w", encoding="utf-8") as f:
        f.write(f"# Hierarchical FAISS Evaluation - {run_name}\n\n")
        f.write("## Configuration\n")
        f.write(f"- Top documents: {args.top_docs}\n")
        f.write(f"- Top chunks: {args.top_chunks}\n")
        f.write(f"- Final top-K: {args.final_top_k}\n")
        f.write(f"- Device: {args.device}\n\n")
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

