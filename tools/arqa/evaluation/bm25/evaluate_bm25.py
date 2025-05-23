"""
This script evaluates a BM25-based retriever on a synthetic QA dataset.

It performs the following steps:
1. Loads a precomputed BM25 index using `BM25Search`.
2. Loads a synthetic dataset of queries and their known relevant document IDs.
3. Runs each query through the retriever and compares retrieved results against the ground truth.
4. Computes metrics
5. Optionally:
   - Saves failed queries to a CSV for review.
   - Generates a Markdown report (`bm25_eval_report.md`) listing each query, its expected answers, results, and whether it was a hit.

Metrics implemented:
--------------------
1. Recall@k (binary): A query counts as a hit (1) if **at least one** relevant document is retrieved
   within the top-k results. It does not consider how many were retrieved or their rank.

2. Precision@k: For each query, calculates the proportion of the top-k retrieved documents that are
   actually relevant. This measures accuracy of the retrieval.

3. Recall@k (per-document): The fraction of relevant documents that were retrieved in the top-k,
   averaged across all queries. This is stricter than binary Recall@k.

Example:
---------
Query has 3 relevant docs: [A, B, C]
Retrieved top-5: [B, X, Y, A, Z]

→ Binary Recall@5 = 1 (B or A were found)
→ Precision@5 = 2 / 5 = 0.4
→ Recall@5 (per-doc) = 2 / 3 ≈ 0.667   

Expected input format (.jsonl):
{
  "question": "¿Qué medicamentos se usan para infección entérica producida por Escherichia coli en caprinos?",
  "relevant_doc_ids": ["2321 ESP", "2082 ESP", "3934 ESP"]
}

"""

import json
from tqdm import tqdm
from tabulate import tabulate
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import os

import sys
from pathlib import Path


ARQA_TOOLS_ARQA = Path(__file__).resolve().parents[3]
sys.path.append(str(ARQA_TOOLS_ARQA))

from bm25_search import BM25Search

def evaluate_bm25(bm25: BM25Search, dataset_path: str, k: int = 10, print_table: bool = True) -> Tuple[float, List[dict]]:
    """
    Evaluate the BM25 retriever on a synthetic QA dataset.

    Returns:
        recall_binary: Recall@k (binary): % of queries with at least 1 hit
        all_results: List of per-query result dictionaries
    """
    total = 0
    recall_hits = 0
    all_results = []

    precision_scores = []
    recall_doc_scores = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Evaluating top-{k} retrieval"):
        sample = json.loads(line)
        question = sample["question"]
        relevant_ids = set(sample["relevant_doc_ids"])

        try:
            retrieved_ids_raw = bm25.search(question, k=k)
            # Normalize BM25 IDs to match format
            retrieved_ids = [doc_id.replace("FT_", "").replace("_ESP", " ESP") for doc_id in retrieved_ids_raw]
            retrieved_set = set(retrieved_ids)

            # Binary hit
            hit = bool(relevant_ids & retrieved_set)
            if hit:
                recall_hits += 1

            # Precision@k: proportion of retrieved docs that are relevant
            precision_k = len(relevant_ids & retrieved_set) / k
            precision_scores.append(precision_k)

            # Recall@k per query: proportion of relevant docs that were retrieved
            if relevant_ids:
                recall_k = len(relevant_ids & retrieved_set) / len(relevant_ids)
            else:
                recall_k = 0.0
            recall_doc_scores.append(recall_k)

        except Exception as e:
            print(f"Error with question: {question}\n{e}")
            retrieved_ids = []
            hit = False
            precision_scores.append(0.0)
            recall_doc_scores.append(0.0)

        all_results.append({
            "question": question,
            "relevant": list(relevant_ids),
            "retrieved": list(retrieved_ids),
            "hit": hit,
            "precision_k": precision_scores[-1],
            "recall_k_doc": recall_doc_scores[-1]
        })
        total += 1

    # Metrics
    recall_binary = recall_hits / total if total else 0.0
    mean_precision = sum(precision_scores) / total if total else 0.0
    mean_recall_doc = sum(recall_doc_scores) / total if total else 0.0

    return recall_binary, mean_precision, mean_recall_doc, all_results


def save_results_markdown(results: List[dict], output_path: Path, recall: float, precision: float, recall_doc: float, k: int) -> None:
    """
    Save the evaluation results as a Markdown file with per-query and global metrics.

    Args:
        results: List of per-query result dictionaries.
        output_path: Path to the Markdown file.
        recall: Binary Recall@k.
        precision: Mean Precision@k.
        recall_doc: Mean Document-level Recall@k.
        k: Top-k retrieved documents per query.
    """
    lines = []
    lines.append(f"# BM25 Evaluation Report\n")
    lines.append(f"**Top-k:** {k}\n")
    lines.append(f"**Binary Recall@{k}:** {recall:.3f}  \n")
    lines.append(f"**Mean Precision@{k}:** {precision:.3f}  \n")
    lines.append(f"**Mean Document-level Recall@{k}:** {recall_doc:.3f}\n")
    lines.append("\n---\n")

    for i, row in enumerate(results, 1):
        question = row["question"]
        relevant = ", ".join(row["relevant"])
        retrieved = ", ".join(row["retrieved"])
        hit = "✅" if row["hit"] else "❌"
        prec_k = f"{row.get('precision_k', 0.0):.2f}"
        recall_k_doc = f"{row.get('recall_k_doc', 0.0):.2f}"

        lines.append(f"## Query {i}\n")
        lines.append(f"**Question:** {question}")
        lines.append(f"\n**Relevant Doc IDs:** {relevant}")
        lines.append(f"\n**Top-{k} Retrieved:** {retrieved}")
        lines.append(f"\n**Precision@{k}:** {prec_k}")
        lines.append(f"\n**Recall@{k} (per-doc):** {recall_k_doc}")
        lines.append(f"\n**Hit:** {hit}\n")
        lines.append("\n---\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nMarkdown report saved to: {output_path}")



def save_failed_queries(results: List[dict], output_path: Path) -> None:
    """
    Print and save the failed queries for inspection.

    Args:
        results: Output from evaluate_bm25.
        output_path: Path to save the failed queries (CSV format).
    """
    df = pd.DataFrame(results)
    df_failed = df[df["hit"] == False]

    if output_path:
        df_failed.to_csv(output_path, index=False)
        print(f"\Failed results saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate BM25 index using a synthetic QA dataset")

    parser.add_argument("--bm25-dir", type=str, default="data/posteriori_resources/bm25_stuffs",
                        help="Directory containing the saved BM25 index files")
    parser.add_argument("--dataset", type=str, default="tools/arqa/tests/retriever/bm25/synthetic_validation_dataset.jsonl",
                        help="Path to the synthetic validation dataset (.jsonl)")
    parser.add_argument("--fasttext-model", type=str, default="../../../models/lang_model",
                        help="Optional path to FastText language model")
    parser.add_argument("--stopwords", type=str, default="data/priori_resources/stopwords.txt",
                        help="Optional path to stopwords list")
    parser.add_argument("--preserve-words", type=str, default="data/priori_resources/preserve_words.txt",
                        help="Optional path to file with words to preserve")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of documents to retrieve per query")
    parser.add_argument("--save-results", type=str, default="tools/arqa/tests/retriever/bm25/bm25_eval_results.csv",
                        help="Optional path to save the evaluation results as CSV")


    args = parser.parse_args()

    # Initialize BM25Search instance
    bm25 = BM25Search(
        directory=args.bm25_dir,
        fasttext_model_path=args.fasttext_model,
        stopwords_path=args.stopwords,
        preserve_words_path=args.preserve_words,
        verbose=True
    )

    # Run evaluation
    recall, precision, recall_doc, results = evaluate_bm25(bm25, args.dataset, k=args.k)

    # Save Markdown report
    md_output = Path("tools/arqa/tests/retriever/bm25/bm25_eval_report.md")
    save_results_markdown(results, md_output, recall, precision, recall_doc, args.k)

    # Save results (optional)
    if args.save_results:
        save_failed_queries(results, Path(args.save_results))
