# === generate_answers_from_dataset.py ===

import os
import json
import argparse
from tqdm import tqdm
import sys

# Config paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tools.rag_pipeline import run_rag_pipeline

DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets/structured_mistral_min3.jsonl")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets/answers_mistral.jsonl")

def main(args):
    input_path = args.input
    output_path = args.output

    with open(input_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
        if args.limit is not None:
            questions = questions[:args.limit]

    # Configuración específica por tipo de retriever
    retriever_kwargs = {
        "bm25_k": args.bm25_k,
        "faiss_k": args.faiss_k,
        "max_docs": args.max_docs,
    }

    if args.retriever == "late":
        retriever_kwargs["fusion_top_k"] = args.fusion_top_k
        retriever_kwargs["alpha"] = args.alpha

    with open(output_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(questions, desc="Generating answers"):
            question = sample["question"]
            try:
                result = run_rag_pipeline(
                    question=question,
                    retriever=args.retriever,
                    **retriever_kwargs
                )
                output = {
                    "question": question,
                    "generated_answer": result["generated_answer"],
                    "retrieved_doc_ids": result["retrieved_doc_ids"],
                    "relevant_doc_ids": sample.get("relevant_doc_ids", []),
                    "context_length": result["context_length"],
                    "retriever_config": result["retriever_config"],
                    "level": sample.get("level", None)
                }
                fout.write(json.dumps(output, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"❌ Error processing question: {question}\n{e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Path to the input .jsonl with questions")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Path to save the output .jsonl with answers")
    parser.add_argument("--limit", type=int, default=None,
                        help="Number of questions to process")

    parser.add_argument("--retriever", type=str, default="late+fallback",
                        help="Retrieval strategy: bm25, faiss, late, late+fallback")
    parser.add_argument("--bm25_k", type=int, default=30)
    parser.add_argument("--faiss_k", type=int, default=15)
    parser.add_argument("--max_docs", type=int, default=10)

    # Params para "late"
    parser.add_argument("--fusion_top_k", type=int, default=10,
                        help="Number of docs to return in late fusion")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for BM25 (alpha) in late fusion")

    args = parser.parse_args()
    main(args)

