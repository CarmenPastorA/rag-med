# tools/rag_orchestrator.py
import os
import argparse
import sys

# Config paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tools.rag_pipeline import run_rag_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--retriever", type=str, default="late+fallback")
    parser.add_argument("--bm25_k", type=int, default=30)
    parser.add_argument("--faiss_k", type=int, default=15)
    parser.add_argument("--max_docs", type=int, default=10)
    args = parser.parse_args()

    result = run_rag_pipeline(
        question=args.question,
        retriever=args.retriever,
        bm25_k=args.bm25_k,
        faiss_k=args.faiss_k,
        max_docs=args.max_docs,
    )

    print("\n--- Result ---")
    print(f"Question: {result['question']}")
    print(f"Answer:\n{result['generated_answer']}")
    print(f"\nRetrieved Docs: {result['retrieved_doc_ids']}")
    print(f"Context Length: {result['context_length']} characters")


