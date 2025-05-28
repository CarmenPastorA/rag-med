# rag_orchestrator.py
import argparse
import os
import sys

# Config paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tools.retrievers import (
    get_bm25_results,
    get_faiss_results,
    get_late_fusion_with_fallback
)
from tools.reader import generate_answer_with_vllm

def run_rag_pipeline(
    question: str,
    retriever_mode: str = "bm25",
    max_docs: int = 10,
    bm25_k: int = 50,
    bm25_score_threshold: float = 8.0,
    faiss_k: int = 50,
    fallback_bm25_k: int = 10,
    fallback_chunks_per_doc: int = 2,
    model: str = "mistralai/Mistral-7B-Instruct-v0.2",
    device: str = "cuda",
    verbose: bool = True
) -> dict:
    """
    Executes a RAG pipeline using BM25, FAISS, or hybrid late+fallback retrieval.

    Args:
        question (str): Natural language query.
        retriever_mode (str): One of 'bm25', 'faiss', or 'late+fallback'.
        max_docs (int): Maximum number of unique doc_ids to include in context.
        bm25_k (int): Number of top BM25 documents to retrieve.
        bm25_score_threshold (float): Minimum BM25 score to keep.
        faiss_k (int): Number of top FAISS chunks to retrieve.
        fallback_bm25_k (int): Max number of BM25-only docs for fallback.
        fallback_chunks_per_doc (int): Number of top chunks per fallback doc.
        model (str): LLM model to use.
        device (str): Device for embedding ('cuda' or 'cpu').
        verbose (bool): Print diagnostics.
    """
    retrieved_docs = []

    if retriever_mode == "bm25":
        print("[🔎] Running BM25 retrieval...")
        doc_ids = get_bm25_results(question, top_k=bm25_k, score_threshold=bm25_score_threshold)
        retrieved_docs = [{"doc_id": doc_id, "content": ""} for doc_id in doc_ids]

    elif retriever_mode == "faiss":
        print("[🔎] Running FAISS retrieval...")
        retrieved_docs = get_faiss_results(question, top_k=faiss_k, device=device)

    elif retriever_mode == "late+fallback":
        print("[🔎] Running Late Fusion + Semantic Fallback (BM25)...")
        retrieved_docs = get_late_fusion_with_fallback(
            question=question,
            bm25_top_n=bm25_k,
            bm25_fallback_top_n=fallback_bm25_k,
            faiss_top_k=faiss_k,
            fallback_chunk_limit=fallback_chunks_per_doc,
            device=device,
            verbose=verbose
        )

    else:
        raise ValueError(f"Unsupported retriever mode: {retriever_mode}")

    # Deduplicate and truncate by document ID
    seen = set()
    unique_docs = []
    for doc in retrieved_docs:
        if doc["doc_id"] not in seen:
            seen.add(doc["doc_id"])
            unique_docs.append(doc)
        if len(unique_docs) >= max_docs:
            break

    context = "\n\n".join(doc["content"] for doc in unique_docs if doc["content"].strip())
    print(f"[ℹ️] Final context built from {len(unique_docs)} documents. Length: {len(context)} characters.")

    print("\n📚 Context passed to the LLM:\n")
    print(context)

    print("[🧠] Generating answer using vLLM...")
    #answer = generate_answer_with_vllm(question=question, context=context, model=model)
    answer = generate_answer_with_vllm(
        question=f"{question}\n",
        context=context,
        model=model
    )

    return {
        "question": question,
        "retrieved_docs": unique_docs,
        "answer": answer
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG pipeline over veterinary data.")
    parser.add_argument("--question", type=str, required=True, help="Question to process")
    parser.add_argument("--retriever", type=str, choices=["bm25", "faiss", "late+fallback"], default="bm25", help="Retriever strategy")
    parser.add_argument("--max_docs", type=int, default=10, help="Maximum number of unique documents to include in context")
    parser.add_argument("--bm25_k", type=int, default=50, help="Number of BM25 documents to retrieve")
    parser.add_argument("--bm25_score_threshold", type=float, default=8.0, help="Minimum BM25 score to accept")
    parser.add_argument("--faiss_k", type=int, default=50, help="Number of FAISS chunks to retrieve")
    parser.add_argument("--fallback_bm25_k", type=int, default=10, help="Number of BM25-only documents to fallback on")
    parser.add_argument("--fallback_chunks_per_doc", type=int, default=2, help="Chunks per BM25 fallback document")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="vLLM model to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device for embeddings ('cuda' or 'cpu')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose diagnostics")

    args = parser.parse_args()

    result = run_rag_pipeline(
        question=args.question,
        retriever_mode=args.retriever,
        max_docs=args.max_docs,
        bm25_k=args.bm25_k,
        bm25_score_threshold=args.bm25_score_threshold,
        faiss_k=args.faiss_k,
        fallback_bm25_k=args.fallback_bm25_k,
        fallback_chunks_per_doc=args.fallback_chunks_per_doc,
        model=args.model,
        device=args.device,
        verbose=args.verbose
    )

    print("\n📌 Question:")
    print(result["question"])

    print("\n📄 Retrieved Documents:")
    for doc in result["retrieved_docs"]:
        print(f"- {doc['doc_id']}: {doc['content'][:150].strip()}...")

    print("\n💬 Final Answer:")
    print(result["answer"])


