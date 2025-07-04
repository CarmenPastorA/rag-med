# tools/rag_pipeline.py

import os
import sys
import tiktoken

# Config paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tools.retrievers import (
    get_bm25_results,
    get_faiss_results,
    get_late_fusion_results,
    get_late_fusion_with_fallback_optimized
)
from tools.reader import get_answer_from_context

def run_rag_pipeline(
    question: str,
    retriever: str = "late",
    bm25_k: int = 30,
    faiss_k: int = 10,
    max_docs: int = 10,
    **kwargs
) -> dict:
    """
    Executes a RAG pipeline using the specified retriever.
    Supports 'bm25', 'faiss', 'late', 'late+fallback'.
    """

    # === 1. Document Retrieval ===
    if retriever == "bm25":
        doc_ids = get_bm25_results(question, top_k=bm25_k)
        docs = [{"doc_id": doc_id, "content": ""} for doc_id in doc_ids[:max_docs]]

    elif retriever == "faiss":
        docs = get_faiss_results(question, top_k=faiss_k)[:max_docs]

    elif retriever == "late":
        fusion_top_k = kwargs.get("fusion_top_k", 8)
        alpha = kwargs.get("alpha", 0.5)
        docs = get_late_fusion_results(
            question,
            bm25_top_n=bm25_k,
            faiss_top_k=faiss_k,
            fusion_top_k=fusion_top_k,
            alpha=alpha,
        )

    elif retriever == "late+fallback":
        docs = get_late_fusion_with_fallback_optimized(
            question,
            bm25_top_n=bm25_k,
            faiss_top_k=faiss_k,
            bm25_fallback_top_n=10,
            fallback_chunk_limit=2,
        )

    else:
        raise ValueError(f"Unsupported retriever: {retriever}")

    # === 2. Prepare context ===
    encoding = tiktoken.get_encoding("gpt2")
    MAX_TOTAL_TOKENS = 4096
    MAX_CONTEXT_TOKENS = 3500  # deja margen para system y prompt del usuario

    context_parts = []
    total_tokens = 0
    doc_ids = []

    for doc in docs:
        chunk = doc["content"]
        chunk_tokens = len(encoding.encode(chunk))
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            break
        context_parts.append(chunk)
        doc_ids.append(doc["doc_id"])
        total_tokens += chunk_tokens

    context = "\n\n".join(context_parts)

    # === 3. Reader: generate answer ===
    generated_answer = get_answer_from_context(question, context)

    return {
        "question": question,
        "generated_answer": generated_answer,
        "retrieved_doc_ids": doc_ids,
        "context_length": len(context),
        "retriever_config": retriever,
    }
