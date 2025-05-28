# test_orchestrator.py

import os
import sys

# Config paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from rag_orchestrator import run_rag_pipeline

if __name__ == "__main__":
    question = "¿Qué medicamentos se pueden utilizar en cerdos para tratar coccidiosis?"

    result = run_rag_pipeline(
        question=question,
        retriever_mode="late+fallback",
        max_docs=10,                 # max unique documents to send to LLM
        bm25_k=30,                   # top N docs from BM25
        bm25_score_threshold=8.0,   # minimum BM25 score to keep a document
        faiss_k=15,                 # top K chunks from FAISS
        fallback_bm25_k=5,          # top BM25-only docs for fallback
        fallback_chunks_per_doc=2,  # top N chunks per fallback doc
        model="mistralai/Mistral-7B-Instruct-v0.2",
        device="cuda",
        verbose=True
    )

    print("\n📌 Question:")
    print(result["question"])

    #print("\n📄 Retrieved Documents:")
    #for doc in result["retrieved_docs"]:
    #    print(f"- {doc['doc_id']}: {doc['content'][:150].strip()}...")

    print("\n🗣️ Final Answer:")
    print(result["answer"])

