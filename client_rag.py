# client_rag.py

import os
import sys

# Config paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(PROJECT_ROOT)

from tools.rag_orchestrator import run_rag_pipeline

question = "¿Qué medicamentos se pueden utilizar en cerdos para tratar coccidiosis?"

result = run_rag_pipeline(
    question=question,
    retriever_mode="late+fallback",
    max_docs=10,
    bm25_k=30,
    faiss_k=15,
    fallback_bm25_k=5,
    fallback_chunks_per_doc=2,
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda",
    verbose=True
)

print("\n📌 Pregunta:")
print(result["question"])

print("\n📄 Documentos recuperados:")
for doc in result["retrieved_docs"]:
    print(f"- {doc['doc_id']}: {doc['content'][:150].strip()}...")

print("\n🧠 Respuesta generada:")
print(result["answer"])
