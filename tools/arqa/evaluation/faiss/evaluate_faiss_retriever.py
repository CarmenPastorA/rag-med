# evaluate_faiss_retriever.py

import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import sys

# Add parent directory to path to access shared modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # /tools/arqa/evaluation/faiss
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))  # root of project
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)

from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

from tools.arqa.faiss_search import FaissSearch
from shared.veterinary_utils.embedding_model import EmbeddingModel
from shared.veterinary_utils.jina_embedding_model import JinaEmbeddingModel
from tools.arqa.evaluation.metrics import precision_at_k, recall_at_k, hit_at_k, mrr, normalized_precision_at_k, n_relevant_retrieved

# === Evaluation configuration ===
K = 50
DEVICE = "cuda"
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets/structured_mistral_min3.jsonl")

EMBEDDING_MODEL_PATH = "intfloat/multilingual-e5-large"
INDEX_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/index.faiss")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/mapping.json")
CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks.json")

#EMBEDDING_MODEL_PATH = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
#INDEX_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff_SapBERT_UMLS_2020AB_all_lang_from_XLMR/chunks_index.faiss")
#MAPPING_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff_SapBERT_UMLS_2020AB_all_lang_from_XLMR/chunks_mapping.json")
#CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff_SapBERT_UMLS_2020AB_all_lang_from_XLMR/chunks_cache.json")

#EMBEDDING_MODEL_PATH = "jinaai/jina-embeddings-v3"
#INDEX_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff_jina_embeddings_v3/chunks_index.faiss")
#MAPPING_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff_jina_embeddings_v3/chunks_mapping.json")
#CHUNKS_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff_jina_embeddings_v3/chunks_cache.json")

# Create a unique name for this evaluation run
RUN_NAME = f"faiss_eval_infloat_{datetime.now().strftime('%Y-%m-%d')}_k{K}"
LOG_DIR =  os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/faiss/logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_JSON = os.path.join(LOG_DIR, f"{RUN_NAME}.json")
LOG_MD = os.path.join(LOG_DIR, f"{RUN_NAME}.md")

# === Load FAISS retriever with embedding model ===
embedding_model = EmbeddingModel(EMBEDDING_MODEL_PATH, DEVICE, 512)
#embedding_model = JinaEmbeddingModel(EMBEDDING_MODEL_PATH, DEVICE, 512)
retriever = FaissSearch(embedding_model)
retriever.load_index(INDEX_PATH, MAPPING_PATH, CHUNKS_PATH)

# === Load evaluation questions ===
with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

# === Run retrieval and evaluate per question ===
results = []

# Filter out the 78 questions with 50+ retrieved documents (too general and not clinically relevant)
#questions = [q for q in questions if len(q["relevant_doc_ids"]) <= 50]
for q in questions:
    query_id = q.get("id")
    query = q["question"]
    expected_ids = q["relevant_doc_ids"]

    # Search using FAISS (no context)
    top_k = retriever.search(query, k=K)
    retrieved_ids = [r["metadata"]["document_id"] for r in top_k]

    # Compute metrics
    results.append({
        "id": query_id,
        "question": query,
        "relevant_doc_ids": expected_ids,
        "retrieved_doc_ids": retrieved_ids,
        "precision@k": precision_at_k(retrieved_ids, expected_ids),
        "recall@k": recall_at_k(retrieved_ids, expected_ids),
        "hit@k": hit_at_k(retrieved_ids, expected_ids),
        "mrr": mrr(retrieved_ids, expected_ids),
        "normalized_precision@k": normalized_precision_at_k(retrieved_ids, expected_ids, k=K),
        "n_relevant_retrieved": n_relevant_retrieved(retrieved_ids, expected_ids)
    })

# === Compute global summary ===
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
    "k": K,
    "embedding_model": EMBEDDING_MODEL_PATH,
    "device": DEVICE,
    "run_name": RUN_NAME
}

# === Save raw results to JSON log ===
with open(LOG_JSON, "w", encoding="utf-8") as f:
    json.dump({"summary": global_metrics, "results": results}, f, indent=2, ensure_ascii=False)

# === Write summary Markdown report ===
with open(LOG_MD, "w", encoding="utf-8") as f:
    f.write(f"# FAISS Evaluation Report - {RUN_NAME}\n\n")
    f.write("## Configuration\n")
    f.write(f"- `Note: Improved FAISS index adding 'Nombre Medicamento: ' at the beginning of each chunk`\n")
    f.write(f"- `Top-k`: {K}\n")
    f.write(f"- `Embedding model`: `{EMBEDDING_MODEL_PATH}`\n")
    f.write(f"- `Device`: `{DEVICE}`\n")
    f.write(f"- `Date`: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    f.write("## Global Results\n\n")
    for k, v in global_metrics.items():
        f.write(f"- **{k.replace('_', ' ').capitalize()}**: {v}\n")

    f.write("\n---\n## Sample Questions (Top 5)\n\n")
    for r in results[:5]:
        f.write(f"- **Question:** {r['question']}\n")
        f.write(f"  - Relevant doc IDs: {r['relevant_doc_ids']}\n")
        f.write(f"  - Retrieved doc IDs: {r['retrieved_doc_ids']}\n")
        f.write(f"  - P@{K}: {r['precision@k']}, R@{K}: {r['recall@k']}, MRR: {r['mrr']}\n\n")

# === Final printout to terminal ===
print("\nEvaluation completed.")
print("Global summary:")
for k, v in global_metrics.items():
    print(f"{k}: {v}")
print(f"\nLogs saved to:\n- JSON: {LOG_JSON}\n- Markdown: {LOG_MD}")
