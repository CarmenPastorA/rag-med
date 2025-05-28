import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
from pathlib import Path
import sys

# === Add project root to path ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # /tools/arqa/evaluation/bm25
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))  # root of project
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)

from shared import dunder_info
dunder_info.inject_dunder(__name__)

from tools.arqa.bm25_search import BM25Search
from tools.arqa.evaluation.metrics import precision_at_k, recall_at_k, hit_at_k, mrr, normalized_precision_at_k, n_relevant_retrieved

# === Configuration ===
K = 10
BM25_DIR = os.path.join(PROJECT_ROOT, "data/posteriori_resources/bm25_stuffs")
FASTTEXT_PATH = os.path.join(PROJECT_ROOT, "models/lang_model")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data/priori_resources/stopwords.txt")
PRESERVE_WORDS_PATH = os.path.join(PROJECT_ROOT, "data/priori_resources/preserve_words.txt")
DATASET_PATH = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/generated_datasets/structured_mistral_min3.jsonl")

RUN_NAME = f"bm25_eval_{datetime.now().strftime('%Y-%m-%d')}_k{K}"
LOG_DIR = os.path.join(PROJECT_ROOT, "tools/arqa/evaluation/bm25/logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_JSON = os.path.join(LOG_DIR, f"{RUN_NAME}.json")
LOG_MD = os.path.join(LOG_DIR, f"{RUN_NAME}.md")

# === Initialize BM25 Retriever ===
bm25 = BM25Search(
    directory=BM25_DIR,
    fasttext_model_path=FASTTEXT_PATH,
    stopwords_path=STOPWORDS_PATH,
    preserve_words_path=PRESERVE_WORDS_PATH,
    verbose=True
)

# === Load synthetic QA dataset ===
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

# === Per-query evaluation ===
results = []

# Filter out the 78 questions with 50+ retrieved documents (too general and not clinically relevant)
#questions = [q for q in questions if len(q["relevant_doc_ids"]) <= 50]
for q in questions:
    query_id = q.get("id")
    query = q["question"]
    relevant_ids = q["relevant_doc_ids"]

    try:
        retrieved_raw = bm25.search(query, k=K)
        retrieved_ids = [doc_id.replace("FT_", "").replace("_ESP", " ESP") for doc_id in retrieved_raw]

        results.append({
            "id": query_id,
            "question": query,
            "relevant_doc_ids": relevant_ids,
            "retrieved_doc_ids": retrieved_ids,
            "precision@k": precision_at_k(retrieved_ids, relevant_ids),
            "recall@k": recall_at_k(retrieved_ids, relevant_ids),
            "hit@k": hit_at_k(retrieved_ids, relevant_ids),
            "mrr": mrr(retrieved_ids, relevant_ids),
            "normalized_precision@k": normalized_precision_at_k(retrieved_ids, relevant_ids, k=K),
            "n_relevant_retrieved": n_relevant_retrieved(retrieved_ids, relevant_ids)
        })
    except Exception as e:
        print(f"Error processing: {query}\n{e}")

# === Aggregate metrics ===
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
    "retriever": "BM25",
    "run_name": RUN_NAME
}

# === Save results to JSON ===
with open(LOG_JSON, "w", encoding="utf-8") as f:
    json.dump({"summary": global_metrics, "results": results}, f, indent=2, ensure_ascii=False)

# === Save Markdown report ===
with open(LOG_MD, "w", encoding="utf-8") as f:
    f.write(f"# BM25 Evaluation Report - {RUN_NAME}\n\n")
    f.write("## Configuration\n")
    f.write(f"- `Top-k`: {K}\n")
    f.write(f"- `Date`: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("## Global Results\n\n")
    for k, v in global_metrics.items():
        f.write(f"- **{k.replace('_', ' ').capitalize()}**: {v}\n")
    f.write("\n---\n## Sample Questions\n\n")
    for r in results[:5]:
        f.write(f"- **Question:** {r['question']}\n")
        f.write(f"  - Relevant doc IDs: {r['relevant_doc_ids']}\n")
        f.write(f"  - Retrieved doc IDs: {r['retrieved_doc_ids']}\n")
        f.write(f"  - P@{K}: {r['precision@k']}, R@{K}: {r['recall@k']}, MRR: {r['mrr']}\n\n")

# === Print global results to terminal ===
print("\nGlobal summary:")
for k, v in global_metrics.items():
    print(f"{k}: {v}")
print(f"\nLogs saved to:\n- JSON: {LOG_JSON}\n- Markdown: {LOG_MD}")
