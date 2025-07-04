# === retrievers.py ===

import os
import sys
import json
import numpy as np
from statistics import mean
from collections import defaultdict

# Config paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from tools.arqa.bm25_search import BM25Search
from tools.arqa.faiss_search import HierarchicalFaissSearch
from shared.veterinary_utils.embedding_model import EmbeddingModel
from shared.veterinary_utils.utils import normalize_doc_id

BM25_DIR = os.path.join(PROJECT_ROOT, "data/posteriori_resources/bm25_stuffs")
FASTTEXT_PATH = os.path.join(PROJECT_ROOT, "models/lang_model")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data/priori_resources/stopwords.txt")
PRESERVE_WORDS_PATH = os.path.join(PROJECT_ROOT, "data/priori_resources/preserve_words.txt")

FAISS_DOCS_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/processed_json")

# === FAISS Essential-level (document-level) index ===
ESSENTIAL_INDEX_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_index.faiss")
ESSENTIAL_MAPPING_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_mapping.json")
ESSENTIAL_CACHE_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/essential_cache.json")

# === FAISS Chunk-level index ===
CHUNKS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_index.faiss")
CHUNKS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_mapping.json")
CHUNKS_CACHE_PATH = os.path.join(PROJECT_ROOT, "data/posteriori_resources/faiss_stuff/chunks_cache.json")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# === Cached resources ===
_bm25_model = None
_embedding_model = None
_faiss_chunks = None
_faiss_search = None
_doc_to_chunks = None  # doc_id -> list of (faiss_id, full_chunk_id)

def get_bm25_results(question: str, top_k: int = 50, score_threshold: float = 8.0) -> list[str]:
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = BM25Search(
            directory=BM25_DIR,
            fasttext_model_path=FASTTEXT_PATH,
            stopwords_path=STOPWORDS_PATH,
            preserve_words_path=PRESERVE_WORDS_PATH,
            verbose=False
        )
    results = _bm25_model.search(question, k=top_k, include_scores=True, translate=True)
    return [doc_id.strip() for doc_id, score in results if score >= score_threshold]

def get_faiss_results(question: str, top_k: int = 10, device: str = "cuda") -> list[dict]:
    global _embedding_model, _faiss_search
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME, device, 512)
    if _faiss_search is None:
        _faiss_search = HierarchicalFaissSearch(_embedding_model)
        _faiss_search.load_indices(
            essential_index_path=ESSENTIAL_INDEX_PATH,
            essential_mapping_path=ESSENTIAL_MAPPING_PATH,
            essential_cache_path=ESSENTIAL_CACHE_PATH,
            chunks_index_path=CHUNKS_INDEX_PATH,
            chunks_mapping_path=CHUNKS_MAPPING_PATH,
            chunks_cache_path=CHUNKS_CACHE_PATH
        )
    return [
        {
            "doc_id": normalize_doc_id(r["metadata"].get("document_id", r["chunk_id"])),
            "content": r["text"]
        }
        for r in _faiss_search.hierarchical_search(query=question, top_documents=10, top_chunks=top_k)

    ]

def get_late_fusion_results(
    question: str,
    bm25_top_n: int = 50,
    faiss_top_k: int = 50,
    fusion_top_k: int = 10,
    alpha: float = 0.5,
    device: str = "cuda"
) -> list[dict]:
    """
    Late Fusion strategy: BM25 and FAISS are executed independently over the full index.
    Their rankings are combined via linear weighted scoring and top-K documents are selected.
    """
    bm25_doc_ids = get_bm25_results(question, top_k=bm25_top_n, score_threshold=0)
    faiss_results = get_faiss_results(question, top_k=faiss_top_k, device=device)

    bm25_scores = {
        normalize_doc_id(doc_id): (bm25_top_n - rank) / bm25_top_n
        for rank, doc_id in enumerate(bm25_doc_ids)
    }

    faiss_scores = {
        normalize_doc_id(doc["doc_id"]): (faiss_top_k - rank) / faiss_top_k
        for rank, doc in enumerate(faiss_results)
    }

    combined_scores = {}
    for doc_id in set(bm25_scores) | set(faiss_scores):
        b_score = bm25_scores.get(doc_id, 0)
        f_score = faiss_scores.get(doc_id, 0)
        combined_scores[doc_id] = alpha * b_score + (1 - alpha) * f_score

    top_doc_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:fusion_top_k]
    top_doc_ids_set = {doc_id for doc_id, _ in top_doc_ids}

    doc_content_map = {
        normalize_doc_id(doc["doc_id"]): doc["content"]
        for doc in faiss_results
        if normalize_doc_id(doc["doc_id"]) in top_doc_ids_set
    }
    return [{"doc_id": doc_id, "content": doc_content_map.get(doc_id, "")} for doc_id in top_doc_ids_set]


def build_doc_to_chunk_index(chunks_mapping_path: str) -> dict[str, list[str]]:
    """ 
    Utility function to build an inverted index mapping each document ID to the list of its associated chunks (with FAISS index IDs). 
    This avoids iterating over all chunks repeatedly and speeds up fallback operations.
    """
    with open(chunks_mapping_path, "r", encoding="utf-8") as f:
        id_to_embedding_info = json.load(f)
    doc_to_chunks = defaultdict(list)
    for faiss_id_str, info in id_to_embedding_info.items():
        doc_id = normalize_doc_id(info["metadata"]["document_id"])
        chunk_id = info["metadata"]["chunk_id"]
        full_chunk_id = f"{doc_id}@{chunk_id}"
        doc_to_chunks[doc_id].append((int(faiss_id_str), full_chunk_id))
    return doc_to_chunks

def get_late_fusion_with_fallback_optimized(
    question: str,
    bm25_top_n: int = 50,
    bm25_fallback_top_n: int = 10,
    faiss_top_k: int = 20,
    fallback_chunk_limit: int = 2,
    device: str = "cuda",
    verbose: bool = False
) -> list[dict]:
    """
    Optimized hybrid retrieval: FAISS semantic + fallback BM25 not in FAISS.
    Uses cached doc-to-chunks mapping and chunks cache for faster lookup.
    """
    global _embedding_model, _faiss_search, _faiss_chunks, _doc_to_chunks

    if _embedding_model is None:
        _embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME, device, 512)
    if _faiss_search is None:
        _faiss_search = HierarchicalFaissSearch(_embedding_model, verbose=False)
        _faiss_search.load_indices(
            essential_index_path=ESSENTIAL_INDEX_PATH,
            essential_mapping_path=ESSENTIAL_MAPPING_PATH,
            essential_cache_path=ESSENTIAL_CACHE_PATH,
            chunks_index_path=CHUNKS_INDEX_PATH,
            chunks_mapping_path=CHUNKS_MAPPING_PATH,
            chunks_cache_path=CHUNKS_CACHE_PATH
        )
    if _faiss_chunks is None:
        with open(CHUNKS_CACHE_PATH, "r", encoding="utf-8") as f:
            _faiss_chunks = json.load(f)
    if _doc_to_chunks is None:
        with open(CHUNKS_MAPPING_PATH, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        from collections import defaultdict
        _doc_to_chunks = defaultdict(list)
        for faiss_id_str, info in mapping.items():
            doc_id = normalize_doc_id(info["metadata"]["document_id"])
            chunk_id = info["metadata"]["chunk_id"]
            full_chunk_id = f"{doc_id}@{chunk_id}"
            _doc_to_chunks[doc_id].append((int(faiss_id_str), full_chunk_id))

    # Retrieve top FAISS chunks
    faiss_results = _faiss_search.search_chunks_only(query=question, top_k=faiss_top_k)
    faiss_chunks = []
    faiss_doc_ids = set()

    for res in faiss_results:
        doc_id = normalize_doc_id(res["metadata"].get("document_id", res["chunk_id"].split("@")[0]))
        faiss_doc_ids.add(doc_id)
        faiss_chunks.append({
            "doc_id": doc_id,
            "chunk_id": res["chunk_id"],
            "content": res["text"].strip()
        })

    # BM25 fallback doc IDs
    bm25_doc_ids = [normalize_doc_id(d) for d in get_bm25_results(question, top_k=bm25_top_n, score_threshold=0)]
    fallback_doc_ids = [doc_id for doc_id in bm25_doc_ids if doc_id not in faiss_doc_ids][:bm25_fallback_top_n]

    # Embed the query once
    query_emb = _embedding_model.get_embeddings([question], convert_to_numpy=True, normalize_embeddings=True).squeeze()

    # Select fallback chunks
    fallback_scores = []
    selected_fallback_chunks = []

    for doc_id in fallback_doc_ids:
        if doc_id not in _doc_to_chunks:
            continue
        similarities = []
        for faiss_id, full_chunk_id in _doc_to_chunks[doc_id]:
            if full_chunk_id not in _faiss_chunks:
                continue
            text = _faiss_chunks[full_chunk_id]["text"]
            chunk_emb = _faiss_search.get_chunk_index().reconstruct(faiss_id)
            score = float(np.dot(chunk_emb, query_emb))
            similarities.append((full_chunk_id, text, score))
        top_chunks = sorted(similarities, key=lambda x: x[2], reverse=True)[:fallback_chunk_limit]
        for chunk_id, text, score in top_chunks:
            fallback_scores.append(score)
            selected_fallback_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "score": score,
                "content": text.strip()
            })

    if verbose:
        print(f"\n [Fallback OPTIMIZED Summary]")
        print(f"BM25 top-N: {bm25_top_n}")
        print(f"Fallback candidates: {len(fallback_doc_ids)}")
        print(f"Chunks added from fallback: {len(selected_fallback_chunks)}")
        if fallback_scores:
            print(f"Score stats — mean: {mean(fallback_scores):.3f}, min: {min(fallback_scores):.3f}, max: {max(fallback_scores):.3f}")

    return faiss_chunks + selected_fallback_chunks



def analyze_bm25_fallback_needs(
    question: str,
    bm25_top_n: int = 50,
    faiss_top_k: int = 20,
    device: str = "cuda"
):
    """
    Analyzes how many of the top BM25 documents are not covered by any chunks 
    retrieved by FAISS, and estimates the number of chunks that would need to 
    be re-embedded for a semantic fallback strategy.

    Args:
        question (str): The user query.
        bm25_top_n (int): Number of top documents to retrieve from BM25.
        faiss_top_k (int): Number of top chunks to retrieve from FAISS.
        device (str): Device to run embeddings on ("cuda" or "cpu").

    Returns:
        None — Prints diagnostic output to console.
    """
    global _embedding_model, _faiss_search, _faiss_chunks

    if _embedding_model is None:
        _embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME, device, 512)

    if _faiss_search is None:
        _faiss_search = HierarchicalFaissSearch(_embedding_model)
        _faiss_search.load_indices(
            essential_index_path=ESSENTIAL_INDEX_PATH,
            essential_mapping_path=ESSENTIAL_MAPPING_PATH,
            essential_cache_path=ESSENTIAL_CACHE_PATH,
            chunks_index_path=CHUNKS_INDEX_PATH,
            chunks_mapping_path=CHUNKS_MAPPING_PATH,
            chunks_cache_path=CHUNKS_CACHE_PATH
        )

    if _faiss_chunks is None:
        with open(CHUNKS_CACHE_PATH, "r", encoding="utf-8") as f:
            _faiss_chunks = json.load(f)

    # Retrieve top-N BM25 document IDs (normalized)
    bm25_doc_ids = [normalize_doc_id(d) for d in get_bm25_results(question, top_k=bm25_top_n, score_threshold=0)]

    # Retrieve FAISS results and extract their document IDs
    faiss_doc_ids = set([
        normalize_doc_id(r["metadata"].get("document_id", r["chunk_id"].split("@")[0]))
        for r in _faiss_search.search(query=question, k=faiss_top_k)
    ])

    # Identify BM25 documents not covered by FAISS
    uncovered = [doc_id for doc_id in bm25_doc_ids if doc_id not in faiss_doc_ids]

    print(f"\n📌 Out of the top {bm25_top_n} BM25 documents, {len(uncovered)} are not covered by FAISS.")

    # Count total chunks from uncovered documents (for re-embedding in semantic fallback)
    total_chunks_to_reembed = 0
    for doc_id in uncovered:
        doc_chunks = [cid for cid in _faiss_chunks if cid.startswith(doc_id)]
        print(f"- {doc_id}: {len(doc_chunks)} chunks")
        total_chunks_to_reembed += len(doc_chunks)

    print(f"\n🔁 Total chunks to re-embed for semantic fallback: {total_chunks_to_reembed}")


if __name__ == "__main__":
    q = "¿Qué medicamentos se pueden usar en gallinas para coccidiosis?"
    
    print("\n=== Original fallback ===")
    original = get_late_fusion_with_fallback(q, verbose=True)
    print(f"Total chunks retrieved (original): {len(original)}")

    print("\n=== Optimized fallback ===")
    optimized = get_late_fusion_with_fallback_optimized(q, verbose=True)
    print(f"Total chunks retrieved (optimized): {len(optimized)}")

    # Optional: compare document IDs covered
    original_docs = {c['doc_id'] for c in original}
    optimized_docs = {c['doc_id'] for c in optimized}
    common = original_docs & optimized_docs
    print(f"\n Shared doc_ids: {len(common)} / {len(original_docs)} original, {len(optimized_docs)} optimized")

