# Metric functions to evaluate FAISS retrieval performance

def precision_at_k(retrieved, relevant):
    """Compute Precision@k: proportion of retrieved documents that are relevant"""
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)

def recall_at_k(retrieved, relevant):
    """Compute Recall@k: proportion of relevant documents that were retrieved"""
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)

def hit_at_k(retrieved, relevant):
    """Binary Hit@k: returns 1 if at least one relevant document is in top-k"""
    return int(any(r in relevant for r in retrieved))

def mrr(retrieved, relevant):
    """Mean Reciprocal Rank: inverse rank of the first relevant document"""
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0
