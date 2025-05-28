# Unified metric functions for retriever evaluation

def precision_at_k(retrieved, relevant):
    """
    Precision@k: Proportion of retrieved documents that are actually relevant.
    - Measures how "clean" the top-k results are.
    - High precision means fewer irrelevant documents in the results.
    """
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)

def recall_at_k(retrieved, relevant):
    """
    Recall@k: Proportion of relevant documents that were retrieved.
    - Measures coverage: how much of the ground truth was found.
    - High recall means fewer relevant documents were missed.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)

def hit_at_k(retrieved, relevant):
    """
    Hit@k: Binary score (0 or 1) indicating if at least one relevant document is present in the top-k.
    - Robust metric for evaluating whether the system finds any relevant item at all.
    """
    return int(any(r in relevant for r in retrieved))

def mrr(retrieved, relevant):
    """
    Mean Reciprocal Rank (MRR): Inverse of the rank position of the first relevant document.
    - Captures how early in the list the first relevant result appears.
    - A perfect score (1.0) means the top-1 result is relevant.
    """
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0

def normalized_precision_at_k(retrieved, relevant, k=10):
    """
    Normalized Precision@k: Adjusts precision using min(k, number of relevant documents).
    - Prevents unfair penalization when the total number of relevant documents is less than k.
    - Example: If only 5 relevant docs exist and 4 are retrieved in top-10, precision should be 4/5 not 4/10.
    """
    if not relevant or not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / min(k, len(relevant))

def n_relevant_retrieved(retrieved, relevant):
    """
    Count of relevant documents that were actually retrieved.
    - Raw metric (not normalized) useful for aggregation and detailed diagnostics.
    - Example: Helps track how many relevant documents are retrieved per question.
    """
    return sum(1 for r in retrieved if r in relevant)
