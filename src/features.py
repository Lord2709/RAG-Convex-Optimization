"""
features.py
-----------
Builds the feature matrix X used by both optimizers.

Each candidate document gets a 3-dimensional feature vector:
  [cosine_similarity,  bm25_score,  mmr_diversity]

All features are independently min-max normalized to [0, 1] per query,
so the weight vector w is on a consistent scale across queries.

Note on convexity:
  The diversity signal is computed relative to a *fixed* reference set
  (top-k by cosine similarity) rather than dynamically during selection.
  This keeps the feature matrix X independent of w, which is what allows
  the weight learning problem to be formulated as a convex QP.
  The non-convex optimizer uses the same X but a different objective.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def cosine_similarity(query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
    """Dot product of two L2-normalized vectors equals cosine similarity."""
    return float(np.dot(query_emb, doc_emb))


def mmr_diversity(doc_emb: np.ndarray, reference_embs: List[np.ndarray]) -> float:
    """
    Diversity of doc relative to a reference set.
    Defined as 1 - max_cosine_similarity(doc, reference_set).
    Returns 1.0 if reference set is empty (no similar docs selected yet).
    """
    if not reference_embs:
        return 1.0
    sims = [float(np.dot(doc_emb, ref)) for ref in reference_embs]
    return 1.0 - max(sims)


def normalize_min_max(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Handles the degenerate all-equal case."""
    lo, hi = arr.min(), arr.max()
    if (hi - lo) < 1e-10:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
def build_feature_matrix(
    query: str,
    query_emb: np.ndarray,
    candidate_doc_ids: List[str],
    dense_retriever,
    bm25_retriever,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix X of shape (n_candidates, 3) for a single query.

    Columns
    -------
    0  cosine_similarity  : dense embedding similarity (normalized)
    1  bm25_score         : sparse lexical score (normalized)
    2  mmr_diversity      : 1 - max_sim to top-reference_size docs (normalized)

    Returns
    -------
    X        : np.ndarray of shape (n_candidates, 3), values in [0, 1]
    doc_ids  : list of doc_ids in the same row order as X
    """
    n = len(candidate_doc_ids)
    ref_size = config["retrieval"]["mmr_reference_size"]

    cosine_raw = np.zeros(n, dtype=np.float32)
    bm25_raw = np.zeros(n, dtype=np.float32)
    doc_embs: Dict[str, np.ndarray] = {}

    # Fetch BM25 scores for all candidates in one call
    bm25_scores_map = bm25_retriever.scores_for_ids(query, candidate_doc_ids)

    for i, doc_id in enumerate(candidate_doc_ids):
        emb = dense_retriever.get_embedding(doc_id)
        doc_embs[doc_id] = emb
        cosine_raw[i] = cosine_similarity(query_emb, emb)
        bm25_raw[i] = bm25_scores_map.get(doc_id, 0.0)

    # Build fixed reference set: top-ref_size docs by cosine similarity
    ref_indices = np.argsort(cosine_raw)[::-1][:ref_size]
    reference_embs = [doc_embs[candidate_doc_ids[i]] for i in ref_indices]

    diversity_raw = np.array(
        [mmr_diversity(doc_embs[did], reference_embs) for did in candidate_doc_ids],
        dtype=np.float32,
    )

    # Normalize each feature independently per query
    X = np.column_stack(
        [
            normalize_min_max(cosine_raw),
            normalize_min_max(bm25_raw),
            normalize_min_max(diversity_raw),
        ]
    )
    return X, candidate_doc_ids


# ---------------------------------------------------------------------------
def build_label_vector(
    candidate_doc_ids: List[str], relevant_doc_ids: List[str]
) -> np.ndarray:
    """
    Binary relevance labels: 1 if doc is relevant, 0 otherwise.
    Used as target vector y in the QP.
    """
    relevant_set = set(relevant_doc_ids)
    return np.array(
        [1.0 if did in relevant_set else 0.0 for did in candidate_doc_ids],
        dtype=np.float32,
    )
