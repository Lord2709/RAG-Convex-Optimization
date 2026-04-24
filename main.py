"""
main.py
-------
MSML604 Project: Convex Optimization for RAG Pipeline Tuning
Full pipeline orchestrator — reads everything from config.yaml.

Pipeline stages
---------------
1. Load TriviaQA corpus + train/test queries
2. Build BM25 and Dense (FAISS) retrieval indices
3. Extract feature matrices for training queries
4. Learn weights via:
     a) Convex QP         (CVXPY, guaranteed global optimum)
     b) Non-Convex opt    (Nelder-Mead on Recall@k directly)
     c) Cluster-Adaptive  (K-means + per-cluster QP, K ∈ {2,3,4})  [NEW]
     d) Equal-weight      (uniform baseline)
     e) Cosine-only       (single-signal baseline)
     f) BM25-only         (single-signal baseline)
5. Evaluate all systems on held-out test queries (Recall@k, Precision@k, MRR)
6. End-to-end LLM evaluation via Groq (Exact Match, F1)
7. Save full results to results/results.json

Run
---
    python main.py                         # full run
    python main.py --skip-llm             # skip Groq generation step
    python main.py --config my_config.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import TriviaQALoader
from src.evaluator import evaluate_generation, evaluate_retrieval, print_results_table
from src.features import build_feature_matrix, build_label_vector, compute_query_profile
from src.llm_generator import create_generator
from src.optimizer import (
    ClusterAdaptiveOptimizer,
    ConvexOptimizer,
    EqualWeightBaseline,
    NonConvexOptimizer,
    SingleSignalBaseline,
    run_k_path,
    run_lambda_path,
)
from src.retriever import BM25Retriever, DenseRetriever


# ---------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


# ---------------------------------------------------------------------------
def retrieve_candidates(
    query: str,
    query_emb: np.ndarray,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    n_candidates: int,
) -> List[str]:
    """
    Merge BM25 and dense top-n candidates into a single deduplicated list,
    ranked by a simple round-robin fusion.
    """
    bm25_hits = bm25.retrieve(query, top_k=n_candidates)
    dense_hits = dense.retrieve(query, top_k=n_candidates)

    seen = set()
    merged = []
    for (b_id, _), (d_id, _) in zip(bm25_hits, dense_hits):
        for doc_id in (b_id, d_id):
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc_id)
    for doc_id, _ in bm25_hits + dense_hits:
        if doc_id not in seen and len(merged) < n_candidates:
            seen.add(doc_id)
            merged.append(doc_id)

    return merged[:n_candidates]


# ---------------------------------------------------------------------------
def rank_with_weights(
    X: np.ndarray,
    candidate_ids: List[str],
    optimizer,
    top_k: int,
    query_profile: np.ndarray = None,
) -> List[str]:
    """
    Rank candidates using optimizer.score().

    For ClusterAdaptiveOptimizer, passes query_profile so the correct
    cluster's weights are used.  All other optimizers ignore query_profile.
    """
    if isinstance(optimizer, ClusterAdaptiveOptimizer):
        scores = optimizer.score(X, query_profile)
    else:
        scores = optimizer.score(X)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [candidate_ids[i] for i in top_indices]


# ---------------------------------------------------------------------------
def run_pipeline(config: Dict[str, Any], skip_llm: bool = False) -> None:
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(exist_ok=True)

    # ── Stage 1: Load data ────────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 1  Loading TriviaQA")
    logging.info("=" * 60)
    loader = TriviaQALoader(config)
    corpus, train_queries, test_queries = loader.load()

    # ── Stage 2: Build indices ────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 2  Building retrieval indices")
    logging.info("=" * 60)
    bm25 = BM25Retriever(config)
    bm25.index(corpus)

    dense = DenseRetriever(config)
    dense.index(corpus)

    # ── Stage 3: Extract training features ───────────────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 3  Extracting training features")
    logging.info("=" * 60)
    n_candidates = config["retrieval"]["num_candidates"]
    top_k = config["retrieval"]["top_k"]

    train_feature_matrices: List[np.ndarray] = []
    train_label_vectors: List[np.ndarray] = []
    train_candidate_ids: List[List[str]] = []
    train_relevant_ids: List[List[str]] = []
    train_query_profiles: List[np.ndarray] = []   # NEW — one (3,) per train query

    t0 = time.time()
    for i, q in enumerate(train_queries):
        q_emb = dense.encode_query(q["question"])
        candidates = retrieve_candidates(
            q["question"], q_emb, bm25, dense, n_candidates
        )
        X, ordered_ids = build_feature_matrix(
            q["question"], q_emb, candidates, dense, bm25, config
        )
        y = build_label_vector(ordered_ids, q["relevant_doc_ids"])

        train_feature_matrices.append(X)
        train_label_vectors.append(y)
        train_candidate_ids.append(ordered_ids)
        train_relevant_ids.append(q["relevant_doc_ids"])
        train_query_profiles.append(compute_query_profile(X))   # NEW

        if (i + 1) % 25 == 0:
            logging.info("  Processed %d/%d train queries", i + 1, len(train_queries))

    # Stack for QP training
    X_train = np.vstack(train_feature_matrices)
    y_train = np.concatenate(train_label_vectors)
    query_profiles_train = np.vstack(train_query_profiles)   # (n_train, 3)

    logging.info(
        "Feature matrix  shape=%s  positives=%.1f%%",
        X_train.shape,
        100 * y_train.mean(),
    )
    logging.info("Feature extraction took %.1fs", time.time() - t0)

    # ── Stage 4a: Global weight optimization ──────────────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 4a  Global weight learning")
    logging.info("=" * 60)

    # (a) Convex QP — global
    convex_opt = ConvexOptimizer(config)
    convex_opt.fit(X_train, y_train)

    # (b) Non-convex
    nonconvex_opt = NonConvexOptimizer(config)
    nonconvex_opt.fit(train_feature_matrices, train_relevant_ids, train_candidate_ids)

    # (c-e) Baselines
    equal_baseline = EqualWeightBaseline()
    cosine_baseline = SingleSignalBaseline("cosine")
    bm25_baseline = SingleSignalBaseline("bm25")

    # ── Stage 4b: Cluster-Adaptive QP (K-path) ───────────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 4b  Cluster-Adaptive QP  (K-path: K=2,3,4)")
    logging.info("=" * 60)

    k_values_adaptive = config.get("adaptive", {}).get("k_path", [2, 3, 4])
    k_path = run_k_path(
        config=config,
        query_profiles=query_profiles_train,
        feature_matrices=train_feature_matrices,
        label_vectors=train_label_vectors,
        global_weights=convex_opt.weights,
        k_values=k_values_adaptive,
    )

    # Pick the best K for the main comparison using inertia elbow heuristic:
    # largest drop in inertia between consecutive K values.
    inertias = {k: opt.inertia for k, opt in k_path.items()}
    sorted_ks = sorted(inertias.keys())
    if len(sorted_ks) >= 2:
        drops = {
            sorted_ks[i + 1]: inertias[sorted_ks[i]] - inertias[sorted_ks[i + 1]]
            for i in range(len(sorted_ks) - 1)
        }
        best_k = max(drops, key=drops.get)
    else:
        best_k = sorted_ks[0]

    logging.info(
        "Inertia elbow → best K = %d  (inertia drop from K=%d: %.4f)",
        best_k,
        best_k - 1,
        drops.get(best_k, 0.0),
    )

    best_adaptive_opt = k_path[best_k]

    # Build the full optimizer dict — includes all K variants + original systems
    optimizers = {
        "Convex QP": convex_opt,
        "Non-Convex": nonconvex_opt,
        "Equal Weights": equal_baseline,
        "Cosine Only": cosine_baseline,
        "BM25 Only": bm25_baseline,
        f"Adaptive QP (K={best_k})": best_adaptive_opt,
    }

    # Also add all K variants for a detailed comparison table
    for k, opt in k_path.items():
        if k != best_k:
            optimizers[f"Adaptive QP (K={k})"] = opt

    # ── Stage 4c: Lambda path analysis ───────────────────────────────────
    # Run BEFORE Stage 5 so we have test feature matrices available.
    # We pre-collect test features here for the lambda sweep, then reuse
    # them in Stage 5 to avoid computing them twice.
    logging.info("=" * 60)
    logging.info("STAGE 4c  Lambda path analysis (regularization robustness)")
    logging.info("=" * 60)

    # Pre-extract all test features (reused in Stage 5)
    test_feature_matrices_all: List[np.ndarray] = []
    test_candidate_ids_all: List[List[str]] = []
    test_relevant_ids_prefetch: List[List[str]] = []
    test_query_profiles_all: List[np.ndarray] = []
    test_queries_cache = []   # store (q_emb, ordered_ids, X) for Stage 5 reuse

    for q in test_queries:
        q_emb = dense.encode_query(q["question"])
        candidates = retrieve_candidates(
            q["question"], q_emb, bm25, dense, n_candidates
        )
        X_t, ordered_ids_t = build_feature_matrix(
            q["question"], q_emb, candidates, dense, bm25, config
        )
        test_feature_matrices_all.append(X_t)
        test_candidate_ids_all.append(ordered_ids_t)
        test_relevant_ids_prefetch.append(q["relevant_doc_ids"])
        test_query_profiles_all.append(compute_query_profile(X_t))
        test_queries_cache.append((q, ordered_ids_t, X_t))

    lambda_values = config.get("adaptive", {}).get(
        "lambda_path", [0.001, 0.01, 0.1, 1.0]
    )
    lambda_path_results = run_lambda_path(
        config=config,
        X_train=X_train,
        y_train=y_train,
        test_feature_matrices=test_feature_matrices_all,
        test_candidate_ids=test_candidate_ids_all,
        test_relevant_ids=test_relevant_ids_prefetch,
        lambda_values=lambda_values,
        k=top_k,
    )

    # ── Stage 5: Retrieval evaluation on test queries ─────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 5  Retrieval evaluation on %d test queries", len(test_queries))
    logging.info("=" * 60)

    system_retrieved: Dict[str, List[List[str]]] = {k: [] for k in optimizers}
    test_relevant_ids: List[List[str]] = []

    llm_context_docs: List[List[Dict]] = []
    llm_questions: List[str] = []
    llm_gold_answers: List[List[str]] = []

    doc_id_to_doc = {doc["doc_id"]: doc for doc in corpus}

    # Track which cluster each test query lands in (for analysis)
    test_cluster_assignments: Dict[int, List[int]] = {
        k: [] for k in k_values_adaptive
    }

    # Reuse pre-extracted test features from Stage 4c — avoids re-encoding
    for i, (q, ordered_ids, X) in enumerate(test_queries_cache):
        test_relevant_ids.append(q["relevant_doc_ids"])

        # Query profile already computed in Stage 4c
        q_profile = test_query_profiles_all[i]

        # Record cluster assignment for each K (for the results JSON)
        for k, opt in k_path.items():
            test_cluster_assignments[k].append(opt.get_cluster_id(q_profile))

        # Rank with all systems
        for system_name, opt in optimizers.items():
            ranked = rank_with_weights(X, ordered_ids, opt, top_k, q_profile)
            system_retrieved[system_name].append(ranked)

        # Store top-k docs from best Adaptive QP for LLM eval
        adaptive_key = f"Adaptive QP (K={best_k})"
        adaptive_ranked = system_retrieved[adaptive_key][-1]
        context_docs = [
            doc_id_to_doc[did] for did in adaptive_ranked if did in doc_id_to_doc
        ]
        llm_context_docs.append(context_docs)
        llm_questions.append(q["question"])
        llm_gold_answers.append(q["answers"])

    # Compute retrieval metrics
    k_values = config["evaluation"]["k_values"]
    retrieval_results: Dict[str, Dict[str, float]] = {}
    for system_name in optimizers:
        retrieval_results[system_name] = evaluate_retrieval(
            system_retrieved[system_name], test_relevant_ids, k_values
        )

    logging.info("\nRetrieval Results:")
    print_results_table(retrieval_results)

    # ── Stage 6: LLM generation evaluation ───────────────────────────────
    generation_results: Dict[str, Dict[str, float]] = {}

    if not skip_llm:
        logging.info("=" * 60)
        logging.info("STAGE 6  End-to-end LLM evaluation (Groq)")
        logging.info("=" * 60)
        n_llm = min(config["llm"]["num_eval_queries"], len(test_queries))
        try:
            provider = config["llm"]["provider"]
            generator = create_generator(config)
            predictions = generator.generate_batch(
                llm_questions[:n_llm], llm_context_docs[:n_llm]
            )
            label = f"Adaptive QP K={best_k} ({provider.capitalize()})"
            generation_results[label] = evaluate_generation(
                predictions, llm_gold_answers[:n_llm]
            )
            logging.info("Generation results: %s", generation_results)
        except EnvironmentError as e:
            logging.warning("Skipping LLM eval: %s", e)
    else:
        logging.info("STAGE 6  Skipped (--skip-llm flag)")

    # ── Stage 7: Save results ─────────────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 7  Saving results")
    logging.info("=" * 60)

    output = {
        "retrieval": retrieval_results,
        "generation": generation_results,
        "weights": {
            "convex_qp": convex_opt.weights.tolist(),
            "nonconvex": nonconvex_opt.weights.tolist(),
            "equal": equal_baseline.weights.tolist(),
            # Per-cluster weights for each K
            "adaptive_k_path": {
                str(k): opt.summary() for k, opt in k_path.items()
            },
        },
        "adaptive_analysis": {
            "inertias": {str(k): opt.inertia for k, opt in k_path.items()},
            "best_k": best_k,
            "test_cluster_assignments": {
                str(k): v for k, v in test_cluster_assignments.items()
            },
        },
        "lambda_path": lambda_path_results,
        "config_snapshot": {
            "num_train_queries": len(train_queries),
            "num_test_queries": len(test_queries),
            "corpus_size": len(corpus),
            "lambda_reg": config["optimization"]["lambda_reg"],
            "top_k": top_k,
            "adaptive_k_path": k_values_adaptive,
        },
    }

    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logging.info("Results saved to %s", out_path)
    logging.info("Pipeline complete.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSML604 RAG Optimization Pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["output"]["log_level"])
    run_pipeline(cfg, skip_llm=args.skip_llm)