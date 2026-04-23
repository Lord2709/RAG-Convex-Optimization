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
     c) Equal-weight      (uniform baseline)
     d) Cosine-only       (single-signal baseline)
     e) BM25-only         (single-signal baseline)
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

# Ensure src/ is on the path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import TriviaQALoader
from src.evaluator import evaluate_generation, evaluate_retrieval, print_results_table
from src.features import build_feature_matrix, build_label_vector
from src.llm_generator import create_generator
from src.optimizer import (
    ConvexOptimizer,
    EqualWeightBaseline,
    NonConvexOptimizer,
    SingleSignalBaseline,
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
    ranked by a simple round-robin fusion (interleave BM25 and dense results).
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
    # Fill remaining slots from each list if one is exhausted
    for doc_id, _ in bm25_hits + dense_hits:
        if doc_id not in seen and len(merged) < n_candidates:
            seen.add(doc_id)
            merged.append(doc_id)

    return merged[:n_candidates]


# ---------------------------------------------------------------------------
def rank_with_weights(
    X: np.ndarray, candidate_ids: List[str], optimizer, top_k: int
) -> List[str]:
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

        if (i + 1) % 25 == 0:
            logging.info("  Processed %d/%d train queries", i + 1, len(train_queries))

    # Stack for QP training
    X_train = np.vstack(train_feature_matrices)
    y_train = np.concatenate(train_label_vectors)
    logging.info(
        "Feature matrix  shape=%s  positives=%.1f%%",
        X_train.shape,
        100 * y_train.mean(),
    )
    logging.info("Feature extraction took %.1fs", time.time() - t0)

    # ── Stage 4: Weight optimization ─────────────────────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 4  Learning retrieval weights")
    logging.info("=" * 60)

    # (a) Convex QP
    convex_opt = ConvexOptimizer(config)
    convex_opt.fit(X_train, y_train)

    # (b) Non-convex (same features, different objective: Recall@k)
    nonconvex_opt = NonConvexOptimizer(config)
    nonconvex_opt.fit(train_feature_matrices, train_relevant_ids, train_candidate_ids)

    # (c-e) Baselines — no training needed
    equal_baseline = EqualWeightBaseline()
    cosine_baseline = SingleSignalBaseline("cosine")
    bm25_baseline = SingleSignalBaseline("bm25")

    optimizers = {
        "Convex QP": convex_opt,
        "Non-Convex": nonconvex_opt,
        "Equal Weights": equal_baseline,
        "Cosine Only": cosine_baseline,
        "BM25 Only": bm25_baseline,
    }

    # ── Stage 5: Retrieval evaluation on test queries ─────────────────────
    logging.info("=" * 60)
    logging.info("STAGE 5  Retrieval evaluation on %d test queries", len(test_queries))
    logging.info("=" * 60)

    # Collect retrieved lists per system
    system_retrieved: Dict[str, List[List[str]]] = {k: [] for k in optimizers}
    test_relevant_ids: List[List[str]] = []

    # Also store context docs for LLM eval (use convex QP retrieval)
    llm_context_docs: List[List[Dict]] = []
    llm_questions: List[str] = []
    llm_gold_answers: List[List[str]] = []

    doc_id_to_doc = {doc["doc_id"]: doc for doc in corpus}

    for i, q in enumerate(test_queries):
        q_emb = dense.encode_query(q["question"])
        candidates = retrieve_candidates(
            q["question"], q_emb, bm25, dense, n_candidates
        )
        X, ordered_ids = build_feature_matrix(
            q["question"], q_emb, candidates, dense, bm25, config
        )
        test_relevant_ids.append(q["relevant_doc_ids"])

        for system_name, opt in optimizers.items():
            ranked = rank_with_weights(X, ordered_ids, opt, top_k)
            system_retrieved[system_name].append(ranked)

        # Store top-k docs from Convex QP for LLM eval
        convex_ranked = system_retrieved["Convex QP"][-1]
        context_docs = [
            doc_id_to_doc[did] for did in convex_ranked if did in doc_id_to_doc
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
            label = f"Convex QP ({provider.capitalize()})"
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
        },
        "config_snapshot": {
            "num_train_queries": len(train_queries),
            "num_test_queries": len(test_queries),
            "corpus_size": len(corpus),
            "lambda_reg": config["optimization"]["lambda_reg"],
            "top_k": top_k,
        },
    }

    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logging.info("Results saved to %s", out_path)
    logging.info("Pipeline complete.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MSML604 RAG Optimization Pipeline"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Groq LLM generation evaluation",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["output"]["log_level"])
    run_pipeline(cfg, skip_llm=args.skip_llm)
