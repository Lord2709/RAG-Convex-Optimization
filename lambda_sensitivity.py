"""
lambda_sensitivity.py
---------------------
Lambda (regularization) sensitivity analysis for the Convex QP optimizer.

Sweeps lambda values across several orders of magnitude, solves the QP at each
value, and records:
  - Learned weights [w_cosine, w_bm25, w_diversity]  (regularization path)
  - QP optimal objective value
  - Recall@5 on training data  (proxy for retrieval quality at each lambda)

Outputs
-------
  results/lambda_results.json        — raw numbers for every lambda
  results/figures/fig8_lambda_sensitivity.png  — four-panel figure:
      (a) Regularization path (weights vs log-lambda)
      (b) QP optimal objective value vs log-lambda
      (c) Recall@5 vs log-lambda
      (d) Weight composition as a stacked area chart

Usage
-----
  # After running main.py at least once (generates results/X_train.npz):
  python lambda_sensitivity.py

  # Custom config or lambda grid:
  python lambda_sensitivity.py --config config.yaml --lambdas 0.0001 0.001 0.01 0.1 1.0 10.0

  # If X_train.npz is missing, pass --recompute to rebuild features from scratch
  # (requires the dataset download; takes a few minutes):
  python lambda_sensitivity.py --recompute
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml

# ── Ensure src/ is importable ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Plot style (mirrors generate_plots.py) ────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

FEATURE_NAMES = ["Cosine Similarity", "BM25 Score", "MMR Diversity"]
FEATURE_COLORS = ["#2563EB", "#DC2626", "#16A34A"]

DEFAULT_LAMBDAS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


# ── QP solver ─────────────────────────────────────────────────────────────────

def solve_qp(X: np.ndarray, y: np.ndarray, lam: float, solver_name: str = "OSQP"):
    """
    Solve:  min  ||Xw - y||^2 + lam * ||w||^2
            s.t. w >= 0,  sum(w) = 1

    Returns (weights, optimal_value, status).
    """
    n_features = X.shape[1]
    w = cp.Variable(n_features)
    objective = cp.Minimize(cp.sum_squares(X @ w - y) + lam * cp.sum_squares(w))
    constraints = [w >= 0, cp.sum(w) == 1]
    problem = cp.Problem(objective, constraints)

    solver = getattr(cp, solver_name, cp.OSQP)
    problem.solve(solver=solver, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        logger.warning("Lambda=%.5f  solver status: %s — falling back to equal weights", lam, problem.status)
        weights = np.ones(n_features) / n_features
    else:
        weights = np.clip(np.array(w.value).flatten(), 0, None)
        if weights.sum() > 1e-10:
            weights /= weights.sum()
        else:
            weights = np.ones(n_features) / n_features

    return weights, float(problem.value) if problem.value is not None else float("nan"), problem.status


def recall_at_k_from_matrix(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                              n_candidates: int, k: int = 5) -> float:
    """
    Compute Recall@k on training data using a stacked feature matrix.

    X is (n_train * n_candidates, 3), y is (n_train * n_candidates,).
    We treat each block of n_candidates rows as one query.
    """
    n_queries = len(y) // n_candidates
    recalls = []
    for i in range(n_queries):
        start, end = i * n_candidates, (i + 1) * n_candidates
        X_q = X[start:end]
        y_q = y[start:end]
        scores = X_q @ weights
        top_k_idx = np.argsort(scores)[::-1][:k]
        n_relevant = int(y_q.sum())
        if n_relevant == 0:
            continue
        hits = int(y_q[top_k_idx].sum())
        recalls.append(hits / n_relevant)
    return float(np.mean(recalls)) if recalls else 0.0


# ── Feature re-computation (fallback when X_train.npz is missing) ─────────────

def recompute_features(config: dict) -> tuple:
    """Re-run Stages 1–3 of the main pipeline to produce X_train, y_train."""
    import time
    from src.data_loader import TriviaQALoader
    from src.features import build_feature_matrix, build_label_vector
    from src.retriever import BM25Retriever, DenseRetriever

    logger.info("Recomputing features from scratch (this may take a few minutes)...")
    loader = TriviaQALoader(config)
    corpus, train_queries, _ = loader.load()

    bm25 = BM25Retriever(config)
    bm25.index(corpus)
    dense = DenseRetriever(config)
    dense.index(corpus)

    n_candidates = config["retrieval"]["num_candidates"]
    Xs, ys = [], []
    t0 = time.time()
    for i, q in enumerate(train_queries):
        q_emb = dense.encode_query(q["question"])
        # Simple round-robin candidate merge (same as main.py)
        bm25_hits = bm25.retrieve(q["question"], top_k=n_candidates)
        dense_hits = dense.retrieve(q["question"], top_k=n_candidates)
        seen, merged = set(), []
        for (b_id, _), (d_id, _) in zip(bm25_hits, dense_hits):
            for doc_id in (b_id, d_id):
                if doc_id not in seen:
                    seen.add(doc_id)
                    merged.append(doc_id)
        for doc_id, _ in bm25_hits + dense_hits:
            if doc_id not in seen and len(merged) < n_candidates:
                seen.add(doc_id)
                merged.append(doc_id)
        candidates = merged[:n_candidates]

        X, ordered_ids = build_feature_matrix(q["question"], q_emb, candidates, dense, bm25, config)
        y = build_label_vector(ordered_ids, q["relevant_doc_ids"])
        Xs.append(X)
        ys.append(y)
        if (i + 1) % 25 == 0:
            logger.info("  Processed %d/%d train queries", i + 1, len(train_queries))

    X_train = np.vstack(Xs)
    y_train = np.concatenate(ys)
    logger.info("Features ready  shape=%s  time=%.1fs", X_train.shape, time.time() - t0)
    return X_train, y_train, config["retrieval"]["num_candidates"]


# ── Main sensitivity sweep ─────────────────────────────────────────────────────

def run_sensitivity(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambdas: List[float],
    n_candidates: int,
    solver_name: str = "OSQP",
    k: int = 5,
) -> List[dict]:
    """
    For each lambda, solve the QP and record weights + metrics.
    Returns a list of result dicts sorted by lambda.
    """
    records = []
    logger.info("Running lambda sweep over %d values...", len(lambdas))
    for lam in sorted(lambdas):
        weights, opt_val, status = solve_qp(X_train, y_train, lam, solver_name)
        recall5 = recall_at_k_from_matrix(X_train, y_train, weights, n_candidates, k=k)

        record = {
            "lambda": lam,
            "log10_lambda": float(np.log10(lam)),
            "weights": weights.tolist(),
            "w_cosine": float(weights[0]),
            "w_bm25": float(weights[1]),
            "w_diversity": float(weights[2]),
            "optimal_value": opt_val,
            "recall_at_k": recall5,
            "k": k,
            "solver_status": status,
        }
        records.append(record)
        logger.info(
            "  λ=%8.4f  cosine=%.3f  bm25=%.3f  div=%.3f  Recall@%d=%.3f  obj=%.4f",
            lam, weights[0], weights[1], weights[2], k, recall5, opt_val,
        )

    return records


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sensitivity(records: List[dict], out_path: Path, title_suffix: str = "") -> None:
    """Generate the four-panel fig8 sensitivity figure."""

    log_lambdas = [r["log10_lambda"] for r in records]
    lambdas     = [r["lambda"] for r in records]
    w_cosine    = [r["w_cosine"] for r in records]
    w_bm25      = [r["w_bm25"] for r in records]
    w_diversity = [r["w_diversity"] for r in records]
    obj_vals    = [r["optimal_value"] for r in records]
    recall_vals = [r["recall_at_k"] for r in records]
    k           = records[0]["k"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Figure 8 — Lambda (λ) Regularization Sensitivity Analysis\n"
        f"QP: min ‖Xw−y‖² + λ‖w‖²  s.t. w≥0, Σw=1{title_suffix}",
        fontsize=12, y=1.01,
    )

    # ── Panel (a): Regularization path ───────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(log_lambdas, w_cosine,   "o-", color=FEATURE_COLORS[0],
            linewidth=2, markersize=6, label=FEATURE_NAMES[0])
    ax.plot(log_lambdas, w_bm25,     "s-", color=FEATURE_COLORS[1],
            linewidth=2, markersize=6, label=FEATURE_NAMES[1])
    ax.plot(log_lambdas, w_diversity, "^-", color=FEATURE_COLORS[2],
            linewidth=2, markersize=6, label=FEATURE_NAMES[2])
    ax.axhline(1/3, color="gray", linestyle=":", linewidth=1.2,
               label="Equal weight (1/3)")

    # Mark the default lambda (0.01)
    default_log = np.log10(0.01)
    ax.axvline(default_log, color="black", linestyle="--", linewidth=1,
               alpha=0.6, label=f"λ=0.01 (default)")

    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("Learned Weight  w*")
    ax.set_title("(a) Regularization Path")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ── Panel (b): Optimal objective value ────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(log_lambdas, obj_vals, "D-", color="#9333EA",
            linewidth=2, markersize=6)
    ax.axvline(default_log, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("QP Optimal Objective  ‖Xw*−y‖² + λ‖w*‖²")
    ax.set_title("(b) Objective Value vs λ\n(increases as regularization tightens)")

    # Annotate minimum
    min_idx = int(np.argmin(obj_vals))
    ax.annotate(
        f"min @ λ={lambdas[min_idx]:.4f}",
        xy=(log_lambdas[min_idx], obj_vals[min_idx]),
        xytext=(log_lambdas[min_idx] + 0.4, obj_vals[min_idx] + (max(obj_vals) - min(obj_vals)) * 0.1),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    # ── Panel (c): Recall@k ───────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(log_lambdas, recall_vals, "o-", color="#EA580C",
            linewidth=2, markersize=6)
    ax.axvline(default_log, color="black", linestyle="--", linewidth=1,
               alpha=0.6, label=f"λ=0.01 (default)")

    # Annotate maximum recall
    max_idx = int(np.argmax(recall_vals))
    ax.annotate(
        f"peak @ λ={lambdas[max_idx]:.4f}\nRecall={recall_vals[max_idx]:.3f}",
        xy=(log_lambdas[max_idx], recall_vals[max_idx]),
        xytext=(log_lambdas[max_idx] + 0.3,
                recall_vals[max_idx] - (max(recall_vals) - min(recall_vals)) * 0.3),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel(f"Recall@{k}  (training set)")
    ax.set_title(f"(c) Training Recall@{k} vs λ\n(proxy for retrieval quality)")
    ax.legend(loc="lower left", fontsize=9)

    # ── Panel (d): Stacked area chart ─────────────────────────────────────────
    ax = axes[1, 1]
    log_arr = np.array(log_lambdas)
    wc = np.array(w_cosine)
    wb = np.array(w_bm25)
    wd = np.array(w_diversity)

    ax.stackplot(
        log_arr,
        wc, wb, wd,
        labels=FEATURE_NAMES,
        colors=FEATURE_COLORS,
        alpha=0.75,
    )
    ax.axvline(default_log, color="black", linestyle="--", linewidth=1.2,
               alpha=0.8, label="λ=0.01 (default)")
    ax.set_xlabel("log₁₀(λ)")
    ax.set_ylabel("Weight Composition")
    ax.set_title("(d) Weight Composition vs λ\n(stacked area — shows how regularization blends signals)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved %s", out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lambda sensitivity analysis for Convex QP")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS,
                        help="Lambda values to sweep (space-separated floats)")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute X_train from scratch even if X_train.npz exists")
    parser.add_argument("--k", type=int, default=5,
                        help="k value for Recall@k metric (default: 5)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["output"]["results_dir"])
    features_path = results_dir / "X_train.npz"
    solver_name = config["optimization"]["solver"]

    # Load or recompute features
    n_candidates = config["retrieval"]["num_candidates"]
    if not args.recompute and features_path.exists():
        logger.info("Loading training features from %s", features_path)
        data = np.load(features_path)
        X_train = data["X_train"]
        y_train = data["y_train"]
        logger.info("  X_train shape: %s   positives=%.1f%%", X_train.shape, 100 * y_train.mean())
    else:
        if not features_path.exists():
            logger.info("X_train.npz not found — recomputing features from scratch.")
        X_train, y_train, n_candidates = recompute_features(config)
        # Save for future runs
        np.savez(features_path, X_train=X_train, y_train=y_train)
        logger.info("Saved features to %s", features_path)

    # Run sweep
    records = run_sensitivity(
        X_train, y_train,
        lambdas=args.lambdas,
        n_candidates=n_candidates,
        solver_name=solver_name,
        k=args.k,
    )

    # Save raw results
    json_path = results_dir / "lambda_results.json"
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Lambda sweep results saved to %s", json_path)

    # Generate figure
    fig_path = results_dir / "figures" / "fig8_lambda_sensitivity.png"
    n_train = config["dataset"]["num_train_queries"]
    n_test = config["dataset"]["num_test_queries"]
    plot_sensitivity(
        records,
        fig_path,
        title_suffix=f"  |  TriviaQA: {n_train} train / {n_test} test queries",
    )

    # Print summary table
    print("\n" + "=" * 75)
    print(f"{'λ':>10}  {'w_cosine':>10}  {'w_bm25':>10}  {'w_div':>10}  {'Recall@%d' % args.k:>12}  {'Obj Value':>12}")
    print("-" * 75)
    for r in records:
        print(f"{r['lambda']:>10.4f}  {r['w_cosine']:>10.4f}  {r['w_bm25']:>10.4f}  "
              f"{r['w_diversity']:>10.4f}  {r['recall_at_k']:>12.4f}  {r['optimal_value']:>12.4f}")
    print("=" * 75)
    print(f"\nFigure saved to: {fig_path.resolve()}")
    print(f"Results saved to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
