"""
optimizer.py
------------
Two optimizers for learning retrieval signal weights w = [w_cosine, w_bm25, w_div].

ConvexOptimizer
  Solves a constrained Quadratic Program (QP) via CVXPY.
  Objective : minimize  ||Xw - y||^2 + lambda * ||w||^2
  Constraints:  w >= 0,  sum(w) = 1   (probability simplex)
  This is strongly convex → unique global minimum, solved efficiently by OSQP.

NonConvexOptimizer
  Directly maximizes Recall@k using gradient-free Nelder-Mead (scipy).
  The objective is non-differentiable (rank-based), justifying the use of a
  derivative-free method.  Uses the same pre-computed feature matrices as the
  QP so both optimizers are compared fairly.

ClusterAdaptiveOptimizer
  Extends ConvexOptimizer to learn K separate weight vectors — one per cluster
  of queries with similar retrieval behavior.

  Training:
    1. Compute a (3,) query profile = mean(X_i) for each training query i.
       This summarizes whether the query leans semantic, lexical, or diverse.
    2. Run KMeans on the (n_train, 3) profile matrix → K clusters.
    3. For each cluster, stack only the feature matrices from member queries
       and solve a fresh ConvexQP → cluster-specific weight vector w_k.

  Inference:
    Assign test query to nearest centroid (L2), use that cluster's w_k.

  Convexity argument:
    Each per-cluster QP is independently strongly convex (same guarantees as
    the global ConvexOptimizer).  Clustering is a fixed preprocessing step —
    it does not affect the convex structure of any individual QP.
    We therefore have K convex problems, each with a provably unique global
    optimum.

  K-path analysis:
    run_k_path() solves for K ∈ {2, 3, 4} and returns per-K weights +
    inertia, enabling a principled model-selection argument.

run_lambda_path  [NEW]
  Regularization path analysis for the global ConvexOptimizer.
  Solves the QP for λ ∈ {0.001, 0.01, 0.1, 1.0} and records:
    - learned weights at each λ
    - QP optimal value at each λ
    - Recall@k on held-out test queries at each λ

  Purpose: show that Recall@10 is flat across λ values, proving that the
  global QP solution is near-optimal and robust — i.e., the performance
  ceiling is the feature space, not regularization choice.  This directly
  supports the argument that adaptive weights cannot improve over global
  weights given the current 3-feature representation.

After fitting, all optimizers expose a .score(X) method for ranking candidates.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

N_FEATURES = 3  # cosine, bm25, diversity
FEATURE_NAMES = ["cosine", "bm25", "diversity"]


# ---------------------------------------------------------------------------
def _project_simplex(w: np.ndarray) -> np.ndarray:
    """Project onto the non-negative probability simplex (w >= 0, sum = 1)."""
    w = np.abs(w)
    total = w.sum()
    if total < 1e-10:
        return np.ones(len(w)) / len(w)
    return w / total


# ---------------------------------------------------------------------------
class ConvexOptimizer:
    """
    Constrained QP via CVXPY.
    This is a convex problem — the objective is strongly convex (l2 reg + quadratic),
    and constraints are linear, so CVXPY guarantees a unique global optimum.
    """

    def __init__(self, config: Dict[str, Any]):
        self.lam = config["optimization"]["lambda_reg"]
        self.solver_name = config["optimization"]["solver"]
        self.weights: np.ndarray | None = None
        self.optimal_value: float | None = None
        self.solve_status: str = "not_solved"

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (n_total_candidates, 3)  — stacked feature matrices from all train queries
        y : (n_total_candidates,)    — binary relevance labels (0 or 1)

        Returns
        -------
        weights : (3,) optimal weight vector on the simplex
        """
        w = cp.Variable(N_FEATURES)

        objective = cp.Minimize(
            cp.sum_squares(X @ w - y) + self.lam * cp.sum_squares(w)
        )
        constraints = [w >= 0, cp.sum(w) == 1]

        problem = cp.Problem(objective, constraints)
        solver = getattr(cp, self.solver_name, cp.OSQP)
        problem.solve(solver=solver, verbose=False)

        self.solve_status = problem.status
        self.optimal_value = problem.value

        if problem.status not in ("optimal", "optimal_inaccurate"):
            logger.warning(
                "QP solver returned status '%s'. Falling back to equal weights.",
                problem.status,
            )
            self.weights = np.ones(N_FEATURES) / N_FEATURES
        else:
            self.weights = np.clip(np.array(w.value).flatten(), 0, None)
            self.weights /= self.weights.sum()

        self._log_weights("Convex QP")
        logger.info(
            "QP  optimal_value=%.6f  status=%s", self.optimal_value, self.solve_status
        )
        return self.weights

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights

    def _log_weights(self, label: str) -> None:
        parts = [f"{n}={v:.4f}" for n, v in zip(FEATURE_NAMES, self.weights)]
        logger.info("%s weights:  %s", label, "  ".join(parts))


# ---------------------------------------------------------------------------
class ClusterAdaptiveOptimizer:
    """
    Query-adaptive weight learning via K-means clustering + per-cluster QP.

    Each cluster gets its own convex QP — K independent strongly-convex
    problems, each with a guaranteed unique global optimum.

    At inference time, a test query is assigned to its nearest cluster
    centroid (by L2 distance in profile space), and that cluster's weight
    vector is used for ranking.
    """

    def __init__(self, config: Dict[str, Any], n_clusters: int = 3):
        self.config = config
        self.n_clusters = n_clusters
        self.lam = config["optimization"]["lambda_reg"]
        self.min_cluster_size: int = config.get("adaptive", {}).get(
            "min_cluster_size", 10
        )
        self.seed: int = config.get("dataset", {}).get("seed", 42)

        # Filled after fit()
        self.cluster_weights: List[np.ndarray] = []   # shape: (K, 3)
        self.centroids: np.ndarray | None = None       # shape: (K, 3)
        self.cluster_labels: np.ndarray | None = None  # shape: (n_train,)
        self.cluster_qp_values: List[float] = []       # optimal QP value per cluster
        self.cluster_sizes: List[int] = []             # num queries per cluster
        self.inertia: float = 0.0                      # KMeans inertia
        self.weights: np.ndarray | None = None         # fallback global weights

    # ------------------------------------------------------------------
    def fit(
        self,
        query_profiles: np.ndarray,
        feature_matrices: List[np.ndarray],
        label_vectors: List[np.ndarray],
        global_weights: np.ndarray,
    ) -> "ClusterAdaptiveOptimizer":
        """
        Parameters
        ----------
        query_profiles  : (n_train, 3) — one profile vector per training query
                          Each profile = mean(X_i, axis=0), summarizing the
                          query's retrieval behavior across its candidates.
        feature_matrices: list of X_i, one per training query
        label_vectors   : list of y_i (binary relevance), one per training query
        global_weights  : (3,) fallback — used when a cluster is too small

        Returns
        -------
        self
        """
        n_train = len(feature_matrices)
        assert query_profiles.shape == (n_train, N_FEATURES), (
            f"query_profiles must be ({n_train}, {N_FEATURES}), "
            f"got {query_profiles.shape}"
        )

        # ── Step 1: Cluster query profiles ────────────────────────────
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=10,
        )
        self.cluster_labels = kmeans.fit_predict(query_profiles)
        self.centroids = kmeans.cluster_centers_     # (K, 3)
        self.inertia = float(kmeans.inertia_)

        logger.info(
            "KMeans (K=%d)  inertia=%.4f  cluster_sizes=%s",
            self.n_clusters,
            self.inertia,
            np.bincount(self.cluster_labels).tolist(),
        )

        # ── Step 2: Per-cluster QP ─────────────────────────────────────
        self.cluster_weights = []
        self.cluster_qp_values = []
        self.cluster_sizes = []

        for k in range(self.n_clusters):
            member_indices = np.where(self.cluster_labels == k)[0]
            cluster_size = len(member_indices)
            self.cluster_sizes.append(cluster_size)

            if cluster_size < self.min_cluster_size:
                # Cluster too small — fall back to global weights
                logger.warning(
                    "Cluster %d has only %d members (< min=%d). "
                    "Using global weights as fallback.",
                    k, cluster_size, self.min_cluster_size,
                )
                self.cluster_weights.append(global_weights.copy())
                self.cluster_qp_values.append(float("nan"))
                continue

            # Stack feature matrices and labels for this cluster
            X_k = np.vstack([feature_matrices[i] for i in member_indices])
            y_k = np.concatenate([label_vectors[i] for i in member_indices])

            # Solve per-cluster QP (same formulation as global ConvexOptimizer)
            w_var = cp.Variable(N_FEATURES)
            objective = cp.Minimize(
                cp.sum_squares(X_k @ w_var - y_k) + self.lam * cp.sum_squares(w_var)
            )
            constraints = [w_var >= 0, cp.sum(w_var) == 1]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status not in ("optimal", "optimal_inaccurate"):
                logger.warning(
                    "Cluster %d QP returned status '%s'. Using global weights.",
                    k, problem.status,
                )
                w_k = global_weights.copy()
                qp_val = float("nan")
            else:
                w_k = np.clip(np.array(w_var.value).flatten(), 0, None)
                w_k /= w_k.sum()
                qp_val = float(problem.value)

            self.cluster_weights.append(w_k)
            self.cluster_qp_values.append(qp_val)

            parts = [f"{n}={v:.4f}" for n, v in zip(FEATURE_NAMES, w_k)]
            logger.info(
                "Cluster %d  (n=%d)  weights: %s  qp_val=%.6f",
                k, cluster_size, "  ".join(parts), qp_val,
            )

        # Store a flat weight for any single-query fallback in score()
        self.weights = global_weights.copy()
        return self

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, query_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Score candidates for a single query.

        Parameters
        ----------
        X             : (n_candidates, 3) feature matrix for this query
        query_profile : (3,) profile vector for this query.
                        If None, falls back to global weights (graceful degradation).

        Returns
        -------
        scores : (n_candidates,) ranking scores
        """
        if query_profile is None or self.centroids is None:
            return X @ self.weights

        # Assign to nearest centroid by L2 distance
        dists = np.linalg.norm(self.centroids - query_profile, axis=1)
        cluster_id = int(np.argmin(dists))
        return X @ self.cluster_weights[cluster_id]

    # ------------------------------------------------------------------
    def get_cluster_id(self, query_profile: np.ndarray) -> int:
        """Return the nearest cluster index for a given query profile."""
        dists = np.linalg.norm(self.centroids - query_profile, axis=1)
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a serializable summary of fit results."""
        return {
            "n_clusters": self.n_clusters,
            "inertia": self.inertia,
            "cluster_sizes": self.cluster_sizes,
            "cluster_weights": [w.tolist() for w in self.cluster_weights],
            "cluster_qp_values": self.cluster_qp_values,
            "centroids": self.centroids.tolist() if self.centroids is not None else [],
        }


# ---------------------------------------------------------------------------
def run_k_path(
    config: Dict[str, Any],
    query_profiles: np.ndarray,
    feature_matrices: List[np.ndarray],
    label_vectors: List[np.ndarray],
    global_weights: np.ndarray,
    k_values: List[int] = [2, 3, 4],
) -> Dict[int, ClusterAdaptiveOptimizer]:
    """
    Regularization-path style analysis over cluster counts K ∈ k_values.

    Solves the full ClusterAdaptiveOptimizer for each K, returning a dict
    keyed by K. This enables:
      - Inertia elbow curve (model selection)
      - Per-K weight diversity analysis (do clusters learn different weights?)
      - Comparison of retrieval metrics across K values

    Parameters
    ----------
    config          : full config dict
    query_profiles  : (n_train, 3)
    feature_matrices: list of X_i per training query
    label_vectors   : list of y_i per training query
    global_weights  : (3,) fallback weights from global ConvexOptimizer
    k_values        : list of K values to sweep

    Returns
    -------
    k_path : dict mapping K → fitted ClusterAdaptiveOptimizer
    """
    k_path: Dict[int, ClusterAdaptiveOptimizer] = {}

    logger.info("=" * 50)
    logger.info("K-PATH ANALYSIS  K values = %s", k_values)
    logger.info("=" * 50)

    for k in k_values:
        logger.info("Fitting ClusterAdaptiveOptimizer  K=%d ...", k)
        opt = ClusterAdaptiveOptimizer(config, n_clusters=k)
        opt.fit(query_profiles, feature_matrices, label_vectors, global_weights)
        k_path[k] = opt

        logger.info(
            "K=%d  inertia=%.4f  cluster_sizes=%s",
            k, opt.inertia, opt.cluster_sizes,
        )

    # Log weight diversity: how different are the per-cluster weights?
    logger.info("\nWeight diversity across K values:")
    for k, opt in k_path.items():
        W = np.array(opt.cluster_weights)          # (K, 3)
        diversity = float(np.std(W, axis=0).mean())  # mean std across features
        logger.info(
            "  K=%d  mean_weight_std=%.4f  (higher = clusters learned more distinct weights)",
            k, diversity,
        )

    return k_path


# ---------------------------------------------------------------------------
class NonConvexOptimizer:
    """
    Gradient-free optimizer that directly maximizes Recall@k.
    The objective (ranking-based recall) is non-differentiable and non-convex,
    which is why we use a derivative-free method (Nelder-Mead).

    Uses the same pre-computed feature matrices as ConvexOptimizer for a fair
    comparison — the only difference is the choice of objective function.
    """

    def __init__(self, config: Dict[str, Any]):
        self.method = config["optimization"]["nonconvex_method"]
        self.maxiter = config["optimization"]["nonconvex_maxiter"]
        k_vals = config["evaluation"]["k_values"]
        self.k = k_vals[len(k_vals) // 2]
        self.weights: np.ndarray | None = None
        self.converged: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        feature_matrices: List[np.ndarray],
        relevant_doc_ids_list: List[List[str]],
        candidate_doc_ids_list: List[List[str]],
    ) -> np.ndarray:
        n_queries = len(feature_matrices)

        def neg_recall(w_raw: np.ndarray) -> float:
            w = _project_simplex(w_raw)
            total = 0.0
            for X, relevant, candidates in zip(
                feature_matrices, relevant_doc_ids_list, candidate_doc_ids_list
            ):
                scores = X @ w
                top_k_ids = [candidates[i] for i in np.argsort(scores)[::-1][: self.k]]
                hits = len(set(top_k_ids) & set(relevant))
                total += hits / max(len(relevant), 1)
            return -total / n_queries

        w0 = np.ones(N_FEATURES) / N_FEATURES
        result = minimize(
            neg_recall,
            w0,
            method=self.method,
            options={"maxiter": self.maxiter, "xatol": 1e-5, "fatol": 1e-5},
        )

        self.converged = result.success
        self.weights = _project_simplex(result.x)

        self._log_weights("Non-Convex")
        logger.info(
            "Non-Convex  neg_recall@%d=%.4f  converged=%s",
            self.k, result.fun, self.converged,
        )
        return self.weights

    # ------------------------------------------------------------------
    def score(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights

    def _log_weights(self, label: str) -> None:
        parts = [f"{n}={v:.4f}" for n, v in zip(FEATURE_NAMES, self.weights)]
        logger.info("%s weights:  %s", label, "  ".join(parts))


# ---------------------------------------------------------------------------
class EqualWeightBaseline:
    """Uniform baseline: w = [1/3, 1/3, 1/3]. No training needed."""

    def __init__(self):
        self.weights = np.ones(N_FEATURES) / N_FEATURES

    def score(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


class SingleSignalBaseline:
    """Uses only one signal (cosine, bm25, or diversity). Useful as ablation."""

    SIGNAL_MAP = {"cosine": 0, "bm25": 1, "diversity": 2}

    def __init__(self, signal: str):
        if signal not in self.SIGNAL_MAP:
            raise ValueError(f"signal must be one of {list(self.SIGNAL_MAP.keys())}")
        self.weights = np.zeros(N_FEATURES)
        self.weights[self.SIGNAL_MAP[signal]] = 1.0

    def score(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


# ---------------------------------------------------------------------------
def run_lambda_path(
    config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_feature_matrices: List[np.ndarray],
    test_candidate_ids: List[List[str]],
    test_relevant_ids: List[List[str]],
    lambda_values: List[float] = [0.001, 0.01, 0.1, 1.0],
    k: int = 10,
) -> Dict[str, Any]:
    """
    Regularization path analysis for the global ConvexOptimizer.

    Solves the QP for each λ in lambda_values, recording:
      - learned weights w(λ)         — how does regularization shift the solution?
      - QP optimal value f*(λ)       — objective value at the solution
      - Recall@k on test queries     — does retrieval performance change with λ?

    Key insight this proves
    ----------------------
    If Recall@k is approximately flat across λ values, the global QP solution
    is ROBUST — the optimizer is not overfitting to λ=0.01, and the performance
    ceiling is determined by the feature space, not the regularization.

    This directly supports the argument:
      "Adaptive weights cannot improve over global weights because the 3-feature
       representation is the binding constraint, not the optimization."

    Parameters
    ----------
    config               : full config dict (used to build ConvexOptimizer instances)
    X_train              : (n_total_candidates, 3) stacked training feature matrix
    y_train              : (n_total_candidates,) stacked binary labels
    test_feature_matrices: list of X_i per test query  (for Recall@k evaluation)
    test_candidate_ids   : list of candidate doc_id lists per test query
    test_relevant_ids    : list of relevant doc_id lists per test query
    lambda_values        : list of λ values to sweep
    k                    : Recall@k to compute at each λ

    Returns
    -------
    results : dict with keys:
        "lambda_values"  : list of λ values swept
        "weights"        : list of (3,) weight vectors, one per λ
        "qp_values"      : list of QP optimal values, one per λ
        "recall_at_k"    : list of Recall@k scores, one per λ
        "weight_stability": float — mean L2 distance between consecutive w(λ)
                            Near 0 → weights barely change → solution is stable
    """
    import copy

    lambda_results = {
        "lambda_values": [],
        "weights": [],
        "qp_values": [],
        "recall_at_k": [],
    }

    logger.info("=" * 50)
    logger.info("LAMBDA PATH ANALYSIS  λ values = %s", lambda_values)
    logger.info("=" * 50)

    prev_weights = None

    for lam in lambda_values:
        # Build a temporary config with this λ, leave everything else intact
        tmp_config = copy.deepcopy(config)
        tmp_config["optimization"]["lambda_reg"] = lam

        # Solve QP
        opt = ConvexOptimizer(tmp_config)
        opt.fit(X_train, y_train)

        # Evaluate Recall@k on test queries
        recall_scores = []
        for X_test, candidates, relevant in zip(
            test_feature_matrices, test_candidate_ids, test_relevant_ids
        ):
            scores = opt.score(X_test)
            top_k_ids = [
                candidates[i] for i in np.argsort(scores)[::-1][:k]
            ]
            hits = len(set(top_k_ids) & set(relevant))
            recall_scores.append(hits / max(len(relevant), 1))

        recall_at_k = float(np.mean(recall_scores))

        lambda_results["lambda_values"].append(lam)
        lambda_results["weights"].append(opt.weights.tolist())
        lambda_results["qp_values"].append(float(opt.optimal_value))
        lambda_results["recall_at_k"].append(recall_at_k)

        parts = [f"{n}={v:.4f}" for n, v in zip(FEATURE_NAMES, opt.weights)]
        logger.info(
            "λ=%.4f  weights=[%s]  qp_val=%.6f  Recall@%d=%.4f",
            lam, "  ".join(parts), opt.optimal_value, k, recall_at_k,
        )

    # Weight stability: mean L2 distance between consecutive weight vectors
    # Near 0 → regularization barely changes the solution → it's already robust
    weight_array = np.array(lambda_results["weights"])   # (n_lambdas, 3)
    if len(weight_array) >= 2:
        diffs = np.linalg.norm(np.diff(weight_array, axis=0), axis=1)
        stability = float(diffs.mean())
    else:
        stability = 0.0

    lambda_results["weight_stability"] = stability

    # Summary
    recall_arr = np.array(lambda_results["recall_at_k"])
    logger.info(
        "Lambda path summary:  Recall@%d range=[%.4f, %.4f]  "
        "std=%.4f  weight_stability=%.4f",
        k,
        recall_arr.min(),
        recall_arr.max(),
        float(recall_arr.std()),
        stability,
    )
    logger.info(
        "Interpretation: %s",
        "FLAT — global QP is near-optimal; feature space is the bottleneck."
        if recall_arr.std() < 0.01
        else "VARIABLE — solution is sensitive to λ; consider tuning.",
    )

    return lambda_results