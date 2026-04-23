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

After fitting, both expose a .score(X) method for ranking candidates.
"""

import logging
from typing import Any, Dict, List, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

N_FEATURES = 3  # cosine, bm25, diversity
FEATURE_NAMES = ["cosine", "bm25", "diversity"]


# ---------------------------------------------------------------------------
def _project_simplex(w: np.ndarray) -> np.ndarray:
    """Project onto the non-negative probability simplex (w >= 0, sum = 1)."""
    w = np.abs(w)          # ensure non-negative before projection
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
            self.weights /= self.weights.sum()  # re-normalize after clipping

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
        # Use the middle k value as the optimization target
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
        """
        Parameters
        ----------
        feature_matrices       : list of X matrices, one per training query
        relevant_doc_ids_list  : list of relevant doc_id sets, one per query
        candidate_doc_ids_list : list of candidate doc_id lists, one per query

        Returns
        -------
        weights : (3,) weight vector on the simplex
        """
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
            return -total / n_queries  # negate because scipy minimizes

        # Start from uniform weights
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
            self.k,
            result.fun,
            self.converged,
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
