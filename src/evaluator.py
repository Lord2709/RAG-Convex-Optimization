"""
evaluator.py
------------
Retrieval and generation evaluation metrics.

Retrieval metrics  (computed on doc_ids):
  Recall@k       : fraction of relevant docs found in top-k
  Precision@k    : fraction of top-k docs that are relevant
  MRR            : Mean Reciprocal Rank

Generation metrics (computed on text strings):
  Exact Match    : prediction matches any gold answer after normalization
  Token F1       : token overlap between prediction and best-matching gold answer
"""

import logging
import re
import string
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top-k results."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """Reciprocal of the rank of the first relevant document."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]],
    k_values: List[int],
) -> Dict[str, float]:
    """
    Aggregate retrieval metrics across all queries.

    Parameters
    ----------
    all_retrieved : list of retrieved doc_id lists (one per query)
    all_relevant  : list of relevant doc_id lists (one per query)
    k_values      : list of k thresholds to evaluate at

    Returns
    -------
    metrics dict, e.g. {"Recall@5": 0.42, "Precision@5": 0.18, "MRR": 0.51}
    """
    metrics: Dict[str, float] = {}
    for k in k_values:
        recalls = [recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
        precs = [precision_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
        metrics[f"Recall@{k}"] = float(np.mean(recalls))
        metrics[f"Precision@{k}"] = float(np.mean(precs))

    mrrs = [reciprocal_rank(r, rel) for r, rel in zip(all_retrieved, all_relevant)]
    metrics["MRR"] = float(np.mean(mrrs))
    return metrics


# ---------------------------------------------------------------------------
# Text normalization 
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold_answers: List[str]) -> float:
    """1.0 if the normalized prediction matches any normalized gold answer."""
    pred = _normalize(prediction)
    return float(any(_normalize(ans) == pred for ans in gold_answers))


def token_f1(prediction: str, gold_answers: List[str]) -> float:
    """Token-level F1 between prediction and the best-matching gold answer."""
    pred_tokens = _normalize(prediction).split()
    best = 0.0
    for ans in gold_answers:
        gold_tokens = _normalize(ans).split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        prec = len(common) / len(pred_tokens) if pred_tokens else 0.0
        rec = len(common) / len(gold_tokens) if gold_tokens else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best = max(best, f1)
    return best


def evaluate_generation(
    predictions: List[str], gold_answers: List[List[str]]
) -> Dict[str, float]:
    """
    Aggregate EM and F1 across all predictions.

    Parameters
    ----------
    predictions  : list of model-generated answer strings
    gold_answers : list of gold answer lists (each query can have multiple valid answers)
    """
    ems = [exact_match(p, g) for p, g in zip(predictions, gold_answers)]
    f1s = [token_f1(p, g) for p, g in zip(predictions, gold_answers)]
    return {
        "ExactMatch": float(np.mean(ems)),
        "F1": float(np.mean(f1s)),
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_results_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a nicely formatted comparison table of all systems."""
    # Collect all metric names
    all_metrics = []
    for metrics in results.values():
        for m in metrics:
            if m not in all_metrics:
                all_metrics.append(m)

    col_w = 14
    sys_w = 22
    header = f"{'System':<{sys_w}}" + "".join(f"{m:>{col_w}}" for m in all_metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for system, metrics in results.items():
        row = f"{system:<{sys_w}}"
        for m in all_metrics:
            val = metrics.get(m, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * len(header) + "\n")
