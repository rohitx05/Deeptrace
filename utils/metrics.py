"""Metric computation helpers for evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    brier_score_loss,
)
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Compute all detection metrics.

    Args:
        y_true: Ground truth binary labels (0=real, 1=fake)
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for the positive class (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["average_precision"] = 0.0
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None) -> str:
    """Generate a per-class classification report."""
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def compute_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute ROC curve data points."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximizes Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def print_metrics(metrics: dict, dataset_name: str = ""):
    """Pretty-print metrics to log."""
    header = f"--- {dataset_name} Metrics ---" if dataset_name else "--- Metrics ---"
    logger.info(header)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            logger.info(f"  Confusion Matrix:\n{np.array(value)}")
        else:
            logger.info(f"  {key}: {value:.4f}")


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a scalar metric.

    Args:
        y_true:      Ground-truth binary labels.
        y_prob:      Predicted probabilities.
        metric_fn:   Callable(y_true, y_prob) -> float  (e.g. roc_auc_score).
        n_bootstrap: Number of resamples.
        ci:          Confidence level (default 0.95 → 95% CI).
        seed:        Random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except ValueError:
            pass  # skip degenerate resamples (single-class)
    scores = np.array(scores)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100))
    return lower, upper
