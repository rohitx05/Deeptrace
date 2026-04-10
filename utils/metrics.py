"""Metric computation helpers for evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
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
