"""
Uncertainty Estimation via Monte Carlo Dropout.
Runs N stochastic forward passes at inference time to estimate epistemic uncertainty.
No extra parameters — wraps any model with dropout layers.
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MCDropoutWrapper:
    """
    Monte Carlo Dropout uncertainty estimation.

    At inference: enables dropout, runs N forward passes, computes:
    - mean prediction
    - epistemic uncertainty (std)
    - predictive entropy
    """

    def __init__(self, model: nn.Module, n_passes: int = 10):
        self.model = model
        self.n_passes = n_passes

    def _enable_dropout(self):
        """Enable dropout layers during eval mode."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(self, **kwargs) -> dict:
        """
        Run N stochastic forward passes and aggregate.

        Args:
            **kwargs: same as model.forward() (images, dct, frames, mode, etc.)

        Returns:
            dict with:
                - binary_pred: (B,) mean fake probability
                - confidence: (B,) mean confidence
                - uncertainty: (B,) epistemic uncertainty (std of predictions)
                - predictive_entropy: (B,) entropy of mean prediction
                - all original prediction keys (from last pass)
                - mc_predictions: (N, B) all N predictions
        """
        self.model.eval()
        self._enable_dropout()

        all_binary_preds = []
        all_confidences = []
        last_predictions = None

        with torch.no_grad():
            for _ in range(self.n_passes):
                predictions = self.model(**kwargs)
                all_binary_preds.append(predictions["binary_pred"].cpu())
                all_confidences.append(predictions["confidence"].cpu())
                last_predictions = predictions

        # Stack: (N, B)
        preds_stack = torch.stack(all_binary_preds, dim=0)
        conf_stack = torch.stack(all_confidences, dim=0)

        # Compute statistics
        mean_pred = preds_stack.mean(dim=0)  # (B,)
        std_pred = preds_stack.std(dim=0)    # (B,) — epistemic uncertainty
        mean_conf = conf_stack.mean(dim=0)   # (B,)

        # Predictive entropy: -Σ p·log(p)
        p = mean_pred.clamp(1e-7, 1 - 1e-7)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())

        # Build result
        result = {}
        for k, v in last_predictions.items():
            result[k] = v

        result["binary_pred"] = mean_pred.to(last_predictions["binary_pred"].device)
        result["confidence"] = mean_conf.to(last_predictions["confidence"].device)
        result["uncertainty"] = std_pred.to(last_predictions["binary_pred"].device)
        result["predictive_entropy"] = entropy.to(last_predictions["binary_pred"].device)
        result["mc_predictions"] = preds_stack  # (N, B)

        # Restore model to eval mode
        self.model.eval()

        return result


def compute_calibration_metrics(predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.

    Args:
        predictions: (N,) predicted probabilities
        labels: (N,) binary labels

    Returns:
        dict with ECE, bin accuracies, bin confidences
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(labels[mask].mean())
            bin_confidences.append(predictions[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    bin_counts = np.array(bin_counts)
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)

    # Expected Calibration Error
    total = bin_counts.sum()
    ece = (bin_counts / max(total, 1) * np.abs(bin_accuracies - bin_confidences)).sum()

    return {
        "ece": float(ece),
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_confidences": bin_confidences.tolist(),
        "bin_counts": bin_counts.tolist(),
    }
