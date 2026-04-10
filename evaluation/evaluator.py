"""
Evaluation module for deepfake detection.
Computes per-dataset and cross-dataset metrics with report generation.
"""

import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils.metrics import compute_metrics, compute_roc_curve, print_metrics, find_optimal_threshold
from utils.device import get_device

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate deepfake detection model on various datasets."""

    def __init__(self, model, device=None, use_amp: bool = True):
        self.model = model
        self.device = device or get_device()
        self.use_amp = use_amp
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader, dataset_name: str = "test") -> dict:
        """
        Evaluate on a single dataset.

        Returns:
            dict with metrics, predictions, and labels
        """
        all_labels = []
        all_probs = []
        all_manip_true = []
        all_manip_pred = []

        pbar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            mode = "video" if "frames" in batch else "image"

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if mode == "video":
                    preds = self.model(frames=batch["frames"], dct_frames=batch.get("dct_frames"), mode="video")
                else:
                    preds = self.model(images=batch["image"], dct=batch.get("dct"), mode="image")

            probs = preds["binary_pred"].cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_labels.extend(labels)
            all_probs.extend(probs)

            manip_true = batch["manipulation_type"].cpu().numpy()
            manip_pred = preds["manipulation_pred"].cpu().numpy()
            all_manip_true.extend(manip_true)
            all_manip_pred.extend(manip_pred)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(all_labels, all_probs)
        all_preds = (all_probs > optimal_threshold).astype(int)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics["optimal_threshold"] = optimal_threshold
        metrics["dataset"] = dataset_name
        metrics["num_samples"] = len(all_labels)

        print_metrics(metrics, dataset_name)

        return {
            "metrics": metrics,
            "labels": all_labels.tolist(),
            "probs": all_probs.tolist(),
            "preds": all_preds.tolist(),
        }

    def generate_report(self, results: dict, output_dir: str = "results/"):
        """Generate evaluation report with plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_name = results["metrics"]["dataset"]

        # Save metrics JSON
        with open(output_path / f"{dataset_name}_metrics.json", "w") as f:
            json.dump(results["metrics"], f, indent=2)

        labels = np.array(results["labels"])
        probs = np.array(results["probs"])
        preds = np.array(results["preds"])

        # ROC curve
        roc_data = compute_roc_curve(labels, probs)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(roc_data["fpr"], roc_data["tpr"], "b-", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — {dataset_name} (AUC={results['metrics'].get('roc_auc', 0):.4f})", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f"{dataset_name}_roc.png", dpi=150)
        plt.close()

        # Confusion matrix
        cm = np.array(results["metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {dataset_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / f"{dataset_name}_confusion_matrix.png", dpi=150)
        plt.close()

        # Probability distribution
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(probs[labels == 0], bins=50, alpha=0.6, label="Real", color="green")
        ax.hist(probs[labels == 1], bins=50, alpha=0.6, label="Fake", color="red")
        ax.axvline(results["metrics"]["optimal_threshold"], color="black", linestyle="--",
                   label=f"Threshold={results['metrics']['optimal_threshold']:.3f}")
        ax.set_xlabel("Fake Probability", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Score Distribution — {dataset_name}", fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"{dataset_name}_distribution.png", dpi=150)
        plt.close()

        logger.info(f"Evaluation report saved to {output_path}")
