"""
Zero-Shot Evaluation of DeepTrace (v2_clip_finetune) on FaceForensics++ C23.
Computes overall and per-manipulation metrics without any fine-tuning.

Manipulation Subsets:
- FaceSwap
- Deepfakes
- Face2Face
- NeuralTextures
- FaceShifter
- DeepFakeDetection
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from datasets.multisource import MultiSourceDataset
from utils.checkpoint import load_checkpoint
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ffpp_zeroshot")


def evaluate_records(model, records, device, batch_size=64):
    ds = MultiSourceDataset(records, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            imgs = batch["image"].to(device)
            dcts = batch["dct"].to(device)
            lbls = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs, dct=dcts)
                if isinstance(out, dict):
                    probs = out.get("binary_pred", torch.sigmoid(out.get("binary_logit", out.get("logits"))))
                else:
                    probs = torch.sigmoid(out)

            all_preds.extend(probs.detach().cpu().numpy().reshape(-1).tolist())
            all_labels.extend(lbls.detach().cpu().numpy().reshape(-1).tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)
    y_pred = (y_prob >= 0.5).astype(float)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "f1_score": float(f1),
        "brier_score": float(brier),
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


def main():
    device = get_device()
    logger.info(f"Running FaceForensics++ Zero-Shot Benchmark on {device}")

    # Load DeepTrace Checkpoint
    ckpt_path = "checkpoints/v2_clip_finetune/best_model.pth"
    model = DeepfakeDetector()
    load_checkpoint(ckpt_path, model, device=device)
    model.to(device)
    model.eval()

    # Load FF++ Manifest
    manifest_df = pd.read_csv("manifests/ffpp_c23_manifest.csv")
    logger.info(f"Loaded manifest with {len(manifest_df):,} face frames")

    # Sample 1,000 real images for evaluation
    reals = manifest_df[manifest_df["manipulation_type"] == "real"].sample(n=2000, random_state=42)
    real_records = [(r["filepath"], 0, "real") for _, r in reals.iterrows()]

    results_per_method = {}
    all_eval_records = list(real_records)

    manip_types = ["FaceSwap", "Deepfakes", "Face2Face", "NeuralTextures", "FaceShifter", "DeepFakeDetection"]

    for m in manip_types:
        fakes = manifest_df[manifest_df["manipulation_type"] == m].sample(n=2000, random_state=42)
        fake_records = [(r["filepath"], 1, m) for _, r in fakes.iterrows()]
        all_eval_records.extend(fake_records)

        # Evaluate real vs this specific manipulation
        subset_records = real_records + fake_records
        logger.info(f"Evaluating Zero-Shot: Real vs. {m} (N={len(subset_records)})...")
        res = evaluate_records(model, subset_records, device)
        results_per_method[m] = res
        logger.info(f"  --> {m} Zero-Shot Acc: {res['accuracy']*100:.2f}%, AUC: {res['roc_auc']:.4f}")

    # Overall FF++ Evaluation
    logger.info("Evaluating Overall FF++ Benchmark...")
    overall_res = evaluate_records(model, all_eval_records, device)
    logger.info(f"=== OVERALL FF++ ZERO-SHOT ACCURACY: {overall_res['accuracy']*100:.2f}%, ROC-AUC: {overall_res['roc_auc']:.4f} ===")

    final_report = {
        "model": "DeepTrace (v2_clip_finetune) Zero-Shot",
        "checkpoint": ckpt_path,
        "overall": overall_res,
        "per_manipulation": results_per_method,
    }

    out_file = Path("results/benchmark_eval_v2/ffpp_zeroshot_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"FF++ Zero-Shot results saved to: {out_file}")


if __name__ == "__main__":
    main()
