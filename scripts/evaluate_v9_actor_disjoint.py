"""
DeepTrace V9: Strictly Actor-Disjoint Leak-Free Evaluation Engine.

Evaluates the clean V9 model (trained with 0% actor overlap) against the
strictly held-out 400 test actors (1,420 fakes per cohort, independent real pools).
"""

import sys
import json
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from scripts.train_v9_actor_disjoint import V9ActorDisjointDetector
from utils.checkpoint import load_checkpoint
from utils.device import get_device, AMPContext
from utils.actor_splits import get_actor_disjoint_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_v9")


class EvalDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (160, 160), (128, 128, 128))
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def find_optimal_threshold_youden(targets, probs):
    fpr, tpr, thresholds = roc_curve(targets, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(j_scores[best_idx]), float(tpr[best_idx]), float(1.0 - fpr[best_idx])


def evaluate_model(model, samples, device):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = EvalDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    all_logits, all_targets = [], []
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with AMPContext(device=device, enabled=True):
                out = model(images, return_spectral_details=False)
                logits = out["binary_logit"].cpu().numpy()
            all_logits.extend(logits)
            all_targets.extend(targets.numpy())

    logits = np.array(all_logits)
    targets = np.array(all_targets)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

    auc = float(roc_auc_score(targets, probs))
    acc_raw = float(accuracy_score(targets, probs >= 0.5))
    opt_thresh, j_stat, tpr_opt, tnr_opt = find_optimal_threshold_youden(targets, probs)
    acc_cal = float(accuracy_score(targets, probs >= opt_thresh))
    f1_cal = float(f1_score(targets, probs >= opt_thresh))
    brier = float(brier_score_loss(targets, probs))

    return {
        "roc_auc": auc,
        "accuracy_default_05": acc_raw,
        "optimal_threshold_youden": opt_thresh,
        "accuracy_calibrated": acc_cal,
        "f1_calibrated": f1_cal,
        "brier_score": brier,
        "sensitivity_tpr": tpr_opt,
        "specificity_tnr": tnr_opt,
        "real_logit_mean": float(logits[targets == 0].mean()),
        "fake_logit_mean": float(logits[targets == 1].mean()),
        "n_samples": len(samples),
    }


def main():
    device = get_device()
    logger.info("=== DeepTrace V9 Actor-Disjoint Evaluation ===")

    manifest_path = Path("manifests/ffpp_c23_manifest.csv")
    df = pd.read_csv(manifest_path)
    col = "filepath" if "filepath" in df.columns else "image_path"
    split = get_actor_disjoint_split(df, filepath_col=col)

    base_detector = DeepfakeDetector()
    model_v9 = V9ActorDisjointDetector(base_detector)
    ckpt_v9 = "checkpoints/v9_actor_disjoint/best_model.pth"
    load_checkpoint(ckpt_v9, model_v9, device=device)
    model_v9.to(device).eval()

    test_real_paths = split["test_reals"][col].tolist()
    cohort_names = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]

    report = {
        "evaluation_name": "DeepTrace V9 Clean Actor-Disjoint Evaluation (100% Leak-Free)",
        "cohorts": {},
    }
    all_aucs, all_accs = [], []

    for i, c_name in enumerate(cohort_names):
        test_fake_paths = split["test_fakes"][c_name][col].tolist()
        n_fake = len(test_fake_paths)
        rng = random.Random(42 + i)
        real_sample = rng.sample(test_real_paths, min(n_fake, len(test_real_paths)))
        n_eval = min(len(real_sample), n_fake)

        samples = [(p, 0) for p in real_sample[:n_eval]] + [(p, 1) for p in test_fake_paths[:n_eval]]
        res_v9 = evaluate_model(model_v9, samples, device)
        report["cohorts"][c_name] = res_v9
        all_aucs.append(res_v9["roc_auc"])
        all_accs.append(res_v9["accuracy_calibrated"])

        logger.info(
            f"V9 [{c_name}] (N={len(samples)} Balanced):\n"
            f"  ROC-AUC: {res_v9['roc_auc']:.4f} | Acc @ 0.5: {res_v9['accuracy_default_05']*100:.2f}%\n"
            f"  Optimal Threshold: t* = {res_v9['optimal_threshold_youden']:.4f} | Cal. Acc: {res_v9['accuracy_calibrated']*100:.2f}%\n"
            f"  Mean Logits: Real = {res_v9['real_logit_mean']:.2f} | Fake = {res_v9['fake_logit_mean']:.2f}"
        )

    macro_auc = float(np.mean(all_aucs))
    macro_acc = float(np.mean(all_accs))
    report["macro_averages"] = {
        "macro_roc_auc": macro_auc,
        "macro_calibrated_accuracy": macro_acc,
    }
    logger.info(f"\n🏆 V9 Clean Macro AUC (5 Cohorts): {macro_auc:.4f} | Macro Calibrated Acc: {macro_acc*100:.2f}%")

    out_file = Path("results/benchmark_eval_v7/v9_clean_actor_disjoint_eval.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"V9 Evaluation Saved: {out_file}")


if __name__ == "__main__":
    main()
