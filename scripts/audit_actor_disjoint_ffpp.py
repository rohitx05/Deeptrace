"""
Actor-Disjoint Leak-Free Evaluation on FaceForensics++ c23.

Uses utils/actor_splits.py for split consistency with training.
Independent real pools per cohort. DFD excluded. Youden's Index calibration.
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
from scripts.train_v7_sota_spectral import V7SOTADetector
from utils.checkpoint import load_checkpoint
from utils.device import get_device, AMPContext
from utils.actor_splits import get_actor_disjoint_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("actor_disjoint_eval")


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
                if hasattr(model, "spectral_combiner"):
                    out = model(images, return_spectral_details=False)
                else:
                    out = model(images)
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
    logger.info("=== Actor-Disjoint Leak-Free Evaluation ===")

    manifest_path = Path("manifests/ffpp_c23_manifest.csv")
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    col = "filepath" if "filepath" in df.columns else "image_path"

    # Get actor-disjoint split
    split = get_actor_disjoint_split(df, filepath_col=col)

    # Load models
    base_detector = DeepfakeDetector()
    model_v7 = V7SOTADetector(base_detector)
    load_checkpoint("checkpoints/v7_sota_spectral/best_model.pth", model_v7, device=device)
    model_v7.to(device).eval()

    # Also load V2 as zero-shot baseline
    model_v2 = DeepfakeDetector()
    load_checkpoint("checkpoints/v2_clip_finetune/best_model.pth", model_v2, device=device)
    model_v2.to(device).eval()

    # Test reals: all available test real frames
    test_real_paths = split["test_reals"][col].tolist()
    logger.info(f"Test real pool: {len(test_real_paths)} frames from {len(split['test_actors'])} actors")

    cohort_names = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]
    report = {
        "description": "Actor-Disjoint Leak-Free Evaluation (shuffled 60/40, seed=42, DFD excluded)",
        "split": {"train_actors": len(split["train_actors"]), "test_actors": len(split["test_actors"])},
        "cohorts": {},
    }

    all_v7_aucs = []
    all_v2_aucs = []

    for i, c_name in enumerate(cohort_names):
        test_fake_df = split["test_fakes"][c_name]
        test_fake_paths = test_fake_df[col].tolist()

        # Independent real sampling per cohort (seed = 42 + cohort_index)
        # Each cohort gets a different random sample of test reals
        # Sample size = min(len(test_fake_paths), len(test_real_paths))
        n_fake = len(test_fake_paths)
        rng = random.Random(42 + i)
        real_sample = rng.sample(test_real_paths, min(n_fake, len(test_real_paths)))

        n_eval = min(len(real_sample), n_fake)
        samples = [(p, 0) for p in real_sample[:n_eval]] + [(p, 1) for p in test_fake_paths[:n_eval]]

        logger.info(f"\n--- {c_name}: {n_eval} real + {n_eval} fake = {2*n_eval} balanced samples ---")

        res_v7 = evaluate_model(model_v7, samples, device)
        res_v2 = evaluate_model(model_v2, samples, device)

        report["cohorts"][c_name] = {
            "v7_multisource": res_v7,
            "v2_zeroshot": res_v2,
            "n_test_videos": test_fake_df["video_id"].nunique() if "video_id" in test_fake_df.columns else -1,
            "n_balanced_samples": 2 * n_eval,
        }

        all_v7_aucs.append(res_v7["roc_auc"])
        all_v2_aucs.append(res_v2["roc_auc"])

        logger.info(
            f"{c_name} Results:\n"
            f"  V7: AUC={res_v7['roc_auc']:.4f} | Acc@0.5={res_v7['accuracy_default_05']*100:.2f}% | "
            f"Cal.Acc@{res_v7['optimal_threshold_youden']:.3f}={res_v7['accuracy_calibrated']*100:.2f}% | "
            f"Brier={res_v7['brier_score']:.4f}\n"
            f"  V2: AUC={res_v2['roc_auc']:.4f} | Acc@0.5={res_v2['accuracy_default_05']*100:.2f}%"
        )

    # Macro averages
    report["macro_averages"] = {
        "v7_macro_auc": float(np.mean(all_v7_aucs)),
        "v2_macro_auc": float(np.mean(all_v2_aucs)),
    }

    logger.info(f"\n=== MACRO AVERAGES (5 Cohorts, DFD Excluded) ===")
    logger.info(f"V7 Macro AUC: {np.mean(all_v7_aucs):.4f}")
    logger.info(f"V2 Macro AUC: {np.mean(all_v2_aucs):.4f}")

    # Save report
    out_dir = Path("results/benchmark_eval_v7")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "actor_disjoint_leak_free_eval.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved: {out_file}")


if __name__ == "__main__":
    main()
