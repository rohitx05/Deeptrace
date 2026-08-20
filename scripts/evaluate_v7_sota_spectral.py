"""
DeepTrace V7 SOTA Multi-Spectral Comprehensive Benchmark Evaluation.
Evaluates checkpoints/v7_sota_spectral/best_model.pth across:
1. Kaggle In-Domain Test Split (N=20,000)
2. FaceForensics++ c23 Multi-Cohort Benchmark (N=14,000 across all 6 cohorts)
3. Outputs JSON report to results/benchmark_eval_v7/v7_sota_spectral_evaluation.json
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from scripts.train_v7_sota_spectral import V7SOTADetector
from utils.device import get_device, UnifiedAMPContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v7_spectral_eval")


class EvalImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label)
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


def evaluate_samples(model, samples, device, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = EvalImageDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    all_preds, all_targets = [], []
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with UnifiedAMPContext(device=device, enabled=True):
                outputs = model(images, return_spectral_details=False)
                probs = torch.sigmoid(outputs["binary_logit"]).cpu().numpy()

            all_preds.extend(probs)
            all_targets.extend(targets.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    acc = float(accuracy_score(targets, preds >= 0.5))
    try:
        auc = float(roc_auc_score(targets, preds))
    except Exception:
        auc = 0.5
    f1 = float(f1_score(targets, preds >= 0.5))
    brier = float(brier_score_loss(targets, preds))
    cm = confusion_matrix(targets, preds >= 0.5).tolist()

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "f1_score": f1,
        "brier_score": brier,
        "confusion_matrix": cm,
        "n_samples": len(samples),
    }


def main():
    device = get_device()
    logger.info(f"=== Starting V7 SOTA Multi-Spectral Comprehensive Benchmark on {device} ===")

    ckpt_path = Path("checkpoints/v7_sota_spectral/best_model.pth")
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at: {ckpt_path}")
        return

    # Load Model
    base_detector = DeepfakeDetector()
    model = V7SOTADetector(base_detector)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    results = {
        "model": "DeepTrace V7 SOTA Multi-Spectral",
        "checkpoint": str(ckpt_path),
        "cohorts": {},
    }

    # 1. In-Domain Kaggle Test Split
    k_test = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/test")
    if k_test.exists():
        k_reals = list((k_test / "real").glob("*.jpg"))
        k_fakes = list((k_test / "fake").glob("*.jpg"))
        kaggle_samples = [(str(p), 0) for p in k_reals] + [(str(p), 1) for p in k_fakes]
        logger.info(f"Evaluating In-Domain Kaggle Test Split (N={len(kaggle_samples)})...")
        results["kaggle_in_domain"] = evaluate_samples(model, kaggle_samples, device)
        logger.info(
            f"Kaggle Test: Acc={results['kaggle_in_domain']['accuracy']*100:.2f}% | "
            f"AUC={results['kaggle_in_domain']['roc_auc']:.5f} | "
            f"F1={results['kaggle_in_domain']['f1_score']:.4f} | "
            f"Brier={results['kaggle_in_domain']['brier_score']:.6f}"
        )

    # 2. FaceForensics++ c23 Cohorts
    ffpp_csv = Path("manifests/ffpp_c23_manifest.csv")
    if ffpp_csv.exists():
        df = pd.read_csv(ffpp_csv)
        col = "filepath" if "filepath" in df.columns else "image_path"
        ff_reals = [(str(p), 0) for p in df[df["manipulation_type"] == "real"][col].tolist()[:2000]]

        all_ff_fakes = []
        cohort_names = ["FaceSwap", "Deepfakes", "Face2Face", "NeuralTextures", "FaceShifter", "DeepFakeDetection"]

        for c_name in cohort_names:
            c_fakes = [(str(p), 1) for p in df[df["manipulation_type"] == c_name][col].tolist()[:2000]]
            all_ff_fakes.extend(c_fakes)

            cohort_samples = ff_reals + c_fakes
            logger.info(f"Evaluating FF++ Cohort: {c_name} (N={len(cohort_samples)})...")
            res = evaluate_samples(model, cohort_samples, device)
            results["cohorts"][c_name] = res
            logger.info(f"  -> {c_name}: Acc={res['accuracy']*100:.2f}% | AUC={res['roc_auc']:.4f} | F1={res['f1_score']:.4f}")

        # Overall FF++ Aggregate Benchmark
        overall_samples = ff_reals + all_ff_fakes
        logger.info(f"Evaluating Overall FF++ Benchmark (N={len(overall_samples)})...")
        results["ffpp_overall"] = evaluate_samples(model, overall_samples, device)
        logger.info(
            f"Overall FF++: Acc={results['ffpp_overall']['accuracy']*100:.2f}% | "
            f"AUC={results['ffpp_overall']['roc_auc']:.4f} | "
            f"F1={results['ffpp_overall']['f1_score']:.4f}"
        )

    out_json = Path("results/benchmark_eval_v7/v7_sota_spectral_evaluation.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"=== V7 Evaluation Complete! Saved report to: {out_json} ===")


if __name__ == "__main__":
    main()
