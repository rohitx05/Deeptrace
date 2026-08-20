"""
DeepTrace V8 Unified Multi-Expert Super-Ensemble Evaluation.
Combines:
1. V7 SOTA Multi-Spectral & SBI Specialist (checkpoints/v7_sota_spectral/best_model.pth)
2. V5 SRM Residual & Boundary Specialist (checkpoints/v5_srm_residual/best_model.pth)
3. V3 Macro Multi-Source Specialist (checkpoints/v3_e2e_multisource/best_model.pth)
4. V2 In-Domain CLIP Anchor (checkpoints/v2_clip_finetune/best_model.pth)

All checkpoints loaded in READ-ONLY mode. Zero overwriting or modification.
Outputs benchmark report to results/benchmark_eval_v8/v8_super_ensemble_evaluation.json
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from scripts.train_v7_sota_spectral import V7SOTADetector
from utils.device import get_device, AMPContext
from utils.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v8_super_ensemble")


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


class V8SuperEnsemble:
    def __init__(self, device):
        self.device = device
        logger.info("Loading Frozen Checkpoints for V8 Super-Ensemble in READ-ONLY mode...")

        # 1. Load V7 SOTA Multi-Spectral
        base_v7 = DeepfakeDetector()
        self.model_v7 = V7SOTADetector(base_v7)
        load_checkpoint("checkpoints/v7_sota_spectral/best_model.pth", self.model_v7, device=device)
        self.model_v7.to(device).eval()

        # 2. Load V5 SRM Residual
        self.model_v5 = DeepfakeDetector()
        load_checkpoint("checkpoints/v5_srm_residual/best_model.pth", self.model_v5, device=device)
        self.model_v5.to(device).eval()

        # 3. Load V3 Macro Multi-Source
        self.model_v3 = DeepfakeDetector()
        load_checkpoint("checkpoints/v3_e2e_multisource/best_model.pth", self.model_v3, device=device)
        self.model_v3.to(device).eval()

        # 4. Load V2 CLIP Anchor
        self.model_v2 = DeepfakeDetector()
        load_checkpoint("checkpoints/v2_clip_finetune/best_model.pth", self.model_v2, device=device)
        self.model_v2.to(device).eval()

        # Ensemble Weights & Temperature
        # Higher weight on V7 (Multi-Spectral) and V5 (Residual) with stabilizing anchors (V3, V2)
        self.w_v7 = 0.45
        self.w_v5 = 0.30
        self.w_v3 = 0.15
        self.w_v2 = 0.10
        self.temperature = 0.873507

        logger.info(f"V8 Super-Ensemble Initialized: Weights [V7: {self.w_v7}, V5: {self.w_v5}, V3: {self.w_v3}, V2: {self.w_v2}], T={self.temperature}")

    def predict_batch(self, images):
        with torch.no_grad():
            with AMPContext(device=self.device, enabled=True):
                out_v7 = self.model_v7(images, return_spectral_details=False)
                logit_v7 = out_v7["binary_logit"].squeeze(-1) if out_v7["binary_logit"].dim() > 1 else out_v7["binary_logit"]

                out_v5 = self.model_v5(images, compute_clip_alignment_loss=False)
                logit_v5 = out_v5["binary_logit"].squeeze(-1) if out_v5["binary_logit"].dim() > 1 else out_v5["binary_logit"]

                out_v3 = self.model_v3(images, compute_clip_alignment_loss=False)
                logit_v3 = out_v3["binary_logit"].squeeze(-1) if out_v3["binary_logit"].dim() > 1 else out_v3["binary_logit"]

                out_v2 = self.model_v2(images, compute_clip_alignment_loss=False)
                logit_v2 = out_v2["binary_logit"].squeeze(-1) if out_v2["binary_logit"].dim() > 1 else out_v2["binary_logit"]

                ensemble_logit = (
                    self.w_v7 * logit_v7 +
                    self.w_v5 * logit_v5 +
                    self.w_v3 * logit_v3 +
                    self.w_v2 * logit_v2
                )

                calibrated_prob = torch.sigmoid(ensemble_logit / self.temperature)
                return calibrated_prob.cpu().numpy()


def evaluate_dataset(ensemble, samples, device, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = EvalImageDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    all_preds, all_targets = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        probs = ensemble.predict_batch(images)
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
    logger.info(f"=== Starting DeepTrace V8 Unified Multi-Expert Super-Ensemble Evaluation on {device} ===")

    ensemble = V8SuperEnsemble(device)

    results = {
        "model": "DeepTrace V8 Unified Multi-Expert Super-Ensemble",
        "weights": {"V7_spectral": 0.45, "V5_residual": 0.30, "V3_macro": 0.15, "V2_anchor": 0.10},
        "temperature": 0.873507,
        "cohorts": {},
    }

    # 1. In-Domain Kaggle Test Split
    k_test = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/test")
    if k_test.exists():
        k_reals = list((k_test / "real").glob("*.jpg"))
        k_fakes = list((k_test / "fake").glob("*.jpg"))
        kaggle_samples = [(str(p), 0) for p in k_reals] + [(str(p), 1) for p in k_fakes]
        logger.info(f"Evaluating In-Domain Kaggle Test Split (N={len(kaggle_samples)})...")
        results["kaggle_in_domain"] = evaluate_dataset(ensemble, kaggle_samples, device)
        logger.info(
            f"V8 Kaggle Test: Acc={results['kaggle_in_domain']['accuracy']*100:.2f}% | "
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
            res = evaluate_dataset(ensemble, cohort_samples, device)
            results["cohorts"][c_name] = res
            logger.info(f"  -> {c_name}: Acc={res['accuracy']*100:.2f}% | AUC={res['roc_auc']:.4f} | F1={res['f1_score']:.4f}")

        # Overall FF++ Aggregate Benchmark
        overall_samples = ff_reals + all_ff_fakes
        logger.info(f"Evaluating Overall FF++ Benchmark (N={len(overall_samples)})...")
        results["ffpp_overall"] = evaluate_dataset(ensemble, overall_samples, device)
        logger.info(
            f"V8 Overall FF++: Acc={results['ffpp_overall']['accuracy']*100:.2f}% | "
            f"AUC={results['ffpp_overall']['roc_auc']:.4f} | "
            f"F1={results['ffpp_overall']['f1_score']:.4f}"
        )

    out_json = Path("results/benchmark_eval_v8/v8_super_ensemble_evaluation.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"=== V8 Super-Ensemble Evaluation Complete! Saved report to: {out_json} ===")


if __name__ == "__main__":
    main()
