"""
Step 4: Dual-Stream Multimodal Ensemble & Calibrated Decision Fusion (V6).
Fuses predictions from:
1. Macro Synthesis Stream: DeepTrace V3-E2E (Spatial + Global 2D-DCT + CLIP)
2. Microscopic Residual Stream: DeepTrace V5-SRM (Spatial + SRM High-Pass Residuals + CLIP)
Evaluates on:
1. Kaggle In-Domain Test Split (N=20,000)
2. FaceForensics++ 6 Cohorts (FaceSwap, Deepfakes, Face2Face, NeuralTextures, FaceShifter, DeepFakeDetection)
3. FaceForensics++ Overall Benchmark (N=14,000)
Outputs: results/benchmark_eval_v6/v6_ensemble_evaluation.json
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v6_ensemble")


SRM_KERNELS = torch.tensor([
    [[-1.0, -2.0, -1.0],
     [-2.0, 12.0, -2.0],
     [-1.0, -2.0, -1.0]],
    [[-1.0,  2.0, -1.0],
     [-2.0,  4.0, -2.0],
     [-1.0,  2.0, -1.0]],
    [[-1.0, -2.0, -1.0],
     [ 2.0,  4.0,  2.0],
     [-1.0, -2.0, -1.0]],
], dtype=torch.float32).unsqueeze(1)


def compute_dct_tensor_fast(img_tensor):
    fft = torch.fft.fft2(img_tensor, norm="ortho")
    dct_approx = torch.log(torch.abs(fft.real) + 1e-6)
    min_val = dct_approx.amin(dim=(-2, -1), keepdim=True)
    max_val = dct_approx.amax(dim=(-2, -1), keepdim=True)
    return (dct_approx - min_val) / (max_val - min_val + 1e-6)


def compute_srm_residual_tensor(img_tensor):
    gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    kernels = SRM_KERNELS.to(img_tensor.device)
    residuals = F.conv2d(gray, kernels, padding=1)
    min_val = residuals.amin(dim=(-2, -1), keepdim=True)
    max_val = residuals.amax(dim=(-2, -1), keepdim=True)
    norm_residuals = (residuals - min_val) / (max_val - min_val + 1e-6)
    return norm_residuals


class ImageListDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (160, 160), (128, 128, 128))

        if self.transform:
            img = self.transform(pil_img)
        else:
            img = transforms.ToTensor()(pil_img.resize((160, 160)))

        return {"image": img, "label": label}


def get_ensemble_predictions(model_v3, model_v5, loader, device, weight_v3=0.60):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)

            dcts = compute_dct_tensor_fast(imgs)
            srm_res = compute_srm_residual_tensor(imgs)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out_v3 = model_v3(imgs, dct=dcts)
                logit_v3 = out_v3["binary_logit"].squeeze(-1)

                out_v5 = model_v5(imgs, dct=srm_res)
                logit_v5 = out_v5["binary_logit"].squeeze(-1)

                # Calibrated Convex Logit Fusion
                fused_logit = weight_v3 * logit_v3 + (1.0 - weight_v3) * logit_v5
                fused_prob = torch.sigmoid(fused_logit)

            all_probs.extend(fused_prob.cpu().numpy().tolist())
            all_labels.extend(lbls.cpu().numpy().tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr = (probs_arr >= 0.50).astype(int)

    acc = accuracy_score(labels_arr, preds_arr)
    auc = roc_auc_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.5
    f1 = f1_score(labels_arr, preds_arr, zero_division=0)
    brier = brier_score_loss(labels_arr, probs_arr)
    cm = confusion_matrix(labels_arr, preds_arr).tolist()

    return {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "f1_score": float(f1),
        "brier_score": float(brier),
        "confusion_matrix": cm,
        "n_samples": len(labels_arr),
    }


def main():
    device = get_device()
    logger.info(f"Starting Dual-Stream V6 Ensemble Evaluation on {device}")

    # Load Model 1: V3-E2E
    ckpt_v3 = "checkpoints/v3_e2e_multisource/best_model.pth"
    model_v3 = DeepfakeDetector()
    load_checkpoint(ckpt_v3, model_v3, device=device)
    model_v3.to(device)
    model_v3.eval()

    # Load Model 2: V5-SRM
    ckpt_v5 = "checkpoints/v5_srm_residual/best_model.pth"
    model_v5 = DeepfakeDetector()
    load_checkpoint(ckpt_v5, model_v5, device=device)
    model_v5.to(device)
    model_v5.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = {
        "model": "DeepTrace (v6_dual_stream_ensemble)",
        "components": {
            "macro_model": ckpt_v3,
            "residual_model": ckpt_v5,
        },
        "cohorts": {},
    }

    # 1. In-Domain Kaggle Test Split (N=20,000)
    logger.info("Evaluating In-Domain Kaggle Test Split (N=20,000)...")
    kaggle_dir = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/test")
    kaggle_samples = []
    for p in (kaggle_dir / "real").glob("*.jpg"):
        kaggle_samples.append((str(p), 0))
    for p in (kaggle_dir / "fake").glob("*.jpg"):
        kaggle_samples.append((str(p), 1))

    kaggle_loader = DataLoader(ImageListDataset(kaggle_samples, transform=eval_transform), batch_size=64, shuffle=False, num_workers=2)
    kaggle_res = get_ensemble_predictions(model_v3, model_v5, kaggle_loader, device, weight_v3=0.60)
    results["kaggle_in_domain"] = kaggle_res
    logger.info(f"  --> Kaggle Test: Acc = {kaggle_res['accuracy']*100:.2f}%, AUC = {kaggle_res['roc_auc']:.5f}, F1 = {kaggle_res['f1_score']:.4f}")

    # 2. FaceForensics++ Cohorts
    manifest_csv = Path("manifests/ffpp_c23_manifest.csv")
    df_ffpp = pd.read_csv(manifest_csv)
    col = "filepath" if "filepath" in df_ffpp.columns else "image_path"

    real_paths = df_ffpp[df_ffpp["manipulation_type"] == "real"][col].tolist()[5000:7000]

    manip_types = ["FaceSwap", "Deepfakes", "Face2Face", "NeuralTextures", "FaceShifter", "DeepFakeDetection"]
    overall_samples = [(p, 0) for p in real_paths]

    for manip in manip_types:
        fake_paths = df_ffpp[df_ffpp["manipulation_type"] == manip][col].tolist()[5000:7000]
        cohort_samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        overall_samples.extend([(p, 1) for p in fake_paths])

        loader = DataLoader(ImageListDataset(cohort_samples, transform=eval_transform), batch_size=64, shuffle=False, num_workers=2)
        res = get_ensemble_predictions(model_v3, model_v5, loader, device, weight_v3=0.50)
        results["cohorts"][manip] = res
        logger.info(f"  --> FF++ {manip}: Acc = {res['accuracy']*100:.2f}%, AUC = {res['roc_auc']:.4f}, F1 = {res['f1_score']:.4f}")

    # 3. Overall FF++ (N=14,000)
    logger.info("Evaluating Overall FaceForensics++ Benchmark (N=14,000)...")
    overall_loader = DataLoader(ImageListDataset(overall_samples, transform=eval_transform), batch_size=64, shuffle=False, num_workers=2)
    overall_res = get_ensemble_predictions(model_v3, model_v5, overall_loader, device, weight_v3=0.50)
    results["ffpp_overall"] = overall_res
    logger.info(f"  --> FF++ Overall: Acc = {overall_res['accuracy']*100:.2f}%, AUC = {overall_res['roc_auc']:.4f}, F1 = {overall_res['f1_score']:.4f}")

    # Save to JSON
    out_file = Path("results/benchmark_eval_v6/v6_ensemble_evaluation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"=== Dual-Stream V6 Ensemble Evaluation Saved: {out_file} ===")


if __name__ == "__main__":
    main()
