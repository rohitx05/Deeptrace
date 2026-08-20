"""
Evaluation script for V5-SRM High-Pass Residual model.
Evaluates `checkpoints/v5_srm_residual/best_model.pth` on:
1. In-domain Kaggle test split (N=20,000)
2. FaceForensics++ 6 cohorts (FaceSwap, Deepfakes, Face2Face, NeuralTextures, FaceShifter, DeepFakeDetection)
3. FaceForensics++ Overall (N=14,000)
Outputs: results/benchmark_eval_v5/v5_srm_evaluation.json
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v5_srm_eval")


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


def evaluate_loader(model, loader, device):
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            srm_residuals = compute_srm_residual_tensor(imgs)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs, dct=srm_residuals)
                logits = out["binary_logit"].squeeze(-1)
                probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().tolist())
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
    logger.info(f"Starting Evaluation of V5-SRM Model on {device}")

    ckpt_path = "checkpoints/v5_srm_residual/best_model.pth"
    model = DeepfakeDetector()
    load_checkpoint(ckpt_path, model, device=device)
    model.to(device)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = {
        "model": "DeepTrace (v5_srm_residual)",
        "checkpoint": ckpt_path,
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
    kaggle_res = evaluate_loader(model, kaggle_loader, device)
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
        res = evaluate_loader(model, loader, device)
        results["cohorts"][manip] = res
        logger.info(f"  --> FF++ {manip}: Acc = {res['accuracy']*100:.2f}%, AUC = {res['roc_auc']:.4f}, F1 = {res['f1_score']:.4f}")

    # 3. Overall FF++ (N=14,000)
    logger.info("Evaluating Overall FaceForensics++ Benchmark (N=14,000)...")
    overall_loader = DataLoader(ImageListDataset(overall_samples, transform=eval_transform), batch_size=64, shuffle=False, num_workers=2)
    overall_res = evaluate_loader(model, overall_loader, device)
    results["ffpp_overall"] = overall_res
    logger.info(f"  --> FF++ Overall: Acc = {overall_res['accuracy']*100:.2f}%, AUC = {overall_res['roc_auc']:.4f}, F1 = {overall_res['f1_score']:.4f}")

    # Save to JSON
    out_file = Path("results/benchmark_eval_v5/v5_srm_evaluation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"=== V5-SRM Evaluation Complete & Saved: {out_file} ===")


if __name__ == "__main__":
    main()
