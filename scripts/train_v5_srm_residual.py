"""
Hour 4: SRM High-Pass Residual Filtering & Multi-Scale Discontinuity Pipeline (V5).
Applies 3 SRM high-pass residual kernels (Laplacian, Horizontal Edge, Vertical Edge)
directly to the input stream to strip face identity and expose Poisson blending seams.
Trains for 8 epochs on RTX 4050 GPU.
Saves: checkpoints/v5_srm_residual/best_model.pth
"""

import sys
import math
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v5_srm_train")


# Fixed SRM High-Pass Kernels for Deepfake Seam Extraction (Steganalysis Rich Models)
SRM_KERNELS = torch.tensor([
    # 1. 3x3 Laplacian / High-Pass Filter
    [[-1.0, -2.0, -1.0],
     [-2.0, 12.0, -2.0],
     [-1.0, -2.0, -1.0]],
    # 2. 3x3 Horizontal Edge Discontinuity
    [[-1.0,  2.0, -1.0],
     [-2.0,  4.0, -2.0],
     [-1.0,  2.0, -1.0]],
    # 3. 3x3 Vertical Edge Discontinuity
    [[-1.0, -2.0, -1.0],
     [ 2.0,  4.0,  2.0],
     [-1.0, -2.0, -1.0]],
], dtype=torch.float32).unsqueeze(1)  # shape: (3, 1, 3, 3)


def compute_srm_residual_tensor(img_tensor):
    """
    Computes 3-channel high-pass residual map from RGB input (B, 3, H, W).
    Converts RGB to grayscale, applies SRM kernels, normalizes between [0, 1].
    """
    # Grayscale conversion: (B, 1, H, W)
    gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    kernels = SRM_KERNELS.to(img_tensor.device)
    residuals = F.conv2d(gray, kernels, padding=1)
    
    # Normalize per image
    min_val = residuals.amin(dim=(-2, -1), keepdim=True)
    max_val = residuals.amax(dim=(-2, -1), keepdim=True)
    norm_residuals = (residuals - min_val) / (max_val - min_val + 1e-6)
    return norm_residuals


class SRMDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (path, binary_label, manip_class_idx, sample_weight)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, binary_lbl, manip_idx, weight = self.samples[idx]
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (160, 160), (128, 128, 128))

        if self.transform:
            img = self.transform(pil_img)
        else:
            img = transforms.ToTensor()(pil_img.resize((160, 160)))

        return {
            "image": img,
            "binary_label": torch.tensor(binary_lbl, dtype=torch.float32),
            "manip_label": torch.tensor(manip_idx, dtype=torch.long),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
        }


class HardSampleFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, weights):
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce_loss
        weighted_loss = focal_loss * weights
        return weighted_loss.mean()


def build_srm_dataset():
    logger.info("Constructing SRM Residual Multi-Source Dataset...")

    train_samples = []
    val_samples = []

    # 1. Kaggle In-Domain Reals & StyleGAN Fakes (Anchor)
    kaggle_train = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/train")
    kaggle_val = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/valid")

    k_reals = list((kaggle_train / "real").glob("*.jpg"))[:5000]
    k_fakes = list((kaggle_train / "fake").glob("*.jpg"))[:5000]

    for p in k_reals:
        train_samples.append((str(p), 0, 0, 1.0))
    for p in k_fakes:
        train_samples.append((str(p), 1, 1, 1.0))

    for p in list((kaggle_val / "real").glob("*.jpg"))[:1000]:
        val_samples.append((str(p), 0, 0, 1.0))
    for p in list((kaggle_val / "fake").glob("*.jpg"))[:1000]:
        val_samples.append((str(p), 1, 1, 1.0))

    # 2. FaceForensics++ Manifest with High Weighting on FaceSwap / Face2Face
    ffpp_csv = Path("manifests/ffpp_c23_manifest.csv")
    if ffpp_csv.exists():
        df_ffpp = pd.read_csv(ffpp_csv)
        col = "filepath" if "filepath" in df_ffpp.columns else "image_path"

        ff_reals = df_ffpp[df_ffpp["manipulation_type"] == "real"][col].tolist()[:5000]
        for p in ff_reals:
            train_samples.append((str(p), 0, 0, 1.0))
        for p in df_ffpp[df_ffpp["manipulation_type"] == "real"][col].tolist()[5000:6000]:
            val_samples.append((str(p), 0, 0, 1.0))

        hard_types = [
            ("FaceSwap", 2, 3.0),
            ("FaceShifter", 2, 3.0),
            ("Deepfakes", 3, 2.0),
            ("DeepFakeDetection", 3, 3.0),
            ("Face2Face", 4, 2.0),
            ("NeuralTextures", 4, 2.0),
        ]

        for manip, c_idx, w_val in hard_types:
            f_paths = df_ffpp[df_ffpp["manipulation_type"] == manip][col].tolist()
            train_f = f_paths[:2000]
            val_f = f_paths[5000:5500]

            for p in train_f:
                train_samples.append((str(p), 1, c_idx, w_val))
            for p in val_f:
                val_samples.append((str(p), 1, c_idx, 1.0))

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    logger.info(f"SRM Train Samples: {len(train_samples)} | Val Samples: {len(val_samples)}")
    return train_samples, val_samples


def main():
    device = get_device()
    logger.info(f"=== Starting Hour 4 SRM High-Pass Residual Training on {device} ===")

    out_dir = Path("checkpoints/v5_srm_residual")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and load from V4 Best Model
    model = DeepfakeDetector()
    ckpt_v4 = "checkpoints/v4_seam_hardmining/best_model.pth"
    load_checkpoint(ckpt_v4, model, device=device)
    model.to(device)

    # Differential Learning Rates
    optimizer_grouped_parameters = [
        {"params": [p for p in model.spatial_encoder.parameters() if p.requires_grad], "lr": 3e-5, "weight_decay": 1e-4},
        {"params": [p for p in model.frequency_encoder.parameters() if p.requires_grad], "lr": 3e-5, "weight_decay": 1e-4},
        {"params": [p for p in model.clip_alignment.parameters() if p.requires_grad], "lr": 5e-6, "weight_decay": 1e-4},
        {"params": list(model.fusion.parameters()) + list(model.detection_head.parameters()), "lr": 1e-4, "weight_decay": 1e-4},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    focal_criterion = HardSampleFocalLoss(gamma=2.0, alpha=0.25)
    manip_criterion = nn.CrossEntropyLoss()

    train_samples, val_samples = build_srm_dataset()

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(SRMDataset(train_samples, transform=transform), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(SRMDataset(val_samples, transform=val_transform), batch_size=64, shuffle=False, num_workers=2)

    best_val_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        all_train_probs, all_train_labels = [], []

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            b_lbls = batch["binary_label"].to(device, non_blocking=True)
            m_lbls = batch["manip_label"].to(device, non_blocking=True)
            weights = batch["sample_weight"].to(device, non_blocking=True)

            # Compute SRM High-Pass Residual map as the frequency stream input
            srm_residuals = compute_srm_residual_tensor(imgs)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                # Feed RGB to spatial branch, and SRM high-pass edge residual to frequency branch
                out = model(imgs, dct=srm_residuals)
                b_logit = out["binary_logit"].squeeze(-1)
                m_logits = out["manipulation_logits"]

                loss_b = focal_criterion(b_logit, b_lbls, weights)
                loss_m = manip_criterion(m_logits, m_lbls)
                loss = loss_b + 0.20 * loss_m

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(b_lbls)
            probs = torch.sigmoid(b_logit).detach()
            preds = (probs >= 0.50).float()
            correct += (preds == b_lbls).sum().item()
            total += len(b_lbls)

            all_train_probs.extend(probs.cpu().tolist())
            all_train_labels.extend(b_lbls.cpu().tolist())

            if (step + 1) % 200 == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()
        train_acc = correct / max(total, 1)
        train_auc = roc_auc_score(all_train_labels, all_train_probs) if len(np.unique(all_train_labels)) > 1 else 0.5

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_probs, all_val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                b_lbls = batch["binary_label"].to(device)
                m_lbls = batch["manip_label"].to(device)
                weights = batch["sample_weight"].to(device)
                srm_residuals = compute_srm_residual_tensor(imgs)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(imgs, dct=srm_residuals)
                    b_logit = out["binary_logit"].squeeze(-1)
                    m_logits = out["manipulation_logits"]

                    loss_b = focal_criterion(b_logit, b_lbls, weights)
                    loss_m = manip_criterion(m_logits, m_lbls)
                    loss = loss_b + 0.20 * loss_m

                val_loss += loss.item() * len(b_lbls)
                probs = torch.sigmoid(b_logit)
                preds = (probs >= 0.50).float()
                val_correct += (preds == b_lbls).sum().item()
                val_total += len(b_lbls)

                all_val_probs.extend(probs.cpu().tolist())
                all_val_labels.extend(b_lbls.cpu().tolist())

        val_acc = val_correct / max(val_total, 1)
        val_auc = roc_auc_score(all_val_labels, all_val_probs) if len(np.unique(all_val_labels)) > 1 else 0.5
        val_f1 = f1_score(all_val_labels, (np.array(all_val_probs) >= 0.50).astype(int))

        logger.info(f"--> Epoch [{epoch}/{epochs}] | Train Acc: {train_acc*100:.2f}% AUC: {train_auc:.4f} | Val Acc: {val_acc*100:.2f}% AUC: {val_auc:.4f} F1: {val_f1:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = out_dir / "best_model.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_auc": float(val_auc), "val_acc": float(val_acc), "val_f1": float(val_f1)},
                path=str(save_path),
                scaler=scaler,
                scheduler=scheduler,
            )
            logger.info(f"  [SAVED] New best V5-SRM checkpoint with Val AUC = {val_auc:.5f} -> {save_path}")

    logger.info(f"=== V5-SRM Training Complete. Peak Val AUC = {best_val_auc:.5f} ===")


if __name__ == "__main__":
    main()
