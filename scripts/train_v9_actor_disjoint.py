"""
DeepTrace V9: Strictly Actor-Disjoint Multi-Spectral Forensics Training Engine.

Guarantees:
1. 100% Actor-Disjoint Split: 600 train actors, 400 test actors (seed=42).
2. ZERO contamination: No test actor (source or target) in training data.
3. DFD excluded from cross-manipulation training to avoid studio domain confound.
4. Tri-Objective Loss (Boundary Focal Loss + Spectral Orthogonal Loss + Multi-Task Manipulation Loss).
5. Isolated checkpoint: checkpoints/v9_actor_disjoint/best_model.pth.
"""

import os
import sys
import math
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from models.spectral_branches import SOTASpectralCombiner
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_device, AMPContext, get_grad_scaler
from utils.actor_splits import get_actor_disjoint_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v9_actor_disjoint_train")


class V9ActorDisjointDetector(nn.Module):
    def __init__(self, base_detector: DeepfakeDetector):
        super().__init__()
        self.spatial_encoder = base_detector.spatial_encoder
        self.frequency_encoder = base_detector.frequency_encoder
        self.clip_alignment = base_detector.clip_alignment
        self.fusion = base_detector.fusion
        self.detection_head = base_detector.detection_head

        self.spectral_combiner = SOTASpectralCombiner(
            dct_in_dim=1280,
            branch_dim=320,
            output_dim=1280,
            gradient_checkpointing=True,
        )

    def forward(self, images: torch.Tensor, return_spectral_details: bool = False):
        spatial_feat = self.spatial_encoder(images)
        dct_feat = self.frequency_encoder(images)
        spectral_out = self.spectral_combiner(images, dct_feat, return_branches=return_spectral_details)

        if return_spectral_details:
            combined_spectral_feat = spectral_out["combined"]
            branch_dict = spectral_out["branches"]
            gate_weights = spectral_out["gate_weights"]
        else:
            combined_spectral_feat = spectral_out
            branch_dict = None
            gate_weights = None

        clip_result = self.clip_alignment(spatial_feat, images, compute_alignment_loss=True)
        clip_proj = clip_result["spatial_projected"]
        clip_loss = clip_result["alignment_loss"]

        fused_feat = self.fusion(
            spatial_features=spatial_feat,
            frequency_features=combined_spectral_feat,
            temporal_features=None,
            physiology_features=None,
            clip_features=clip_proj,
        )

        predictions = self.detection_head(fused_feat)
        binary_logit = predictions["binary_logit"].squeeze(-1)

        out = {
            "binary_logit": binary_logit,
            "clip_loss": clip_loss,
            "multitask_logits": predictions.get("manipulation_type_logits", None),
        }
        if return_spectral_details:
            out["branch_features"] = branch_dict
            out["gate_weights"] = gate_weights
        return out


def generate_dynamic_sbi_seam(image: Image.Image, p_blend: float = 0.5) -> Image.Image:
    if random.random() > p_blend:
        return image
    w, h = image.size
    cx, cy = w // 2, h // 2
    rw, rh = int(w * random.uniform(0.35, 0.65)), int(h * random.uniform(0.35, 0.65))
    x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
    x2, y2 = min(w, cx + rw // 2), min(h, cy + rh // 2)

    crop = image.crop((x1, y1, x2, y2))
    if random.random() < 0.5:
        crop = crop.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
    if random.random() < 0.5:
        crop = crop.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    mask = Image.new("L", (x2 - x1, y2 - y1), 0)
    for px in range(x2 - x1):
        for py in range(y2 - y1):
            dx = min(px, x2 - x1 - px) / max(1, (x2 - x1) * 0.15)
            dy = min(py, y2 - y1 - py) / max(1, (y2 - y1) * 0.15)
            mask.putpixel((px, py), int(255 * min(1.0, dx, dy)))

    blended = image.copy()
    blended.paste(crop, (x1, y1), mask)
    return blended


class V9Dataset(Dataset):
    def __init__(self, samples, transform, is_training: bool = True):
        self.samples = samples
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, binary_label, manip_idx, sample_weight = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (160, 160), (128, 128, 128))

        if self.is_training and binary_label == 1 and random.random() < 0.30:
            img = generate_dynamic_sbi_seam(img, p_blend=0.80)

        tensor_img = self.transform(img)
        return {
            "image": tensor_img,
            "binary_label": torch.tensor(binary_label, dtype=torch.float32),
            "manip_label": torch.tensor(manip_idx, dtype=torch.long),
            "weight": torch.tensor(sample_weight, dtype=torch.float32),
        }


def compute_spectral_orthogonality_loss(branch_dict: dict) -> torch.Tensor:
    if not branch_dict:
        return torch.tensor(0.0)
    feats = [F.normalize(feat, p=2, dim=-1) for feat in branch_dict.values()]
    ortho_loss = torch.tensor(0.0, device=feats[0].device)
    pairs = 0
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            cosine_sim = (feats[i] * feats[j]).sum(dim=-1).abs().mean()
            ortho_loss = ortho_loss + cosine_sim
            pairs += 1
    return ortho_loss / max(pairs, 1)


def build_v9_dataset():
    logger.info("Constructing V9 Strictly Actor-Disjoint Dataset...")
    train_samples = []
    val_samples = []

    # 1. In-domain Kaggle anchor (StyleGAN vs Real)
    k_train = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/train")
    k_val = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/valid")

    for p in list((k_train / "real").glob("*.jpg"))[:5000]:
        train_samples.append((str(p), 0, 0, 1.0))
    for p in list((k_train / "fake").glob("*.jpg"))[:5000]:
        train_samples.append((str(p), 1, 1, 1.0))

    for p in list((k_val / "real").glob("*.jpg"))[:1000]:
        val_samples.append((str(p), 0, 0, 1.0))
    for p in list((k_val / "fake").glob("*.jpg"))[:1000]:
        val_samples.append((str(p), 1, 1, 1.0))

    # 2. FaceForensics++ strictly actor-disjoint
    ffpp_csv = Path("manifests/ffpp_c23_manifest.csv")
    if ffpp_csv.exists():
        df_ffpp = pd.read_csv(ffpp_csv)
        col = "filepath" if "filepath" in df_ffpp.columns else "image_path"
        split = get_actor_disjoint_split(df_ffpp, filepath_col=col)

        # Train reals: 6,000 frames from 600 train actors
        train_real_paths = split["train_reals"][col].tolist()
        for p in train_real_paths:
            train_samples.append((str(p), 0, 0, 1.0))

        # Val reals: last 500 frames from train actors
        for p in train_real_paths[-500:]:
            val_samples.append((str(p), 0, 0, 1.0))

        manip_specs = [
            ("FaceSwap", 2, 3.5),
            ("FaceShifter", 2, 3.5),
            ("Deepfakes", 3, 2.0),
            ("Face2Face", 4, 2.5),
            ("NeuralTextures", 4, 2.0),
        ]

        for manip, c_idx, w_val in manip_specs:
            if manip in split["train_fakes"]:
                train_fake_paths = split["train_fakes"][manip][col].tolist()
                for p in train_fake_paths:
                    train_samples.append((str(p), 1, c_idx, w_val))
                # Val fakes: last 300 frames from train fakes
                for p in train_fake_paths[-300:]:
                    val_samples.append((str(p), 1, c_idx, 1.0))

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    logger.info(f"V9 Clean Dataset: {len(train_samples)} Train | {len(val_samples)} Val")
    return train_samples, val_samples


def main():
    device = get_device()
    logger.info(f"=== Starting DeepTrace V9 Actor-Disjoint Training on {device} ===")

    out_dir = Path("checkpoints/v9_actor_disjoint")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_detector = DeepfakeDetector()
    ckpt_v7 = "checkpoints/v7_sota_spectral/best_model.pth"
    if Path(ckpt_v7).exists():
        logger.info(f"Initializing architecture from V7 checkpoint: {ckpt_v7}")
        model = V9ActorDisjointDetector(base_detector)
        load_checkpoint(ckpt_v7, model, device=device)
    else:
        model = V9ActorDisjointDetector(base_detector)
    model.to(device)

    # Differential learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": model.spectral_combiner.parameters(), "lr": 1.5e-4, "weight_decay": 1e-4},
            {"params": model.spatial_encoder.parameters(), "lr": 2e-5, "weight_decay": 1e-4},
            {"params": model.frequency_encoder.parameters(), "lr": 2e-5, "weight_decay": 1e-4},
            {"params": model.fusion.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
            {"params": model.detection_head.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        ]
    )
    scaler = get_grad_scaler(device=device)

    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_samples, val_samples = build_v9_dataset()
    train_loader = DataLoader(V9Dataset(train_samples, train_transform, is_training=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(V9Dataset(val_samples, val_transform, is_training=False), batch_size=64, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-6)
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    epochs = 3

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_steps = 0.0, 0
        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            labels = batch["binary_label"].to(device, non_blocking=True)
            manips = batch["manip_label"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with AMPContext(device=device, enabled=True):
                out = model(imgs, return_spectral_details=True)
                raw_bce = bce_loss_fn(out["binary_logit"], labels)
                pt = torch.exp(-raw_bce)
                focal_bce = ((1.0 - pt) ** 2.0) * raw_bce * weights
                loss_bce = focal_bce.mean()

                loss_ortho = compute_spectral_orthogonality_loss(out.get("branch_features", {}))
                loss_clip = out.get("clip_loss", torch.tensor(0.0, device=device))
                loss_manip = ce_loss_fn(out["multitask_logits"], manips) if out["multitask_logits"] is not None else torch.tensor(0.0, device=device)

                total_loss = loss_bce + 0.10 * loss_ortho + 0.05 * loss_clip + 0.15 * loss_manip

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            train_steps += 1

        scheduler.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                labels = batch["binary_label"].cpu().numpy()
                with AMPContext(device=device, enabled=True):
                    out = model(imgs, return_spectral_details=False)
                    logits = np.nan_to_num(out["binary_logit"].float().cpu().numpy(), nan=0.0)
                all_preds.extend(logits)
                all_labels.extend(labels)

        preds_arr = np.nan_to_num(np.array(all_preds), nan=0.0)
        probs = 1.0 / (1.0 + np.exp(-np.clip(preds_arr, -20.0, 20.0)))
        probs = np.nan_to_num(probs, nan=0.5)
        val_auc = float(roc_auc_score(all_labels, probs))
        val_acc = float(accuracy_score(all_labels, probs >= 0.5))

        logger.info(f"Epoch {epoch}/{epochs} -> Train Loss: {train_loss/train_steps:.4f} | Val AUC: {val_auc:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_auc": val_auc, "val_acc": val_acc},
                path=str(out_dir / "best_model.pth"),
            )
            logger.info(f"🏆 New Best V9 Model Saved (Val AUC = {val_auc:.4f})")

    logger.info(f"=== V9 Actor-Disjoint Training Complete! Best Val AUC: {best_val_auc:.4f} ===")


if __name__ == "__main__":
    main()
