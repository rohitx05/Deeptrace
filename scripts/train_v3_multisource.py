"""
Stage 1: Multi-Source Fine-Tuning (V3).
Trains on a 50/50 mixture of Kaggle (StyleGAN) and FF++ (Deepfakes, Face2Face, FaceSwap, NeuralTextures).

Training Configuration:
- Base Checkpoint: checkpoints/v2_clip_finetune/best_model.pth
- Trainable: Fusion Layer + Detection Head + Projection Heads
- Frozen: EfficientNet Spatial Encoder, DCT Frequency Encoder, CLIP ViT Backbone
- Optimizer: AdamW(lr=1.5e-5, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Loss: BCEWithLogitsLoss (Real vs. Fake) + CrossEntropyLoss (Manipulation Type classification)
- Checkpoint Output: checkpoints/v3_multisource_finetune/best_model.pth
"""

import os
import sys
import json
import logging
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from datasets.multisource import MultiSourceDataset
from utils.checkpoint import load_checkpoint
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v3_multisource")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_cohort_records(kaggle_root="data/kaggle_realfake", manifest_path="manifests/ffpp_c23_manifest.csv", split="train", max_per_class=10000):
    """Collect balanced image records from Kaggle and FaceForensics++ with video-isolated splits."""
    records = []
    kaggle_p = Path(kaggle_root) / "real_vs_fake" / "real-vs-fake" / split

    # 1. Kaggle Reals (FFHQ)
    if kaggle_p.exists():
        k_real = sorted(list((kaggle_p / "real").glob("*.jpg")))[:max_per_class]
        for p in k_real:
            records.append((str(p), 0, "real"))

        # 2. Kaggle Fakes (StyleGAN)
        k_fake = sorted(list((kaggle_p / "fake").glob("*.jpg")))[:max_per_class]
        for p in k_fake:
            records.append((str(p), 1, "StyleGAN"))

    # 3. FF++ Reals & Fakes from Manifest
    mf_p = Path(manifest_path)
    if mf_p.exists():
        import pandas as pd
        df = pd.read_csv(mf_p)
        
        # Deterministic 80/20 video split
        for manip in df["manipulation_type"].unique():
            sub_df = df[df["manipulation_type"] == manip].sort_values("filepath")
            n = len(sub_df)
            n_train = int(n * 0.8)
            
            if split == "train":
                chosen_df = sub_df.iloc[:n_train]
            else:
                chosen_df = sub_df.iloc[n_train:]
                
            sample_n = min(len(chosen_df), max_per_class if manip == "real" else max_per_class // 5)
            sampled = chosen_df.sample(n=sample_n, random_state=42)
            
            label = 0 if manip == "real" else 1
            for _, r in sampled.iterrows():
                records.append((r["filepath"], label, manip))

    random.seed(42)
    random.shuffle(records)
    logger.info(f"Loaded {len(records)} records for {split} split")
    return records


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            dcts = batch["dct"].to(device)
            lbls = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs, dct=dcts)
                if isinstance(out, dict):
                    probs = out.get("binary_pred", torch.sigmoid(out.get("binary_logit", out.get("logits"))))
                else:
                    probs = torch.sigmoid(out)

            all_preds.extend(probs.squeeze(-1).cpu().numpy().tolist())
            all_labels.extend(lbls.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)
    y_pred = (y_prob >= 0.5).astype(float)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def train(epochs=3, batch_size=32, lr=1.5e-5, seed=42):
    set_seed(seed)
    device = get_device()
    logger.info(f"Starting V3 Multi-Source Fine-Tuning on {device}")

    # Load Phase-2 Checkpoint
    ckpt_path = "checkpoints/v2_clip_finetune/best_model.pth"
    model = DeepfakeDetector()
    load_checkpoint(ckpt_path, model, device=device)
    model.to(device)

    # Freeze feature backbones (Spatial, Frequency, CLIP, Temporal, Physiology)
    for p in model.spatial_encoder.parameters():
        p.requires_grad = False
    for p in model.frequency_encoder.parameters():
        p.requires_grad = False
    for p in model.clip_alignment.parameters():
        p.requires_grad = False
    for p in model.temporal_model.parameters():
        p.requires_grad = False
    for p in model.physiology_encoder.parameters():
        p.requires_grad = False

    # Trainable: Fusion + Detection Head
    for p in model.fusion.parameters():
        p.requires_grad = True
    for p in model.detection_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Datasets
    train_records = build_cohort_records(split="train")
    val_records = build_cohort_records(split="val", max_per_class=2000)

    train_ds = MultiSourceDataset(train_records, is_train=True)
    val_ds = MultiSourceDataset(val_records, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path("checkpoints/v3_multisource_finetune")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_model.pth"
    best_auc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"V3 Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            imgs = batch["image"].to(device)
            dcts = batch["dct"].to(device)
            lbls = batch["label"].to(device).unsqueeze(-1)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs, dct=dcts)
                logits = out["binary_logit"] if isinstance(out, dict) else out
                loss = criterion_bce(logits, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        val_res = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} Multi-Source Val Acc: {val_res['accuracy']*100:.2f}%, AUC: {val_res['roc_auc']:.4f}")

        if val_res["roc_auc"] > best_auc:
            best_auc = val_res["roc_auc"]
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"Saved new best V3 checkpoint (Val AUC: {best_auc:.4f})")

    logger.info(f"V3 Multi-Source Fine-Tuning Complete! Best Checkpoint saved to: {best_ckpt}")


if __name__ == "__main__":
    train()
