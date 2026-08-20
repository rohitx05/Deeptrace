"""
Hour 1: End-to-End Multi-Source Fine-Tuning Pipeline (DeepTrace V3-E2E)

Key Features:
1. End-to-end unfreezing of top spatial & DCT frequency backbone stages (5-7) + CLIP blocks (10-11)
2. Differential learning rates:
   - Spatial stages 5-7 & Conv Head: 1.0e-5
   - Frequency stages 5-7 & Conv Head: 1.0e-5
   - CLIP ViT-B/32 blocks 10-11: 5.0e-6
   - Fusion Transformer & Multi-task Detection Head: 5.0e-5
3. Focal Loss (gamma=2.0, alpha=0.25) to prevent easy GAN samples from dominating gradients
4. Balanced 40,000-image dataset (50% Kaggle StyleGAN/FFHQ + 50% FF++ FaceSwap/Deepfakes/Face2Face/NeuralTextures/FaceShifter/DFD)
5. Mixed-precision AMP (FP16) training on RTX 4050 Laptop GPU
"""

import sys
import time
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("v3_e2e_train.log", mode="w"),
    ]
)
logger = logging.getLogger("v3_e2e_train")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalBinaryLoss(nn.Module):
    """Binary Focal Loss with logits input."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_factor * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


def compute_dct_tensor_fast(img_tensor):
    """Compute 2D-DCT log-magnitude spectrum tensor (B, 3, H, W)."""
    fft = torch.fft.fft2(img_tensor, norm="ortho")
    dct_approx = torch.log(torch.abs(fft.real) + 1e-6)
    min_val = dct_approx.amin(dim=(-2, -1), keepdim=True)
    max_val = dct_approx.amax(dim=(-2, -1), keepdim=True)
    return (dct_approx - min_val) / (max_val - min_val + 1e-6)


class MultiSourceBalancedDataset(Dataset):
    """Combines Kaggle and FF++ frames with balanced manipulation indexing."""
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (img_path, binary_label, manip_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, binary_label, manip_idx = self.samples[idx]
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", (160, 160), (128, 128, 128))

        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)

        return {
            "image": img_tensor,
            "binary_label": torch.tensor(binary_label, dtype=torch.float32),
            "manip_label": torch.tensor(manip_idx, dtype=torch.long),
        }


def build_datasets():
    logger.info("Building balanced multi-source dataset (Kaggle + FaceForensics++)...")
    
    # 1. Kaggle Samples (StyleGAN + FFHQ Real)
    kaggle_dir = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake")
    kaggle_train_real = list((kaggle_dir / "train/real").glob("*.jpg"))
    kaggle_train_fake = list((kaggle_dir / "train/fake").glob("*.jpg"))
    kaggle_val_real = list((kaggle_dir / "valid/real").glob("*.jpg"))
    kaggle_val_fake = list((kaggle_dir / "valid/fake").glob("*.jpg"))

    random.seed(42)
    random.shuffle(kaggle_train_real)
    random.shuffle(kaggle_train_fake)
    random.shuffle(kaggle_val_real)
    random.shuffle(kaggle_val_fake)

    # 2. FF++ Manifest
    manifest_csv = Path("manifests/ffpp_c23_manifest.csv")
    if manifest_csv.exists():
        df_ffpp = pd.read_csv(manifest_csv)
    else:
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    # Manipulation type mapping:
    # 0 = Real, 1 = Deepfakes, 2 = Face2Face, 3 = FaceSwap, 4 = NeuralTextures, (others -> Deepfakes)
    manip_map = {
        "real": 0,
        "Deepfakes": 1,
        "Face2Face": 2,
        "FaceSwap": 3,
        "NeuralTextures": 4,
        "FaceShifter": 3,
        "DeepFakeDetection": 1,
    }

    ffpp_by_type = {}
    for manip, grp in df_ffpp.groupby("manipulation_type"):
        col = "filepath" if "filepath" in grp.columns else "image_path"
        paths = grp[col].tolist()
        random.shuffle(paths)
        ffpp_by_type[manip] = paths

    # Assemble 40,000 Training Samples:
    # - 10,000 Kaggle Real (Label 0, Manip 0)
    # - 10,000 Kaggle Fake StyleGAN (Label 1, Manip 1)
    # - 4,000 FF++ Real (Label 0, Manip 0)
    # - 3,200 FF++ FaceSwap (Label 1, Manip 3)
    # - 3,200 FF++ Deepfakes (Label 1, Manip 1)
    # - 3,200 FF++ Face2Face (Label 1, Manip 2)
    # - 3,200 FF++ NeuralTextures (Label 1, Manip 4)
    # - 3,200 FF++ FaceShifter (Label 1, Manip 3)
    train_samples = []
    
    # Kaggle train
    for p in kaggle_train_real[:10000]:
        train_samples.append((str(p), 0, 0))
    for p in kaggle_train_fake[:10000]:
        train_samples.append((str(p), 1, 1))

    # FF++ train
    for p in ffpp_by_type.get("real", [])[:4000]:
        train_samples.append((p, 0, 0))
    for p in ffpp_by_type.get("FaceSwap", [])[:3200]:
        train_samples.append((p, 1, 3))
    for p in ffpp_by_type.get("Deepfakes", [])[:3200]:
        train_samples.append((p, 1, 1))
    for p in ffpp_by_type.get("Face2Face", [])[:3200]:
        train_samples.append((p, 1, 2))
    for p in ffpp_by_type.get("NeuralTextures", [])[:3200]:
        train_samples.append((p, 1, 4))
    for p in ffpp_by_type.get("FaceShifter", [])[:3200]:
        train_samples.append((p, 1, 3))

    random.shuffle(train_samples)

    # Assemble 4,400 Validation Samples:
    val_samples = []
    for p in kaggle_val_real[:1000]:
        val_samples.append((str(p), 0, 0))
    for p in kaggle_val_fake[:1000]:
        val_samples.append((str(p), 1, 1))
    for p in ffpp_by_type.get("real", [])[4000:4800]:
        val_samples.append((p, 0, 0))
    for p in ffpp_by_type.get("FaceSwap", [])[3200:3600]:
        val_samples.append((p, 1, 3))
    for p in ffpp_by_type.get("Deepfakes", [])[3200:3600]:
        val_samples.append((p, 1, 1))
    for p in ffpp_by_type.get("Face2Face", [])[3200:3600]:
        val_samples.append((p, 1, 2))
    for p in ffpp_by_type.get("NeuralTextures", [])[3200:3600]:
        val_samples.append((p, 1, 4))
    for p in ffpp_by_type.get("FaceShifter", [])[3200:3600]:
        val_samples.append((p, 1, 3))

    random.shuffle(val_samples)

    logger.info(f"Total Train Samples: {len(train_samples)} (Reals: {sum(1 for s in train_samples if s[1]==0)}, Fakes: {sum(1 for s in train_samples if s[1]==1)})")
    logger.info(f"Total Val Samples:   {len(val_samples)} (Reals: {sum(1 for s in val_samples if s[1]==0)}, Fakes: {sum(1 for s in val_samples if s[1]==1)})")

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

    train_dataset = MultiSourceBalancedDataset(train_samples, transform=train_transform)
    val_dataset = MultiSourceBalancedDataset(val_samples, transform=val_transform)

    return train_dataset, val_dataset


def configure_unfreezing_and_optimizer(model):
    """
    Unfreezes top layers of Spatial & Frequency encoders and CLIP ViT blocks 10-11.
    Sets up differential parameter groups with specialized learning rates.
    """
    # 1. Spatial Encoder: Freeze stages 0-4, unfreeze stages 5, 6, 7 and conv_head
    for p in model.spatial_encoder.parameters():
        p.requires_grad = False
    
    spatial_trainable = []
    for name, param in model.spatial_encoder.backbone.named_parameters():
        if any(f"blocks.{i}" in name for i in [4, 5, 6]) or "conv_head" in name or "bn2" in name:
            param.requires_grad = True
            spatial_trainable.append(param)

    # 2. Frequency Encoder: Freeze stages 0-4, unfreeze stages 5, 6, 7 and conv_head
    for p in model.frequency_encoder.parameters():
        p.requires_grad = False

    freq_trainable = []
    for name, param in model.frequency_encoder.backbone.named_parameters():
        if any(f"blocks.{i}" in name for i in [4, 5, 6]) or "conv_head" in name or "bn2" in name:
            param.requires_grad = True
            freq_trainable.append(param)

    # 3. CLIP ViT-B/32: Unfreeze blocks 10, 11 + ln_post + projection
    clip_trainable = []
    if hasattr(model, "clip_alignment") and model.clip_alignment is not None:
        for p in model.clip_alignment.parameters():
            p.requires_grad = False
        if hasattr(model.clip_alignment, "spatial_projection"):
            for p in model.clip_alignment.spatial_projection.parameters():
                p.requires_grad = True
                clip_trainable.append(p)
        if hasattr(model.clip_alignment, "clip_visual") and model.clip_alignment.clip_visual is not None:
            for name, param in model.clip_alignment.clip_visual.named_parameters():
                if any(f"resblocks.{i}" in name for i in [10, 11]) or "ln_post" in name or "proj" in name:
                    param.requires_grad = True
                    clip_trainable.append(param)

    # 4. Fusion and Detection Head: Fully Trainable
    fusion_trainable = list(model.fusion.parameters()) + list(model.detection_head.parameters())
    for p in fusion_trainable:
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Param Configuration: {trainable_params/1e6:.2f}M trainable / {total_params/1e6:.2f}M total ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"  Spatial Trainable:   {sum(p.numel() for p in spatial_trainable)/1e6:.2f}M (LR = 1.0e-5)")
    logger.info(f"  Frequency Trainable: {sum(p.numel() for p in freq_trainable)/1e6:.2f}M (LR = 1.0e-5)")
    logger.info(f"  CLIP Trainable:      {sum(p.numel() for p in clip_trainable)/1e6:.2f}M (LR = 5.0e-6)")
    logger.info(f"  Fusion + Head:       {sum(p.numel() for p in fusion_trainable)/1e6:.2f}M (LR = 5.0e-5)")

    param_groups = [
        {"params": spatial_trainable, "lr": 1.0e-5, "weight_decay": 1e-4},
        {"params": freq_trainable, "lr": 1.0e-5, "weight_decay": 1e-4},
        {"params": clip_trainable, "lr": 5.0e-6, "weight_decay": 1e-4},
        {"params": fusion_trainable, "lr": 5.0e-5, "weight_decay": 1e-4},
    ]

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


def train_e2e():
    set_seed(42)
    device = get_device()
    logger.info(f"Starting Hour 1 End-to-End Multi-Source Fine-Tuning on {device}")

    # Build model from best V2 checkpoint
    model = DeepfakeDetector()
    ckpt_path = "checkpoints/v2_clip_finetune/best_model.pth"
    if Path(ckpt_path).exists():
        logger.info(f"Loading base weights from: {ckpt_path}")
        load_checkpoint(ckpt_path, model, device=device)
    else:
        logger.warning(f"Base checkpoint not found at {ckpt_path}. Using random init.")
    
    model.to(device)

    # Configure unfreezing and optimizer
    optimizer = configure_unfreezing_and_optimizer(model)

    # Datasets and Loaders
    train_dataset, val_dataset = build_datasets()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    num_epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    focal_criterion = FocalBinaryLoss(gamma=2.0, alpha=0.25)
    manip_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path("checkpoints/v3_e2e_multisource")
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    total_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device)
            bin_lbls = batch["binary_label"].to(device)
            manip_lbls = batch["manip_label"].to(device)

            # Compute DCT on GPU
            with torch.no_grad():
                dcts = compute_dct_tensor_fast(imgs)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(imgs, dct=dcts)
                bin_logits = out["binary_logit"].squeeze(-1)
                loss_binary = focal_criterion(bin_logits, bin_lbls)
                loss_manip = manip_criterion(out["manipulation_logits"], manip_lbls)
                loss = loss_binary + 0.3 * loss_manip

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            with torch.no_grad():
                probs = torch.sigmoid(bin_logits).cpu().numpy()
                train_preds.extend(probs.tolist())
                train_targets.extend(bin_lbls.cpu().numpy().tolist())

            if (step + 1) % 250 == 0 or (step + 1) == len(train_loader):
                logger.info(f"Epoch [{epoch}/{num_epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()
        train_auc = roc_auc_score(train_targets, train_preds)
        train_acc = accuracy_score(train_targets, (np.array(train_preds) >= 0.5).astype(int))
        avg_train_loss = running_loss / len(train_loader)

        # Validation Pass
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                bin_lbls = batch["binary_label"].to(device)
                manip_lbls = batch["manip_label"].to(device)

                dcts = compute_dct_tensor_fast(imgs)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(imgs, dct=dcts)
                    bin_logits = out["binary_logit"].squeeze(-1)
                    loss_bin = focal_criterion(bin_logits, bin_lbls)
                    loss_m = manip_criterion(out["manipulation_logits"], manip_lbls)
                    v_loss = loss_bin + 0.3 * loss_m

                val_loss += v_loss.item()
                probs = torch.sigmoid(bin_logits).cpu().numpy()
                val_preds.extend(probs.tolist())
                val_targets.extend(bin_lbls.cpu().numpy().tolist())

        val_auc = roc_auc_score(val_targets, val_preds)
        val_acc = accuracy_score(val_targets, (np.array(val_preds) >= 0.5).astype(int))
        val_f1 = f1_score(val_targets, (np.array(val_preds) >= 0.5).astype(int))
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"--> Epoch [{epoch}/{num_epochs}] ({epoch_time:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100:.2f}% AUC: {train_auc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.2f}% AUC: {val_auc:.4f} F1: {val_f1:.4f}"
        )

        # Save Best Model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_path = out_dir / "best_model.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": float(avg_val_loss), "val_auc": float(val_auc), "val_acc": float(val_acc), "val_f1": float(val_f1)},
                path=str(best_path),
                scaler=scaler,
                scheduler=scheduler,
            )
            logger.info(f"  [SAVED] New best checkpoint saved with Val AUC = {val_auc:.5f} -> {best_path}")

    total_time = time.time() - total_start_time
    logger.info(f"=== Training Complete in {total_time/60:.2f} mins. Peak Val AUC: {best_val_auc:.5f} ===")


if __name__ == "__main__":
    train_e2e()
