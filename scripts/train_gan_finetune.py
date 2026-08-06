r"""
Phase 1 — GAN Style Fine-Tuning Script.

Loads V1 best_model.pth, unfreezes spatial + frequency encoders,
trains with generator attribution loss active on StyleGAN-labelled data.

Usage:
    .venv\Scripts\python.exe scripts/train_gan_finetune.py
    .venv\Scripts\python.exe scripts/train_gan_finetune.py --epochs 15 --batch_size 16
    .venv\Scripts\python.exe scripts/train_gan_finetune.py --resume checkpoints/v1_gan_finetune/last.pth
"""

import os
# Must be set BEFORE numpy/torch import — prevents OpenBLAS RAM explosion with multiple workers
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.detector import DeepfakeDetector
from datasets.kaggle_realfake import KaggleRealFakeDataset
from utils.checkpoint import save_checkpoint
from utils.device import get_device, get_grad_scaler, empty_cache
from utils.metrics import compute_metrics


def collate_fn(batch):
    """Skip None fields (cache misses) so collate doesn't crash."""
    from torch.utils.data.dataloader import default_collate
    # filter out None values per key
    filtered = []
    for item in batch:
        filtered.append({k: v for k, v in item.items() if v is not None
                         and not isinstance(v, str)})
    # Only keep keys present in ALL items
    keys = set(filtered[0].keys()) if filtered else set()
    for item in filtered:
        keys &= set(item.keys())
    filtered = [{k: item[k] for k in keys} for item in filtered]
    return default_collate(filtered)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gan_finetune")


# ─── LOSS ────────────────────────────────────────────────────────────────────

class GANFinetuneLoss(nn.Module):
    """
    Binary CE + generator attribution CE (StyleGAN vs unknown fake).
    Generator head is a lightweight 2-layer MLP attached at training time only.
    """
    def __init__(self, gen_weight: float = 0.4, label_smoothing: float = 0.05):
        super().__init__()
        self.gen_weight = gen_weight
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-1)

    def forward(self, binary_logit, labels, gen_logits, gen_labels):
        # Binary real/fake loss
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logit.squeeze(-1), labels.float()
        )
        # Generator attribution loss (only on fake samples, ignores real with -1)
        fake_mask = labels > 0.5
        if fake_mask.any() and gen_logits is not None:
            gen_loss = self.ce(gen_logits[fake_mask], gen_labels[fake_mask])
        else:
            gen_loss = torch.tensor(0.0, device=binary_logit.device)

        total = binary_loss + self.gen_weight * gen_loss
        return total, binary_loss.item(), gen_loss.item()


# ─── GENERATOR HEAD ──────────────────────────────────────────────────────────

class GeneratorHead(nn.Module):
    """Lightweight 2-class head (StyleGAN=0 vs Unknown=3→1) on top of fused features."""
    def __init__(self, in_dim: int = 512, num_classes: int = 4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/kaggle_realfake/best_model.pth")
    parser.add_argument("--data_root", default="data/kaggle_realfake")
    parser.add_argument("--stylegan_dir", default=r"D:\deepfake_data\kaggle_realfake\real_vs_fake\real-vs-fake\train\fake",
                        help="Folder containing your downloaded GAN images on D: drive")
    parser.add_argument("--out_dir", default="checkpoints/v1_gan_finetune")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulation", type=int, default=4,   help="gradient accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers. Use 0 if Windows pagefile/spawn errors return.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gan_only", action="store_true",
                        help="Train on 10k real + 10k StyleGAN only (fast mode, ~8 min/epoch)")
    parser.add_argument("--no_channels_last", action="store_true",
                        help="Disable channels_last memory format optimization")
    parser.add_argument("--fused_optimizer", action="store_true",
                        help="Use fused AdamW. Disabled by default because some PyTorch/CUDA builds crash with AMP.")
    args = parser.parse_args()

    device = get_device(prefer_cuda=True)
    use_channels_last = device.type == "cuda" and not args.no_channels_last
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("CUDA optimizations enabled: cudnn.benchmark, TF32, channels_last=%s", use_channels_last)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    # extra_fake_dirs: your manually downloaded GAN images on D: drive
    extra_dirs = [args.stylegan_dir] if args.stylegan_dir and Path(args.stylegan_dir).exists() else []
    if extra_dirs:
        logger.info(f"Extra GAN dir: {args.stylegan_dir}")
    else:
        logger.warning(f"stylegan_dir not found or empty: {args.stylegan_dir} — training without extra GAN images")

    train_ds = KaggleRealFakeDataset(args.data_root, split="train", image_size=args.image_size,
                                     extra_fake_dirs=extra_dirs)
    val_ds   = KaggleRealFakeDataset(args.data_root, split="val",   image_size=args.image_size)

    # Log StyleGAN coverage
    sg_count = sum(1 for s in train_ds.samples if s["generator_type"] == 0)
    logger.info(f"Train: {len(train_ds)} images | {sg_count} StyleGAN ({sg_count/max(len(train_ds),1)*100:.1f}%)")

    # ── GAN-only fast mode: 10k real + 10k StyleGAN ───────────────────────────
    if args.gan_only:
        from torch.utils.data import Subset
        import random
        gan_idx  = [i for i, s in enumerate(train_ds.samples) if s["generator_type"] == 0]
        real_idx = [i for i, s in enumerate(train_ds.samples) if s["label"] == 0]
        random.shuffle(real_idx)
        keep = gan_idx + real_idx[:len(gan_idx)]   # equal real/GAN
        random.shuffle(keep)
        train_ds = Subset(train_ds, keep)
        logger.info(f"GAN-only mode: {len(keep)} images ({len(gan_idx)} GAN + {len(gan_idx)} real)")

    pin = device.type == "cuda"
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs = {"persistent_workers": True, "prefetch_factor": 2}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                              collate_fn=collate_fn, **worker_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size*2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin,
                              collate_fn=collate_fn, **worker_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading V1 checkpoint: {args.checkpoint}")
    import yaml
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    model = DeepfakeDetector(config=model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"V1 weights loaded | missing={len(missing)} unexpected={len(unexpected)}")

    # Freeze temporal + physiology + CLIP backbone (not projection)
    for name, param in model.named_parameters():
        if any(m in name for m in ("temporal_model", "physiology_encoder", "clip_visual")):
            param.requires_grad = False

    # Attach generator head
    gen_head = GeneratorHead(in_dim=512, num_classes=4).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in gen_head.parameters())
    logger.info(f"Trainable params: {trainable/1e6:.1f}M")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    params = list(filter(lambda p: p.requires_grad, model.parameters())) + \
             list(gen_head.parameters())
    optimizer_kwargs = {"lr": args.lr, "weight_decay": 1e-4}
    if args.fused_optimizer and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    use_amp  = device.type == "cuda"
    scaler   = get_grad_scaler(enabled=use_amp)
    criterion = GANFinetuneLoss(gen_weight=0.4)

    writer = SummaryWriter(log_dir=f"logs/v1_gan_finetune")

    start_epoch = 0
    best_auc    = -1.0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        gen_head.load_state_dict(ckpt.get("gen_head_state", gen_head.state_dict()))
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        gen_head.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = bin_loss_sum = gen_loss_sum = 0.0
        correct = total = 0

        logger.info(f"Epoch {epoch+1}/{args.epochs} starting...")
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(pbar):
            images      = batch["image"].to(device, non_blocking=True)
            dct         = batch["dct"].to(device, non_blocking=True)
            labels      = batch["label"].to(device, non_blocking=True)
            gen_labels  = batch["generator_type"].to(device, non_blocking=True)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
                dct = dct.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(images=images, dct=dct, compute_clip_alignment_loss=False)
                # Get fused representation for generator head
                fused = out.get("fused_features", None)
                gen_logits = gen_head(fused) if fused is not None else None

                loss, bl, gl = criterion(out["binary_logit"], labels, gen_logits, gen_labels)
                loss = loss / args.accumulation

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss   += loss.item() * args.accumulation
            bin_loss_sum += bl
            gen_loss_sum += gl
            preds = (torch.sigmoid(out["binary_logit"].squeeze(-1)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss/(step+1):.4f}",
                "acc":  f"{correct/max(total,1):.4f}",
            }, refresh=False)

            # Also log every 200 steps
            if (step + 1) % 200 == 0:
                logger.info(
                    f"  step {step+1}/{len(train_loader)} | "
                    f"loss={total_loss/(step+1):.4f} acc={correct/max(total,1):.4f}"
                )

        pbar.close()

        scheduler.step()
        train_acc = correct / total

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        gen_head.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="  Val", unit="batch", dynamic_ncols=True, leave=False)
            for batch in val_pbar:
                images = batch["image"].to(device, non_blocking=True)
                dct    = batch["dct"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                if use_channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                    dct = dct.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out = model(images=images, dct=dct, compute_clip_alignment_loss=False)
                    probs = torch.sigmoid(out["binary_logit"].squeeze(-1))
                all_preds.extend(probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        import numpy as np
        y_true = np.array(all_labels)
        y_prob = np.array(all_preds)
        y_pred = (y_prob > 0.5).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        val_auc = metrics.get("roc_auc", metrics.get("auc", 0.0))
        val_acc = metrics.get("accuracy", 0.0)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={total_loss/len(train_loader):.4f} "
            f"bin={bin_loss_sum/len(train_loader):.4f} "
            f"gen={gen_loss_sum/len(train_loader):.4f} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )

        writer.add_scalars("loss", {"train": total_loss/len(train_loader)}, epoch)
        writer.add_scalars("auc",  {"val": val_auc}, epoch)
        writer.add_scalars("acc",  {"train": train_acc, "val": val_acc}, epoch)

        # Save last checkpoint (always)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "gen_head_state": gen_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_auc": best_auc,
        }, out_dir / "last.pth")

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "gen_head_state": gen_head.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
            }, out_dir / "best_model.pth")
            logger.info(f"  New best AUC: {best_auc:.4f} -> saved best_model.pth")

        empty_cache()

    writer.close()
    logger.info(f"\nPhase 1 complete. Best val AUC: {best_auc:.4f}")
    logger.info(f"Checkpoint: {out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
