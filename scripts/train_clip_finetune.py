r"""
Phase 2 — CLIP Partial Unfreeze Fine-Tuning Script.

Loads Phase 1 best_model.pth (GAN fine-tuned), unfreezes CLIP ViT blocks 10-11
+ ln_post + proj, and continues training with full CLIP alignment loss enabled.

Goal: improve cross-generator generalisation by letting CLIP's highest-level
semantic representation adapt to the deepfake detection task.

Freeze config (matches AGENT_HANDOVER.md Phase 2 plan):
  FROZEN:   temporal_model, physiology_encoder, CLIP blocks 0-9
  TRAINED:  spatial+freq encoders, fusion, detection_head, CLIP blocks 10-11
            + ln_post + proj, clip_projection head
  VRAM est: ~5.5 GB  (reduce --batch_size to 8 if OOM)

Usage:
    .venv\Scripts\python.exe scripts/train_clip_finetune.py
    .venv\Scripts\python.exe scripts/train_clip_finetune.py --epochs 10 --batch_size 16
    .venv\Scripts\python.exe scripts/train_clip_finetune.py --resume checkpoints/v2_clip_finetune/last.pth
"""

import os
# Must be set BEFORE numpy/torch import — prevents OpenBLAS RAM explosion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.detector import DeepfakeDetector
from datasets.kaggle_realfake import KaggleRealFakeDataset
from utils.device import get_device, get_grad_scaler, empty_cache
from utils.metrics import compute_metrics


# ─── COLLATE ─────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Skip None fields (cache misses) so collate doesn't crash."""
    from torch.utils.data.dataloader import default_collate
    filtered = []
    for item in batch:
        filtered.append({k: v for k, v in item.items()
                         if v is not None and not isinstance(v, str)})
    keys = set(filtered[0].keys()) if filtered else set()
    for item in filtered:
        keys &= set(item.keys())
    filtered = [{k: item[k] for k in keys} for item in filtered]
    return default_collate(filtered)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clip_finetune")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   default="checkpoints/v1_gan_finetune/best_model.pth",
                        help="Phase 1 GAN-finetuned checkpoint to start from")
    parser.add_argument("--data_root",    default="data/kaggle_realfake")
    parser.add_argument("--stylegan_dir", default=r"D:\deepfake_data\kaggle_realfake\real_vs_fake\real-vs-fake\train\fake",
                        help="Extra GAN images folder on D: drive")
    parser.add_argument("--out_dir",      default="checkpoints/v2_clip_finetune")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=16,
                        help="Reduce to 8 if OOM (CLIP unfreeze adds ~1 GB VRAM)")
    parser.add_argument("--lr",           type=float, default=1e-5,
                        help="Main LR for spatial/freq/fusion/detection_head layers")
    parser.add_argument("--clip_lr",      type=float, default=5e-6,
                        help="Separate (lower) LR for unfrozen CLIP blocks")
    parser.add_argument("--clip_loss_weight", type=float, default=0.3,
                        help="Weight on alignment_loss relative to binary BCE")
    parser.add_argument("--accumulation", type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--image_size",   type=int,   default=160)
    parser.add_argument("--num_workers",  type=int,   default=2,
                        help="DataLoader workers. Use 0 if Windows spawn errors.")
    parser.add_argument("--num_clip_blocks", type=int, default=2,
                        help="Number of trailing CLIP ViT blocks to unfreeze")
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Resume from a Phase 2 last.pth checkpoint")
    parser.add_argument("--no_channels_last", action="store_true")
    parser.add_argument("--fused_optimizer",  action="store_true",
                        help="Fused AdamW (disabled by default; can crash on some builds)")
    args = parser.parse_args()

    device = get_device(prefer_cuda=True)
    use_channels_last = device.type == "cuda" and not args.no_channels_last
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        logger.info("CUDA optimizations: cudnn.benchmark, TF32, channels_last=%s", use_channels_last)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    extra_dirs = [args.stylegan_dir] if args.stylegan_dir and Path(args.stylegan_dir).exists() else []
    if extra_dirs:
        logger.info(f"Extra GAN dir: {args.stylegan_dir}")
    else:
        logger.warning(f"stylegan_dir not found: {args.stylegan_dir} — training without extra GAN images")

    train_ds = KaggleRealFakeDataset(args.data_root, split="train", image_size=args.image_size,
                                     extra_fake_dirs=extra_dirs)
    val_ds   = KaggleRealFakeDataset(args.data_root, split="val",   image_size=args.image_size)

    sg_count = sum(1 for s in train_ds.samples if s["generator_type"] == 0)
    logger.info(f"Train: {len(train_ds)} images | {sg_count} StyleGAN ({sg_count/max(len(train_ds),1)*100:.1f}%)")

    pin = device.type == "cuda"
    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs = {"persistent_workers": True, "prefetch_factor": 2}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True,
                              collate_fn=collate_fn, **worker_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin,
                              collate_fn=collate_fn, **worker_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info(f"Loading Phase 1 checkpoint: {args.checkpoint}")
    import yaml
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    model = DeepfakeDetector(config=model_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Phase 1 weights loaded | missing={len(missing)} unexpected={len(unexpected)}")

    # ── Freeze strategy ───────────────────────────────────────────────────────
    # Step 1: freeze temporal + physiology (always frozen in Phases 1 & 2)
    for name, param in model.named_parameters():
        if any(m in name for m in ("temporal_model", "physiology_encoder")):
            param.requires_grad = False

    # Step 2: freeze ALL clip_visual params (partial_unfreeze will selectively re-enable)
    if hasattr(model, "clip_alignment") and model.clip_alignment is not None:
        for param in model.clip_alignment.clip_visual.parameters():
            param.requires_grad = False

    # Step 3: call partial_unfreeze to open last num_clip_blocks
    clip_module = getattr(model, "clip_alignment", None)
    if clip_module is not None:
        n_unfrozen = clip_module.partial_unfreeze(num_blocks=args.num_clip_blocks)
        logger.info(f"CLIP partial unfreeze: {n_unfrozen/1e6:.2f}M params unlocked "
                    f"({args.num_clip_blocks} blocks + ln_post + proj)")
    else:
        logger.warning("Model has no clip_alignment module — CLIP unfreeze skipped")

    # Param count report
    total_trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    clip_trainable    = sum(p.numel() for p in clip_module.parameters() if p.requires_grad) \
                        if clip_module else 0
    logger.info(f"Trainable params total: {total_trainable/1e6:.1f}M "
                f"(CLIP portion: {clip_trainable/1e6:.2f}M)")

    # ── Optimiser: two param groups (main LR vs CLIP LR) ──────────────────────
    # Collect CLIP visual params that are now trainable
    clip_param_ids = set()
    if clip_module is not None and clip_module.clip_visual is not None:
        for p in clip_module.clip_visual.parameters():
            if p.requires_grad:
                clip_param_ids.add(id(p))

    clip_params  = [p for p in model.parameters() if p.requires_grad and id(p) in clip_param_ids]
    other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in clip_param_ids]

    optimizer_kwargs = {"weight_decay": 1e-4}
    if args.fused_optimizer and device.type == "cuda":
        optimizer_kwargs["fused"] = True

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": clip_params,  "lr": args.clip_lr},
        ],
        **optimizer_kwargs,
    )
    logger.info(f"AdamW: main_lr={args.lr:.1e} ({len(other_params)} param tensors), "
                f"clip_lr={args.clip_lr:.1e} ({len(clip_params)} CLIP param tensors)")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    use_amp  = device.type == "cuda"
    scaler   = get_grad_scaler(enabled=use_amp)
    bce      = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir="logs/v2_clip_finetune")

    start_epoch = 0
    best_auc    = -1.0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("best_auc", -1.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    all_params = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = clip_loss_sum = bin_loss_sum = 0.0
        correct = total = 0

        logger.info(f"Epoch {epoch+1}/{args.epochs} starting...")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                    unit="batch", dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            dct    = batch["dct"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
                dct    = dct.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                # Phase 2: enable CLIP alignment loss
                out = model(images=images, dct=dct, compute_clip_alignment_loss=True)

                bin_loss  = bce(out["binary_logit"].squeeze(-1), labels.float())
                clip_loss = out.get("clip_alignment_loss", torch.tensor(0.0, device=device))
                loss = bin_loss + args.clip_loss_weight * clip_loss
                loss = loss / args.accumulation

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss    += loss.item() * args.accumulation
            bin_loss_sum  += bin_loss.item()
            clip_loss_sum += clip_loss.item() if isinstance(clip_loss, torch.Tensor) else clip_loss
            preds   = (torch.sigmoid(out["binary_logit"].squeeze(-1)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            pbar.set_postfix({
                "loss": f"{total_loss/(step+1):.4f}",
                "acc":  f"{correct/max(total,1):.4f}",
            }, refresh=False)

            if (step + 1) % 200 == 0:
                logger.info(
                    f"  step {step+1}/{len(train_loader)} | "
                    f"loss={total_loss/(step+1):.4f} "
                    f"bin={bin_loss_sum/(step+1):.4f} "
                    f"clip={clip_loss_sum/(step+1):.4f} "
                    f"acc={correct/max(total,1):.4f}"
                )

        pbar.close()
        scheduler.step()
        train_acc = correct / total

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="  Val", unit="batch",
                            dynamic_ncols=True, leave=False)
            for batch in val_pbar:
                images = batch["image"].to(device, non_blocking=True)
                dct    = batch["dct"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                if use_channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                    dct    = dct.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out = model(images=images, dct=dct, compute_clip_alignment_loss=False)
                    probs = torch.sigmoid(out["binary_logit"].squeeze(-1))
                all_preds.extend(probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

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
            f"clip={clip_loss_sum/len(train_loader):.4f} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )

        writer.add_scalars("loss",      {"train": total_loss/len(train_loader)}, epoch)
        writer.add_scalars("clip_loss", {"train": clip_loss_sum/len(train_loader)}, epoch)
        writer.add_scalars("auc",       {"val": val_auc}, epoch)
        writer.add_scalars("acc",       {"train": train_acc, "val": val_acc}, epoch)

        # Save last checkpoint (always)
        torch.save({
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_auc":       best_auc,
        }, out_dir / "last.pth")

        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     val_auc,
                "val_acc":     val_acc,
            }, out_dir / "best_model.pth")
            logger.info(f"  New best AUC: {best_auc:.4f} -> saved best_model.pth")

        empty_cache()

    writer.close()
    logger.info(f"\nPhase 2 complete. Best val AUC: {best_auc:.4f}")
    logger.info(f"Checkpoint: {out_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
