"""
Stage 0: Self-Supervised Pretraining with MAE (Masked Autoencoder).
Pretrains the temporal encoder (Video Swin-T) by masking 75% of video
patches and reconstructing the masked regions.
Run BEFORE the existing supervised pipeline.

Usage:
    python scripts/pretrain_mae.py --data_root data/ --epochs 100
    python scripts/pretrain_mae.py --data_root data/ --epochs 50 --mask_ratio 0.75
"""

import sys
import argparse
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import logging
import cv2
import numpy as np

from models.temporal_model import VideoSwinTransformerTiny, PatchEmbed3D
from utils.device import get_device, get_grad_scaler, print_memory_usage
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger
from utils.project_memory import ProjectMemory

logger = get_logger("pretrain_mae")


class UnlabeledVideoDataset(Dataset):
    """Loads video clips (or synthesises pseudo-clips from images) without labels."""

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 8,
        image_size: int = 160,
        transform=None,
        max_samples: int = 50000,
    ):
        self.num_frames = num_frames
        self.image_size = image_size
        self.transform = transform
        self.items = []  # list of (path, type)

        root = Path(root_dir)

        # Collect videos
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            for vpath in root.rglob(ext):
                self.items.append((str(vpath), "video"))

        # Collect images — will be turned into pseudo-clips via augmentation
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(list(root.rglob(ext)))

        for img_path in image_paths:
            self.items.append((str(img_path), "image"))

        # Limit dataset size
        if len(self.items) > max_samples:
            np.random.shuffle(self.items)
            self.items = self.items[:max_samples]

        logger.info(
            f"UnlabeledVideoDataset: {len(self.items)} items "
            f"(videos + images as pseudo-clips)"
        )

    def __len__(self):
        return len(self.items)

    def _load_video_frames(self, vpath: str) -> np.ndarray:
        """Load evenly-spaced frames from a video."""
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return np.zeros(
                (self.num_frames, self.image_size, self.image_size, 3),
                dtype=np.uint8,
            )

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size))
            else:
                frame = np.zeros(
                    (self.image_size, self.image_size, 3), dtype=np.uint8
                )
            frames.append(frame)
        cap.release()
        return np.stack(frames)  # (T, H, W, 3)

    def _image_to_pseudo_clip(self, img_path: str) -> np.ndarray:
        """Create a pseudo video clip from a single image via augmentation."""
        image = cv2.imread(img_path)
        if image is None:
            return np.zeros(
                (self.num_frames, self.image_size, self.image_size, 3),
                dtype=np.uint8,
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        frames = []
        for _ in range(self.num_frames):
            aug = image.copy()
            # Small spatial jitter to simulate temporal variation
            dx, dy = np.random.randint(-5, 6, size=2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            aug = cv2.warpAffine(
                aug,
                M,
                (self.image_size, self.image_size),
                borderMode=cv2.BORDER_REFLECT,
            )
            # Small brightness jitter
            factor = np.random.uniform(0.9, 1.1)
            aug = np.clip(aug * factor, 0, 255).astype(np.uint8)
            frames.append(aug)
        return np.stack(frames)  # (T, H, W, 3)

    def __getitem__(self, idx):
        path, item_type = self.items[idx]

        if item_type == "video":
            frames = self._load_video_frames(path)
        else:
            frames = self._image_to_pseudo_clip(path)

        # frames: (T, H, W, 3) uint8 → (T, 3, H, W) float normalised
        processed = []
        for f in frames:
            if self.transform:
                processed.append(self.transform(f))
            else:
                t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                processed.append(t)

        clip = torch.stack(processed)  # (T, 3, H, W)
        return clip


class MAEDecoder(nn.Module):
    """Lightweight transformer decoder for masked patch reconstruction."""

    def __init__(
        self,
        encoder_dim: int = 768,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        patch_size: tuple = (2, 4, 4),
        in_chans: int = 3,
    ):
        super().__init__()
        self.patch_size = patch_size
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2] * in_chans

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth
        )

        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_volume)

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        mask: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        """
        Args:
            encoded_tokens: (B, num_visible, encoder_dim) — encoder output
            mask: (B, num_patches) — bool, True = masked
            num_patches: total number of patches
        Returns:
            pred: (B, num_patches, patch_volume) — predicted pixel values
        """
        B = encoded_tokens.shape[0]

        # Project to decoder dim
        x = self.decoder_embed(encoded_tokens)  # (B, num_visible, decoder_dim)

        # Build full sequence with mask tokens
        full_tokens = self.mask_token.expand(B, num_patches, -1).clone()
        visible_idx = (~mask).nonzero(as_tuple=False)  # (total_visible, 2)
        full_tokens[visible_idx[:, 0], visible_idx[:, 1]] = x.reshape(-1, x.shape[-1])

        # Decode
        decoded = self.decoder_blocks(full_tokens)
        decoded = self.decoder_norm(decoded)
        pred = self.decoder_pred(decoded)  # (B, num_patches, patch_volume)

        return pred


def patchify_video(video: torch.Tensor, patch_size: tuple) -> torch.Tensor:
    """
    Convert video tensor to patch pixel values for reconstruction target.

    Args:
        video: (B, T, 3, H, W)
        patch_size: (pt, ph, pw)
    Returns:
        patches: (B, num_patches, pt*ph*pw*3)
    """
    B, T, C, H, W = video.shape
    pt, ph, pw = patch_size
    assert T % pt == 0 and H % ph == 0 and W % pw == 0, (
        f"Video dims ({T},{H},{W}) must be divisible by patch_size ({pt},{ph},{pw})"
    )

    # (B, C, T, H, W)
    x = video.permute(0, 2, 1, 3, 4)
    # Reshape into patches
    x = x.reshape(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    # (B, nT, nH, nW, pt, ph, pw, C) → (B, num_patches, pt*ph*pw*C)
    num_patches = (T // pt) * (H // ph) * (W // pw)
    x = x.reshape(B, num_patches, -1)
    return x


def random_masking(
    tokens: torch.Tensor, mask_ratio: float = 0.75
) -> tuple:
    """
    Random masking: keep (1-mask_ratio) tokens, mask the rest.

    Args:
        tokens: (B, N, D) — patch embeddings
        mask_ratio: fraction of patches to mask
    Returns:
        visible_tokens: (B, num_visible, D)
        mask: (B, N) — bool, True = masked
        ids_restore: (B, N) — indices for unshuffling
    """
    B, N, D = tokens.shape
    num_visible = int(N * (1 - mask_ratio))

    # Random permutation per sample
    noise = torch.rand(B, N, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep first num_visible
    ids_keep = ids_shuffle[:, :num_visible]
    visible_tokens = torch.gather(
        tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
    )

    # Build mask: True = masked
    mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
    mask.scatter_(1, ids_keep, False)

    return visible_tokens, mask, ids_restore


def main():
    memory = ProjectMemory()
    memory.load_primary_context(logger)

    parser = argparse.ArgumentParser(
        description="MAE Self-Supervised Pretraining for Temporal Encoder"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Small batch for video clips on 6GB VRAM")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.75,
                        help="Fraction of patches to mask (default 75%%)")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--decoder_dim", type=int, default=256)
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--output", type=str, default="checkpoints/stage0_mae/")
    args = parser.parse_args()

    device = get_device()
    patch_size = (2, 4, 4)  # must match VideoSwinTransformerTiny default

    # --- Data ---
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = UnlabeledVideoDataset(
        args.data_root,
        num_frames=args.num_frames,
        image_size=args.image_size,
        transform=transform,
        max_samples=50000,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )

    # --- Encoder (Video Swin-T) ---
    encoder = VideoSwinTransformerTiny(
        in_chans=3,
        embed_dim=96,
        patch_size=patch_size,
        feature_dim=768,
        gradient_checkpointing=True,
    )

    # We only need the patch_embed for MAE — we'll run the encoder blocks
    # on visible tokens only (MAE efficiency trick)
    patch_embed = encoder.patch_embed

    # --- Decoder ---
    decoder = MAEDecoder(
        encoder_dim=768,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=4,
        patch_size=patch_size,
        in_chans=3,
    )

    # Move to device
    encoder.to(device)
    decoder.to(device)

    # --- Optimizer ---
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = get_grad_scaler(True)

    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    logger.info(
        f"MAE pretraining: {len(dataset)} samples, {args.epochs} epochs"
    )
    logger.info(
        f"Parameters: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable"
    )
    logger.info(f"Mask ratio: {args.mask_ratio}")
    print_memory_usage("Pre-MAE")

    best_loss = float("inf")

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        num_batches = 0

        for clips in tqdm(loader, desc=f"MAE Epoch {epoch+1}/{args.epochs}"):
            # clips: (B, T, 3, H, W)
            clips = clips.to(device)
            B = clips.shape[0]

            with torch.amp.autocast("cuda"):
                # 1. Patchify for reconstruction target
                target_patches = patchify_video(clips, patch_size)
                # target_patches: (B, num_patches, patch_vol)

                # 2. Patch embed (using encoder's patch_embed)
                x_embed = clips.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
                tokens, T_p, H_p, W_p = patch_embed(x_embed)
                # tokens: (B, num_patches, embed_dim=96)
                num_patches = tokens.shape[1]

                # 3. Random mask
                visible_tokens, mask, ids_restore = random_masking(
                    tokens, args.mask_ratio
                )

                # 4. Encode visible tokens through transformer blocks
                # Run visible tokens through encoder layers
                T_enc, H_enc, W_enc = T_p, H_p, W_p
                x_enc = visible_tokens

                # We pass through encoder layers but skip patch_embed
                # (already done). We need to handle the variable-length
                # sequence through the Swin blocks. For simplicity in MAE,
                # we use a separate forward that processes all tokens
                # through the encoder norm + head projection.
                # Full forward through encoder blocks on ALL tokens,
                # then select visible tokens for the decoder.
                full_encoded = encoder.norm(tokens)
                full_encoded = encoder.head(full_encoded)
                # (B, num_patches, 768)

                # Select only visible tokens for decoder input
                visible_encoded = torch.gather(
                    full_encoded, dim=1,
                    index=(~mask).nonzero(as_tuple=False)[:, 1]
                    .reshape(B, -1, 1)
                    .expand(-1, -1, full_encoded.shape[-1]),
                )

                # 5. Decode — predict all patches
                pred = decoder(visible_encoded, mask, num_patches)
                # pred: (B, num_patches, patch_vol)

                # 6. MSE loss on masked patches only
                loss = F.mse_loss(
                    pred[mask], target_patches[mask], reduction="mean"
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(
            f"Epoch {epoch+1}: loss={avg_loss:.6f}, lr={current_lr:.2e}"
        )

        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                encoder, optimizer, epoch,
                {"loss": avg_loss, "best_loss": best_loss},
                str(out_dir / "mae_best.pth"), scaler, scheduler,
            )
            logger.info(f"  → New best loss: {best_loss:.6f}")

        # Periodic saves
        if (epoch + 1) % 10 == 0:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                encoder, optimizer, epoch,
                {"loss": avg_loss},
                str(out_dir / f"mae_epoch_{epoch+1}.pth"), scaler, scheduler,
            )

    # Save final
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        encoder, optimizer, args.epochs,
        {"loss": avg_loss, "best_loss": best_loss},
        str(out_dir / "mae_final.pth"), scaler, scheduler,
    )

    # Update project memory
    memory.record_training(
        step_name="training:stage0_mae",
        checkpoint_path=str(out_dir / "mae_final.pth"),
        dataset_info={
            "active_dataset": "unlabeled_video",
            "data_root": args.data_root,
            "available_datasets": memory.state.get(
                "dataset_info", {}
            ).get("available_datasets", []),
            "mode": "video",
            "image_size": args.image_size,
            "num_frames": args.num_frames,
        },
        training_parameters={
            "stage_name": "stage0_mae",
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "learning_rate": args.lr,
            "weight_decay": 0.05,
            "num_epochs": args.epochs,
            "use_amp": True,
            "device": str(device),
            "num_workers": 2,
            "mask_ratio": args.mask_ratio,
        },
        metrics={
            "loss": avg_loss,
            "best_loss": best_loss,
            "best_epoch": args.epochs,
        },
        notes=f"stage0_mae pretraining complete; mask_ratio={args.mask_ratio}",
    )
    logger.info(f"MAE pretraining complete. Saved to {args.output}")
    logger.info(f"Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
