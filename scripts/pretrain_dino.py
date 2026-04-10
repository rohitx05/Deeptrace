"""
Stage 0: Self-Supervised Pretraining with DINO.
Pretrains the spatial encoder using DINO (self-distillation with no labels).
Run BEFORE the existing 4-stage supervised pipeline.

Usage:
    python scripts/pretrain_dino.py --data_root data/ --epochs 100
"""

import sys
import argparse
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import yaml
import logging
import cv2
import glob
import numpy as np

from models.spatial_encoder import SpatialEncoder
from utils.device import get_device, get_grad_scaler, print_memory_usage
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger
from utils.project_memory import ProjectMemory

logger = get_logger("pretrain_dino")


class UnlabeledFaceDataset(Dataset):
    """Simple dataset that loads face images without labels."""

    def __init__(self, root_dir: str, transform=None, max_samples: int = 50000):
        self.transform = transform
        self.image_paths = []

        # Recursively find all images
        root = Path(root_dir)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.image_paths.extend(list(root.rglob(ext)))

        # Also extract frames from videos
        video_exts = ["*.mp4", "*.avi", "*.mov"]
        video_paths = []
        for ext in video_exts:
            video_paths.extend(list(root.rglob(ext)))

        # Sample frames from videos (max 5 per video)
        for vpath in video_paths[:200]:  # limit to 200 videos
            cap = cv2.VideoCapture(str(vpath))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                for idx in np.linspace(0, total - 1, min(5, total), dtype=int):
                    self.image_paths.append((str(vpath), int(idx)))  # (video_path, frame_idx)
            cap.release()

        if len(self.image_paths) > max_samples:
            np.random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_samples]

        logger.info(f"UnlabeledFaceDataset: {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        item = self.image_paths[idx]

        if isinstance(item, tuple):
            # Video frame
            vpath, frame_idx = item
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                frame = np.zeros((160, 160, 3), dtype=np.uint8)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(str(item))
            if image is None:
                image = np.zeros((160, 160, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (160, 160))

        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
            return view1, view2

        return image, image


class DINOLoss(nn.Module):
    """DINO self-distillation loss with centering."""

    def __init__(self, out_dim: int, teacher_temp: float = 0.04, student_temp: float = 0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        student_probs = F.log_softmax(student_out / self.student_temp, dim=-1)
        teacher_probs = F.softmax((teacher_out - self.center) / self.teacher_temp, dim=-1).detach()
        loss = -torch.sum(teacher_probs * student_probs, dim=-1).mean()

        # Update center (EMA)
        self.center = 0.9 * self.center + 0.1 * teacher_out.mean(dim=0, keepdim=True)
        return loss


def main():
    memory = ProjectMemory()
    memory.load_primary_context(logger)

    parser = argparse.ArgumentParser(description="DINO Self-Supervised Pretraining")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dim", type=int, default=256)
    parser.add_argument("--output", type=str, default="checkpoints/stage0_dino/")
    args = parser.parse_args()

    device = get_device()

    # Augmentations: two random crops with different augmentations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(160, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = UnlabeledFaceDataset(args.data_root, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Student and teacher
    student = SpatialEncoder(pretrained=True)
    student_head = nn.Sequential(nn.Linear(1280, 512), nn.ReLU(), nn.Linear(512, args.out_dim))

    teacher = copy.deepcopy(student)
    teacher_head = copy.deepcopy(student_head)
    for p in teacher.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False

    student.to(device)
    student_head.to(device)
    teacher.to(device)
    teacher_head.to(device)

    criterion = DINOLoss(args.out_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(student_head.parameters()),
        lr=args.lr, weight_decay=0.04
    )
    scaler = get_grad_scaler(True)

    logger.info(f"DINO pretraining: {len(dataset)} samples, {args.epochs} epochs")
    print_memory_usage("Pre-DINO")

    ema_momentum = 0.996

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0

        for view1, view2 in tqdm(loader, desc=f"DINO Epoch {epoch+1}/{args.epochs}"):
            view1, view2 = view1.to(device), view2.to(device)

            with torch.amp.autocast("cuda"):
                s1 = student_head(student(view1))
                s2 = student_head(student(view2))
                with torch.no_grad():
                    t1 = teacher_head(teacher(view1))
                    t2 = teacher_head(teacher(view2))
                loss = (criterion(s1, t2) + criterion(s2, t1)) / 2

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update teacher
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = ema_momentum * tp.data + (1 - ema_momentum) * sp.data
                for tp, sp in zip(teacher_head.parameters(), student_head.parameters()):
                    tp.data = ema_momentum * tp.data + (1 - ema_momentum) * sp.data

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            out_dir = Path(args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(student, optimizer, epoch, {"loss": avg_loss},
                          str(out_dir / f"dino_epoch_{epoch+1}.pth"), scaler)

    # Save final
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(student, optimizer, args.epochs, {"loss": avg_loss},
                    str(out_dir / "dino_final.pth"), scaler)
    memory.record_training(
        step_name="training:stage0_dino",
        checkpoint_path=str(out_dir / "dino_final.pth"),
        dataset_info={
            "active_dataset": "unlabeled_faces",
            "data_root": args.data_root,
            "available_datasets": memory.state.get("dataset_info", {}).get("available_datasets", []),
            "mode": "image",
            "image_size": 160,
            "num_frames": 1,
        },
        training_parameters={
            "stage_name": "stage0_dino",
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "learning_rate": args.lr,
            "weight_decay": 0.04,
            "num_epochs": args.epochs,
            "use_amp": True,
            "device": str(device),
            "num_workers": 2,
        },
        metrics={
            "loss": avg_loss,
            "best_epoch": args.epochs,
        },
        notes="stage0_dino pretraining complete",
    )
    logger.info(f"DINO pretraining complete. Saved to {args.output}")


if __name__ == "__main__":
    main()
