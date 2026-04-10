"""
Core Trainer class for deepfake detection.
Handles training loop, validation, mixed precision, gradient accumulation,
checkpointing, LR scheduling, and early stopping.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import logging
import yaml

from training.losses import DeepfakeLoss
from utils.device import get_device, get_grad_scaler, print_memory_usage, empty_cache
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator with mixed precision and gradient accumulation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        stage_name: str = "default",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.stage_name = stage_name

        # Training config
        train_cfg = config.get("training", {})
        self.num_epochs = train_cfg.get("num_epochs", 50)
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.accumulation_steps = train_cfg.get("gradient_accumulation_steps", 8)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 3)
        self.patience = train_cfg.get("early_stopping_patience", 10)

        # Paths
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / stage_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(train_cfg.get("log_dir", "logs")) / stage_name

        # Device
        hw_cfg = config.get("hardware", {})
        self.device = get_device(hw_cfg.get("device", "cuda") == "cuda")
        self.model = self.model.to(self.device)

        # Mixed precision
        self.use_amp = hw_cfg.get("mixed_precision", True)
        self.scaler = get_grad_scaler(self.use_amp)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        scheduler_type = train_cfg.get("scheduler", "cosine")
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.num_epochs - self.warmup_epochs
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=15, gamma=0.1
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=5, factor=0.5
            )

        # Loss
        loss_cfg = config.get("loss", {})
        self.criterion = DeepfakeLoss(
            binary_weight=loss_cfg.get("binary_weight", 1.0),
            manipulation_weight=loss_cfg.get("manipulation_type_weight", 0.5),
            clip_alignment_weight=loss_cfg.get("clip_alignment_weight", 0.3),
            consistency_weight=loss_cfg.get("consistency_weight", 0.2),
        )

        # Tensorboard
        self.writer = SummaryWriter(self.log_dir)

        # State
        self.best_val_auc = 0.0
        self.best_val_metrics = {}
        self.last_val_metrics = {}
        self.best_epoch = None
        self.best_checkpoint_path = str(self.checkpoint_dir / "best_model.pth")
        self.epochs_without_improvement = 0
        self.global_step = 0

    def train(self, resume_from: str = None) -> dict:
        """Run the full training loop."""
        start_epoch = 0

        if resume_from:
            meta = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scaler, self.scheduler, self.device
            )
            start_epoch = meta["epoch"] + 1
            self.best_val_auc = meta["metrics"].get("val_auc", 0.0)
            logger.info(f"Resumed from epoch {start_epoch}")

        logger.info(f"Starting training: {self.stage_name}")
        logger.info(f"  Epochs: {self.num_epochs}, LR: {self.lr}, AMP: {self.use_amp}")
        logger.info(f"  Batch size: {self.train_loader.batch_size} x {self.accumulation_steps} accumulation")
        print_memory_usage("Pre-training")

        for epoch in range(start_epoch, self.num_epochs):
            # Warmup LR
            if epoch < self.warmup_epochs:
                warmup_lr = self.lr * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # Train
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate(epoch)
            self.last_val_metrics = dict(val_metrics)

            # Log
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Scheduler
            if epoch >= self.warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("loss", 0))
                else:
                    self.scheduler.step()

            # Checkpoint
            val_auc = val_metrics.get("roc_auc", val_metrics.get("accuracy", 0))
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_metrics = dict(val_metrics)
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    str(self.checkpoint_dir / "best_model.pth"),
                    self.scaler, self.scheduler,
                )
            else:
                self.epochs_without_improvement += 1

            # Periodic checkpoint
            if (epoch + 1) % 5 == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    str(self.checkpoint_dir / f"epoch_{epoch + 1}.pth"),
                    self.scaler, self.scheduler,
                )

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.writer.close()
        logger.info(f"Training complete. Best val AUC: {self.best_val_auc:.4f}")
        if not self.best_val_metrics:
            self.best_val_metrics = dict(self.last_val_metrics)
        return {
            "best_val_auc": self.best_val_auc,
            "best_val_metrics": self.best_val_metrics,
            "last_val_metrics": self.last_val_metrics,
            "best_epoch": self.best_epoch,
            "best_checkpoint_path": self.best_checkpoint_path,
            "checkpoint_dir": str(self.checkpoint_dir),
        }

    def _train_epoch(self, epoch: int) -> dict:
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Determine mode
            mode = "video" if "frames" in batch else "image"

            # Forward
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if mode == "video":
                    predictions = self.model(frames=batch["frames"], dct_frames=batch.get("dct_frames"), mode="video")
                else:
                    predictions = self.model(images=batch["image"], dct=batch.get("dct"), mode="image")

                targets = {
                    "label": batch["label"],
                    "manipulation_type": batch["manipulation_type"],
                }
                losses = self.criterion(predictions, targets)
                loss = losses["total"] / self.accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Track metrics
            total_loss += losses["total"].item()
            probs = predictions["binary_pred"].detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = batch["label"].cpu().numpy()
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

            pbar.set_postfix(loss=f'{losses["total"].item():.4f}')

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Val]")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            mode = "video" if "frames" in batch else "image"

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if mode == "video":
                    predictions = self.model(frames=batch["frames"], dct_frames=batch.get("dct_frames"), mode="video")
                else:
                    predictions = self.model(images=batch["image"], dct=batch.get("dct"), mode="image")

                targets = {"label": batch["label"], "manipulation_type": batch["manipulation_type"]}
                losses = self.criterion(predictions, targets)

            total_loss += losses["total"].item()
            probs = predictions["binary_pred"].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = batch["label"].cpu().numpy()
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics["loss"] = total_loss / max(len(self.val_loader), 1)
        return metrics

    def _log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log metrics to tensorboard and console."""
        lr = self.optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.4f} | "
            f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.4f} "
            f"AUC={val_metrics.get('roc_auc', 0):.4f} | LR={lr:.6f}"
        )

        # Tensorboard
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"val/{k}", v, epoch)
        self.writer.add_scalar("learning_rate", lr, epoch)

        print_memory_usage(f"Epoch {epoch + 1}")
