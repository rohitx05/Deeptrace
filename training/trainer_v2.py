"""
V2 Trainer — extends Trainer with generator attribution loss and adversarial augmentations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from training.losses_v2 import ExtendedDeepfakeLoss
from utils.device import get_device, get_grad_scaler, print_memory_usage, empty_cache
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import compute_metrics
from models.generator_head import GeneratorFingerprintHead

logger = logging.getLogger(__name__)


class TrainerV2:
    """V2 Training orchestrator with extended losses and adversarial augmentations."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        stage_name: str = "default",
        use_adversarial: bool = False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.stage_name = stage_name
        self.use_adversarial = use_adversarial

        train_cfg = config.get("training", {})
        self.num_epochs = train_cfg.get("num_epochs", 50)
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.accumulation_steps = train_cfg.get("gradient_accumulation_steps", 8)
        self.warmup_epochs = train_cfg.get("warmup_epochs", 3)
        self.patience = train_cfg.get("early_stopping_patience", 10)

        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / stage_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(train_cfg.get("log_dir", "logs")) / stage_name

        hw_cfg = config.get("hardware", {})
        self.device = get_device(hw_cfg.get("device", "cuda") == "cuda")
        self.model = self.model.to(self.device)

        self.use_amp = hw_cfg.get("mixed_precision", True)
        self.scaler = get_grad_scaler(self.use_amp)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.num_epochs - self.warmup_epochs, 1)
        )

        loss_cfg = config.get("loss", {})
        self.criterion = ExtendedDeepfakeLoss(
            binary_weight=loss_cfg.get("binary_weight", 1.0),
            manipulation_weight=loss_cfg.get("manipulation_type_weight", 0.5),
            generator_weight=loss_cfg.get("generator_weight", 0.3),
            clip_alignment_weight=loss_cfg.get("clip_alignment_weight", 0.3),
            confidence_weight=loss_cfg.get("confidence_weight", 0.2),
            identity_weight=loss_cfg.get("identity_weight", 0.2),
        )

        self.writer = SummaryWriter(self.log_dir)
        self.best_val_auc = 0.0
        self.best_val_metrics = {}
        self.last_val_metrics = {}
        self.best_epoch = None
        self.best_checkpoint_path = str(self.checkpoint_dir / "best_model.pth")
        self.epochs_without_improvement = 0
        self.global_step = 0

    def train(self, resume_from: str = None) -> dict:
        start_epoch = 0
        if resume_from:
            meta = load_checkpoint(resume_from, self.model, self.optimizer, self.scaler, self.scheduler, self.device)
            start_epoch = meta["epoch"] + 1
            self.best_val_auc = meta["metrics"].get("val_auc", 0.0)

        logger.info(f"Training V2: {self.stage_name}, adversarial={self.use_adversarial}")
        logger.info(f"  Epochs: {self.num_epochs}, LR: {self.lr}, AMP: {self.use_amp}")
        print_memory_usage("Pre-training")

        for epoch in range(start_epoch, self.num_epochs):
            if epoch < self.warmup_epochs:
                warmup_lr = self.lr * (epoch + 1) / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)
            self.last_val_metrics = dict(val_metrics)
            self._log_epoch(epoch, train_metrics, val_metrics)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            val_auc = val_metrics.get("roc_auc", val_metrics.get("accuracy", 0))
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_metrics = dict(val_metrics)
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics,
                              str(self.checkpoint_dir / "best_model.pth"), self.scaler, self.scheduler)
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % 5 == 0:
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics,
                              str(self.checkpoint_dir / f"epoch_{epoch+1}.pth"), self.scaler, self.scheduler)

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
        self.model.train()
        total_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            mode = "video" if "frames" in batch else "image"

            # Build targets with generator type mapping
            targets = {"label": batch["label"], "manipulation_type": batch["manipulation_type"]}
            if "generator_type" not in batch:
                # Auto-map from manipulation_type
                targets["generator_type"] = torch.tensor([
                    GeneratorFingerprintHead.map_manipulation_to_generator(int(mt))
                    for mt in batch["manipulation_type"]
                ], device=self.device)
            else:
                targets["generator_type"] = batch["generator_type"]

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if mode == "video":
                    predictions = self.model(frames=batch["frames"], dct_frames=batch.get("dct_frames"), mode="video")
                else:
                    predictions = self.model(images=batch["image"], dct=batch.get("dct"), mode="image")

                losses = self.criterion(predictions, targets)
                loss = losses["total"] / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += losses["total"].item()
            probs = predictions["binary_pred"].detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels = batch["label"].cpu().numpy()
            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

            pbar.set_postfix(loss=f'{losses["total"].item():.4f}')

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics["loss"] = total_loss / max(len(self.train_loader), 1)
        return metrics

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            mode = "video" if "frames" in batch else "image"

            targets = {"label": batch["label"], "manipulation_type": batch["manipulation_type"]}
            targets["generator_type"] = torch.tensor([
                GeneratorFingerprintHead.map_manipulation_to_generator(int(mt))
                for mt in batch["manipulation_type"]
            ], device=self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if mode == "video":
                    predictions = self.model(frames=batch["frames"], dct_frames=batch.get("dct_frames"), mode="video")
                else:
                    predictions = self.model(images=batch["image"], dct=batch.get("dct"), mode="image")
                losses = self.criterion(predictions, targets)

            total_loss += losses["total"].item()
            probs = predictions["binary_pred"].cpu().numpy()
            all_labels.extend(batch["label"].cpu().numpy())
            all_preds.extend((probs > 0.5).astype(int))
            all_probs.extend(probs)

        metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        metrics["loss"] = total_loss / max(len(self.val_loader), 1)
        return metrics

    def _log_epoch(self, epoch, train_m, val_m):
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1}: Train Loss={train_m['loss']:.4f} Acc={train_m['accuracy']:.4f} | "
            f"Val Loss={val_m['loss']:.4f} Acc={val_m['accuracy']:.4f} AUC={val_m.get('roc_auc',0):.4f} | LR={lr:.6f}"
        )
        for k, v in train_m.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_m.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"val/{k}", v, epoch)
        self.writer.add_scalar("learning_rate", lr, epoch)
        print_memory_usage(f"Epoch {epoch+1}")
