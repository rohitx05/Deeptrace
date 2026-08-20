"""Checkpoint save/load with optimizer state and epoch tracking."""

import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: str,
    scaler=None,
    scheduler=None,
):
    """Save a training checkpoint."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, save_path)
    logger.info(f"Checkpoint saved: {save_path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scaler=None,
    scheduler=None,
    device: torch.device = None,
) -> dict:
    """Load a training checkpoint. Returns metadata dict."""
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)

    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("model_state")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    model.load_state_dict(state_dict)
    logger.info(f"Model weights loaded from {path}")

    opt_state = checkpoint.get("optimizer_state_dict") or checkpoint.get("optimizer_state")
    if optimizer and opt_state is not None:
        optimizer.load_state_dict(opt_state)

    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {
            "val_auc": checkpoint.get("val_auc"),
            "val_acc": checkpoint.get("val_acc"),
        }),
    }
