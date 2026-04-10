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

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Model weights loaded from {path}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }
