"""Device utilities: GPU detection, memory monitoring, AMP context."""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def get_memory_info() -> dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "total": 0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def print_memory_usage(prefix: str = ""):
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    info = get_memory_info()
    logger.info(
        f"{prefix} GPU Memory: {info['allocated']:.2f}GB allocated, "
        f"{info['reserved']:.2f}GB reserved, {info['total']:.1f}GB total"
    )


def empty_cache():
    """Clear GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class AMPContext:
    """Automatic Mixed Precision context manager."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()

    def __enter__(self):
        if self.enabled:
            self.ctx = torch.amp.autocast("cuda")
            return self.ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            return self.ctx.__exit__(*args)


def get_grad_scaler(enabled: bool = True) -> torch.amp.GradScaler:
    """Get gradient scaler for mixed precision training."""
    return torch.amp.GradScaler("cuda", enabled=enabled and torch.cuda.is_available())