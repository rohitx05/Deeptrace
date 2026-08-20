"""
Cross-Platform Device Utilities:
Supports NVIDIA CUDA (Windows/Linux), Apple Silicon MPS (macOS M1/M2/M3/M4), and CPU fallback.
Includes automatic mixed precision (AMP) context and memory profiling.
"""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Auto-detects the best available hardware accelerator:
    1. NVIDIA CUDA (Windows / Linux)
    2. Apple Silicon MPS (Metal Performance Shaders on macOS M1/M2/M3/M4)
    3. CPU fallback
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU (Metal Performance Shaders - MPS / M4)")
            return device

    device = torch.device("cpu")
    logger.info("Using CPU")
    return device


def get_memory_info() -> dict:
    """Get current GPU / accelerator memory usage."""
    if torch.cuda.is_available():
        return {
            "device_type": "cuda",
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "reserved": torch.cuda.memory_reserved() / 1e9,
            "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS uses unified memory dynamically managed by macOS
        return {
            "device_type": "mps",
            "allocated": torch.mps.current_allocated_memory() / 1e9 if hasattr(torch.mps, "current_allocated_memory") else 0,
            "reserved": torch.mps.driver_allocated_memory() / 1e9 if hasattr(torch.mps, "driver_allocated_memory") else 0,
            "total": 16.0,  # Apple Silicon Unified Memory
        }
    return {"device_type": "cpu", "allocated": 0, "reserved": 0, "total": 0}


def print_memory_usage(prefix: str = ""):
    """Print current accelerator memory usage."""
    info = get_memory_info()
    if info["device_type"] == "cuda":
        logger.info(
            f"{prefix} CUDA VRAM: {info['allocated']:.2f}GB allocated, "
            f"{info['reserved']:.2f}GB reserved, {info['total']:.1f}GB total"
        )
    elif info["device_type"] == "mps":
        logger.info(f"{prefix} Apple Silicon MPS Unified Memory: {info['allocated']:.2f}GB allocated")


def empty_cache():
    """Clear accelerator memory cache across CUDA and MPS."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


class AMPContext:
    """
    Unified Automatic Mixed Precision (AMP) Context Manager.
    Automatically handles 'cuda', 'mps' (BFloat16 / FP16 on PyTorch 2.3+), and CPU.
    """

    def __init__(self, device: torch.device = None, enabled: bool = True):
        self.device = device if device is not None else get_device()
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return self

        if self.device.type == "cuda":
            self.ctx = torch.amp.autocast("cuda")
            return self.ctx.__enter__()
        elif self.device.type == "mps":
            # PyTorch 2.3+ supports MPS autocast
            if hasattr(torch.amp, "autocast"):
                try:
                    self.ctx = torch.amp.autocast("mps")
                    return self.ctx.__enter__()
                except Exception:
                    pass
        return self

    def __exit__(self, *args):
        if hasattr(self, "ctx") and self.ctx is not None:
            return self.ctx.__exit__(*args)


UnifiedAMPContext = AMPContext


def get_grad_scaler(device: torch.device = None, enabled: bool = True) -> torch.amp.GradScaler:
    """
    Get gradient scaler for mixed precision training.
    CUDA utilizes standard GradScaler; MPS utilizes standard or unscaled backprop.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    else:
        # MPS / CPU do not require CUDA-specific float16 scaling (often uses BF16/FP32 natively)
        return torch.amp.GradScaler("cuda", enabled=False)