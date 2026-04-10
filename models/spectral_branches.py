"""
Multi-Spectral Frequency Analysis Branches.
Extends the existing DCT-only FrequencyEncoder with FFT, Wavelet, and Noise Residual branches.
All branches run in parallel and are combined via SpectralCombiner.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import logging

logger = logging.getLogger(__name__)


# ─── Lightweight CNN Block ───────────────────────────────────────────────────

class SpectralCNNBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU + residual."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class LightweightSpectralCNN(nn.Module):
    """4-block CNN: in_ch → 32 → 64 → 128 → 256 → GAP → feature_dim."""

    def __init__(self, in_channels: int, feature_dim: int = 256):
        super().__init__()
        self.blocks = nn.Sequential(
            SpectralCNNBlock(in_channels, 32, stride=2),
            SpectralCNNBlock(32, 64, stride=2),
            SpectralCNNBlock(64, 128, stride=2),
            SpectralCNNBlock(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# ─── FFT Branch ──────────────────────────────────────────────────────────────

class FFTBranch(nn.Module):
    """2D FFT magnitude + phase → lightweight CNN → 256d."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        # 2 channels: magnitude + phase
        self.cnn = LightweightSpectralCNN(in_channels=2, feature_dim=feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) RGB image tensor (normalized)
        Returns:
            (B, feature_dim) FFT features
        """
        # Convert to grayscale
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]  # (B, H, W)

        # 2D FFT
        fft = torch.fft.fft2(gray)
        fft_shifted = torch.fft.fftshift(fft)

        magnitude = torch.log1p(torch.abs(fft_shifted))
        phase = torch.angle(fft_shifted)

        # Stack as 2-channel input
        fft_input = torch.stack([magnitude, phase], dim=1)  # (B, 2, H, W)
        return self.cnn(fft_input)


# ─── Wavelet Branch ──────────────────────────────────────────────────────────

class WaveletBranch(nn.Module):
    """
    Haar Wavelet (DWT) decomposition → 4 sub-bands (LL, LH, HL, HH) → CNN → 256d.
    Pure PyTorch implementation (no pywt dependency).
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        # 4 channels: LL, LH, HL, HH
        self.cnn = LightweightSpectralCNN(in_channels=4, feature_dim=feature_dim)

        # Haar wavelet filters (fixed, non-trainable)
        self.register_buffer("ll_filter", torch.tensor([[1, 1], [1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4.0)
        self.register_buffer("lh_filter", torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4.0)
        self.register_buffer("hl_filter", torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4.0)
        self.register_buffer("hh_filter", torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4.0)

    def haar_dwt(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Haar DWT, returns (B, 4, H/2, W/2)."""
        # x: (B, 1, H, W)
        ll = F.conv2d(x, self.ll_filter, stride=2)
        lh = F.conv2d(x, self.lh_filter, stride=2)
        hl = F.conv2d(x, self.hl_filter, stride=2)
        hh = F.conv2d(x, self.hh_filter, stride=2)
        return torch.cat([ll, lh, hl, hh], dim=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) normalized image
        Returns:
            (B, feature_dim) wavelet features
        """
        gray = (0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]).unsqueeze(1)
        wavelet = self.haar_dwt(gray)  # (B, 4, H/2, W/2)
        return self.cnn(wavelet)


# ─── Noise Residual Branch ───────────────────────────────────────────────────

class NoiseResidualBranch(nn.Module):
    """
    SRM (Steganalysis Rich Model) high-pass filters → CNN → 256d.
    Extracts noise residuals that reveal manipulation artifacts.
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # SRM filters (3 standard high-pass kernels)
        srm1 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32) / 4.0

        srm2 = torch.tensor([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ], dtype=torch.float32) / 12.0

        srm3 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32)

        # Each filter applied per RGB channel → 3×3=9 output channels
        filters = torch.stack([srm1, srm2, srm3])  # (3, 5, 5)
        # Expand for 3 input channels
        self.register_buffer(
            "srm_filters",
            filters.unsqueeze(1).repeat(1, 3, 1, 1)  # (3, 3, 5, 5) — groups=1
        )

        self.cnn = LightweightSpectralCNN(in_channels=3, feature_dim=feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) normalized image
        Returns:
            (B, feature_dim) noise residual features
        """
        # Apply SRM filters
        residuals = F.conv2d(image, self.srm_filters, padding=2)  # (B, 3, H, W)
        return self.cnn(residuals)


# ─── Spectral Combiner ──────────────────────────────────────────────────────

class SpectralCombiner(nn.Module):
    """
    Combines DCT (from existing FrequencyEncoder) + FFT + Wavelet + Noise Residual
    into a single 1280d spectral feature vector.
    """

    def __init__(
        self,
        dct_dim: int = 1280,
        fft_dim: int = 256,
        wavelet_dim: int = 256,
        noise_dim: int = 256,
        output_dim: int = 1280,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        self.fft_branch = FFTBranch(feature_dim=fft_dim)
        self.wavelet_branch = WaveletBranch(feature_dim=wavelet_dim)
        self.noise_branch = NoiseResidualBranch(feature_dim=noise_dim)

        total_dim = dct_dim + fft_dim + wavelet_dim + noise_dim  # 2048
        self.combiner = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

        self.output_dim = output_dim

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SpectralCombiner: {total_params / 1e6:.1f}M params, output_dim={output_dim}")

    def forward(self, image: torch.Tensor, dct_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) for FFT/Wavelet/Noise
            dct_features: (B, 1280) from existing FrequencyEncoder

        Returns:
            (B, output_dim) combined spectral features
        """
        if self.gradient_checkpointing and self.training:
            fft_feat = grad_checkpoint(self.fft_branch, image, use_reentrant=False)
            wav_feat = grad_checkpoint(self.wavelet_branch, image, use_reentrant=False)
            noise_feat = grad_checkpoint(self.noise_branch, image, use_reentrant=False)
        else:
            fft_feat = self.fft_branch(image)
            wav_feat = self.wavelet_branch(image)
            noise_feat = self.noise_branch(image)

        combined = torch.cat([dct_features, fft_feat, wav_feat, noise_feat], dim=-1)
        return self.combiner(combined)
