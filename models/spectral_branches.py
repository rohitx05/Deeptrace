"""
Multi-Spectral Frequency Analysis Branches (2025-2026 SOTA).
Implements:
1. Continuous Phase & Spatial Phase Reconstruction (SPR) FFT Branch (4-channel).
2. 2-Level Wavelet Packet Sub-Band Decomposition Branch (7 sub-bands).
3. 9-Channel Forensic Filter Bank (5 SRM High-Pass + 4 Directional Gabor).
4. Learnable Spectral Gating Network (LSGN) with Dual-Pooling (GAP + GMP).
5. Cross-platform support (CUDA / MPS / CPU) with gradient checkpointing.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import logging

logger = logging.getLogger(__name__)


# ─── Dual-Pooling Lightweight Spectral CNN ───────────────────────────────────

class SOTASpectralCNNBlock(nn.Module):
    """Conv → GroupNorm → GELU → Conv → GroupNorm + Residual."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        num_groups = min(8, out_ch)
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class DualPoolingSpectralBackbone(nn.Module):
    """
    4-Stage Spectral Backbone with Dual-Pooling (GAP + GMP):
    GAP captures global lattice artifacts (StyleGAN, Diffusion).
    GMP captures localized 2-pixel boundary seams (FaceSwap, Face2Face Poisson seams).
    """

    def __init__(self, in_channels: int, feature_dim: int = 320):
        super().__init__()
        self.stage1 = SOTASpectralCNNBlock(in_channels, 32, stride=2)
        self.stage2 = SOTASpectralCNNBlock(32, 64, stride=2)
        self.stage3 = SOTASpectralCNNBlock(64, 128, stride=2)
        self.stage4 = SOTASpectralCNNBlock(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        # 256 (GAP) + 256 (GMP) = 512d -> projected to feature_dim (320d)
        self.proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        gap_feat = self.gap(x).flatten(1)
        gmp_feat = self.gmp(x).flatten(1)
        fused = torch.cat([gap_feat, gmp_feat], dim=-1)
        return self.proj(fused)


# ─── 1. SOTA Continuous Phase & SPR FFT Branch ──────────────────────────────

class SOTAFFTBranch(nn.Module):
    """
    Continuous Phase & Spatial Phase Reconstruction (SPR) Branch:
    Channels:
    0: Log Magnitude Spectrum: log(1 + |F(u, v)|)
    1: Continuous Phase Cosine: cos(θ(u, v))
    2: Continuous Phase Sine: sin(θ(u, v))
    3: Spatial Phase Reconstruction (SPR) Map: Re(ifft2(e^(jθ)))
    """

    def __init__(self, feature_dim: int = 320):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone = DualPoolingSpectralBackbone(in_channels=4, feature_dim=feature_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Convert RGB to Grayscale
        gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]  # (B, H, W)

        # 2D FFT
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)

        eps = 1e-8
        mag = torch.log1p(torch.abs(fft_shift))
        phase = torch.angle(fft_shift)

        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Spatial Phase Reconstruction (SPR): inverse FFT of unit-amplitude phase
        unit_phase_fft = torch.exp(1j * torch.angle(fft))
        spr_map = torch.fft.ifft2(unit_phase_fft).real.clamp(-3.0, 3.0)

        # Stack 4 channels (B, 4, H, W)
        fft_input = torch.stack([mag, cos_phase, sin_phase, spr_map], dim=1)
        return self.backbone(fft_input)


# ─── 2. SOTA 2-Level Wavelet Packet Branch ───────────────────────────────────

class SOTAWaveletBranch(nn.Module):
    """
    2-Level Haar Wavelet Packet Decomposition (Pure PyTorch, 7 Sub-Bands):
    Level 1: LH1, HL1, HH1 (H/2, W/2)
    Level 2: LL2, LH2, HL2, HH2 (H/4, W/4) upsampled to (H/2, W/2)
    Captures multi-scale high-frequency noise & blending boundaries.
    """

    def __init__(self, feature_dim: int = 320):
        super().__init__()
        self.feature_dim = feature_dim

        # Haar wavelet filter bank (fixed buffers)
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4.0
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4.0
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32).view(1, 1, 2, 2) / 4.0

        self.register_buffer("ll_f", ll)
        self.register_buffer("lh_f", lh)
        self.register_buffer("hl_f", hl)
        self.register_buffer("hh_f", hh)

        # 7-channel input to backbone
        self.backbone = DualPoolingSpectralBackbone(in_channels=7, feature_dim=feature_dim)

    def dwt_step(self, x: torch.Tensor):
        ll = F.conv2d(x, self.ll_f, stride=2)
        lh = F.conv2d(x, self.lh_f, stride=2)
        hl = F.conv2d(x, self.hl_f, stride=2)
        hh = F.conv2d(x, self.hh_f, stride=2)
        return ll, lh, hl, hh

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = (0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]).unsqueeze(1)

        # Level 1 DWT -> (B, 1, H/2, W/2) each
        ll1, lh1, hl1, hh1 = self.dwt_step(gray)

        # Level 2 DWT on LL1 -> (B, 1, H/4, W/4) each
        ll2, lh2, hl2, hh2 = self.dwt_step(ll1)

        # Upsample Level 2 bands to Level 1 resolution (H/2, W/2)
        h_half, w_half = lh1.shape[-2:]
        ll2_up = F.interpolate(ll2, size=(h_half, w_half), mode="bilinear", align_corners=False)
        lh2_up = F.interpolate(lh2, size=(h_half, w_half), mode="bilinear", align_corners=False)
        hl2_up = F.interpolate(hl2, size=(h_half, w_half), mode="bilinear", align_corners=False)
        hh2_up = F.interpolate(hh2, size=(h_half, w_half), mode="bilinear", align_corners=False)

        # Concatenate 7 sub-bands (B, 7, H/2, W/2)
        wavelet_stack = torch.cat([lh1, hl1, hh1, ll2_up, lh2_up, hl2_up, hh2_up], dim=1)
        return self.backbone(wavelet_stack)


# ─── 3. SOTA 9-Channel SRM & Gabor Forensic Bank ─────────────────────────────

class SOTANoiseResidualBranch(nn.Module):
    """
    9-Channel Steganalysis & Multi-Orientation Gabor Bank:
    - 5 SRM High-Pass Kernels (Laplacian, 2nd-order, 3rd-order, EDGE3x3, SQUARE5x5)
    - 4 Directional Gabor Filters (0°, 45°, 90°, 135°)
    Extracts Poisson boundary contours and blending discontinuity residuals.
    """

    def __init__(self, feature_dim: int = 320):
        super().__init__()
        self.feature_dim = feature_dim

        # 1. SRM 5-kernel bank (5x5)
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
            [0, 1, -2, 1, 0],
            [0, -2, 4, -2, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32) / 4.0

        srm4 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 0, 1, 0],
            [0, -2, 0, 2, 0],
            [0, -1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32) / 4.0

        srm5 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, -2, -1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 2, 1, 0],
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32) / 4.0

        # 2. Gabor 4-Orientation Kernels (5x5, theta = 0, 45, 90, 135 deg)
        gabor_filters = []
        for theta in [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]:
            g_kernel = self._create_gabor_kernel(kernel_size=5, sigma=1.5, theta=theta, lambd=3.0, gamma=0.5)
            gabor_filters.append(g_kernel)

        all_filters = torch.stack([srm1, srm2, srm3, srm4, srm5] + gabor_filters)  # (9, 5, 5)
        # Apply across grayscale image: (9, 1, 5, 5)
        self.register_buffer("filter_bank", all_filters.unsqueeze(1))

        self.backbone = DualPoolingSpectralBackbone(in_channels=9, feature_dim=feature_dim)

    @staticmethod
    def _create_gabor_kernel(kernel_size: int, sigma: float, theta: float, lambd: float, gamma: float) -> torch.Tensor:
        half_k = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half_k, half_k + 1, dtype=torch.float32),
            torch.arange(-half_k, half_k + 1, dtype=torch.float32),
            indexing="ij",
        )
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)
        gb = torch.exp(-0.5 * (x_theta**2 + (gamma * y_theta)**2) / (sigma**2)) * torch.cos(
            2 * math.pi * x_theta / lambd
        )
        gb = gb - gb.mean()
        norm = torch.sqrt(torch.sum(gb**2)) + 1e-8
        return gb / norm

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        gray = (0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]).unsqueeze(1)
        residuals = F.conv2d(gray, self.filter_bank, padding=2)  # (B, 9, H, W)
        return self.backbone(residuals)


# ─── 4. SOTA Learnable Spectral Gating Network (LSGN) Combiner ───────────────

class SOTASpectralCombiner(nn.Module):
    """
    Learnable Spectral Gating Network (LSGN):
    - Balanced 320-d projections for DCT, FFT, Wavelet, and SRM branches ($320 \times 4 = 1280\text{d}$).
    - Softmax modal gating network dynamically weighs each forensic modality per image.
    - Dual pooling captures both macro frequency distributions and localized 2-pixel boundary seams.
    - Returns combined 1280-d feature vector + individual branch embeddings for orthogonal loss.
    """

    def __init__(
        self,
        dct_in_dim: int = 1280,
        branch_dim: int = 320,
        output_dim: int = 1280,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.branch_dim = branch_dim
        self.output_dim = output_dim

        # Project DCT (1280) -> balanced 320d space
        self.dct_proj = nn.Sequential(
            nn.Linear(dct_in_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
        )

        self.fft_branch = SOTAFFTBranch(feature_dim=branch_dim)
        self.wavelet_branch = SOTAWaveletBranch(feature_dim=branch_dim)
        self.noise_branch = SOTANoiseResidualBranch(feature_dim=branch_dim)

        # Learnable Gating Router (4 modalities: DCT, FFT, Wavelet, SRM)
        self.gate_network = nn.Sequential(
            nn.Linear(branch_dim * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1),
        )

        # Master Multimodal Spectral Combiner
        self.combiner = nn.Sequential(
            nn.Linear(branch_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SOTASpectralCombiner initialized: {total_params / 1e6:.2f}M params, output_dim={output_dim}")

    def forward(self, image: torch.Tensor, dct_features: torch.Tensor, return_branches: bool = False):
        """
        Args:
            image: (B, 3, H, W) normalized image tensor
            dct_features: (B, 1280) from FrequencyEncoder
            return_branches: if True, returns dictionary with branch embeddings for Orthogonal Loss
        """
        dct_feat = self.dct_proj(dct_features)

        if self.gradient_checkpointing and self.training:
            fft_feat = grad_checkpoint(self.fft_branch, image, use_reentrant=False)
            wav_feat = grad_checkpoint(self.wavelet_branch, image, use_reentrant=False)
            srm_feat = grad_checkpoint(self.noise_branch, image, use_reentrant=False)
        else:
            fft_feat = self.fft_branch(image)
            wav_feat = self.wavelet_branch(image)
            srm_feat = self.noise_branch(image)

        # Stack representations: (B, 4, 320)
        branch_feats = torch.cat([dct_feat, fft_feat, wav_feat, srm_feat], dim=-1)  # (B, 1280)

        # Compute dynamic gating weights (B, 4)
        gate_weights = self.gate_network(branch_feats)  # (B, 4)
        w_dct, w_fft, w_wav, w_srm = gate_weights.unbind(dim=-1)

        # Modulated features
        gated_feats = torch.cat([
            dct_feat * w_dct.unsqueeze(-1),
            fft_feat * w_fft.unsqueeze(-1),
            wav_feat * w_wav.unsqueeze(-1),
            srm_feat * w_srm.unsqueeze(-1),
        ], dim=-1)

        combined = self.combiner(gated_feats)

        if return_branches:
            return {
                "combined": combined,
                "gate_weights": gate_weights,
                "branches": {
                    "dct": dct_feat,
                    "fft": fft_feat,
                    "wavelet": wav_feat,
                    "srm": srm_feat,
                },
            }
        return combined


# Backward compatibility aliases
FFTBranch = SOTAFFTBranch
WaveletBranch = SOTAWaveletBranch
NoiseResidualBranch = SOTANoiseResidualBranch
SpectralCombiner = SOTASpectralCombiner
