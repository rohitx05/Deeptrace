"""
Video Swin Transformer Tiny for temporal consistency modeling.
Lighter than TimeSformer with shifted-window attention.
Implemented from scratch for 6GB VRAM compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import rearrange
import logging
import math

logger = logging.getLogger(__name__)


class WindowAttention3D(nn.Module):
    """3D shifted window multi-head self-attention."""

    def __init__(self, dim: int, window_size: tuple, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (T_w, H_w, W_w)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, window_volume, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach()  # store for visualization
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """A single Swin Transformer block with 3D shifted windows."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (2, 7, 7),
        shift_size: tuple = (0, 0, 0),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, T*H*W, C)
            T, H, W: temporal and spatial dimensions
        """
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.view(B, T, H, W, C)

        # Pad if needed
        pad_t = (self.window_size[0] - T % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))

        Tp, Hp, Wp = x.shape[1], x.shape[2], x.shape[3]

        # Shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))

        # Partition windows
        x = self._window_partition(x)  # (num_windows*B, window_volume, C)

        # Attention
        x = self.attn(x)

        # Reverse window partition
        x = self._window_reverse(x, Tp, Hp, Wp)

        # Reverse shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))

        # Remove padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :T, :H, :W, :]

        x = x.reshape(B, L, C)

        # Residual + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition into non-overlapping 3D windows."""
        B, T, H, W, C = x.shape
        wt, wh, ww = self.window_size
        x = x.view(B, T // wt, wt, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wt * wh * ww, C)
        return x

    def _window_reverse(self, windows: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """Reverse window partition."""
        wt, wh, ww = self.window_size
        B = int(windows.shape[0] / (T // wt * H // wh * W // ww))
        x = windows.view(B, T // wt, H // wh, W // ww, wt, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
        return x


class PatchEmbed3D(nn.Module):
    """Video to 3D patch embedding."""

    def __init__(self, patch_size: tuple = (2, 4, 4), in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            x: (B, T'*H'*W', embed_dim), T', H', W'
        """
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', C)
        x = self.norm(x)
        return x, T, H, W


class VideoSwinTransformerTiny(nn.Module):
    """
    Video Swin Transformer Tiny — lightweight temporal consistency model.
    Optimized for 6GB VRAM with 8 frames at 160x160.
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: tuple = (8, 5, 5),
        patch_size: tuple = (2, 4, 4),
        feature_dim: int = 768,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.gradient_checkpointing = gradient_checkpointing
        self.feature_dim = feature_dim
        self.num_layers = len(depths)

        # Patch embedding
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim)

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            dim = embed_dim * (2 ** i)
            layer_blocks = nn.ModuleList()

            for j in range(depths[i]):
                shift = (0, 0, 0) if j % 2 == 0 else (
                    window_size[0] // 2,
                    window_size[1] // 2,
                    window_size[2] // 2,
                )
                layer_blocks.append(
                    SwinTransformerBlock3D(
                        dim=dim,
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=shift,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                    )
                )
            self.layers.append(layer_blocks)

            # Downsample (except last layer)
            if i < self.num_layers - 1:
                self.layers.append(nn.ModuleList([PatchMerging3D(dim)]))

        # Final norm + projection
        final_dim = embed_dim * (2 ** (self.num_layers - 1))
        self.norm = nn.LayerNorm(final_dim)
        self.head = nn.Linear(final_dim, feature_dim) if final_dim != feature_dim else nn.Identity()

        self.apply(self._init_weights)
        logger.info(f"VideoSwinTransformerTiny: embed_dim={embed_dim}, depths={depths}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 3, H, W) video frames

        Returns:
            features: (B, feature_dim)
        """
        B = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        x, T, H, W = self.patch_embed(x)

        for layer_blocks in self.layers:
            for block in layer_blocks:
                if isinstance(block, SwinTransformerBlock3D):
                    if self.gradient_checkpointing and self.training:
                        x = grad_checkpoint(block, x, T, H, W, use_reentrant=False)
                    else:
                        x = block(x, T, H, W)
                elif isinstance(block, PatchMerging3D):
                    x, T, H, W = block(x, T, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)

        return x


class PatchMerging3D(nn.Module):
    """3D patch merging (downsampling) layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> tuple:
        B, L, C = x.shape
        x = x.view(B, T, H, W, C)

        # Only merge spatial dims (keep temporal)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H += pad_h
            W += pad_w

        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, T, H/2, W/2, 4*C)

        new_H, new_W = H // 2, W // 2
        x = x.view(B, T * new_H * new_W, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x, T, new_H, new_W
