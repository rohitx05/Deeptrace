"""
Frequency Artifact Encoder using EfficientNet-B0 on DCT input.
Detects compression and generation artifacts in the frequency domain.
"""

import torch
import torch.nn as nn
import timm
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import logging

logger = logging.getLogger(__name__)


class FrequencyEncoder(nn.Module):
    """EfficientNet-B0 backbone for DCT frequency feature extraction."""

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 1280,
        dct_channels: int = 3,
        drop_rate: float = 0.2,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.feature_dim = feature_dim

        # Load EfficientNet-B0
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            global_pool="avg",
            in_chans=dct_channels,  # DCT channels (Y, Cb, Cr)
        )

        # If pretrained and DCT channels differ from RGB, adapt first conv
        if pretrained and dct_channels != 3:
            old_conv = self.backbone.conv_stem
            self.backbone.conv_stem = nn.Conv2d(
                dct_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Initialize with averaged pretrained weights
            with torch.no_grad():
                weight = old_conv.weight.mean(dim=1, keepdim=True)
                self.backbone.conv_stem.weight.copy_(weight.repeat(1, dct_channels, 1, 1))

        logger.info(f"FrequencyEncoder: EfficientNet-B0 on DCT input ({dct_channels} channels)")

    def forward(self, dct_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dct_input: (B, C_dct, H, W) DCT coefficient tensor
        Returns:
            features: (B, 1280) frequency feature vector
        """
        if self.gradient_checkpointing and self.training:
            features = grad_checkpoint(self.backbone, dct_input, use_reentrant=False)
        else:
            features = self.backbone(dct_input)

        return features
