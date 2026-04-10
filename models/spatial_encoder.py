"""
Spatial Artifact Encoder using EfficientNet-B0.
Extracts spatial manipulation artifacts from face images.
"""

import torch
import torch.nn as nn
import timm
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import logging

logger = logging.getLogger(__name__)


class SpatialEncoder(nn.Module):
    """EfficientNet-B0 backbone for spatial feature extraction."""

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 1280,
        drop_rate: float = 0.2,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.feature_dim = feature_dim

        # Load EfficientNet-B0 without classifier head
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # remove classifier
            drop_rate=drop_rate,
            global_pool="avg",
        )

        logger.info(f"SpatialEncoder: EfficientNet-B0 loaded (pretrained={pretrained})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor
        Returns:
            features: (B, 1280) spatial feature vector
        """
        if self.gradient_checkpointing and self.training:
            features = grad_checkpoint(self.backbone, x, use_reentrant=False)
        else:
            features = self.backbone(x)

        return features

    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """
        Get intermediate feature maps for GradCAM.
        Returns dict with 'features' and 'final_conv' keys.
        """
        # Access EfficientNet's blocks for intermediate features
        features = {}
        out = x

        # Forward through individual stages
        for name, module in self.backbone.named_children():
            if name == "global_pool":
                features["final_conv"] = out
                out = module(out)
            elif name == "classifier":
                continue
            else:
                out = module(out)

        features["features"] = out
        return features
