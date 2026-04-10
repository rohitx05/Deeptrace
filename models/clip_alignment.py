"""
CLIP Embedding Alignment module.
Frozen CLIP ViT-B/32 + learnable projection to improve
generalization to unseen deepfake generators.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CLIPAlignmentModule(nn.Module):
    """
    Frozen CLIP image encoder with a learnable alignment head.
    Projects spatial features into CLIP embedding space so the detector
    can leverage CLIP's visual-semantic knowledge for better generalization.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained_dataset: str = "openai",
        feature_dim: int = 512,
        projection_dim: int = 256,
        spatial_feature_dim: int = 1280,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim

        # Load CLIP model
        try:
            import open_clip

            clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_dataset
            )
            self.clip_visual = clip_model.visual
            logger.info(f"CLIPAlignment: loaded {model_name} (pretrained={pretrained_dataset})")
        except ImportError:
            logger.warning("open-clip-torch not installed, using random projection")
            self.clip_visual = None
            feature_dim = projection_dim  # fallback

        # Freeze CLIP
        if freeze_clip and self.clip_visual is not None:
            for param in self.clip_visual.parameters():
                param.requires_grad = False
            self.clip_visual.eval()
            logger.info("CLIP encoder frozen (no gradients)")

        # Learnable projection: map spatial features → CLIP space
        self.spatial_projection = nn.Sequential(
            nn.Linear(spatial_feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # CLIP feature projection (match dimensions)
        if self.clip_visual is not None:
            self.clip_projection = nn.Sequential(
                nn.Linear(feature_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
            )

    @torch.no_grad()
    def get_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP features (no gradient).

        Args:
            images: (B, 3, H, W) normalized images

        Returns:
            clip_features: (B, feature_dim)
        """
        if self.clip_visual is None:
            return torch.zeros(images.size(0), self.projection_dim, device=images.device)

        # CLIP expects 224x224 — resize if needed
        if images.shape[-1] != 224:
            images = nn.functional.interpolate(images, size=224, mode="bilinear", align_corners=False)

        features = self.clip_visual(images)
        return features.float()

    def forward(self, spatial_features: torch.Tensor, images: torch.Tensor) -> dict:
        """
        Args:
            spatial_features: (B, spatial_feature_dim) from SpatialEncoder
            images: (B, 3, H, W) input images for CLIP

        Returns:
            dict with:
                - clip_projected: (B, projection_dim) projected CLIP features
                - spatial_projected: (B, projection_dim) projected spatial features
                - alignment_loss: scalar alignment loss
        """
        # Project spatial features
        spatial_proj = self.spatial_projection(spatial_features)
        spatial_proj = nn.functional.normalize(spatial_proj, dim=-1)

        # Get and project CLIP features
        clip_feats = self.get_clip_features(images)

        if self.clip_visual is not None:
            clip_proj = self.clip_projection(clip_feats)
        else:
            clip_proj = clip_feats

        clip_proj = nn.functional.normalize(clip_proj, dim=-1)

        # Cosine alignment loss (encourage spatial features to align with CLIP)
        alignment_loss = 1.0 - (spatial_proj * clip_proj).sum(dim=-1).mean()

        return {
            "clip_projected": clip_proj,
            "spatial_projected": spatial_proj,
            "alignment_loss": alignment_loss,
        }
