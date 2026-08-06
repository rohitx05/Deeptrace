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

        # Learnable projection: map spatial features -> CLIP space
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

    # ─── Phase 2: Partial CLIP Unfreeze ──────────────────────────────────────

    def partial_unfreeze(self, num_blocks: int = 2) -> int:
        """
        Phase 2 CLIP Partial Unfreeze.

        Unfreezes the last ``num_blocks`` transformer blocks of the CLIP ViT
        plus the ``ln_post`` layer-norm and the output ``proj`` weight.
        All earlier blocks remain frozen to conserve VRAM (~5.5 GB estimate).

        Args:
            num_blocks: Number of trailing transformer blocks to unfreeze.
                        Default=2 -> blocks 10 & 11 in ViT-B/32 (12 total).

        Returns:
            Number of parameters unfrozen (int).
        """
        if self.clip_visual is None:
            logger.warning("partial_unfreeze: clip_visual is None — skipping")
            return 0

        unfrozen = 0

        # ── Transformer blocks ────────────────────────────────────────────────
        # open_clip ViT stores blocks under .transformer.resblocks (ModuleList)
        resblocks = None
        backbone = getattr(self.clip_visual, "transformer", None)
        if backbone is not None:
            resblocks = getattr(backbone, "resblocks", None)

        if resblocks is not None:
            total = len(resblocks)
            unfreeze_from = max(0, total - num_blocks)
            logger.info(
                f"partial_unfreeze: unfreezing CLIP blocks "
                f"{unfreeze_from}-{total-1} / {total} total"
            )
            for block in resblocks[unfreeze_from:]:
                for param in block.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen += param.numel()
        else:
            logger.warning(
                "partial_unfreeze: could not find resblocks — "
                "architecture may differ from expected open_clip ViT"
            )

        # ── ln_post (post-norm before projection) ─────────────────────────────
        ln_post = getattr(self.clip_visual, "ln_post", None)
        if ln_post is not None:
            for param in ln_post.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen += param.numel()
            logger.info("partial_unfreeze: ln_post unfrozen")

        # ── proj (output projection weight, is a raw Tensor not a Module) ─────
        proj = getattr(self.clip_visual, "proj", None)
        if proj is not None and isinstance(proj, torch.Tensor):
            if not proj.requires_grad:
                proj.requires_grad_(True)
                unfrozen += proj.numel()
            logger.info("partial_unfreeze: proj unfrozen")

        # Switch to train mode for the unfrozen layers
        self.clip_visual.train()
        logger.info(f"partial_unfreeze: {unfrozen/1e6:.2f}M params unfrozen total")
        return unfrozen

    # ─── Feature Extraction ──────────────────────────────────────────────────

    def get_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract CLIP features. Uses no_grad unless CLIP blocks are unfrozen
        (partial_unfreeze was called), in which case gradients flow normally.

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

    def forward(
        self,
        spatial_features: torch.Tensor,
        images: torch.Tensor,
        compute_alignment_loss: bool = True,
    ) -> dict:
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

        if compute_alignment_loss:
            # Get and project CLIP features
            clip_feats = self.get_clip_features(images)

            if self.clip_visual is not None:
                clip_proj = self.clip_projection(clip_feats)
            else:
                clip_proj = clip_feats

            clip_proj = nn.functional.normalize(clip_proj, dim=-1)

            # Cosine alignment loss (encourage spatial features to align with CLIP)
            alignment_loss = 1.0 - (spatial_proj * clip_proj).sum(dim=-1).mean()
        else:
            # Detection uses spatial_projected as the CLIP token; GAN fine-tune does not
            # include alignment_loss, so the frozen CLIP visual pass can be skipped.
            clip_proj = torch.zeros_like(spatial_proj)
            alignment_loss = torch.tensor(0.0, device=spatial_features.device)

        return {
            "clip_projected": clip_proj,
            "spatial_projected": spatial_proj,
            "alignment_loss": alignment_loss,
        }
