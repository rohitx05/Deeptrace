"""
Full Deepfake Detector — assembles all components.
Handles both image and video inputs.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import logging

from models.spatial_encoder import SpatialEncoder
from models.frequency_encoder import FrequencyEncoder
from models.physiology_encoder import PhysiologyEncoder
from models.temporal_model import VideoSwinTransformerTiny
from models.clip_alignment import CLIPAlignmentModule
from models.fusion import MultimodalFusion
from models.detection_head import DetectionHead

logger = logging.getLogger(__name__)


class DeepfakeDetector(nn.Module):
    """
    Multimodal Deepfake Detector.

    Architecture:
        Image mode:  Spatial + Frequency + CLIP → Fusion → Detection Head
        Video mode:  Spatial + Frequency + Temporal + Physiology + CLIP → Fusion → Detection Head
    """

    def __init__(self, config_path: str = None, config: dict = None):
        super().__init__()

        # Load config
        if config is None:
            if config_path is None:
                config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        self.config = config

        # --- Build components ---

        # Spatial Encoder
        sc = config.get("spatial_encoder", {})
        self.spatial_encoder = SpatialEncoder(
            pretrained=sc.get("pretrained", True),
            feature_dim=sc.get("feature_dim", 1280),
            drop_rate=sc.get("drop_rate", 0.2),
            gradient_checkpointing=sc.get("gradient_checkpointing", True),
        )

        # Frequency Encoder
        fc = config.get("frequency_encoder", {})
        self.frequency_encoder = FrequencyEncoder(
            pretrained=fc.get("pretrained", True),
            feature_dim=fc.get("feature_dim", 1280),
            dct_channels=fc.get("dct_channels", 3),
            drop_rate=fc.get("drop_rate", 0.2),
            gradient_checkpointing=fc.get("gradient_checkpointing", True),
        )

        # Temporal Model (Video Swin Transformer)
        tc = config.get("temporal_model", {})
        self.temporal_model = VideoSwinTransformerTiny(
            embed_dim=tc.get("embed_dim", 96),
            depths=tc.get("depths", [2, 2, 6, 2]),
            num_heads=tc.get("num_heads", [3, 6, 12, 24]),
            window_size=tuple(tc.get("window_size", [8, 5, 5])),
            patch_size=tuple(tc.get("patch_size", [2, 4, 4])),
            feature_dim=tc.get("feature_dim", 768),
            drop_rate=tc.get("drop_rate", 0.1),
            attn_drop_rate=tc.get("attn_drop_rate", 0.1),
            gradient_checkpointing=tc.get("gradient_checkpointing", True),
        )

        # Physiology Encoder
        pc = config.get("physiology_encoder", {})
        self.physiology_encoder = PhysiologyEncoder(
            feature_dim=pc.get("feature_dim", 64),
            hidden_dim=pc.get("hidden_dim", 128),
            num_layers=pc.get("num_layers", 2),
        )

        # CLIP Alignment
        cc = config.get("clip_alignment", {})
        self.clip_alignment = CLIPAlignmentModule(
            model_name=cc.get("model_name", "ViT-B-32"),
            pretrained_dataset=cc.get("pretrained_dataset", "openai"),
            feature_dim=cc.get("feature_dim", 512),
            projection_dim=cc.get("projection_dim", 256),
            spatial_feature_dim=sc.get("feature_dim", 1280),
            freeze_clip=cc.get("freeze_clip", True),
        )

        # Fusion
        fuse_cfg = config.get("fusion", {})
        self.fusion = MultimodalFusion(
            spatial_dim=sc.get("feature_dim", 1280),
            frequency_dim=fc.get("feature_dim", 1280),
            temporal_dim=tc.get("feature_dim", 768),
            physiology_dim=pc.get("feature_dim", 64),
            clip_projection_dim=cc.get("projection_dim", 256),
            hidden_dim=fuse_cfg.get("hidden_dim", 512),
            num_heads=fuse_cfg.get("num_heads", 8),
            num_layers=fuse_cfg.get("num_layers", 2),
            dropout=fuse_cfg.get("dropout", 0.1),
        )

        # Detection Head
        hc = config.get("detection_head", {})
        self.detection_head = DetectionHead(
            input_dim=fuse_cfg.get("hidden_dim", 512),
            hidden_dim=hc.get("hidden_dim", 256),
            num_manipulation_types=hc.get("num_manipulation_types", 5),
            dropout=hc.get("dropout", 0.3),
        )

        # Log param count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"DeepfakeDetector: {total_params / 1e6:.1f}M total params, "
            f"{trainable_params / 1e6:.1f}M trainable"
        )

    def forward(
        self,
        images: torch.Tensor = None,
        dct: torch.Tensor = None,
        frames: torch.Tensor = None,
        dct_frames: torch.Tensor = None,
        mode: str = "image",
    ) -> dict:
        """
        Forward pass supporting both image and video modes.

        Image mode:
            images: (B, 3, H, W)
            dct: (B, 3, H, W)

        Video mode:
            frames: (B, T, 3, H, W)
            dct_frames: (B, T, 3, H, W)

        Returns:
            dict with predictions + intermediate features
        """
        if mode == "video" and frames is not None:
            return self._forward_video(frames, dct_frames)
        else:
            return self._forward_image(images, dct)

    def _forward_image(self, images: torch.Tensor, dct: torch.Tensor = None) -> dict:
        """Process a single image."""
        # Spatial features
        spatial_features = self.spatial_encoder(images)  # (B, 1280)

        # Frequency features
        if dct is not None:
            freq_features = self.frequency_encoder(dct)  # (B, 1280)
        else:
            freq_features = torch.zeros_like(spatial_features)

        # CLIP alignment
        clip_result = self.clip_alignment(spatial_features, images)

        # Fusion (no temporal or physiology for images)
        fused = self.fusion(
            spatial_features=spatial_features,
            frequency_features=freq_features,
            temporal_features=None,
            physiology_features=None,
            clip_features=clip_result["spatial_projected"],
        )

        # Detection
        predictions = self.detection_head(fused)
        predictions["clip_alignment_loss"] = clip_result["alignment_loss"]
        predictions["spatial_features"] = spatial_features
        predictions["fused_features"] = fused

        return predictions

    def _forward_video(self, frames: torch.Tensor, dct_frames: torch.Tensor = None) -> dict:
        """Process a video (multiple frames)."""
        B, T, C, H, W = frames.shape

        # Process middle frame for spatial + frequency
        mid_idx = T // 2
        mid_frame = frames[:, mid_idx]  # (B, 3, H, W)

        spatial_features = self.spatial_encoder(mid_frame)

        if dct_frames is not None:
            mid_dct = dct_frames[:, mid_idx]
            freq_features = self.frequency_encoder(mid_dct)
        else:
            freq_features = torch.zeros_like(spatial_features)

        # Temporal features from all frames
        temporal_features = self.temporal_model(frames)  # (B, 768)

        # Physiology features
        physiology_features = self.physiology_encoder(frames)  # (B, 64)

        # CLIP alignment
        clip_result = self.clip_alignment(spatial_features, mid_frame)

        # Fusion with all modalities
        fused = self.fusion(
            spatial_features=spatial_features,
            frequency_features=freq_features,
            temporal_features=temporal_features,
            physiology_features=physiology_features,
            clip_features=clip_result["spatial_projected"],
        )

        # Detection
        predictions = self.detection_head(fused)
        predictions["clip_alignment_loss"] = clip_result["alignment_loss"]
        predictions["spatial_features"] = spatial_features
        predictions["temporal_features"] = temporal_features
        predictions["fused_features"] = fused

        return predictions

    def freeze_module(self, module_name: str):
        """Freeze a specific module by name."""
        module = getattr(self, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
            logger.info(f"Frozen: {module_name}")

    def unfreeze_module(self, module_name: str):
        """Unfreeze a specific module by name."""
        module = getattr(self, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
            logger.info(f"Unfrozen: {module_name}")
