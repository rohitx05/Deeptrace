"""
DeepfakeDetectorV2 — Extended detector assembling ALL modules.
Backward-compatible: original DeepfakeDetector is untouched.
"""

import torch
import torch.nn as nn
import yaml
from pathlib import Path
import logging

from models.spatial_encoder import SpatialEncoder
from models.frequency_encoder import FrequencyEncoder
from models.spectral_branches import SpectralCombiner
from models.physiology_encoder import PhysiologyEncoder
from models.temporal_model import VideoSwinTransformerTiny
from models.clip_alignment import CLIPAlignmentModule
from models.identity_encoder import IdentityEncoder
from models.rag_retrieval import RAGRetrieval
from models.fusion_transformer import MultimodalTransformerFusion
from models.detection_head_v2 import ExtendedDetectionHead

logger = logging.getLogger(__name__)


class DeepfakeDetectorV2(nn.Module):
    """
    Extended Multimodal Deepfake Detector V2.

    NEW modules over V1:
        - Multi-Spectral Analysis (FFT + Wavelet + Noise Residual + DCT)
        - Identity Consistency (ArcFace + temporal stability)
        - RAG Retrieval (FAISS artifact database)
        - Multimodal Transformer Fusion (8-token)
        - Extended Detection Head (+ generator attribution)
        - MC Dropout Uncertainty (wrapper, not a module)

    Architecture:
        Image mode:  Spatial + MultiSpectral + CLIP + RAG → TransformerFusion → ExtendedHead
        Video mode:  + Temporal + Physiology + Identity → TransformerFusion → ExtendedHead
    """

    def __init__(self, config_path: str = None, config: dict = None):
        super().__init__()

        if config is None:
            if config_path is None:
                config_path = Path(__file__).parent.parent / "configs" / "model_config_v2.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        self.config = config

        # ─── EXISTING modules (unchanged) ──────────────────────────────

        sc = config.get("spatial_encoder", {})
        self.spatial_encoder = SpatialEncoder(
            pretrained=sc.get("pretrained", True),
            feature_dim=sc.get("feature_dim", 1280),
            drop_rate=sc.get("drop_rate", 0.2),
            gradient_checkpointing=sc.get("gradient_checkpointing", True),
        )

        fc = config.get("frequency_encoder", {})
        self.frequency_encoder = FrequencyEncoder(
            pretrained=fc.get("pretrained", True),
            feature_dim=fc.get("feature_dim", 1280),
            dct_channels=fc.get("dct_channels", 3),
            drop_rate=fc.get("drop_rate", 0.2),
            gradient_checkpointing=fc.get("gradient_checkpointing", True),
        )

        tc = config.get("temporal_model", {})
        self.temporal_model = VideoSwinTransformerTiny(
            embed_dim=tc.get("embed_dim", 96),
            depths=tc.get("depths", [2, 2, 6, 2]),
            num_heads=tc.get("num_heads", [3, 6, 12, 24]),
            window_size=tuple(tc.get("window_size", [8, 5, 5])),
            patch_size=tuple(tc.get("patch_size", [2, 4, 4])),
            feature_dim=tc.get("feature_dim", 768),
            gradient_checkpointing=tc.get("gradient_checkpointing", True),
        )

        pc = config.get("physiology_encoder", {})
        self.physiology_encoder = PhysiologyEncoder(
            feature_dim=pc.get("feature_dim", 64),
            hidden_dim=pc.get("hidden_dim", 128),
            num_layers=pc.get("num_layers", 2),
        )

        cc = config.get("clip_alignment", {})
        self.clip_alignment = CLIPAlignmentModule(
            model_name=cc.get("model_name", "ViT-B-32"),
            pretrained_dataset=cc.get("pretrained_dataset", "openai"),
            feature_dim=cc.get("feature_dim", 512),
            projection_dim=cc.get("projection_dim", 256),
            spatial_feature_dim=sc.get("feature_dim", 1280),
            freeze_clip=cc.get("freeze_clip", True),
        )

        # ─── NEW modules ──────────────────────────────────────────────

        # Multi-Spectral Analysis
        spec_cfg = config.get("spectral_combiner", {})
        self.spectral_combiner = SpectralCombiner(
            dct_dim=fc.get("feature_dim", 1280),
            fft_dim=spec_cfg.get("fft_dim", 256),
            wavelet_dim=spec_cfg.get("wavelet_dim", 256),
            noise_dim=spec_cfg.get("noise_dim", 256),
            output_dim=spec_cfg.get("output_dim", 1280),
            gradient_checkpointing=spec_cfg.get("gradient_checkpointing", True),
        )

        # Identity Consistency
        id_cfg = config.get("identity_encoder", {})
        self.identity_encoder = IdentityEncoder(
            feature_dim=id_cfg.get("feature_dim", 128),
            embedding_dim=id_cfg.get("embedding_dim", 512),
            freeze_backbone=id_cfg.get("freeze_backbone", True),
        )

        # RAG Retrieval
        rag_cfg = config.get("rag_retrieval", {})
        self.rag_retrieval = RAGRetrieval(
            input_dim=sc.get("feature_dim", 1280),
            query_dim=rag_cfg.get("query_dim", 512),
            output_dim=rag_cfg.get("output_dim", 256),
            top_k=rag_cfg.get("top_k", 8),
        )

        # Multimodal Transformer Fusion
        fuse_cfg = config.get("fusion_transformer", {})
        self.fusion = MultimodalTransformerFusion(
            hidden_dim=fuse_cfg.get("hidden_dim", 512),
            spatial_dim=sc.get("feature_dim", 1280),
            spectral_dim=spec_cfg.get("output_dim", 1280),
            temporal_dim=tc.get("feature_dim", 768),
            physiology_dim=pc.get("feature_dim", 64),
            clip_dim=cc.get("projection_dim", 256),
            identity_dim=id_cfg.get("feature_dim", 128),
            rag_dim=rag_cfg.get("output_dim", 256),
            num_heads=fuse_cfg.get("num_heads", 8),
            num_layers=fuse_cfg.get("num_layers", 4),
            dropout=fuse_cfg.get("dropout", 0.1),
        )

        # Extended Detection Head
        hc = config.get("detection_head", {})
        self.detection_head = ExtendedDetectionHead(
            input_dim=fuse_cfg.get("hidden_dim", 512),
            spectral_dim=spec_cfg.get("output_dim", 1280),
            hidden_dim=hc.get("hidden_dim", 256),
            num_manipulation_types=hc.get("num_manipulation_types", 5),
            num_generator_types=hc.get("num_generator_types", 4),
            dropout=hc.get("dropout", 0.3),
        )

        # ─── Log stats ────────────────────────────────────────────────

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        logger.info(
            f"DeepfakeDetectorV2: {total_params/1e6:.1f}M total, "
            f"{trainable_params/1e6:.1f}M trainable, {frozen_params/1e6:.1f}M frozen"
        )

    def forward(
        self,
        images: torch.Tensor = None,
        dct: torch.Tensor = None,
        frames: torch.Tensor = None,
        dct_frames: torch.Tensor = None,
        mode: str = "image",
    ) -> dict:
        if mode == "video" and frames is not None:
            return self._forward_video(frames, dct_frames)
        else:
            return self._forward_image(images, dct)

    def _forward_image(self, images: torch.Tensor, dct: torch.Tensor = None) -> dict:
        """Image mode: Spatial + MultiSpectral + CLIP + RAG → Fusion → Head."""
        # Spatial
        spatial_feat = self.spatial_encoder(images)  # (B, 1280)

        # Frequency (DCT)
        if dct is not None:
            dct_feat = self.frequency_encoder(dct)
        else:
            dct_feat = torch.zeros_like(spatial_feat)

        # Multi-Spectral (combines DCT + FFT + Wavelet + Noise Residual)
        spectral_feat = self.spectral_combiner(images, dct_feat)  # (B, 1280)

        # CLIP alignment
        clip_result = self.clip_alignment(spatial_feat, images)

        # RAG retrieval
        rag_context = self.rag_retrieval(spatial_feat)  # (B, 256)

        # Multimodal Transformer Fusion
        fused = self.fusion(
            spatial_features=spatial_feat,
            spectral_features=spectral_feat,
            temporal_features=None,
            physiology_features=None,
            clip_features=clip_result["spatial_projected"],
            identity_features=None,
            rag_features=rag_context,
        )

        # Extended Detection Head
        predictions = self.detection_head(fused, spectral_features=spectral_feat)
        predictions["clip_alignment_loss"] = clip_result["alignment_loss"]
        predictions["spatial_features"] = spatial_feat
        predictions["spectral_features"] = spectral_feat
        predictions["fused_features"] = fused

        return predictions

    def _forward_video(self, frames: torch.Tensor, dct_frames: torch.Tensor = None) -> dict:
        """Video mode: ALL modalities → Fusion → Head."""
        B, T, C, H, W = frames.shape
        mid_idx = T // 2
        mid_frame = frames[:, mid_idx]

        # Spatial (middle frame)
        spatial_feat = self.spatial_encoder(mid_frame)

        # DCT + Multi-Spectral (middle frame)
        if dct_frames is not None:
            dct_feat = self.frequency_encoder(dct_frames[:, mid_idx])
        else:
            dct_feat = torch.zeros_like(spatial_feat)
        spectral_feat = self.spectral_combiner(mid_frame, dct_feat)

        # Temporal
        temporal_feat = self.temporal_model(frames)

        # Physiology
        physiology_feat = self.physiology_encoder(frames)

        # Identity Consistency
        identity_feat = self.identity_encoder(frames)

        # CLIP
        clip_result = self.clip_alignment(spatial_feat, mid_frame)

        # RAG
        rag_context = self.rag_retrieval(spatial_feat)

        # Multimodal Transformer Fusion (all 7 modalities + CLS)
        fused = self.fusion(
            spatial_features=spatial_feat,
            spectral_features=spectral_feat,
            temporal_features=temporal_feat,
            physiology_features=physiology_feat,
            clip_features=clip_result["spatial_projected"],
            identity_features=identity_feat,
            rag_features=rag_context,
        )

        # Extended Detection Head
        predictions = self.detection_head(fused, spectral_features=spectral_feat)
        predictions["clip_alignment_loss"] = clip_result["alignment_loss"]
        predictions["spatial_features"] = spatial_feat
        predictions["spectral_features"] = spectral_feat
        predictions["temporal_features"] = temporal_feat
        predictions["fused_features"] = fused

        return predictions

    def freeze_module(self, module_name: str):
        module = getattr(self, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False
            logger.info(f"Frozen: {module_name}")

    def unfreeze_module(self, module_name: str):
        module = getattr(self, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True
            logger.info(f"Unfrozen: {module_name}")

    def load_v1_checkpoint(self, v1_path: str):
        """Load weights from a V1 DeepfakeDetector checkpoint (compatible keys transfer)."""
        state = torch.load(v1_path, map_location="cpu")
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        # Filter to matching keys
        own_state = self.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k] = v
                loaded += 1
        self.load_state_dict(own_state, strict=False)
        logger.info(f"Loaded {loaded} weights from V1 checkpoint (of {len(state)} available)")
