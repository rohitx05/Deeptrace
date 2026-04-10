"""
Identity Consistency Encoder.
Frozen ArcFace ResNet-18 extracts per-frame identity embeddings.
Temporal stability scoring detects identity flickering in deepfakes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class IdentityEncoder(nn.Module):
    """
    Frozen face recognition model + temporal identity stability scoring.

    Architecture:
        1. Pretrained ArcFace/FaceNet (frozen) → 512d per frame
        2. Pairwise cosine similarity across frames → stability matrix
        3. MLP → 128d identity consistency feature
    """

    def __init__(
        self,
        feature_dim: int = 128,
        embedding_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        # Load pretrained face recognition model
        try:
            from facenet_pytorch import InceptionResnetV1
            self.backbone = InceptionResnetV1(pretrained="vggface2")
            self.backbone_type = "facenet"
            self.raw_embedding_dim = 512
            logger.info("IdentityEncoder: loaded FaceNet (VGGFace2)")
        except ImportError:
            logger.warning("facenet-pytorch not available, using lightweight fallback")
            self.backbone = self._build_fallback_backbone()
            self.backbone_type = "fallback"
            self.raw_embedding_dim = 512

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Projection to common embedding space
        self.embedding_proj = nn.Linear(self.raw_embedding_dim, embedding_dim)

        # Stability scoring MLP
        # Input: mean similarity + std similarity + min similarity + embedding mean
        self.stability_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
        )

    def _build_fallback_backbone(self):
        """Lightweight fallback if no pretrained face model available."""
        import timm
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg")
        return nn.Sequential(model, nn.Linear(1280, 512))

    @torch.no_grad()
    def extract_identity(self, face: torch.Tensor) -> torch.Tensor:
        """
        Extract identity embedding from a single face image.

        Args:
            face: (B, 3, H, W)
        Returns:
            embedding: (B, embedding_dim) L2-normalized
        """
        if self.backbone_type == "facenet":
            # FaceNet expects 160x160
            if face.shape[-1] != 160:
                face = F.interpolate(face, size=160, mode="bilinear", align_corners=False)
            raw = self.backbone(face)
        else:
            raw = self.backbone(face)

        embedding = self.embedding_proj(raw)
        return F.normalize(embedding, dim=-1)

    def compute_temporal_stability(self, embeddings: torch.Tensor) -> dict:
        """
        Compute identity stability across temporal embeddings.

        Args:
            embeddings: (B, T, embedding_dim) per-frame embeddings

        Returns:
            dict with stability metrics
        """
        B, T, D = embeddings.shape

        # Pairwise cosine similarity matrix
        # (B, T, D) @ (B, D, T) → (B, T, T)
        sim_matrix = torch.bmm(embeddings, embeddings.transpose(1, 2))

        # Extract upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(T, T, device=embeddings.device), diagonal=1).bool()
        pairwise_sims = sim_matrix[:, mask]  # (B, T*(T-1)/2)

        mean_sim = pairwise_sims.mean(dim=-1)  # (B,) — high = consistent identity
        std_sim = pairwise_sims.std(dim=-1)    # (B,) — high = flickering identity
        min_sim = pairwise_sims.min(dim=-1)[0] # (B,) — low = identity swap detected

        return {
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "min_similarity": min_sim,
            "embeddings_mean": embeddings.mean(dim=1),  # (B, D)
        }

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, 3, H, W) video frames

        Returns:
            identity_features: (B, feature_dim) identity consistency features
        """
        B, T, C, H, W = frames.shape

        # Extract per-frame identity embeddings
        flat_frames = frames.reshape(B * T, C, H, W)
        flat_embeddings = self.extract_identity(flat_frames)  # (B*T, embedding_dim)
        embeddings = flat_embeddings.reshape(B, T, -1)  # (B, T, embedding_dim)

        # Compute temporal stability
        stability = self.compute_temporal_stability(embeddings)

        # Build stability feature vector
        stability_input = torch.cat([
            stability["embeddings_mean"],                    # (B, embedding_dim)
            stability["mean_similarity"].unsqueeze(-1),      # (B, 1)
            stability["std_similarity"].unsqueeze(-1),       # (B, 1)
            stability["min_similarity"].unsqueeze(-1),       # (B, 1)
        ], dim=-1)  # (B, embedding_dim + 3)

        identity_features = self.stability_mlp(stability_input)  # (B, feature_dim)
        return identity_features

    def forward_single(self, image: torch.Tensor) -> torch.Tensor:
        """For image mode — returns zero features (no temporal stability)."""
        B = image.size(0)
        return torch.zeros(B, self.feature_dim, device=image.device)
