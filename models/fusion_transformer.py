"""
Multimodal Transformer Fusion.
Replaces simple concatenation with an 8-token transformer encoder.
Processes: [CLS] + spatial + spectral + temporal + physiology + clip + identity + rag_context.
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class ModalityGate(nn.Module):
    """Learnable sigmoid gate per modality — handles missing modalities gracefully."""

    def __init__(self, dim: int, num_modalities: int):
        super().__init__()
        self.gates = nn.Parameter(torch.ones(num_modalities))  # init open

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Multiply each token by its gate value. tokens: (B, num_modalities, dim)."""
        gates = torch.sigmoid(self.gates).unsqueeze(0).unsqueeze(-1)  # (1, M, 1)
        return tokens * gates


class TransformerFusionLayer(nn.Module):
    """Single transformer encoder layer with pre-norm."""

    def __init__(self, dim: int, num_heads: int = 8, ffn_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ffn_ratio), dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, self.attn_weights = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class MultimodalTransformerFusion(nn.Module):
    """
    8-token multimodal transformer encoder.

    Tokens:
        0: CLS (learnable)
        1: Spatial features
        2: Spectral features (multi-spectral combiner output)
        3: Temporal features
        4: Physiology features
        5: CLIP features
        6: Identity features
        7: RAG context features

    Missing modalities are replaced with learned default tokens and gated down.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        spatial_dim: int = 1280,
        spectral_dim: int = 1280,
        temporal_dim: int = 768,
        physiology_dim: int = 64,
        clip_dim: int = 256,
        identity_dim: int = 128,
        rag_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = 8  # CLS + 7 modalities

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Modality projections → hidden_dim
        self.proj_spatial = nn.Linear(spatial_dim, hidden_dim)
        self.proj_spectral = nn.Linear(spectral_dim, hidden_dim)
        self.proj_temporal = nn.Linear(temporal_dim, hidden_dim)
        self.proj_physiology = nn.Linear(physiology_dim, hidden_dim)
        self.proj_clip = nn.Linear(clip_dim, hidden_dim)
        self.proj_identity = nn.Linear(identity_dim, hidden_dim)
        self.proj_rag = nn.Linear(rag_dim, hidden_dim)

        # Modality type embeddings (like positional but for modality type)
        self.modality_embeddings = nn.Parameter(torch.randn(self.num_tokens, hidden_dim) * 0.02)

        # Default tokens for missing modalities
        self.default_tokens = nn.ParameterDict({
            "temporal": nn.Parameter(torch.randn(hidden_dim) * 0.02),
            "physiology": nn.Parameter(torch.randn(hidden_dim) * 0.02),
            "identity": nn.Parameter(torch.randn(hidden_dim) * 0.02),
            "rag": nn.Parameter(torch.randn(hidden_dim) * 0.02),
        })

        # Modality gating
        self.gate = ModalityGate(hidden_dim, 7)  # 7 modalities (not CLS)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerFusionLayer(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultimodalTransformerFusion: {total_params/1e6:.1f}M params, "
            f"{num_layers} layers × {num_heads} heads, dim={hidden_dim}"
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        spectral_features: torch.Tensor,
        temporal_features: torch.Tensor = None,
        physiology_features: torch.Tensor = None,
        clip_features: torch.Tensor = None,
        identity_features: torch.Tensor = None,
        rag_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: (B, spatial_dim)
            spectral_features: (B, spectral_dim)
            temporal_features: (B, temporal_dim) or None
            physiology_features: (B, physiology_dim) or None
            clip_features: (B, clip_dim) or None
            identity_features: (B, identity_dim) or None
            rag_features: (B, rag_dim) or None

        Returns:
            fused: (B, hidden_dim) — CLS token output
        """
        B = spatial_features.size(0)
        device = spatial_features.device

        # Project all modalities
        tok_spatial = self.proj_spatial(spatial_features)      # (B, D)
        tok_spectral = self.proj_spectral(spectral_features)   # (B, D)

        tok_temporal = (
            self.proj_temporal(temporal_features) if temporal_features is not None
            else self.default_tokens["temporal"].unsqueeze(0).expand(B, -1)
        )
        tok_physiology = (
            self.proj_physiology(physiology_features) if physiology_features is not None
            else self.default_tokens["physiology"].unsqueeze(0).expand(B, -1)
        )
        tok_clip = (
            self.proj_clip(clip_features) if clip_features is not None
            else torch.zeros(B, self.hidden_dim, device=device)
        )
        tok_identity = (
            self.proj_identity(identity_features) if identity_features is not None
            else self.default_tokens["identity"].unsqueeze(0).expand(B, -1)
        )
        tok_rag = (
            self.proj_rag(rag_features) if rag_features is not None
            else self.default_tokens["rag"].unsqueeze(0).expand(B, -1)
        )

        # Stack modality tokens: (B, 7, D)
        modality_tokens = torch.stack([
            tok_spatial, tok_spectral, tok_temporal,
            tok_physiology, tok_clip, tok_identity, tok_rag
        ], dim=1)

        # Apply modality gating
        modality_tokens = self.gate(modality_tokens)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, modality_tokens], dim=1)  # (B, 8, D)

        # Add modality embeddings
        tokens = tokens + self.modality_embeddings.unsqueeze(0)

        # Transformer encoding
        for layer in self.layers:
            tokens = layer(tokens)

        # Extract CLS token
        cls_output = self.final_norm(tokens[:, 0])  # (B, D)

        return cls_output
