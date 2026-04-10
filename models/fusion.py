"""
Cross-Attention Fusion module.
Fuses spatial ↔ frequency features, then merges with temporal features.
"""

import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block between two feature streams."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention: query attends to context.

        Args:
            query: (B, dim) or (B, N, dim)
            context: (B, dim) or (B, N, dim)

        Returns:
            output: same shape as query
        """
        # Handle 2D inputs by adding sequence dim
        squeeze = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze = True
        if context.dim() == 2:
            context = context.unsqueeze(1)

        B, N, C = query.shape
        _, M, _ = context.shape

        # Cross-attention
        residual = query
        query_normed = self.norm1(query)
        context_normed = self.norm2(context)

        q = self.q_proj(query_normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context_normed).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context_normed).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach()  # for visualization
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = residual + self.dropout(out)

        # FFN
        out = out + self.ffn(self.norm3(out))

        if squeeze:
            out = out.squeeze(1)

        return out


class MultimodalFusion(nn.Module):
    """
    Fuse spatial, frequency, temporal, physiology, and CLIP features.

    Stage 1: Spatial ↔ Frequency cross-attention
    Stage 2: Fused ↔ Temporal cross-attention (if temporal available)
    Stage 3: Concatenate with physiology + CLIP projections
    """

    def __init__(
        self,
        spatial_dim: int = 1280,
        frequency_dim: int = 1280,
        temporal_dim: int = 768,
        physiology_dim: int = 64,
        clip_projection_dim: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project to common dimension
        self.spatial_proj = nn.Linear(spatial_dim, hidden_dim)
        self.frequency_proj = nn.Linear(frequency_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)

        # Cross-attention layers
        self.sf_attention_layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.temporal_attention_layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Final fusion
        total_dim = hidden_dim + physiology_dim + clip_projection_dim
        self.final_projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.output_dim = hidden_dim
        logger.info(f"MultimodalFusion: hidden_dim={hidden_dim}, total_input={total_dim}")

    def forward(
        self,
        spatial_features: torch.Tensor,
        frequency_features: torch.Tensor,
        temporal_features: torch.Tensor = None,
        physiology_features: torch.Tensor = None,
        clip_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: (B, spatial_dim)
            frequency_features: (B, frequency_dim)
            temporal_features: (B, temporal_dim) or None
            physiology_features: (B, physiology_dim) or None
            clip_features: (B, clip_projection_dim) or None

        Returns:
            fused: (B, hidden_dim)
        """
        B = spatial_features.size(0)
        device = spatial_features.device

        # Project to common space
        spatial = self.spatial_proj(spatial_features)
        frequency = self.frequency_proj(frequency_features)

        # Spatial ↔ Frequency cross-attention
        for attn_layer in self.sf_attention_layers:
            spatial = attn_layer(spatial, frequency)
            frequency = attn_layer(frequency, spatial)

        # Average spatial and frequency
        fused = (spatial + frequency) / 2.0

        # Fused ↔ Temporal cross-attention
        if temporal_features is not None:
            temporal = self.temporal_proj(temporal_features)
            for attn_layer in self.temporal_attention_layers:
                fused = attn_layer(fused, temporal)

        # Concatenate additional features
        components = [fused]

        if physiology_features is not None:
            components.append(physiology_features)
        else:
            components.append(torch.zeros(B, 64, device=device))

        if clip_features is not None:
            components.append(clip_features)
        else:
            components.append(torch.zeros(B, 256, device=device))

        combined = torch.cat(components, dim=-1)
        output = self.final_projection(combined)

        return output
