"""
Attention weight visualization for fusion and temporal transformer.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Visualize attention weights from fusion and temporal modules."""

    def __init__(self, model):
        self.model = model

    def extract_fusion_attention(self) -> list:
        """Extract attention weights from cross-attention fusion layers."""
        attention_maps = []
        try:
            for layer in self.model.fusion.sf_attention_layers:
                attn_w = getattr(layer, "attn_weights", None)
                if attn_w is not None:
                    attention_maps.append({
                        "type": "spatial_frequency",
                        "weights": attn_w.cpu().numpy(),
                    })

            for layer in self.model.fusion.temporal_attention_layers:
                attn_w = getattr(layer, "attn_weights", None)
                if attn_w is not None:
                    attention_maps.append({
                        "type": "temporal_fusion",
                        "weights": attn_w.cpu().numpy(),
                    })
        except Exception as e:
            logger.warning(f"Could not extract fusion attention: {e}")

        return attention_maps

    def extract_temporal_attention(self) -> list:
        """Extract attention weights from Video Swin Transformer."""
        attention_maps = []
        try:
            for i, layer_blocks in enumerate(self.model.temporal_model.layers):
                for j, block in enumerate(layer_blocks):
                    if hasattr(block, "attn"):
                        attn_w = getattr(block.attn, "attn_weights", None)
                        if attn_w is not None:
                            attention_maps.append({
                                "layer": i,
                                "block": j,
                                "weights": attn_w.cpu().numpy(),
                            })
        except Exception as e:
            logger.warning(f"Could not extract temporal attention: {e}")

        return attention_maps

    def visualize_attention(
        self,
        attention_maps: list,
        save_dir: str = "results/attention/",
        prefix: str = "",
    ):
        """Generate attention heatmap visualizations."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for i, attn_map in enumerate(attention_maps):
            weights = attn_map["weights"]

            # Average over batch and heads
            if weights.ndim == 4:  # (B, num_heads, N, M)
                avg_attn = weights[0].mean(axis=0)  # (N, M)
            elif weights.ndim == 3:
                avg_attn = weights[0]  # (N, M)
            else:
                continue

            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            sns.heatmap(avg_attn, cmap="viridis", ax=ax, cbar=True)

            label = attn_map.get("type", f"layer_{attn_map.get('layer', i)}")
            ax.set_title(f"Attention — {label}", fontsize=14)
            ax.set_xlabel("Key", fontsize=11)
            ax.set_ylabel("Query", fontsize=11)

            plt.tight_layout()
            filename = f"{prefix}attention_{label}_{i}.png"
            plt.savefig(save_path / filename, dpi=150)
            plt.close()

        logger.info(f"Saved {len(attention_maps)} attention visualizations to {save_path}")

    def generate_full_report(self, save_dir: str = "results/attention/"):
        """Generate complete attention visualization report."""
        fusion_attn = self.extract_fusion_attention()
        temporal_attn = self.extract_temporal_attention()

        self.visualize_attention(fusion_attn, save_dir, prefix="fusion_")
        self.visualize_attention(temporal_attn, save_dir, prefix="temporal_")

        return {
            "num_fusion_maps": len(fusion_attn),
            "num_temporal_maps": len(temporal_attn),
        }
