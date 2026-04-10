"""
Generator Fingerprint Attribution Head.
Predicts generator type from spectral features: GAN / Diffusion / FaceSwap / Unknown.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

GENERATOR_TYPES = ["GAN", "Diffusion", "FaceSwap", "Unknown"]


class GeneratorFingerprintHead(nn.Module):
    """
    Predicts the generator type from spectral (frequency) features.
    Spectral domain carries unique fingerprints per generator architecture.

    Classes:
        0: GAN (StyleGAN, ProGAN, etc.)
        1: Diffusion (Stable Diffusion, DALL-E, etc.)
        2: FaceSwap (traditional 3D model-based)
        3: Unknown (novel generator)
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Spectral fingerprint projection for retrieval embedding
        self.fingerprint_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        types_str = ', '.join(GENERATOR_TYPES)
        logger.info(f"GeneratorFingerprintHead: {num_classes} classes ({types_str})")

    def forward(self, spectral_features: torch.Tensor) -> dict:
        """
        Args:
            spectral_features: (B, input_dim) from SpectralCombiner

        Returns:
            dict with:
                - generator_logits: (B, num_classes)
                - generator_pred: (B,) predicted class index
                - generator_fingerprint: (B, 128) fingerprint embedding
        """
        logits = self.head(spectral_features)
        pred = logits.argmax(dim=-1)
        fingerprint = self.fingerprint_proj(spectral_features)

        return {
            "generator_logits": logits,
            "generator_pred": pred,
            "generator_fingerprint": fingerprint,
        }

    @staticmethod
    def map_manipulation_to_generator(manipulation_type: int) -> int:
        """
        Map existing manipulation_type labels to generator categories.

        Current manipulation types:
            0: real → not used (only fake samples go through this head)
            1: Deepfakes → GAN (0)
            2: Face2Face → GAN (0)
            3: FaceSwap → FaceSwap (2)
            4: NeuralTextures → GAN (0)

        Override this for datasets with diffusion-based fakes.
        """
        mapping = {0: 3, 1: 0, 2: 0, 3: 2, 4: 0}
        try:
            mt_int = int(manipulation_type)
        except (TypeError, ValueError):
            mt_int = 3
        return mapping.get(mt_int, 3)  # default to Unknown
