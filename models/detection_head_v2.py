"""
Extended Detection Head (V2).
Adds generator attribution and uncertainty-aware branches to the existing detection head.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

GENERATOR_TYPES = ["GAN", "Diffusion", "FaceSwap", "Unknown"]


class ExtendedDetectionHead(nn.Module):
    """
    5-branch multi-task prediction head:
    - Binary: real vs fake (sigmoid)
    - Manipulation type: 5-way classification (softmax)
    - Generator attribution: 4-way classification (GAN/Diffusion/FaceSwap/Unknown)
    - Confidence: calibrated confidence score
    - Uncertainty readiness: dropout kept active for MC Dropout at inference
    """

    def __init__(
        self,
        input_dim: int = 512,
        spectral_dim: int = 1280,
        hidden_dim: int = 256,
        num_manipulation_types: int = 5,
        num_generator_types: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Shared backbone (from fused features)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 1. Binary head: real vs fake
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 2. Manipulation type head
        self.manipulation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_manipulation_types),
        )

        # 3. Generator attribution head (from SPECTRAL features for fingerprint)
        self.generator_head = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_generator_types),
        )

        # 4. Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"ExtendedDetectionHead: input={input_dim}, spectral={spectral_dim}, "
            f"manip_types={num_manipulation_types}, gen_types={num_generator_types}"
        )

    def forward(self, fused_features: torch.Tensor, spectral_features: torch.Tensor = None) -> dict:
        """
        Args:
            fused_features: (B, input_dim) from fusion transformer
            spectral_features: (B, spectral_dim) from spectral combiner

        Returns:
            dict with all prediction outputs
        """
        shared = self.shared(fused_features)

        # Binary
        binary_logit = self.binary_head(shared)
        binary_prob = torch.sigmoid(binary_logit).squeeze(-1)

        # Manipulation type
        manipulation_logits = self.manipulation_head(shared)
        manipulation_pred = manipulation_logits.argmax(dim=-1)

        # Confidence
        confidence = self.confidence_head(shared).squeeze(-1)

        result = {
            "binary_logit": binary_logit,
            "binary_pred": binary_prob,
            "manipulation_logits": manipulation_logits,
            "manipulation_pred": manipulation_pred,
            "confidence": confidence,
        }

        # Generator attribution (from spectral features)
        if spectral_features is not None:
            generator_logits = self.generator_head(spectral_features)
            result["generator_logits"] = generator_logits
            result["generator_pred"] = generator_logits.argmax(dim=-1)
        else:
            B = fused_features.size(0)
            result["generator_logits"] = torch.zeros(B, 4, device=fused_features.device)
            result["generator_pred"] = torch.zeros(B, dtype=torch.long, device=fused_features.device)

        return result
