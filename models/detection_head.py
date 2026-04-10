"""
Multi-task Detection Head.
Binary classification + manipulation type + confidence score.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class DetectionHead(nn.Module):
    """
    Multi-task prediction head:
    - Binary: real vs fake (sigmoid)
    - Manipulation type: 5-way classification (softmax)
    - Confidence: calibrated confidence score
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_manipulation_types: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Binary head: real vs fake
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Manipulation type head
        self.manipulation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_manipulation_types),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"DetectionHead: input={input_dim}, hidden={hidden_dim}, "
            f"types={num_manipulation_types}"
        )

    def forward(self, fused_features: torch.Tensor) -> dict:
        """
        Args:
            fused_features: (B, input_dim) from fusion module

        Returns:
            dict with:
                - binary_logit: (B, 1) raw logit for binary classification
                - binary_pred: (B,) probability of being fake
                - manipulation_logits: (B, num_types) logits per type
                - manipulation_pred: (B,) predicted type index
                - confidence: (B,) confidence score [0, 1]
        """
        shared = self.shared(fused_features)

        # Binary prediction
        binary_logit = self.binary_head(shared)
        binary_prob = torch.sigmoid(binary_logit).squeeze(-1)

        # Manipulation type prediction
        manipulation_logits = self.manipulation_head(shared)
        manipulation_pred = manipulation_logits.argmax(dim=-1)

        # Confidence
        confidence = self.confidence_head(shared).squeeze(-1)

        return {
            "binary_logit": binary_logit,
            "binary_pred": binary_prob,
            "manipulation_logits": manipulation_logits,
            "manipulation_pred": manipulation_pred,
            "confidence": confidence,
        }
