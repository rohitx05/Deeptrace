"""
Multi-task loss functions for deepfake detection.
Weighted BCE (binary) + CE (manipulation type) + CLIP alignment + consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DeepfakeLoss(nn.Module):
    """
    Multi-task loss combining:
    1. Binary Cross-Entropy for real/fake classification
    2. Cross-Entropy for manipulation type classification
    3. CLIP alignment loss
    4. Temporal consistency regularization (video mode)
    """

    def __init__(
        self,
        binary_weight: float = 1.0,
        manipulation_weight: float = 0.5,
        clip_alignment_weight: float = 0.3,
        consistency_weight: float = 0.2,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.binary_weight = binary_weight
        self.manipulation_weight = manipulation_weight
        self.clip_alignment_weight = clip_alignment_weight
        self.consistency_weight = consistency_weight

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Args:
            predictions: dict from DeepfakeDetector.forward()
            targets: dict with 'label' (B,) and 'manipulation_type' (B,)

        Returns:
            dict with individual losses and total loss
        """
        losses = {}

        # Binary loss
        binary_logit = predictions["binary_logit"].squeeze(-1)
        labels = targets["label"].float()
        losses["binary"] = F.binary_cross_entropy_with_logits(binary_logit, labels)

        # Manipulation type loss (only for fake samples)
        fake_mask = labels > 0.5
        if fake_mask.any():
            manip_logits = predictions["manipulation_logits"][fake_mask]
            manip_targets = targets["manipulation_type"][fake_mask]
            losses["manipulation"] = self.ce(manip_logits, manip_targets)
        else:
            losses["manipulation"] = torch.tensor(0.0, device=binary_logit.device)

        # CLIP alignment loss
        if "clip_alignment_loss" in predictions:
            losses["clip_alignment"] = predictions["clip_alignment_loss"]
        else:
            losses["clip_alignment"] = torch.tensor(0.0, device=binary_logit.device)

        # Confidence calibration loss (encourage high confidence for correct predictions)
        pred_correct = ((predictions["binary_pred"] > 0.5).float() == labels).float()
        confidence = predictions["confidence"]
        losses["confidence"] = F.mse_loss(confidence, pred_correct.detach())

        # Total weighted loss
        losses["total"] = (
            self.binary_weight * losses["binary"]
            + self.manipulation_weight * losses["manipulation"]
            + self.clip_alignment_weight * losses["clip_alignment"]
            + self.consistency_weight * losses["confidence"]
        )

        return losses
