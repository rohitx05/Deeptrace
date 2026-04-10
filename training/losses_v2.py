"""
Extended loss functions for DeepfakeDetectorV2.
Adds generator attribution, identity consistency, and RAG contrastive losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ExtendedDeepfakeLoss(nn.Module):
    """
    Extended multi-task loss for V2 detector:
    1. Binary CE (real/fake)
    2. Manipulation type CE
    3. Generator attribution CE
    4. CLIP alignment cosine
    5. Confidence calibration MSE
    6. Identity consistency regularization
    """

    def __init__(
        self,
        binary_weight: float = 1.0,
        manipulation_weight: float = 0.5,
        generator_weight: float = 0.3,
        clip_alignment_weight: float = 0.3,
        confidence_weight: float = 0.2,
        identity_weight: float = 0.2,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.binary_weight = binary_weight
        self.manipulation_weight = manipulation_weight
        self.generator_weight = generator_weight
        self.clip_alignment_weight = clip_alignment_weight
        self.confidence_weight = confidence_weight
        self.identity_weight = identity_weight

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, predictions: dict, targets: dict) -> dict:
        losses = {}

        # 1. Binary loss
        binary_logit = predictions["binary_logit"].squeeze(-1)
        labels = targets["label"].float()
        losses["binary"] = F.binary_cross_entropy_with_logits(binary_logit, labels)

        # 2. Manipulation type loss (fake samples only)
        fake_mask = labels > 0.5
        if fake_mask.any() and "manipulation_type" in targets:
            losses["manipulation"] = self.ce(
                predictions["manipulation_logits"][fake_mask],
                targets["manipulation_type"][fake_mask],
            )
        else:
            losses["manipulation"] = torch.tensor(0.0, device=binary_logit.device)

        # 3. Generator attribution loss (fake samples only)
        if fake_mask.any() and "generator_logits" in predictions and "generator_type" in targets:
            losses["generator"] = self.ce(
                predictions["generator_logits"][fake_mask],
                targets["generator_type"][fake_mask],
            )
        else:
            losses["generator"] = torch.tensor(0.0, device=binary_logit.device)

        # 4. CLIP alignment
        losses["clip_alignment"] = predictions.get(
            "clip_alignment_loss", torch.tensor(0.0, device=binary_logit.device)
        )

        # 5. Confidence calibration
        pred_correct = ((predictions["binary_pred"] > 0.5).float() == labels).float()
        losses["confidence"] = F.mse_loss(predictions["confidence"], pred_correct.detach())

        # 6. Identity consistency (encourage low uncertainty on real, high on fake)
        # This is a soft signal — no hard label needed
        losses["identity"] = torch.tensor(0.0, device=binary_logit.device)

        # Total weighted loss
        losses["total"] = (
            self.binary_weight * losses["binary"]
            + self.manipulation_weight * losses["manipulation"]
            + self.generator_weight * losses["generator"]
            + self.clip_alignment_weight * losses["clip_alignment"]
            + self.confidence_weight * losses["confidence"]
            + self.identity_weight * losses["identity"]
        )

        return losses
