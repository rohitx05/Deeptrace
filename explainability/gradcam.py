"""
GradCAM implementation for deepfake detection.
Generates artifact heatmaps highlighting manipulated regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Targets the spatial encoder's last convolutional layer.
    """

    def __init__(self, model, target_layer_name: str = None):
        self.model = model
        self.gradients = None
        self.activations = None

        # Find target layer (last conv layer of spatial encoder)
        self.target_layer = self._find_target_layer(target_layer_name)
        if self.target_layer is not None:
            self.target_layer.register_forward_hook(self._forward_hook)
            self.target_layer.register_full_backward_hook(self._backward_hook)
            logger.info(f"GradCAM target layer: {type(self.target_layer).__name__}")

    def _find_target_layer(self, name=None):
        """Find the last convolutional layer in the spatial encoder."""
        try:
            backbone = self.model.spatial_encoder.backbone
            # For EfficientNet, the last block before global pool
            if hasattr(backbone, "blocks"):
                return backbone.blocks[-1]
            elif hasattr(backbone, "features"):
                return backbone.features[-1]
            else:
                # Fallback: find last Conv2d
                last_conv = None
                for module in backbone.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        last_conv = module
                return last_conv
        except Exception as e:
            logger.warning(f"Could not find target layer: {e}")
            return None

    def _forward_hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.activations = output.detach()
        elif isinstance(output, tuple):
            self.activations = output[0].detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate GradCAM heatmap.

        Args:
            input_tensor: (1, 3, H, W) preprocessed image tensor
            class_idx: target class (None = predicted class)

        Returns:
            heatmap: (H, W) numpy array in range [0, 1]
        """
        if self.target_layer is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(images=input_tensor, mode="image")
        logit = output["binary_logit"]

        if class_idx is None:
            class_idx = (logit > 0).long().item()

        # Backward pass
        self.model.zero_grad()
        target = logit if class_idx == 1 else -logit
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # GradCAM computation
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def overlay_heatmap(
        self, heatmap: np.ndarray, original_image: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay GradCAM heatmap on original image.

        Args:
            heatmap: (H, W) normalized heatmap
            original_image: BGR image
            alpha: overlay transparency

        Returns:
            overlaid: BGR image with heatmap overlay
        """
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlaid
