"""
Step 5: Generate Final Comprehensive Visual GradCAM Artifacts across all manipulation families.
Evaluates:
1. Authentic face
2. StyleGAN GAN synthesis face
3. FaceSwap boundary seam manipulation
4. DeepFakeDetection multi-actor face
Outputs: results/gradcam_v6/
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v6_gradcam")


SRM_KERNELS = torch.tensor([
    [[-1.0, -2.0, -1.0],
     [-2.0, 12.0, -2.0],
     [-1.0, -2.0, -1.0]],
    [[-1.0,  2.0, -1.0],
     [-2.0,  4.0, -2.0],
     [-1.0,  2.0, -1.0]],
    [[-1.0, -2.0, -1.0],
     [ 2.0,  4.0,  2.0],
     [-1.0, -2.0, -1.0]],
], dtype=torch.float32).unsqueeze(1)


def compute_srm_residual_tensor(img_tensor):
    gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
    kernels = SRM_KERNELS.to(img_tensor.device)
    residuals = F.conv2d(gray, kernels, padding=1)
    min_val = residuals.amin(dim=(-2, -1), keepdim=True)
    max_val = residuals.amax(dim=(-2, -1), keepdim=True)
    norm_residuals = (residuals - min_val) / (max_val - min_val + 1e-6)
    return norm_residuals


def main():
    device = get_device()
    out_dir = Path("results/gradcam_v6")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DeepfakeDetector()
    load_checkpoint("checkpoints/v5_srm_residual/best_model.pth", model, device=device)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_samples = [
        ("data/kaggle_realfake/real_vs_fake/real-vs-fake/test/real/00001.jpg", 0, "Authentic Face"),
        ("data/kaggle_realfake/real_vs_fake/real-vs-fake/test/fake/00276TOPP4.jpg", 1, "StyleGAN Synthesis Fake"),
    ]

    manifest_csv = Path("manifests/ffpp_c23_manifest.csv")
    if manifest_csv.exists():
        df = pd.read_csv(manifest_csv)
        col = "filepath" if "filepath" in df.columns else "image_path"
        fswap = df[df["manipulation_type"] == "FaceSwap"][col].tolist()
        dfd = df[df["manipulation_type"] == "DeepFakeDetection"][col].tolist()
        if fswap:
            test_samples.append((fswap[0], 1, "FaceForensics++ FaceSwap Blending"))
        if dfd:
            test_samples.append((dfd[0], 1, "FaceForensics++ DeepFakeDetection"))

    target_layer = model.spatial_encoder.backbone.conv_head

    for img_path, label, title in test_samples:
        if not Path(img_path).exists():
            continue
        raw_pil = Image.open(img_path).convert("RGB")
        img_t = transform(raw_pil).unsqueeze(0).to(device)
        srm_t = compute_srm_residual_tensor(img_t).to(device)

        activations, gradients = [], []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)

        model.zero_grad()
        out = model(img_t, dct=srm_t)
        logits = out["binary_logit"].squeeze(-1)
        prob = torch.sigmoid(logits).item()

        logits.backward()

        h1.remove()
        h2.remove()

        acts = activations[0]
        grads = gradients[0]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(160, 160), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_uint8 = np.uint8(255 * cam)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(raw_pil.resize((160, 160)))
        plt.title(f"Input: {title}\nTrue: {'Fake' if label==1 else 'Real'}", fontsize=9)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(raw_pil.resize((160, 160)))
        plt.imshow(cam_uint8, cmap="jet", alpha=0.5)
        plt.title(f"SRM Seam GradCAM\nP(Fake)={prob:.4f}", fontsize=9)
        plt.axis("off")

        safe_name = title.lower().replace(" ", "_").replace("++", "pp")
        out_path = out_dir / f"gradcam_v6_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Generated GradCAM overlay for {title} -> {out_path} (P={prob:.4f})")

    logger.info("=== Visual Artifacts Complete ===")


if __name__ == "__main__":
    main()
