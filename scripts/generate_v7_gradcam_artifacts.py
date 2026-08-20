"""
Generate High-Resolution GradCAM Visualizations for DeepTrace V7 SOTA Multi-Spectral Model.
Outputs overlays to results/gradcam_v7/
"""

import sys
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from scripts.train_v7_sota_spectral import V7SOTADetector
from utils.device import get_device, AMPContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v7_gradcam")


def generate_gradcam_v7(model, image_tensor, target_layer):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    with AMPContext(device=image_tensor.device, enabled=True):
        out = model(image_tensor, return_spectral_details=False)
        logit = out["binary_logit"]
        prob = torch.sigmoid(logit).item()

    model.zero_grad()
    logit.backward(retain_graph=True)

    h1.remove()
    h2.remove()

    grads = gradients[0].detach().cpu().numpy()[0]  # (C, H, W)
    acts = activations[0].detach().cpu().numpy()[0]  # (C, H, W)

    weights = np.mean(grads, axis=(1, 2))  # (C,)
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, prob


def main():
    device = get_device()
    out_dir = Path("results/gradcam_v7")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    base_detector = DeepfakeDetector()
    model = V7SOTADetector(base_detector)
    ckpt = torch.load("checkpoints/v7_sota_spectral/best_model.pth", map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)

    # Target last conv stage of spatial encoder
    target_layer = model.spatial_encoder.backbone.conv_head

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_cases = [
        {
            "name": "authentic_face",
            "path": "data/kaggle_realfake/real_vs_fake/real-vs-fake/test/real/00001.jpg",
            "title": "Authentic Real Face",
        },
        {
            "name": "stylegan_synthesis_fake",
            "path": "data/kaggle_realfake/real_vs_fake/real-vs-fake/test/fake/00276TOPP4.jpg",
            "title": "StyleGAN Synthesized Face",
        },
        {
            "name": "faceforensicspp_faceswap_blending",
            "path": "D:/datasets/FFpp_c23_extracted/manipulated_sequences/FaceSwap/c23/frames/000_003/0000.png",
            "title": "FF++ FaceSwap Poisson Blending Seam",
        },
        {
            "name": "faceforensicspp_deepfakedetection",
            "path": "D:/datasets/FFpp_c23_extracted/manipulated_sequences/DeepFakeDetection/c23/frames/01__kitchen_pan/0000.png",
            "title": "FF++ DeepFakeDetection Multi-Subject Swap",
        },
    ]

    for tc in test_cases:
        p = Path(tc["path"])
        if not p.exists():
            continue

        pil_img = Image.open(p).convert("RGB")
        orig_np = np.array(pil_img.resize((160, 160)))
        t_img = transform(pil_img).unsqueeze(0).to(device)

        cam, prob = generate_gradcam_v7(model, t_img, target_layer)

        cam_resized = cv2.resize(cam, (160, 160))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.float32(heatmap) * 0.4 + np.float32(orig_np) * 0.6
        overlay = np.uint8(np.clip(overlay, 0, 255))

        out_path = out_dir / f"gradcam_v7_{tc['name']}.png"
        Image.fromarray(overlay).save(out_path)
        logger.info(f"Saved: {out_path} | Pred P(Fake) = {prob:.4f}")


if __name__ == "__main__":
    main()
