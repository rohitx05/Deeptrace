"""
Item 1: Generate GradCAM heatmaps for 1 real + 1 fake test image.
Uses the FIXED GradCAM with DCT input and no-TTA protocol.
Saves overlay PNGs + machine-readable JSON with prediction provenance.
"""

import sys
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from explainability.gradcam import GradCAM
from datasets.transforms import get_val_transforms, apply_dct_transform
from utils.checkpoint import load_checkpoint
from utils.device import get_device


def preprocess_image(img_path, image_size=160):
    """Load and preprocess one image (same pipeline as evaluation)."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read: {img_path}")

    img_resized = cv2.resize(img, (image_size, image_size))
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # RGB tensor
    transform = get_val_transforms(image_size)
    aug = transform(image=rgb)
    rgb_tensor = aug["image"].unsqueeze(0)  # (1, 3, H, W)

    # DCT tensor (same as eval pipeline)
    dct = apply_dct_transform(img_resized)
    dct_norm = (dct - dct.mean()) / (dct.std() + 1e-8)
    dct_tensor = torch.from_numpy(dct_norm).permute(2, 0, 1).float()
    dct_tensor = F.interpolate(
        dct_tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )  # (1, 3, H, W)

    return img, rgb_tensor, dct_tensor


def main():
    device = get_device()
    print(f"Device: {device}")

    # Load model
    ckpt_path = "checkpoints/v2_clip_finetune/best_model.pth"
    model = DeepfakeDetector()
    load_checkpoint(ckpt_path, model, device=device)
    model.to(device)
    model.eval()

    # Load calibration
    calib_path = Path("checkpoints/v2_clip_finetune/calibration.json")
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        temperature = calib["temperature"]
    else:
        temperature = 1.0

    # Create GradCAM
    cam = GradCAM(model)

    # Select test images: 1 real + 1 fake from Kaggle test split
    real_dir = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/test/real")
    fake_dir = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake/test/fake")

    # Pick first sorted image from each class for reproducibility
    real_imgs = sorted(real_dir.glob("*.jpg"))
    fake_imgs = sorted(fake_dir.glob("*.jpg"))

    if not real_imgs or not fake_imgs:
        print("ERROR: No test images found!")
        return

    test_pairs = [
        (real_imgs[0], "real", 0),
        (fake_imgs[0], "fake", 1),
    ]

    output_dir = Path("results/gradcam_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path, label_name, true_label in test_pairs:
        print(f"\n--- Processing: {img_path.name} (true label: {label_name}) ---")

        original_img, rgb_tensor, dct_tensor = preprocess_image(img_path)
        rgb_tensor = rgb_tensor.to(device)
        dct_tensor = dct_tensor.to(device)

        # Generate GradCAM with full metadata
        result = cam.generate_with_metadata(
            rgb_tensor,
            dct_tensor=dct_tensor,
            class_idx=1,  # Always target "fake" logit
            temperature=temperature,
        )

        heatmap = result["heatmap"]
        print(f"  Raw logit: {result['raw_logit']:.4f}")
        print(f"  Raw probability: {result['raw_probability']:.4f}")
        print(f"  Calibrated probability: {result['calibrated_probability']:.4f}")
        print(f"  Predicted class: {'fake' if result['predicted_class'] == 1 else 'real'}")
        print(f"  TTA used: {result['tta_used']}")
        print(f"  DCT provided: {result['dct_provided']}")

        # Create overlay
        overlay = cam.overlay_heatmap(heatmap, original_img, alpha=0.4)

        # Add text caption with prediction metadata
        h, w = overlay.shape[:2]
        caption_h = 60
        canvas = np.zeros((h + caption_h, w, 3), dtype=np.uint8)
        canvas[:h] = overlay
        canvas[h:] = (40, 40, 40)  # Dark grey caption bar

        prob = result["calibrated_probability"]
        pred = "FAKE" if result["predicted_class"] == 1 else "REAL"
        truth = label_name.upper()

        cv2.putText(canvas, f"True: {truth} | Pred: {pred} | P(fake)={prob:.4f}",
                     (10, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, f"T={temperature:.4f} | No TTA | DCT: Yes | Target: fake logit",
                     (10, h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Save
        out_img = output_dir / f"gradcam_{label_name}_{img_path.stem}.png"
        cv2.imwrite(str(out_img), canvas)
        print(f"  Saved: {out_img}")

        # Save JSON provenance
        out_json = output_dir / f"gradcam_{label_name}_{img_path.stem}.json"
        metadata = {
            "source_image": str(img_path),
            "true_label": true_label,
            "true_label_name": label_name,
            "raw_logit": float(result["raw_logit"]),
            "raw_probability": float(result["raw_probability"]),
            "calibrated_probability": float(result["calibrated_probability"]),
            "temperature": temperature,
            "predicted_class": int(result["predicted_class"]),
            "target_class_for_cam": int(result["target_class_for_cam"]),
            "tta_used": result["tta_used"],
            "dct_provided": result["dct_provided"],
            "checkpoint": ckpt_path,
            "protocol": "Single no-TTA forward pass with RGB + DCT input",
        }
        with open(out_json, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: {out_json}")

    print("\nDone! GradCAM overlays saved to results/gradcam_v2/")


if __name__ == "__main__":
    main()
