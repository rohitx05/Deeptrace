"""
Comprehensive Forensic Audit Script
1. Audits Kaggle Real vs Fake dataset for shortcut learning / data leakage (resolution, size, EXIF, color stats, compression, duplicate hashes).
2. Investigates why _new_dataset achieved AUC 0.4514 (analyzes prediction distributions on Real vs Fake).
3. Analyzes 20 misclassified samples to pinpoint error modes.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import hashlib

def compute_dhash(image, hash_size=8):
    """Compute difference hash (dHash) using PIL."""
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized.getdata(), dtype=np.float32).reshape((hash_size, hash_size + 1))
    diff = pixels[:, 1:] > pixels[:, :-1]
    return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))


sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from datasets.transforms import get_val_transforms, apply_dct_transform
from utils.checkpoint import load_checkpoint
from utils.device import get_device

def audit_image_properties(file_list, desc="Dataset"):
    print(f"\n--- Auditing Image Properties: {desc} ({len(file_list)} samples) ---")
    sizes = []
    file_bytes = []
    formats = set()
    modes = set()
    exif_count = 0
    aspect_ratios = []
    
    for p in file_list:
        try:
            im = Image.open(p)
            sizes.append(im.size)
            aspect_ratios.append(im.size[0] / im.size[1])
            file_bytes.append(os.path.getsize(p))
            formats.add(im.format)
            modes.add(im.mode)
            if hasattr(im, "_getexif") and im._getexif():
                exif_count += 1
        except Exception as e:
            pass

    w_arr, h_arr = zip(*sizes) if sizes else ([0], [0])
    b_arr = np.array(file_bytes) / 1024.0 # in KB
    
    print(f"  Formats: {formats}, Color Modes: {modes}")
    print(f"  Width:  min={min(w_arr)}, max={max(w_arr)}, mean={np.mean(w_arr):.1f}")
    print(f"  Height: min={min(h_arr)}, max={max(h_arr)}, mean={np.mean(h_arr):.1f}")
    print(f"  File Size (KB): min={b_arr.min():.1f}KB, max={b_arr.max():.1f}KB, mean={b_arr.mean():.1f}KB, std={b_arr.std():.1f}KB")
    print(f"  EXIF Presence: {exif_count}/{len(file_list)} ({exif_count/len(file_list)*100:.1f}%)")
    
    return {
        "formats": list(formats),
        "modes": list(modes),
        "mean_width": float(np.mean(w_arr)),
        "mean_height": float(np.mean(h_arr)),
        "mean_size_kb": float(b_arr.mean()),
        "std_size_kb": float(b_arr.std()),
        "exif_ratio": float(exif_count / len(file_list))
    }

def main():
    device = get_device()
    print(f"Running audit on device: {device}")
    
    # 1. Image Property Audit on Kaggle Real vs Fake
    k_root = Path(r"data/kaggle_realfake/real_vs_fake/real-vs-fake")
    kaggle_real_files = list((k_root / "train" / "real").glob("*.jpg"))[:500]
    kaggle_fake_files = list((k_root / "train" / "fake").glob("*.jpg"))[:500]
    
    real_stats = audit_image_properties(kaggle_real_files, "Kaggle Real (Train)")
    fake_stats = audit_image_properties(kaggle_fake_files, "Kaggle Fake (Train)")
    
    # 2. Check for Duplicate Hashes between Train and Test
    print("\n--- Checking for Train/Test Near-Duplicates using Perceptual Hash (dHash) ---")
    kaggle_test_real = list((k_root / "test" / "real").glob("*.jpg"))[:500]
    kaggle_train_real = kaggle_real_files[:500]
    
    train_hashes = {compute_dhash(Image.open(p)): p for p in kaggle_train_real}
    dups = 0
    for p in kaggle_test_real:
        h = compute_dhash(Image.open(p))
        if h in train_hashes:
            dups += 1
    print(f"  Exact dHash duplicates in 500 sampled test images vs train: {dups}/500")
    
    # 3. Model Audit on _new_dataset_extracted and Kaggle Test
    ckpt_path = "checkpoints/v2_clip_finetune/best_model.pth"
    print(f"\n--- Loading Model Checkpoint: {ckpt_path} ---")
    model = DeepfakeDetector()
    load_checkpoint(ckpt_path, model, device=device)
    model.to(device)
    model.eval()
    
    # Debug _new_dataset_extracted
    print("\n--- Debugging _new_dataset_extracted predictions ---")
    new_real_files = list(Path("data/_new_dataset_extracted/Test/Real").glob("*.jpg"))[:200]
    new_fake_files = list(Path("data/_new_dataset_extracted/Test/Fake").glob("*.jpg"))[:200]
    
    print(f"Auditing _new_dataset: found {len(new_real_files)} real, {len(new_fake_files)} fake samples")
    
    transform = get_val_transforms(160)
    
    def predict_paths(paths):
        probs = []
        raw_logits = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (160, 160))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dct = apply_dct_transform(img)
            dct_norm = (dct - dct.mean()) / (dct.std() + 1e-8)
            dct_t = torch.from_numpy(dct_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)
            dct_t = F.interpolate(dct_t, size=(160, 160), mode="bilinear", align_corners=False)
            
            aug = transform(image=rgb)
            img_t = aug["image"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    out = model(images=img_t, dct=dct_t, mode="image")
                    logit = out["binary_logit"].item()
                    prob = torch.sigmoid(out["binary_logit"]).item()
                    raw_logits.append(logit)
                    probs.append(prob)
        return np.array(probs), np.array(raw_logits)
    
    new_real_probs, new_real_logits = predict_paths(new_real_files)
    new_fake_probs, new_fake_logits = predict_paths(new_fake_files)
    
    print(f"\n  _new_dataset Real (Target=0): mean_fake_prob={new_real_probs.mean():.4f}, mean_logit={new_real_logits.mean():.4f}")
    print(f"  _new_dataset Fake (Target=1): mean_fake_prob={new_fake_probs.mean():.4f}, mean_logit={new_fake_logits.mean():.4f}")
    
    # 4. Compare with Kaggle Valid Real vs Fake Predictions
    print("\n--- Comparing with Kaggle Valid predictions ---")
    k_test_real_files = list((k_root / "valid" / "real").glob("*.jpg"))[:200]
    k_test_fake_files = list((k_root / "valid" / "fake").glob("*.jpg"))[:200]
    
    k_real_probs, k_real_logits = predict_paths(k_test_real_files)
    k_fake_probs, k_fake_logits = predict_paths(k_test_fake_files)
    
    print(f"  Kaggle Test Real (Target=0): mean_fake_prob={k_real_probs.mean():.4f}, mean_logit={k_real_logits.mean():.4f}")
    print(f"  Kaggle Test Fake (Target=1): mean_fake_prob={k_fake_probs.mean():.4f}, mean_logit={k_fake_logits.mean():.4f}")
    
    # 5. Inspecting What _new_dataset Actually Is
    audit_image_properties(new_real_files, "_new_dataset Real")
    audit_image_properties(new_fake_files, "_new_dataset Fake")

if __name__ == "__main__":
    main()
