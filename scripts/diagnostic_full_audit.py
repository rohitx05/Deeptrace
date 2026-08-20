"""
Comprehensive Diagnostic Script — Items 1-7 from the critique.
No hand-waving. Just data.

1. Non-StyleGAN fake test (using _new_dataset fakes if they're non-StyleGAN, plus checking)
2. Different "real" source test (use _new_dataset reals against Kaggle fakes)
3. Logit std on _new_dataset (pipeline bug check)
4. Recompress Kaggle test to ~10KB and re-eval (isolate compression vs other causes)
5. Actual pixel resolution of _new_dataset vs Kaggle (not just file size)
6. Visual side-by-side: raw pixels after preprocessing pipeline
7. Spot-check 10 _new_dataset "fake" labels by saving viewable samples
"""

import os, sys, json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from datasets.transforms import get_val_transforms, apply_dct_transform
from utils.checkpoint import load_checkpoint
from utils.device import get_device

OUTPUT_DIR = Path("results/diagnostic_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model(device):
    model = DeepfakeDetector()
    load_checkpoint("checkpoints/v2_clip_finetune/best_model.pth", model, device=device)
    model.to(device)
    model.eval()
    return model

def predict_single(model, img_path, transform, device):
    """Return raw logit and sigmoid prob for one image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h_orig, w_orig = img.shape[:2]
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
    return logit, prob

def predict_batch(model, paths, transform, device, desc=""):
    logits, probs = [], []
    for p in paths:
        l, pr = predict_single(model, p, transform, device)
        if l is not None:
            logits.append(l)
            probs.append(pr)
    logits = np.array(logits)
    probs = np.array(probs)
    print(f"  {desc}: N={len(logits)}, mean_logit={logits.mean():.4f}, std_logit={logits.std():.4f}, "
          f"mean_prob={probs.mean():.4f}, std_prob={probs.std():.4f}, "
          f"min_logit={logits.min():.4f}, max_logit={logits.max():.4f}")
    return logits, probs

def main():
    device = get_device()
    print(f"Device: {device}")
    model = load_model(device)
    transform = get_val_transforms(160)

    k_root = Path("data/kaggle_realfake/real_vs_fake/real-vs-fake")
    nd_root = Path("data/_new_dataset_extracted/Test")
    td_root = Path("test_data")

    results = {}

    # =========================================================================
    # TEST 3: Logit standard deviation on _new_dataset (pipeline bug check)
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: Logit distribution on _new_dataset (bug check)")
    print("="*70)
    nd_real = sorted((nd_root / "Real").glob("*.jpg"))[:300]
    nd_fake = sorted((nd_root / "Fake").glob("*.jpg"))[:300]
    nd_real_logits, nd_real_probs = predict_batch(model, nd_real, transform, device, "_new_dataset Real")
    nd_fake_logits, nd_fake_probs = predict_batch(model, nd_fake, transform, device, "_new_dataset Fake")
    results["test3_logit_std"] = {
        "nd_real_mean": float(nd_real_logits.mean()), "nd_real_std": float(nd_real_logits.std()),
        "nd_fake_mean": float(nd_fake_logits.mean()), "nd_fake_std": float(nd_fake_logits.std()),
    }

    # =========================================================================
    # TEST 5: Actual pixel resolution of _new_dataset vs Kaggle
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 5: Actual pixel resolution check (not file size)")
    print("="*70)
    for label, files in [("nd_real", nd_real[:50]), ("nd_fake", nd_fake[:50])]:
        widths, heights = [], []
        for p in files:
            im = Image.open(p)
            widths.append(im.size[0])
            heights.append(im.size[1])
        print(f"  {label}: W range=[{min(widths)}, {max(widths)}], H range=[{min(heights)}, {max(heights)}], "
              f"mean={np.mean(widths):.0f}x{np.mean(heights):.0f}")
    k_real_files = sorted((k_root / "test" / "real").glob("*.jpg"))[:50]
    k_fake_files = sorted((k_root / "test" / "fake").glob("*.jpg"))[:50]
    for label, files in [("kaggle_real", k_real_files), ("kaggle_fake", k_fake_files)]:
        widths, heights = [], []
        for p in files:
            im = Image.open(p)
            widths.append(im.size[0])
            heights.append(im.size[1])
        print(f"  {label}: W range=[{min(widths)}, {max(widths)}], H range=[{min(heights)}, {max(heights)}], "
              f"mean={np.mean(widths):.0f}x{np.mean(heights):.0f}")

    # =========================================================================
    # TEST 4: Recompress Kaggle test images to ~10KB and re-eval
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 4: Recompress Kaggle test to ~10KB, re-evaluate")
    print("="*70)
    recomp_dir = OUTPUT_DIR / "recompressed_kaggle"
    recomp_dir.mkdir(exist_ok=True)

    k_test_real = sorted((k_root / "test" / "real").glob("*.jpg"))[:200]
    k_test_fake = sorted((k_root / "test" / "fake").glob("*.jpg"))[:200]

    def recompress_to_target(src_paths, target_kb=10, out_subdir="real"):
        out_dir = recomp_dir / out_subdir
        out_dir.mkdir(exist_ok=True)
        recomp_paths = []
        for p in src_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            # Binary search for JPEG quality that gives ~target_kb
            best_q, best_diff = 10, 999
            for q in range(5, 96, 5):
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
                kb = len(buf) / 1024
                diff = abs(kb - target_kb)
                if diff < best_diff:
                    best_diff = diff
                    best_q = q
            out_path = out_dir / p.name
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, best_q])
            recomp_paths.append(out_path)
        return recomp_paths

    recomp_real = recompress_to_target(k_test_real, target_kb=10, out_subdir="real")
    recomp_fake = recompress_to_target(k_test_fake, target_kb=10, out_subdir="fake")

    # Verify file sizes
    real_sizes = [os.path.getsize(p)/1024 for p in recomp_real[:20]]
    fake_sizes = [os.path.getsize(p)/1024 for p in recomp_fake[:20]]
    print(f"  Recompressed real: mean={np.mean(real_sizes):.1f}KB (target=10KB)")
    print(f"  Recompressed fake: mean={np.mean(fake_sizes):.1f}KB (target=10KB)")

    # Run original Kaggle test
    print("\n  --- Original Kaggle test (uncompressed) ---")
    orig_real_logits, _ = predict_batch(model, k_test_real, transform, device, "Kaggle Real (orig)")
    orig_fake_logits, _ = predict_batch(model, k_test_fake, transform, device, "Kaggle Fake (orig)")

    # Run recompressed Kaggle test
    print("\n  --- Recompressed Kaggle test (~10KB) ---")
    recomp_real_logits, _ = predict_batch(model, recomp_real, transform, device, "Kaggle Real (recomp)")
    recomp_fake_logits, _ = predict_batch(model, recomp_fake, transform, device, "Kaggle Fake (recomp)")

    # Compute AUC for both
    from sklearn.metrics import roc_auc_score
    orig_labels = np.array([0]*len(orig_real_logits) + [1]*len(orig_fake_logits))
    orig_scores = np.concatenate([orig_real_logits, orig_fake_logits])
    orig_auc = roc_auc_score(orig_labels, orig_scores)

    recomp_labels = np.array([0]*len(recomp_real_logits) + [1]*len(recomp_fake_logits))
    recomp_scores = np.concatenate([recomp_real_logits, recomp_fake_logits])
    recomp_auc = roc_auc_score(recomp_labels, recomp_scores)

    print(f"\n  Original Kaggle 400-sample AUC:      {orig_auc:.4f}")
    print(f"  Recompressed Kaggle 400-sample AUC:  {recomp_auc:.4f}")
    print(f"  AUC drop from compression:           {orig_auc - recomp_auc:.4f}")

    results["test4_compression"] = {
        "original_auc": float(orig_auc),
        "recompressed_auc": float(recomp_auc),
        "auc_drop": float(orig_auc - recomp_auc),
    }

    # =========================================================================
    # TEST 6: Visual side-by-side: raw pixels after preprocessing
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 6: Visual side-by-side comparison")
    print("="*70)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle("Row 1: Kaggle Real | Row 2: Kaggle Fake | Row 3: _new_dataset Real | Row 4: _new_dataset Fake", fontsize=14)

    samples = [
        ("Kaggle Real", k_test_real[:5]),
        ("Kaggle Fake", k_test_fake[:5]),
        ("_new_dataset Real", nd_real[:5]),
        ("_new_dataset Fake", nd_fake[:5]),
    ]
    for row, (label, paths) in enumerate(samples):
        for col, p in enumerate(paths):
            img = cv2.imread(str(p))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(img_rgb)
                kb = os.path.getsize(p) / 1024
                axes[row, col].set_title(f"{p.name}\n{img.shape[1]}x{img.shape[0]} | {kb:.1f}KB", fontsize=8)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=0, labelpad=80)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "visual_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'visual_comparison.png'}")

    # =========================================================================
    # TEST 7: Spot-check 10 _new_dataset "fake" labels
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 7: Spot-check 10 _new_dataset 'fake' images")
    print("="*70)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("_new_dataset 'Fake' Label Spot-Check: Are these actually fake?", fontsize=14)
    for i, p in enumerate(nd_fake[:10]):
        row, col = i // 5, i % 5
        img = cv2.imread(str(p))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img_rgb)
            kb = os.path.getsize(p) / 1024
            logit, prob = predict_single(model, p, transform, device)
            axes[row, col].set_title(f"{p.name}\n{kb:.1f}KB | logit={logit:.1f} prob={prob:.4f}", fontsize=8)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fake_label_spotcheck.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fake_label_spotcheck.png'}")

    # =========================================================================
    # TEST 1 & 2: Cross-source tests
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 1 & 2: Cross-source diagnostic")
    print("="*70)

    # Test 2: Use _new_dataset reals (non-Flickr) paired with Kaggle fakes (StyleGAN)
    # This checks: is the model detecting "fake" or just "not Flickr"?
    print("\n  --- Test 2: _new_dataset Reals vs Kaggle Fakes ---")
    print("  (If model still catches Kaggle fakes but also marks _new_dataset reals as real,")
    print("   it's not just a Flickr detector)")
    nd_real_logits2, _ = predict_batch(model, nd_real[:200], transform, device, "_new_dataset Real (non-Flickr)")
    k_fake_logits2, _ = predict_batch(model, k_test_fake[:200], transform, device, "Kaggle Fake (StyleGAN)")

    cross_labels = np.array([0]*len(nd_real_logits2) + [1]*len(k_fake_logits2))
    cross_scores = np.concatenate([nd_real_logits2, k_fake_logits2])
    cross_auc = roc_auc_score(cross_labels, cross_scores)
    print(f"\n  AUC (_new_dataset Real vs Kaggle Fake): {cross_auc:.4f}")
    print(f"  If this is high: model detects StyleGAN artifacts, not just 'not Flickr'")
    print(f"  If this is low: model is a Flickr-vs-StyleGAN classifier")

    results["test2_cross_source"] = {"auc_nd_real_vs_kaggle_fake": float(cross_auc)}

    # Test 1: _new_dataset fakes are presumably non-StyleGAN
    # Check: do Kaggle reals (Flickr) vs _new_dataset fakes (non-StyleGAN) get separated?
    print("\n  --- Test 1: Kaggle Reals vs _new_dataset Fakes ---")
    print("  (Tests if model detects non-StyleGAN fakes)")
    k_real_logits2, _ = predict_batch(model, k_test_real[:200], transform, device, "Kaggle Real (Flickr)")
    nd_fake_logits2, _ = predict_batch(model, nd_fake[:200], transform, device, "_new_dataset Fake (non-StyleGAN?)")

    cross2_labels = np.array([0]*len(k_real_logits2) + [1]*len(nd_fake_logits2))
    cross2_scores = np.concatenate([k_real_logits2, nd_fake_logits2])
    cross2_auc = roc_auc_score(cross2_labels, cross2_scores)
    print(f"\n  AUC (Kaggle Real vs _new_dataset Fake): {cross2_auc:.4f}")
    print(f"  If this is near 0.5: model cannot detect non-StyleGAN fakes at all")

    results["test1_non_stylegan"] = {"auc_kaggle_real_vs_nd_fake": float(cross2_auc)}

    # =========================================================================
    # TEST 8: Calibration paradox
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 8: Calibration paradox explanation")
    print("="*70)
    # Raw sigmoid outputs are already near 0 or 1 (overconfident but correct)
    # Temperature scaling T=4.39 softens them toward 0.5
    # On a near-perfect classifier, raw Brier is lower because predictions are already sharp+correct
    # Calibrated Brier rises because T pushes correct-but-sharp predictions toward uncertainty
    all_logits = np.concatenate([orig_real_logits, orig_fake_logits])
    all_labels = np.concatenate([np.zeros(len(orig_real_logits)), np.ones(len(orig_fake_logits))])
    raw_probs = 1 / (1 + np.exp(-all_logits))
    calib_probs = 1 / (1 + np.exp(-all_logits / 4.3868))

    from sklearn.metrics import brier_score_loss
    raw_brier = brier_score_loss(all_labels, raw_probs)
    calib_brier = brier_score_loss(all_labels, calib_probs)

    print(f"  Raw sigmoid probs:   mean_real={raw_probs[all_labels==0].mean():.6f}, mean_fake={raw_probs[all_labels==1].mean():.6f}")
    print(f"  Calib sigmoid probs: mean_real={calib_probs[all_labels==0].mean():.6f}, mean_fake={calib_probs[all_labels==1].mean():.6f}")
    print(f"  Raw Brier:  {raw_brier:.6f}")
    print(f"  Calib Brier: {calib_brier:.6f}")
    print(f"  Explanation: The classifier's raw sigmoid outputs are already near 0/1 and correct.")
    print(f"  Temperature T=4.39 divides logits by 4.39, softening them AWAY from 0/1.")
    print(f"  Since the predictions were already correct and sharp, softening them INCREASES error.")
    print(f"  This is expected when a classifier is overconfident-but-right on its training distribution.")
    print(f"  It does NOT mean calibration is working well — it means the test set is too easy")
    print(f"  (same distribution as training), so there's nothing for calibration to fix.")

    results["test8_calibration"] = {
        "raw_brier": float(raw_brier), "calib_brier": float(calib_brier),
        "explanation": "T=4.39 softens already-correct sharp predictions, increasing Brier. Expected on in-distribution data."
    }

    # Save all results
    with open(OUTPUT_DIR / "diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {OUTPUT_DIR / 'diagnostic_results.json'}")

if __name__ == "__main__":
    main()
