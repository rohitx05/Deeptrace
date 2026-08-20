"""
Chunk 0c: Refit calibration for the exact V2 CLIP checkpoint.

Protocol:
1. Run V2 model on Kaggle validation set → collect raw logits
2. Fit temperature scaling on validation logits only
3. Find optimal threshold on validation only
4. Save calibration.json next to V2 checkpoint with full provenance
5. Report raw vs calibrated Brier/ECE on VALIDATION (for fitting)

The TEST set is NOT used here. Test metrics come from chunk 0d.
"""

import sys
import json
import hashlib
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from datasets.kaggle_realfake import KaggleRealFakeDataset
from utils.checkpoint import load_checkpoint
from utils.device import get_device


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        prop = np.mean(mask)
        if prop > 0:
            acc = np.mean(labels[mask])
            conf = np.mean(probs[mask])
            ece += np.abs(conf - acc) * prop
    return float(ece)


def nll_with_temperature(t, logits, labels):
    """Negative log-likelihood at temperature t."""
    scaled = logits / t
    probs = 1.0 / (1.0 + np.exp(-scaled))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(log_loss(labels, probs))


def find_optimal_threshold(probs, labels):
    """Find threshold that maximizes Youden's J statistic on validation."""
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_j, best_t = -1, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t)


def main():
    device = get_device()
    ckpt_path = Path("checkpoints/v2_clip_finetune/best_model.pth")

    # Compute checkpoint hash
    sha256 = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    ckpt_hash = sha256.hexdigest().upper()[:16]
    print(f"Checkpoint: {ckpt_path}")
    print(f"SHA-256 prefix: {ckpt_hash}")

    # Load model
    model = DeepfakeDetector()
    load_checkpoint(str(ckpt_path), model, device=device)
    model.to(device)
    model.eval()

    # Collect validation logits
    val_dataset = KaggleRealFakeDataset(root_dir="data/kaggle_realfake", split="val", image_size=160)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation inference"):
            imgs = batch["image"].to(device)
            dcts = batch["dct"].to(device)
            lbls = batch["label"].cpu().numpy()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(images=imgs, dct=dcts, mode="image")
                logits = out["binary_logit"].squeeze(-1).float().cpu().numpy()
            all_logits.extend(logits)
            all_labels.extend(lbls)

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    print(f"\nValidation set: N={len(all_logits)}")

    # Fit temperature on validation via NLL minimization
    result = minimize_scalar(
        nll_with_temperature,
        bounds=(0.1, 20.0),
        args=(all_logits, all_labels),
        method="bounded",
    )
    fitted_temperature = float(result.x)
    print(f"Fitted temperature: {fitted_temperature:.6f}")

    # Compute raw and calibrated metrics on VALIDATION
    raw_probs = 1.0 / (1.0 + np.exp(-all_logits))
    calib_probs = 1.0 / (1.0 + np.exp(-all_logits / fitted_temperature))

    raw_brier = brier_score_loss(all_labels, raw_probs)
    calib_brier = brier_score_loss(all_labels, calib_probs)
    raw_ece = compute_ece(raw_probs, all_labels)
    calib_ece = compute_ece(calib_probs, all_labels)
    raw_auc = roc_auc_score(all_labels, raw_probs)

    print(f"\n--- Validation Calibration Metrics ---")
    print(f"Raw Brier:        {raw_brier:.6f}")
    print(f"Calibrated Brier: {calib_brier:.6f}")
    print(f"Raw ECE:          {raw_ece:.6f}")
    print(f"Calibrated ECE:   {calib_ece:.6f}")
    print(f"Raw AUC:          {raw_auc:.6f}")

    # Decision: deploy calibrated probabilities only if Brier improves
    use_calibrated = calib_brier < raw_brier
    print(f"\nCalibration improves Brier on validation: {use_calibrated}")
    if not use_calibrated:
        print("DECISION: Will report raw probabilities as primary. Calibrated shown for reference only.")

    # Find optimal threshold on validation (using whichever probs are primary)
    primary_probs = calib_probs if use_calibrated else raw_probs
    optimal_threshold = find_optimal_threshold(primary_probs, all_labels)
    print(f"Optimal threshold (validation, Youden's J): {optimal_threshold:.4f}")

    # Save calibration sidecar
    calib_data = {
        "temperature": fitted_temperature,
        "threshold": optimal_threshold,
        "checkpoint_sha256_prefix": ckpt_hash,
        "fitted_on": "kaggle_realfake_validation",
        "validation_n": int(len(all_labels)),
        "use_calibrated_probabilities": use_calibrated,
        "validation_metrics": {
            "raw_brier": float(raw_brier),
            "calibrated_brier": float(calib_brier),
            "raw_ece": float(raw_ece),
            "calibrated_ece": float(calib_ece),
            "raw_auc": float(raw_auc),
        },
        "decision": (
            "Deploy calibrated probabilities" if use_calibrated
            else "Deploy raw probabilities (calibration worsened Brier on validation)"
        ),
    }

    out_path = ckpt_path.parent / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Also save raw logits for any future reuse
    np.savez(
        ckpt_path.parent / "val_logits.npz",
        logits=all_logits,
        labels=all_labels,
    )
    print(f"Saved: {ckpt_path.parent / 'val_logits.npz'}")


if __name__ == "__main__":
    main()
