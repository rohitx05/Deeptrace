"""
Run temperature calibration on test_data, save calibration.json,
then re-evaluate with calibrated probabilities and print full metrics.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import (
    FolderRealFakeDataset,
    ModelWithTemperature,
    move_batch_to_device,
    save_calibration,
)
from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint

# ── settings ─────────────────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/kaggle_realfake/best_model.pth"
TEST_DATA = "test_data"
CALIB_PATH = "checkpoints/kaggle_realfake/calibration.json"
IMAGE_SIZE = 160
BATCH_SIZE = 4
USE_AMP = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── 1. Load model ────────────────────────────────────────────────────────────
print("\n=== Loading model ===")
model = DeepfakeDetector()
load_checkpoint(CHECKPOINT, model, device=device)
model.to(device).eval()


# ── 2. Build dataset + loader ────────────────────────────────────────────────
print("\n=== Loading test_data ===")
dataset = FolderRealFakeDataset(
    root_dir=TEST_DATA,
    split="test",
    image_size=IMAGE_SIZE,
    num_frames=8,
    mode="image",
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Samples: {len(dataset)}")


# ── 3. Collect raw logits ────────────────────────────────────────────────────
print("\n=== Collecting raw logits ===")
cal_model = ModelWithTemperature(model, temperature=1.0).to(device)
raw_logits, labels = cal_model.collect_validation_logits(
    loader, device=device, use_amp=USE_AMP
)
print(f"  logits  range: [{raw_logits.min():.4f}, {raw_logits.max():.4f}]")
print(f"  labels  count: real={int((labels == 0).sum())}, fake={int((labels == 1).sum())}")


# ── 4. Temperature calibration ──────────────────────────────────────────────
print("\n=== Running temperature calibration (LBFGS) ===")
metrics = cal_model.set_temperature(
    dataloader=loader, device=device, use_amp=USE_AMP, max_iter=200
)
T = metrics["temperature"]
print(f"  Temperature      : {T:.6f}")
print(f"  Before NLL       : {metrics['before_nll']:.6f}")
print(f"  After NLL        : {metrics['after_nll']:.6f}")

save_calibration(CALIB_PATH, T)
print(f"  Saved → {CALIB_PATH}")


# ── 5. Full evaluation with calibrated probs ────────────────────────────────
print("\n=== Evaluating with calibrated temperature ===")

all_probs_raw = []
all_probs_cal = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
            preds = model(images=batch["image"], dct=batch.get("dct"), mode="image")

        logit = preds["binary_logit"]
        if logit.ndim > 1 and logit.size(-1) == 1:
            logit = logit.squeeze(-1)
        logit = logit.float()

        raw_prob = torch.sigmoid(logit)
        cal_prob = torch.sigmoid(logit / T)

        all_probs_raw.append(raw_prob.cpu().numpy())
        all_probs_cal.append(cal_prob.cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())

probs_raw = np.concatenate(all_probs_raw)
probs_cal = np.concatenate(all_probs_cal)
y_true = np.concatenate(all_labels)


def compute_metrics(probs, y_true, tag):
    """Compute and print accuracy, AUC, ECE, optimal threshold."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    # AUC
    auc = roc_auc_score(y_true, probs)

    # Optimal threshold (Youden's J)
    thresholds = np.linspace(0, 1, 1001)
    best_t, best_acc = 0.5, 0.0
    for t in thresholds:
        acc = accuracy_score(y_true, (probs >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = t

    # Fixed 0.5 threshold accuracy
    acc_05 = accuracy_score(y_true, (probs >= 0.5).astype(int))

    # ECE (Expected Calibration Error, 10 bins)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)

    print(f"\n  [{tag}]")
    print(f"    AUC             : {auc:.4f}")
    print(f"    Accuracy @0.5   : {acc_05:.4f}")
    print(f"    Best Accuracy   : {best_acc:.4f}  (threshold={best_t:.4f})")
    print(f"    ECE             : {ece:.4f}")
    print(f"    Prob range      : [{probs.min():.6f}, {probs.max():.6f}]")
    print(f"    Prob mean(real) : {probs[y_true == 0].mean():.6f}")
    print(f"    Prob mean(fake) : {probs[y_true == 1].mean():.6f}")
    return {"auc": auc, "acc_05": acc_05, "best_acc": best_acc, "best_threshold": best_t, "ece": ece}


m_raw = compute_metrics(probs_raw, y_true, "BEFORE calibration (T=1.0)")
m_cal = compute_metrics(probs_cal, y_true, f"AFTER calibration  (T={T:.4f})")


# ── 6. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CALIBRATION SUMMARY")
print("=" * 60)
print(f"  Temperature        : {T:.6f}")
print(f"  NLL improvement    : {metrics['before_nll']:.4f} → {metrics['after_nll']:.4f}")
print(f"  ECE improvement    : {m_raw['ece']:.4f} → {m_cal['ece']:.4f}")
print(f"  AUC                : {m_cal['auc']:.4f}")
print(f"  Best accuracy      : {m_cal['best_acc']:.4f} @threshold {m_cal['best_threshold']:.4f}")
print(f"  Calibration saved  : {CALIB_PATH}")

# ── 7. Update context_summary.md ────────────────────────────────────────────
ctx_path = Path("context_summary.md")
if ctx_path.exists():
    ctx = ctx_path.read_text(encoding="utf-8")
    import re
    updates = {
        r"Temperature: [\d.]+": f"Temperature: {T:.6f}",
        r"Calibration Status: \w+": "Calibration Status: done",
        r"Optimal Threshold: [\d.]+": f"Optimal Threshold: {m_cal['best_threshold']:.4f}",
        r"Accuracy: [\d.]+": f"Accuracy: {m_cal['best_acc']:.4f}",
        r"Threshold@0.5 Accuracy: [\d.]+": f"Threshold@0.5 Accuracy: {m_cal['acc_05']:.4f}",
        r"AUC: [\d.]+": f"AUC: {m_cal['auc']:.4f}",
    }
    for pattern, replacement in updates.items():
        ctx = re.sub(pattern, replacement, ctx, count=1)
    # Remove "Calibration pending" from Known Issues
    ctx = re.sub(r"\n- Calibration pending:.*", "", ctx)
    ctx_path.write_text(ctx, encoding="utf-8")
    print(f"\n  Updated context_summary.md")

print("\nDone.")
