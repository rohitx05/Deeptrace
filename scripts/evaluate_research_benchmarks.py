"""
Comprehensive Research Evaluation Script.
Computes:
1. Kaggle In-Distribution Validation & Test Metrics (Acc, AUC, Precision, Recall, F1)
2. Brier Scores: Raw vs. Temperature-Calibrated (sklearn.metrics.brier_score_loss)
3. Expected Calibration Error (ECE)
4. Confusion Matrix & Class-wise Classification Report
5. Cross-Dataset Zero-Shot Generalization:
   - data/_new_dataset_extracted/Test (10,905 unseen images)
   - test_data/ (99 images: TPDNE StyleGAN fakes + CelebA reals)
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cv2
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import DeepfakeDetector
from datasets.kaggle_realfake import KaggleRealFakeDataset
from datasets.transforms import get_val_transforms, apply_dct_transform
from utils.checkpoint import load_checkpoint
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_benchmarks")


class GenericFolderDataset(Dataset):
    """Loads a flat directory containing Real/ and Fake/ subfolders."""

    def __init__(self, root_dir: str, image_size: int = 160):
        import os
        self.root = Path(root_dir)
        self.image_size = image_size
        self.transform = get_val_transforms(image_size)
        self.samples = []
        
        seen_paths = set()
        class_counts = {0: 0, 1: 0}

        if self.root.exists() and self.root.is_dir():
            for entry in os.listdir(self.root):
                d = self.root / entry
                if d.is_dir():
                    lower_entry = entry.lower()
                    if lower_entry == "real":
                        label = 0
                    elif lower_entry == "fake":
                        label = 1
                    else:
                        continue
                        
                    for f in d.rglob("*"):
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                            f_res = str(f.resolve())
                            if f_res not in seen_paths:
                                seen_paths.add(f_res)
                                self.samples.append((str(f), label))
                                class_counts[label] += 1

        logger.info(f"Loaded {len(self.samples)} images from {root_dir} (Real: {class_counts[0]}, Fake: {class_counts[1]})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dct = apply_dct_transform(img)
        dct_norm = (dct - dct.mean()) / (dct.std() + 1e-8)
        dct_t = torch.from_numpy(dct_norm).permute(2, 0, 1).float()
        dct_t = F.interpolate(dct_t.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)

        aug = self.transform(image=rgb)
        return {
            "image": aug["image"],
            "dct": dct_t,
            "label": torch.tensor(label, dtype=torch.float32),
            "path": path,
        }


def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def plot_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=12)
    plt.ylabel("True Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot: {save_path}")


@torch.no_grad()
def evaluate_dataloader(model, loader, device, temperature=1.0, threshold_05=0.5,
                        calibrated_threshold=None, use_calibrated=False):
    """Evaluate model on a dataloader.
    
    Returns structured results with SEPARATE sections:
    - calibration: continuous metrics (Brier, ECE) — no threshold
    - classification_raw_05: metrics at raw sigmoid threshold 0.5
    - classification_calibrated: metrics at validation-selected calibrated threshold
    """
    model.eval()
    all_raw_logits = []
    all_labels = []

    for batch in tqdm(loader, desc="Inference"):
        imgs = batch["image"].to(device)
        dcts = batch["dct"].to(device)
        lbls = batch["label"].cpu().numpy()

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            preds = model(images=imgs, dct=dcts, mode="image")
            logits = preds["binary_logit"].squeeze(-1).float().cpu().numpy()

        all_raw_logits.extend(logits)
        all_labels.extend(lbls)

    all_raw_logits = np.array(all_raw_logits)
    all_labels = np.array(all_labels)

    raw_probs = 1.0 / (1.0 + np.exp(-all_raw_logits))
    scaled_logits = all_raw_logits / temperature if temperature != 1.0 else all_raw_logits
    calib_probs = 1.0 / (1.0 + np.exp(-scaled_logits))

    # AUC is threshold-independent — use raw probs (monotonic with logits)
    auc = float(roc_auc_score(all_labels, raw_probs)) if len(np.unique(all_labels)) > 1 else 0.0

    # ── CALIBRATION SECTION (no threshold) ──
    brier_raw = float(brier_score_loss(all_labels, raw_probs))
    brier_calib = float(brier_score_loss(all_labels, calib_probs))
    ece_raw = compute_ece(raw_probs, all_labels)
    ece_calib = compute_ece(calib_probs, all_labels)

    calibration_section = {
        "temperature": temperature,
        "raw_brier": brier_raw,
        "calibrated_brier": brier_calib,
        "raw_ece": ece_raw,
        "calibrated_ece": ece_calib,
        "use_calibrated": use_calibrated,
    }

    # ── CLASSIFICATION AT RAW 0.5 THRESHOLD ──
    preds_05 = (raw_probs >= 0.5).astype(int)
    cm_05 = confusion_matrix(all_labels, preds_05)
    report_05 = classification_report(all_labels, preds_05, target_names=["Real", "Fake"], output_dict=True)

    classification_raw_05 = {
        "threshold": 0.5,
        "probability_source": "raw_sigmoid",
        "accuracy": float(accuracy_score(all_labels, preds_05)),
        "precision": float(precision_score(all_labels, preds_05, zero_division=0)),
        "recall": float(recall_score(all_labels, preds_05, zero_division=0)),
        "f1_score": float(f1_score(all_labels, preds_05, zero_division=0)),
        "confusion_matrix": cm_05.tolist(),
        "classification_report": report_05,
    }

    # ── CLASSIFICATION AT CALIBRATED THRESHOLD (if provided) ──
    classification_calibrated = None
    if calibrated_threshold is not None:
        probs_for_thr = calib_probs if use_calibrated else raw_probs
        preds_cal = (probs_for_thr >= calibrated_threshold).astype(int)
        cm_cal = confusion_matrix(all_labels, preds_cal)
        report_cal = classification_report(all_labels, preds_cal, target_names=["Real", "Fake"], output_dict=True)

        classification_calibrated = {
            "threshold": calibrated_threshold,
            "probability_source": "calibrated_sigmoid" if use_calibrated else "raw_sigmoid",
            "accuracy": float(accuracy_score(all_labels, preds_cal)),
            "precision": float(precision_score(all_labels, preds_cal, zero_division=0)),
            "recall": float(recall_score(all_labels, preds_cal, zero_division=0)),
            "f1_score": float(f1_score(all_labels, preds_cal, zero_division=0)),
            "confusion_matrix": cm_cal.tolist(),
            "classification_report": report_cal,
        }

    return {
        "num_samples": len(all_labels),
        "roc_auc": auc,
        "calibration": calibration_section,
        "classification_raw_05": classification_raw_05,
        "classification_calibrated": classification_calibrated,
        "labels": all_labels,
        "raw_probs": raw_probs,
        "calib_probs": calib_probs,
        "cm_raw_05": cm_05,
    }


def main():
    device = get_device()
    logger.info(f"Using device: {device}")

    # 1. Load Model
    ckpt_path = "checkpoints/v2_clip_finetune/best_model.pth"
    model = DeepfakeDetector()
    load_checkpoint(ckpt_path, model, device=device)
    model.to(device)
    model.eval()

    # Load calibration from SAME directory as checkpoint (not V1's calibration)
    calib_json = Path(ckpt_path).parent / "calibration.json"
    if calib_json.exists():
        with open(calib_json) as f:
            calib_data = json.load(f)
        temperature = float(calib_data.get("temperature", 1.0))
        calibrated_threshold = float(calib_data.get("threshold", 0.5))
        use_calibrated = bool(calib_data.get("use_calibrated_probabilities", False))
        logger.info(f"Loaded calibration from {calib_json}")
    else:
        temperature = 1.0
        calibrated_threshold = 0.5
        use_calibrated = False
        logger.warning(f"No calibration.json found at {calib_json}. Using raw probabilities.")

    logger.info(f"Calibration: T={temperature:.4f}, threshold={calibrated_threshold:.4f}, use_calibrated={use_calibrated}")

    results = {}
    output_dir = Path("results/benchmark_eval_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Evaluate on Kaggle Validation Split
    val_dataset = KaggleRealFakeDataset(root_dir="data/kaggle_realfake", split="val", image_size=160)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    logger.info("Evaluating on Kaggle Validation Split (20,000 samples)...")
    val_res = evaluate_dataloader(model, val_loader, device, temperature, 0.5, calibrated_threshold, use_calibrated)
    plot_confusion_matrix(val_res["cm_raw_05"], ["Real", "Fake"], output_dir / "kaggle_val_cm_raw05.png", "Kaggle Val — Raw Sigmoid ≥ 0.5")
    val_clean = {k: v for k, v in val_res.items() if k not in ("labels", "raw_probs", "calib_probs", "cm_raw_05")}
    results["kaggle_validation"] = val_clean

    # 3. Evaluate on Kaggle Test Split
    test_dataset = KaggleRealFakeDataset(root_dir="data/kaggle_realfake", split="test", image_size=160)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    logger.info("Evaluating on Kaggle Test Split (20,000 samples)...")
    test_res = evaluate_dataloader(model, test_loader, device, temperature, 0.5, calibrated_threshold, use_calibrated)
    plot_confusion_matrix(test_res["cm_raw_05"], ["Real", "Fake"], output_dir / "kaggle_test_cm_raw05.png", "Kaggle Test — Raw Sigmoid ≥ 0.5")
    test_clean = {k: v for k, v in test_res.items() if k not in ("labels", "raw_probs", "calib_probs", "cm_raw_05")}
    results["kaggle_test"] = test_clean

    # 4. Evaluate on Cross-Dataset: New Unseen Dataset
    new_test_dir = Path("data/_new_dataset_extracted/Test")
    if new_test_dir.exists():
        logger.info("Evaluating on _new_dataset (exploratory — labels unverified)...")
        cross1_dataset = GenericFolderDataset(new_test_dir, image_size=160)
        cross1_loader = DataLoader(cross1_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
        cross1_res = evaluate_dataloader(model, cross1_loader, device, temperature, 0.5, calibrated_threshold, use_calibrated)
        cross1_clean = {k: v for k, v in cross1_res.items() if k not in ("labels", "raw_probs", "calib_probs", "cm_raw_05")}
        cross1_clean["caveat"] = "Labels unverified. N corrected from 21810 to actual. Exploratory only."
        results["cross_dataset_new_dataset"] = cross1_clean

    # 5. Evaluate on test_data/ (TPDNE StyleGAN + CelebA)
    test_data_dir = Path("test_data")
    if test_data_dir.exists():
        logger.info("Evaluating on test_data/ (same-generator exploratory)...")
        cross2_dataset = GenericFolderDataset(test_data_dir, image_size=160)
        cross2_loader = DataLoader(cross2_dataset, batch_size=32, shuffle=False, num_workers=0)
        cross2_res = evaluate_dataloader(model, cross2_loader, device, temperature, 0.5, calibrated_threshold, use_calibrated)
        cross2_clean = {k: v for k, v in cross2_res.items() if k not in ("labels", "raw_probs", "calib_probs", "cm_raw_05")}
        cross2_clean["caveat"] = "TPDNE is StyleGAN2 — same generator family as training. NOT cross-dataset."
        results["cross_dataset_test_data"] = cross2_clean

    # Save complete JSON (new directory, never overwrite old results)
    out_json = output_dir / "research_benchmarks_v2_corrected.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"All corrected results exported to: {out_json}")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        print("Running smoke test...")
        d1 = GenericFolderDataset("test_data", 160)
        print(f"test_data N={len(d1)}")
        try:
            d2 = GenericFolderDataset("data/_new_dataset_extracted/Test", 160)
            print(f"data/_new_dataset_extracted/Test N={len(d2)}")
        except Exception as e:
            print(f"Could not load new dataset: {e}")
    else:
        main()
