"""
check_and_train_new_dataset.py
==============================
1. Extract the new Kaggle zip (datasetkeagle.zip) to a temp location
2. Fingerprint both datasets (existing + new) using image count & file hashes
3. If datasets match → skip training, just re-calibrate + fix calibration.json
4. If datasets differ → fine-tune the existing model on the new data and re-calibrate

Run from project root:
    python scripts/check_and_train_new_dataset.py
    python scripts/check_and_train_new_dataset.py --zip_path "C:/Users/Udit/Desktop/datasetkeagle.zip"
    python scripts/check_and_train_new_dataset.py --force_train   (skip sameness check)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import hashlib
import argparse
import json
import shutil
import zipfile
import logging
import tempfile
from pathlib import Path

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("check_and_train")

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_ZIP      = r"C:\Users\Udit\Desktop\datasetkeagle.zip"
EXISTING_DATA    = ROOT / "data" / "kaggle_realfake"
CHECKPOINT_PATH  = ROOT / "checkpoints" / "kaggle_realfake" / "best_model.pth"
CALIB_PATH       = ROOT / "checkpoints" / "kaggle_realfake" / "calibration.json"
CONFIG_PATH      = ROOT / "configs" / "config.yaml"
MODEL_CFG_PATH   = ROOT / "configs" / "model_config.yaml"
IMAGE_SIZE       = 160
BATCH_SIZE       = 8           # increased for RTX 4050 GPU
FINETUNE_EPOCHS  = 10          # fine-tune epochs if dataset is new
FINETUNE_LR      = 5e-5        # lower LR for fine-tuning

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Fingerprinting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _collect_image_paths(root: Path) -> list[Path]:
    """Recursively collect all image paths under root."""
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )


def _file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """MD5 of first 1 MB of a file (fast fingerprint)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        data = f.read(chunk)
        h.update(data)
    return h.hexdigest()


def fingerprint_dataset(root: Path) -> dict:
    """
    Returns a fingerprint dict:
        {"real": N, "fake": M, "sample_hashes": [...]}
    where sample_hashes is the MD5-of-first-MB for up to 200 random images.
    """
    paths = _collect_image_paths(root)
    real_count = sum(1 for p in paths if "real" in p.parts[-2].lower())
    fake_count = len(paths) - real_count

    # Sample a stable (sorted) subset for hashing
    sampled = paths[::max(1, len(paths) // 200)][:200]
    hashes = [_file_hash(p) for p in sampled]

    return {
        "total": len(paths),
        "real": real_count,
        "fake": fake_count,
        "sample_hashes": hashes,
    }


def datasets_are_same(fp_existing: dict, fp_new: dict, tolerance: float = 0.05) -> bool:
    """
    Return True if the two datasets look identical:
      - image counts differ by < tolerance
      - overlapping sample hashes > 80%
    """
    if fp_existing["total"] == 0 or fp_new["total"] == 0:
        return False

    count_ratio = abs(fp_existing["total"] - fp_new["total"]) / max(fp_existing["total"], fp_new["total"])
    if count_ratio > tolerance:
        logger.info(f"Image count differs: {fp_existing['total']} vs {fp_new['total']} ({count_ratio*100:.1f}%)")
        return False

    set_existing = set(fp_existing["sample_hashes"])
    set_new = set(fp_new["sample_hashes"])
    if len(set_existing) == 0:
        return False
    overlap = len(set_existing & set_new) / len(set_existing)
    logger.info(f"Hash overlap: {overlap*100:.1f}%  ({len(set_existing & set_new)} / {len(set_existing)})")
    return overlap >= 0.80


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Dataset structure detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_kaggle_root(extracted_root: Path) -> Path | None:
    """
    Find the directory containing real/fake subdirectories inside an extracted zip.
    Handles lowercase (real/fake), capitalized (Real/Fake), and split-based layouts.
    """
    # Pattern 1: direct real/ fake/ (any casing)
    subs_lower = {p.name.lower(): p for p in extracted_root.iterdir() if p.is_dir()}
    if "real" in subs_lower and "fake" in subs_lower:
        return extracted_root

    # Pattern 2: Train/Real, Train/Fake (capitalized) — return root
    for train_name in ("Train", "train", "TRAIN"):
        train_dir = extracted_root / train_name
        if train_dir.exists():
            subs = {p.name.lower() for p in train_dir.iterdir() if p.is_dir()}
            if "real" in subs and "fake" in subs:
                return extracted_root  # root has Train/Validation/Test

    # Pattern 3: nested real_vs_fake/real-vs-fake/
    for candidate in extracted_root.rglob("real_vs_fake"):
        nested = candidate / "real-vs-fake"
        if nested.exists():
            return nested

    # Pattern 4: walk subdirs and find first with real+fake children
    for candidate in sorted(extracted_root.rglob("*")):
        if not candidate.is_dir():
            continue
        subs = {p.name.lower() for p in candidate.iterdir() if p.is_dir()}
        if "real" in subs and "fake" in subs:
            return candidate

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Calibration helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[(tpr - fpr).argmax()])


def run_calibration(model, loader, device, calib_path: Path, use_amp: bool = True):
    """Run temperature scaling and save calibration.json with optimal_threshold."""
    from calibration import ModelWithTemperature, save_calibration

    cal = ModelWithTemperature(model, temperature=1.0).to(device)
    logger.info("Running temperature calibration (LBFGS)…")
    metrics = cal.set_temperature(loader, device=device, use_amp=use_amp, max_iter=200)

    T   = metrics["temperature"]
    thr = metrics.get("threshold")
    if thr is None:
        # Collect probs and compute manually
        all_probs, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    preds = model(images=batch["image"], dct=batch.get("dct"), mode="image")
                logit = preds["binary_logit"]
                if logit.ndim > 1:
                    logit = logit.squeeze(-1)
                prob = torch.sigmoid(logit.float() / T)
                all_probs.append(prob.cpu().numpy())
                all_labels.append(batch["label"].cpu().numpy())
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        thr = compute_optimal_threshold_youden(labels, probs)

    save_calibration(str(calib_path), T, thr)
    logger.info(f"Calibration saved → {calib_path}")
    logger.info(f"  Temperature     : {T:.6f}")
    logger.info(f"  Optimal threshold: {thr:.4f}")
    return T, thr


def fix_calibration_json_if_needed(calib_path: Path):
    """
    If calibration.json exists but is missing 'optimal_threshold' / 'threshold',
    add the fallback value from config.yaml so the pipeline works correctly.
    """
    if not calib_path.exists():
        logger.warning(f"calibration.json not found at {calib_path}")
        return
    with open(calib_path) as f:
        payload = json.load(f)
    if "threshold" not in payload and "optimal_threshold" not in payload:
        # Read from config as fallback
        import yaml
        with open(CONFIG_PATH) as g:
            cfg = yaml.safe_load(g)
        thr = cfg.get("evaluation", {}).get("threshold", 0.1341)
        payload["threshold"] = float(thr)
        with open(calib_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Patched calibration.json → added threshold={thr}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Quick evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_loader(model, loader, device, temperature: float, threshold: float, use_amp: bool = True):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                preds = model(images=batch["image"], dct=batch.get("dct"), mode="image")
            logit = preds["binary_logit"]
            if logit.ndim > 1:
                logit = logit.squeeze(-1)
            prob = torch.sigmoid(logit.float() / temperature)
            all_probs.append(prob.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= threshold).astype(int)
    acc    = accuracy_score(labels, preds)
    auc    = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return {"accuracy": acc, "auc": auc, "n": len(labels)}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Fine-tune
# ══════════════════════════════════════════════════════════════════════════════

def finetune_on_dataset(model, train_loader, val_loader, device, use_amp, epochs, lr,
                        checkpoint_path, start_epoch: int = 0):
    """Fine-tune with tqdm progress bars, ETA, and per-epoch crash-safe checkpoints."""
    import time
    from torch import nn
    from tqdm import tqdm
    from training.losses import DeepfakeLoss

    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DeepfakeLoss(binary_weight=1.0, manipulation_weight=0.5,
                             clip_alignment_weight=0.0, consistency_weight=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # ── Resume optimizer/scheduler state if mid-run ──────────────────────────
    resume_path = Path(checkpoint_path).parent / "finetune_resume.pth"
    best_auc    = 0.0
    if start_epoch > 0 and resume_path.exists():
        state = torch.load(str(resume_path), map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_auc = state.get("best_auc", 0.0)
        logger.info(f"Resumed optimizer state from epoch {start_epoch}")

    n_train = len(train_loader)
    n_val   = len(val_loader)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # ── TRAIN ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            unit="batch",
            ncols=110,
            colour="cyan",
        )
        for batch in train_bar:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                preds = model(images=batch["image"], dct=batch.get("dct"), mode="image")
                targets = {"label": batch["label"], "manipulation_type": batch["manipulation_type"]}
                loss = criterion(preds, targets)["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            prob = preds["binary_pred"].detach().cpu().numpy()
            all_probs.extend(prob)
            all_labels.extend(batch["label"].cpu().numpy())

            # Live loss in bar
            train_bar.set_postfix(loss=f"{loss.item():.4f}",
                                  avg=f"{total_loss/max(train_bar.n,1):.4f}")

        train_acc = accuracy_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))
        scheduler.step()
        train_elapsed = time.time() - epoch_start

        # ── VALIDATE ──────────────────────────────────────────────────────────
        model.eval()
        val_probs, val_labels = [], []

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Val]  ",
            unit="batch",
            ncols=110,
            colour="green",
        )
        with torch.no_grad():
            for batch in val_bar:
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    preds = model(images=batch["image"], dct=batch.get("dct"), mode="image")
                val_probs.extend(preds["binary_pred"].cpu().numpy())
                val_labels.extend(batch["label"].cpu().numpy())

        val_acc = accuracy_score(val_labels, (np.array(val_probs) >= 0.5).astype(int))
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else float("nan")
        total_elapsed = time.time() - epoch_start
        mins, secs = divmod(int(total_elapsed), 60)

        logger.info(
            f"\n{'='*70}\n"
            f"  Epoch {epoch+1}/{epochs} complete  [{mins}m {secs}s]\n"
            f"  Train  loss={total_loss/n_train:.4f}  acc={train_acc*100:.1f}%\n"
            f"  Val    acc={val_acc*100:.1f}%  AUC={val_auc:.4f}\n"
            f"{'='*70}"
        )

        # ── Save crash-recovery state every epoch ─────────────────────────────
        torch.save({
            "epoch":      epoch + 1,
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "scaler":     scaler.state_dict(),
            "best_auc":   best_auc,
        }, str(resume_path))

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_auc > best_auc:
            best_auc = val_auc
            from utils.checkpoint import save_checkpoint
            save_checkpoint(model, optimizer, epoch, {"val_auc": val_auc},
                            str(checkpoint_path), scaler, scheduler)
            logger.info(f"  ✅ New best AUC={best_auc:.4f} → checkpoint saved")
        else:
            logger.info(f"  ➖ AUC={val_auc:.4f} (best={best_auc:.4f})")

    # Clean up resume file when done
    if resume_path.exists():
        resume_path.unlink()
    return best_auc


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Dataset builder for the new zip
# ══════════════════════════════════════════════════════════════════════════════

class GenericSplitDataset(torch.utils.data.Dataset):
    """
    Handles any folder layout where split/label dirs may be capitalized or lowercase.
    e.g., Train/Real, Train/Fake, Validation/Real, Test/Fake, train/real, valid/fake …
    """
    IMAGE_EXTS = IMAGE_EXTS

    # Candidate names for each semantic split
    SPLIT_ALIASES = {
        "train": ["Train", "train", "TRAIN"],
        "val":   ["Validation", "Valid", "valid", "val", "VAL"],
        "test":  ["Test", "test", "TEST"],
    }
    LABEL_ALIASES = {
        0: ["Real", "real", "REAL"],
        1: ["Fake", "fake", "FAKE"],
    }

    def __init__(self, root: Path, split: str = "train", image_size: int = 160):
        from datasets.transforms import get_train_transforms, get_val_transforms, apply_dct_transform as _dct
        self.image_size = image_size
        self.apply_dct = _dct
        self.transform = get_train_transforms(image_size) if split == "train" else get_val_transforms(image_size)
        self.samples = []

        # Resolve the split directory
        split_dir = None
        for name in self.SPLIT_ALIASES.get(split, [split]):
            candidate = root / name
            if candidate.exists():
                split_dir = candidate
                break

        if split_dir is None:
            # Fallback: if root itself has real/fake directly
            subs = {p.name.lower() for p in root.iterdir() if p.is_dir()}
            if "real" in subs and "fake" in subs:
                split_dir = root

        if split_dir is None:
            logger.warning(f"Split dir not found for '{split}' under {root}")
            return

        for label_id, aliases in self.LABEL_ALIASES.items():
            label_dir = None
            for name in aliases:
                candidate = split_dir / name
                if candidate.exists():
                    label_dir = candidate
                    break
            if label_dir is None:
                logger.warning(f"Label dir not found (label={label_id}) under {split_dir}")
                continue
            for img in sorted(label_dir.rglob("*")):
                if img.suffix.lower() in self.IMAGE_EXTS:
                    manip = "Deepfakes" if label_id == 1 else "real"
                    self.samples.append({"path": str(img), "label": label_id, "manipulation_type": manip})

        logger.info(f"GenericSplitDataset [{split}]: {len(self.samples)} images from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        sample = self.samples[idx]
        image = cv2.imread(sample["path"])
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        face = cv2.resize(image, (self.image_size, self.image_size))
        dct = self.apply_dct(face)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=rgb)
        tensor = augmented["image"]
        dct_normalized = (dct - dct.mean()) / (dct.std() + 1e-8)
        dct_tensor = torch.from_numpy(dct_normalized).permute(2, 0, 1).float()
        dct_tensor = torch.nn.functional.interpolate(
            dct_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        ).squeeze(0)
        manip_map = {"real": 0, "Deepfakes": 1}
        return {
            "image": tensor,
            "dct": dct_tensor,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "manipulation_type": torch.tensor(manip_map.get(sample["manipulation_type"], 1), dtype=torch.long),
            "path": sample["path"],
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Dataset loader builder
# ══════════════════════════════════════════════════════════════════════════════

def build_loaders_from_root(data_root: Path, image_size: int, batch_size: int,
                            num_workers: int = 4):
    """
    Build train + val + test loaders using GenericSplitDataset which handles
    any casing of split/label directories (Train/Real, train/real, Validation/Fake …).
    num_workers=4 feeds the GPU without starving it.
    """
    def _loader(ds, shuffle):
        if len(ds) == 0:
            raise ValueError(f"Empty dataset for loader (shuffle={shuffle}) under {data_root}")
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True,
                         persistent_workers=(num_workers > 0), drop_last=shuffle)

    train_ds = GenericSplitDataset(data_root, split="train", image_size=image_size)
    val_ds   = GenericSplitDataset(data_root, split="val",   image_size=image_size)
    test_ds  = GenericSplitDataset(data_root, split="test",  image_size=image_size)

    if len(train_ds) == 0:
        logger.info("No named splits found — splitting full dataset 80/10/10.")
        from calibration import FolderRealFakeDataset
        from torch.utils.data import random_split
        full_ds = FolderRealFakeDataset(str(data_root), split="train", image_size=image_size)
        n = len(full_ds)
        if n == 0:
            raise ValueError(f"No images found under {data_root}")
        n_train = int(0.8 * n)
        n_val   = int(0.1 * n)
        n_test  = n - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Check new dataset vs existing and conditionally train")
    parser.add_argument("--zip_path",    type=str, default=DEFAULT_ZIP)
    parser.add_argument("--existing_data", type=str, default=str(EXISTING_DATA))
    parser.add_argument("--checkpoint",  type=str, default=str(CHECKPOINT_PATH))
    parser.add_argument("--force_train", action="store_true", help="Skip sameness check and always train")
    parser.add_argument("--epochs",       type=int,   default=FINETUNE_EPOCHS)
    parser.add_argument("--lr",           type=float, default=FINETUNE_LR)
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--num_workers",  type=int,   default=4,
                        help="DataLoader workers (default 4, more = better GPU util)")
    parser.add_argument("--resume_epoch", type=int,   default=0,
                        help="Resume training from this epoch (0 = start fresh)")
    args = parser.parse_args()

    zip_path      = Path(args.zip_path)
    existing_data = Path(args.existing_data)
    checkpoint    = Path(args.checkpoint)
    calib_path    = checkpoint.parent / "calibration.json"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    logger.info(f"Device: {device}")

    # ── Step 0: Always fix missing optimal_threshold in calibration.json ─────
    logger.info("\n=== Step 0: Verifying calibration.json ===")
    fix_calibration_json_if_needed(calib_path)

    # ── Step 1: Extract zip ──────────────────────────────────────────────────
    logger.info(f"\n=== Step 1: Extracting {zip_path.name} ===")
    if not zip_path.exists():
        logger.error(f"ZIP not found: {zip_path}")
        sys.exit(1)

    extract_dir = ROOT / "data" / "_new_dataset_extracted"
    if extract_dir.exists() and any(extract_dir.iterdir()):
        logger.info(f"Reusing previously extracted data at {extract_dir} (delete it to re-extract).")
    else:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        logger.info(f"Extracting to {extract_dir} …")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(extract_dir))
        logger.info("Extraction complete.")

    # ── Step 2: Detect dataset root ─────────────────────────────────────────
    logger.info("\n=== Step 2: Detecting dataset structure ===")
    new_data_root = detect_kaggle_root(extract_dir)
    if new_data_root is None:
        logger.error("Could not find real/fake structure in the extracted zip!")
        logger.error("Contents of extracted dir:")
        for p in extract_dir.rglob("*"):
            logger.error(f"  {p.relative_to(extract_dir)}")
        sys.exit(1)
    logger.info(f"New data root detected: {new_data_root}")

    # ── Step 3: Fingerprint both datasets ───────────────────────────────────
    logger.info("\n=== Step 3: Fingerprinting datasets ===")
    fp_existing = fingerprint_dataset(existing_data)
    fp_new      = fingerprint_dataset(new_data_root)
    logger.info(f"Existing dataset : {fp_existing['total']} images  (real={fp_existing['real']} fake={fp_existing['fake']})")
    logger.info(f"New dataset      : {fp_new['total']} images  (real={fp_new['real']} fake={fp_new['fake']})")

    same = datasets_are_same(fp_existing, fp_new) if not args.force_train else False

    # ── Load model (always needed) ───────────────────────────────────────────
    logger.info("\n=== Loading model ===")
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    with open(MODEL_CFG_PATH) as f:
        model_config = yaml.safe_load(f)

    from models.detector import DeepfakeDetector
    from utils.checkpoint import load_checkpoint

    model = DeepfakeDetector(config=model_config)
    load_checkpoint(str(checkpoint), model, device=device)
    model.to(device).eval()
    logger.info("Model loaded.")

    # ── Step 4 (branch): same vs different ──────────────────────────────────
    if same:
        logger.info("\n=== Step 4: Datasets are the SAME — skipping training ===")
        logger.info("Re-calibrating on new data to ensure calibration.json is fresh…")

        # Build a val loader from the new data
        _, val_loader, test_loader = build_loaders_from_root(new_data_root, IMAGE_SIZE, args.batch_size)

        T, thr = run_calibration(model, val_loader, device, calib_path, use_amp)
        logger.info("Evaluating on new test set…")
        metrics = evaluate_on_loader(model, test_loader, device, T, thr, use_amp)
        logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}%  AUC: {metrics['auc']:.4f}  (n={metrics['n']})")

    else:
        logger.info("\n=== Step 4: Datasets are DIFFERENT — fine-tuning on new data ===")

        train_loader, val_loader, test_loader = build_loaders_from_root(
            new_data_root, IMAGE_SIZE, args.batch_size, num_workers=args.num_workers
        )
        logger.info(
            f"Train batches: {len(train_loader)}, "
            f"Val batches: {len(val_loader)}, "
            f"Test batches: {len(test_loader)}"
        )

        # ── Auto-detect resume epoch from saved state ─────────────────────────
        resume_path  = checkpoint.parent / "finetune_resume.pth"
        start_epoch  = args.resume_epoch
        if start_epoch == 0 and resume_path.exists():
            state = torch.load(str(resume_path), map_location="cpu")
            start_epoch = state.get("epoch", 0)
            logger.info(f"Auto-detected resume state: starting from epoch {start_epoch+1}")

        logger.info(f"\nFine-tuning for {args.epochs} epochs (start={start_epoch+1}) at LR={args.lr}…")
        best_auc = finetune_on_dataset(
            model, train_loader, val_loader, device, use_amp,
            epochs=args.epochs, lr=args.lr, checkpoint_path=checkpoint,
            start_epoch=start_epoch,
        )
        logger.info(f"Fine-tuning complete. Best val AUC={best_auc:.4f}")

        # Reload best checkpoint
        logger.info("Reloading best checkpoint…")
        load_checkpoint(str(checkpoint), model, device=device)
        model.to(device).eval()

        # Re-calibrate
        logger.info("\nRe-calibrating temperature on val set…")
        T, thr = run_calibration(model, val_loader, device, calib_path, use_amp)

        # Evaluate on test set
        logger.info("\nEvaluating on test set…")
        metrics = evaluate_on_loader(model, test_loader, device, T, thr, use_amp)
        logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}%  AUC: {metrics['auc']:.4f}  (n={metrics['n']})")

        # Persist dataset fingerprint for future runs
        fp_record = {
            "new_dataset_root": str(new_data_root),
            "total": fp_new["total"],
            "real": fp_new["real"],
            "fake": fp_new["fake"],
        }
        (checkpoint.parent / "dataset_fingerprint.json").write_text(
            json.dumps(fp_record, indent=2)
        )
        logger.info(f"Dataset fingerprint saved → {checkpoint.parent / 'dataset_fingerprint.json'}")

    logger.info("\n✅ Done! Calibration.json is now up-to-date.")
    logger.info(f"   Checkpoint      : {checkpoint}")
    logger.info(f"   Calibration     : {calib_path}")
    with open(calib_path) as f:
        logger.info(f"   Contents        : {json.load(f)}")


if __name__ == "__main__":
    main()
