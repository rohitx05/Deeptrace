"""
Train and Evaluate MesoNet-4 Baseline on the exact same Kaggle Real/Fake Split.
Optimized fast loader (pure RGB, no DCT calculation).
"""

import sys
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mesonet import Meso4
from datasets.transforms import get_train_transforms, get_val_transforms
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mesonet_baseline")


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FastKaggleRGBDataset(Dataset):
    """Fast RGB-only dataset for Kaggle real-vs-fake (skips DCT)."""

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 160):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.is_train = split == "train"

        if self.is_train:
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        folder_name = "valid" if split == "val" else split
        img_root = self.root_dir / "real_vs_fake" / "real-vs-fake" / folder_name
        self.samples = []

        for label_name, label_id in (("real", 0), ("fake", 1)):
            label_dir = img_root / label_name
            if label_dir.exists():
                for p in label_dir.glob("*.jpg"):
                    self.samples.append((str(p), label_id))

        logger.info(f"FastKaggleRGBDataset [{split}]: {len(self.samples)} images")

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
        aug = self.transform(image=rgb)
        return {
            "image": aug["image"],
            "label": torch.tensor(label, dtype=torch.float32),
        }


def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
            all_logits.extend(logits.float().cpu().numpy())
            all_labels.extend(lbls.float().cpu().numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    auc = float(roc_auc_score(all_labels, probs))
    acc = float(accuracy_score(all_labels, preds))
    prec = float(precision_score(all_labels, preds, zero_division=0))
    rec = float(recall_score(all_labels, preds, zero_division=0))
    f1 = float(f1_score(all_labels, preds, zero_division=0))
    brier = float(brier_score_loss(all_labels, probs))
    cm = confusion_matrix(all_labels, preds).tolist()
    report = classification_report(all_labels, preds, target_names=["Real", "Fake"], output_dict=True)

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "brier_score": brier,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def main(seed=42):
    set_seed(seed)
    device = get_device()
    logger.info(f"Training MesoNet-4 baseline on {device} (seed={seed})")

    # Datasets
    train_ds = FastKaggleRGBDataset(root_dir="data/kaggle_realfake", split="train", image_size=160)
    val_ds = FastKaggleRGBDataset(root_dir="data/kaggle_realfake", split="val", image_size=160)
    test_ds = FastKaggleRGBDataset(root_dir="data/kaggle_realfake", split="test", image_size=160)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = Meso4().to(device)
    # Original MesoNet paper: Adam, lr=0.001, no weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_auc = 0.0
    ckpt_dir = Path(f"checkpoints/mesonet_baseline_seed{seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"MesoNet Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
                loss = criterion(logits, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(lbls)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == lbls).sum().item()
            total += len(lbls)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total*100:.2f}%"})

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} Val Acc: {val_metrics['accuracy']*100:.2f}%, Val AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"Saved new best MesoNet checkpoint (AUC: {best_val_auc:.4f})")

    # Evaluate on Test Set
    logger.info("Loading best MesoNet checkpoint for test evaluation...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False))
    test_metrics = evaluate(model, test_loader, device)
    logger.info(f"=== MesoNet-4 Test Results ===")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"Test Brier:    {test_metrics['brier_score']:.4f}")

    results = {
        "model": "MesoNet-4 (Afchar et al., 2018)",
        "seed": seed,
        "epochs": epochs,
        "optimizer": "Adam(lr=0.001, no weight_decay)",
        "scheduler": f"CosineAnnealingLR(T_max={epochs})",
        "params": sum(p.numel() for p in model.parameters()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    out_dir = Path("results/benchmark_eval_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"mesonet_baseline_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"MesoNet results saved to: {out_file}")
    return test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)
