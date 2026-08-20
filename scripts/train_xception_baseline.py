"""
Train and Evaluate Xception Baseline (Rössler et al., 2019) on the exact same Kaggle Real/Fake Split.
Uses timm legacy_xception with ImageNet pretrained initialization.
"""

import sys
import json
import logging
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
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

from scripts.train_mesonet_baseline import FastKaggleRGBDataset
from utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("xception_baseline")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Xception", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images).squeeze(-1)
                probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)
    y_pred = (y_prob >= 0.5).astype(float)

    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=["Real", "Fake"], output_dict=True, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
    }


def main(epochs=5, seed=42, batch_size=64):
    set_seed(seed)
    device = get_device()
    logger.info(f"Training Xception baseline on {device} (seed={seed})")

    # Datasets (image_size=299 for Xception standard, resized efficiently)
    train_ds = FastKaggleRGBDataset(root_dir="data/kaggle_realfake", split="train", image_size=224)
    val_ds = FastKaggleRGBDataset(root_dir="data/kaggle_realfake", split="val", image_size=224)
    test_ds = FastKaggleRGBDataset(root_dir="data/kaggle_realfake", split="test", image_size=224)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Classic Xception (pretrained on ImageNet)
    model = timm.create_model("legacy_xception", pretrained=True, num_classes=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_auc = 0.0
    ckpt_dir = Path(f"checkpoints/xception_baseline_seed{seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Xception Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs).squeeze(-1)
                loss = criterion(logits, lbls)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} Val Acc: {val_metrics['accuracy']*100:.2f}%, Val AUC: {val_metrics['roc_auc']:.4f}")

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save(model.state_dict(), best_ckpt)
            logger.info(f"Saved new best Xception checkpoint (AUC: {best_val_auc:.4f})")

    # Evaluate on Test Set
    logger.info("Loading best Xception checkpoint for test evaluation...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=False))
    test_metrics = evaluate(model, test_loader, device)
    logger.info("=== Xception Test Results ===")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    logger.info(f"Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"Test Brier:    {test_metrics['brier_score']:.4f}")

    results = {
        "model": "Xception (Rössler et al., 2019)",
        "seed": seed,
        "epochs": epochs,
        "params": sum(p.numel() for p in model.parameters()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    out_dir = Path("results/benchmark_eval_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"xception_baseline_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Xception results saved to: {out_file}")


if __name__ == "__main__":
    main()
