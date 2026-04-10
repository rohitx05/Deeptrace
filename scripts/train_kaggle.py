"""
Train on Kaggle Real-vs-Fake image dataset.
Uses the V1 DeepfakeDetector in image mode for simplicity.

Usage:
    python scripts/train_kaggle.py
    python scripts/train_kaggle.py --epochs 20 --batch_size 4
    python scripts/train_kaggle.py --data_root data/kaggle_realfake
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader

from models.detector import DeepfakeDetector
from datasets.kaggle_realfake import KaggleRealFakeDataset
from training.trainer import Trainer
from utils.logger import get_logger
from utils.project_memory import ProjectMemory

logger = get_logger("train_kaggle")


def main():
    memory = ProjectMemory()
    memory.load_primary_context(logger)

    parser = argparse.ArgumentParser(description="Train on Kaggle Real-vs-Fake Dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data_root", type=str, default="data/kaggle_realfake")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    image_size = config["data"]["image_size"]

    # Create datasets
    logger.info("Loading Kaggle Real-vs-Fake dataset...")
    train_dataset = KaggleRealFakeDataset(root_dir=args.data_root, split="train", image_size=image_size)
    val_dataset = KaggleRealFakeDataset(root_dir=args.data_root, split="val", image_size=image_size)

    if len(train_dataset) == 0:
        logger.error(f"No training images found in {args.data_root}!")
        logger.error("Expected: {data_root}/real_vs_fake/real-vs-fake/train/real/ and .../fake/")
        return

    logger.info(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )

    # Create model
    logger.info("Building DeepfakeDetector...")
    model = DeepfakeDetector(config=model_config)

    # Stage 1 approach: freeze heavy modules, train spatial + detection head
    model.freeze_module("frequency_encoder")
    model.freeze_module("temporal_model")
    model.freeze_module("physiology_encoder")
    model.freeze_module("clip_alignment")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, stage_name="kaggle_realfake")
    result = trainer.train(resume_from=args.resume)

    best_metrics = dict(result.get("best_val_metrics", {}))
    best_metrics["best_epoch"] = result.get("best_epoch")
    memory.record_training(
        step_name="training:kaggle_realfake",
        checkpoint_path=result.get("best_checkpoint_path", "checkpoints/kaggle_realfake/best_model.pth"),
        dataset_info={
            "active_dataset": "kaggle_realfake",
            "data_root": args.data_root,
            "available_datasets": ["kaggle_realfake"],
            "mode": "image",
            "image_size": config["data"]["image_size"],
            "num_frames": config["data"]["num_frames"],
        },
        training_parameters={
            "stage_name": "kaggle_realfake",
            "batch_size": config["training"]["batch_size"],
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "num_epochs": config["training"]["num_epochs"],
            "use_amp": config["hardware"]["mixed_precision"],
            "device": config["hardware"]["device"],
            "num_workers": config["hardware"]["num_workers"],
        },
        metrics=best_metrics,
        notes=f"kaggle_realfake training complete; best_val_auc={result['best_val_auc']:.4f}",
    )

    logger.info(f"Training complete! Best val AUC: {result['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
