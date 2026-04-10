"""
Stage 2: Joint Spatial + Frequency Encoder training.
Spatial encoder initialized from Stage 1 checkpoint.
Adds frequency encoder (EfficientNet-B0 on DCT) and CLIP alignment.

Usage:
    python scripts/train_stage2.py --config configs/config.yaml --stage1_ckpt checkpoints/stage1_spatial/best_model.pth
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader

from models.detector import DeepfakeDetector
from datasets.faceforensics import FaceForensicsDataset
from training.trainer import Trainer
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from utils.project_memory import ProjectMemory

logger = get_logger("stage2")


def main():
    memory = ProjectMemory()
    memory.load_primary_context(logger)

    parser = argparse.ArgumentParser(description="Stage 2: Spatial + Frequency Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--stage1_ckpt", type=str, default="checkpoints/stage1_spatial/best_model.pth")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    # Reduce LR for fine-tuning
    config["training"]["learning_rate"] = 5e-5

    # Dataset
    logger.info("Loading dataset...")
    train_dataset = FaceForensicsDataset(
        root_dir=args.data_root, split="train",
        image_size=config["data"]["image_size"], mode="image",
    )
    val_dataset = FaceForensicsDataset(
        root_dir=args.data_root, split="val",
        image_size=config["data"]["image_size"], mode="image",
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"], drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )

    # Model
    logger.info("Building model...")
    model = DeepfakeDetector(config=model_config)

    # Load Stage 1 weights
    if Path(args.stage1_ckpt).exists():
        logger.info(f"Loading Stage 1 checkpoint: {args.stage1_ckpt}")
        load_checkpoint(args.stage1_ckpt, model)

    # Stage 2: Freeze temporal + physiology, train spatial + frequency + CLIP + fusion + head
    model.freeze_module("temporal_model")
    model.freeze_module("physiology_encoder")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Stage 2 trainable params: {trainable / 1e6:.1f}M")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, stage_name="stage2_spatial_frequency")
    result = trainer.train(resume_from=args.resume)

    best_metrics = dict(result.get("best_val_metrics", {}))
    best_metrics["best_epoch"] = result.get("best_epoch")
    memory.record_training(
        step_name="training:stage2_spatial_frequency",
        checkpoint_path=result.get("best_checkpoint_path", "checkpoints/stage2_spatial_frequency/best_model.pth"),
        dataset_info={
            "active_dataset": "faceforensics",
            "data_root": args.data_root,
            "available_datasets": ["faceforensics"],
            "mode": "image",
            "image_size": config["data"]["image_size"],
            "num_frames": config["data"]["num_frames"],
        },
        training_parameters={
            "stage_name": "stage2_spatial_frequency",
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
        notes=f"stage2_spatial_frequency training complete; best_val_auc={result['best_val_auc']:.4f}",
    )

    logger.info(f"Stage 2 complete! Best val AUC: {result['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
