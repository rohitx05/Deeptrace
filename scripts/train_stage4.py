"""
Stage 4: Full Multi-Task End-to-End Fine-Tuning.
All modules unfrozen. Full multi-task loss.
Initialized from Stage 3 checkpoint.

Usage:
    python scripts/train_stage4.py --config configs/config.yaml --stage3_ckpt checkpoints/stage3_temporal/best_model.pth
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader, ConcatDataset

from models.detector import DeepfakeDetector
from datasets.faceforensics import FaceForensicsDataset
from datasets.celebdf import CelebDFDataset
from datasets.dfdc import DFDCDataset
from training.trainer import Trainer
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from utils.project_memory import ProjectMemory

logger = get_logger("stage4")


def main():
    memory = ProjectMemory()
    memory.load_primary_context(logger)

    parser = argparse.ArgumentParser(description="Stage 4: Multi-Task End-to-End Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--stage3_ckpt", type=str, default="checkpoints/stage3_temporal/best_model.pth")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    config["training"]["learning_rate"] = 1e-5  # Very low LR for fine-tuning
    if args.mode == "video":
        config["training"]["batch_size"] = 1
        config["training"]["gradient_accumulation_steps"] = 16

    # Build combined dataset from all available sources
    logger.info("Loading datasets...")
    train_datasets = []
    val_datasets = []

    ds_kwargs = {
        "image_size": config["data"]["image_size"],
        "num_frames": config["data"]["num_frames"],
        "mode": args.mode,
    }

    # FaceForensics++
    ff_path = Path(args.data_root) / "FaceForensics++"
    if ff_path.exists():
        train_datasets.append(FaceForensicsDataset(root_dir=args.data_root, split="train", **ds_kwargs))
        val_datasets.append(FaceForensicsDataset(root_dir=args.data_root, split="val", **ds_kwargs))
        logger.info(f"  FaceForensics++: {len(train_datasets[-1])} train, {len(val_datasets[-1])} val")

    # CelebDF
    cdf_path = Path(args.data_root) / "CelebDF"
    if cdf_path.exists():
        train_datasets.append(CelebDFDataset(root_dir=args.data_root, split="train", **ds_kwargs))
        val_datasets.append(CelebDFDataset(root_dir=args.data_root, split="val", **ds_kwargs))
        logger.info(f"  CelebDF: {len(train_datasets[-1])} train, {len(val_datasets[-1])} val")

    # DFDC
    dfdc_path = Path(args.data_root) / "DFDC"
    if dfdc_path.exists():
        train_datasets.append(DFDCDataset(root_dir=args.data_root, split="train", **ds_kwargs))
        val_datasets.append(DFDCDataset(root_dir=args.data_root, split="val", **ds_kwargs))
        logger.info(f"  DFDC: {len(train_datasets[-1])} train, {len(val_datasets[-1])} val")

    if not train_datasets:
        logger.error("No datasets found! Place datasets in data/ directory.")
        return

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    logger.info(f"Combined: {len(train_dataset)} train, {len(val_dataset)} val samples")

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

    # Load Stage 3 weights
    if Path(args.stage3_ckpt).exists():
        logger.info(f"Loading Stage 3 checkpoint: {args.stage3_ckpt}")
        load_checkpoint(args.stage3_ckpt, model)

    # Stage 4: Unfreeze ALL (except CLIP which is always frozen)
    model.unfreeze_module("spatial_encoder")
    model.unfreeze_module("frequency_encoder")
    model.unfreeze_module("temporal_model")
    model.unfreeze_module("physiology_encoder")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Stage 4 trainable params: {trainable / 1e6:.1f}M")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, stage_name="stage4_multitask")
    result = trainer.train(resume_from=args.resume)

    best_metrics = dict(result.get("best_val_metrics", {}))
    best_metrics["best_epoch"] = result.get("best_epoch")
    memory.record_training(
        step_name="training:stage4_multitask",
        checkpoint_path=result.get("best_checkpoint_path", "checkpoints/stage4_multitask/best_model.pth"),
        dataset_info={
            "active_dataset": "multi_dataset",
            "data_root": args.data_root,
            "available_datasets": [name for name, present in {
                "faceforensics": ff_path.exists(),
                "celebdf": cdf_path.exists(),
                "dfdc": dfdc_path.exists(),
            }.items() if present],
            "mode": args.mode,
            "image_size": config["data"]["image_size"],
            "num_frames": config["data"]["num_frames"],
        },
        training_parameters={
            "stage_name": "stage4_multitask",
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
        notes=f"stage4_multitask training complete; best_val_auc={result['best_val_auc']:.4f}",
    )

    logger.info(f"Stage 4 complete! Best val AUC: {result['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
