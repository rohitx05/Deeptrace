"""
V2 Training Pipeline — 6-Stage Orchestrator.
Single entry point for all training stages of DeepfakeDetectorV2.

Usage:
    python scripts/train_v2.py --stage 1 --data_root data/
    python scripts/train_v2.py --stage 2 --prev_ckpt checkpoints/v2_stage1/best_model.pth
    python scripts/train_v2.py --stage 5 --prev_ckpt checkpoints/v2_stage4/best_model.pth
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader, ConcatDataset

from models.detector_v2 import DeepfakeDetectorV2
from datasets.faceforensics import FaceForensicsDataset
from datasets.celebdf import CelebDFDataset
from datasets.dfdc import DFDCDataset
from training.trainer_v2 import TrainerV2
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from utils.project_memory import ProjectMemory

logger = get_logger("train_v2")


STAGE_CONFIG = {
    1: {
        "name": "v2_stage1_spatial",
        "description": "Spatial encoder + detection head only",
        "lr": 1e-4,
        "mode": "image",
        "batch_size": 2,
        "accumulation": 8,
        "freeze": ["frequency_encoder", "spectral_combiner", "temporal_model",
                    "physiology_encoder", "identity_encoder", "rag_retrieval"],
        "unfreeze": ["spatial_encoder", "fusion", "detection_head"],
        "adversarial": False,
    },
    2: {
        "name": "v2_stage2_spectral",
        "description": "Add multi-spectral + CLIP alignment",
        "lr": 5e-5,
        "mode": "image",
        "batch_size": 2,
        "accumulation": 8,
        "freeze": ["temporal_model", "physiology_encoder", "identity_encoder"],
        "unfreeze": ["spatial_encoder", "frequency_encoder", "spectral_combiner",
                     "fusion", "detection_head", "rag_retrieval"],
        "adversarial": False,
    },
    3: {
        "name": "v2_stage3_temporal",
        "description": "Add temporal + physiology + identity (video mode)",
        "lr": 3e-5,
        "mode": "video",
        "batch_size": 1,
        "accumulation": 16,
        "freeze": ["spatial_encoder", "frequency_encoder", "spectral_combiner"],
        "unfreeze": ["temporal_model", "physiology_encoder", "identity_encoder",
                     "fusion", "detection_head", "rag_retrieval"],
        "adversarial": False,
    },
    4: {
        "name": "v2_stage4_multitask",
        "description": "Full model multi-task fine-tuning",
        "lr": 1e-5,
        "mode": "image",
        "batch_size": 2,
        "accumulation": 8,
        "freeze": [],
        "unfreeze": ["spatial_encoder", "frequency_encoder", "spectral_combiner",
                     "temporal_model", "physiology_encoder", "identity_encoder",
                     "fusion", "detection_head", "rag_retrieval"],
        "adversarial": False,
    },
    5: {
        "name": "v2_stage5_adversarial",
        "description": "Adversarial robustness hardening",
        "lr": 5e-6,
        "mode": "image",
        "batch_size": 2,
        "accumulation": 8,
        "freeze": [],
        "unfreeze": ["spatial_encoder", "frequency_encoder", "spectral_combiner",
                     "temporal_model", "physiology_encoder", "identity_encoder",
                     "fusion", "detection_head", "rag_retrieval"],
        "adversarial": True,
    },
}


def build_datasets(data_root: str, mode: str, config: dict):
    """Build train/val datasets from all available sources."""
    image_size = config["data"]["image_size"]
    num_frames = config["data"]["num_frames"]
    ds_kwargs = {"image_size": image_size, "num_frames": num_frames, "mode": mode}

    train_datasets, val_datasets = [], []
    root = Path(data_root)

    if (root / "FaceForensics++").exists():
        train_datasets.append(FaceForensicsDataset(root_dir=data_root, split="train", **ds_kwargs))
        val_datasets.append(FaceForensicsDataset(root_dir=data_root, split="val", **ds_kwargs))
        logger.info(f"  FF++: {len(train_datasets[-1])} train")

    if (root / "CelebDF").exists():
        train_datasets.append(CelebDFDataset(root_dir=data_root, split="train", **ds_kwargs))
        val_datasets.append(CelebDFDataset(root_dir=data_root, split="val", **ds_kwargs))
        logger.info(f"  CelebDF: {len(train_datasets[-1])} train")

    if (root / "DFDC").exists():
        train_datasets.append(DFDCDataset(root_dir=data_root, split="train", **ds_kwargs))
        val_datasets.append(DFDCDataset(root_dir=data_root, split="val", **ds_kwargs))
        logger.info(f"  DFDC: {len(train_datasets[-1])} train")

    if not train_datasets:
        logger.error("No datasets found! See DATASET_SETUP.md")
        return None, None

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def main():
    memory = ProjectMemory()
    memory.load_primary_context(logger)

    parser = argparse.ArgumentParser(description="DeepfakeDetectorV2 Training Pipeline")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config_v2.yaml")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--prev_ckpt", type=str, default=None, help="Previous stage checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Resume this stage from checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    stage = STAGE_CONFIG[args.stage]
    logger.info(f"\n{'='*60}\nStage {args.stage}: {stage['description']}\n{'='*60}")

    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    # Override from stage config
    config["training"]["learning_rate"] = stage["lr"]
    config["training"]["batch_size"] = stage["batch_size"]
    config["training"]["gradient_accumulation_steps"] = stage["accumulation"]
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    # Datasets
    train_dataset, val_dataset = build_datasets(args.data_root, stage["mode"], config)
    if train_dataset is None:
        return

    train_loader = DataLoader(
        train_dataset, batch_size=stage["batch_size"],
        shuffle=True, num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"].get("pin_memory", True), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=stage["batch_size"],
        shuffle=False, num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"].get("pin_memory", True),
    )

    # Model
    logger.info("Building DeepfakeDetectorV2...")
    model = DeepfakeDetectorV2(config=model_config)

    # Load previous stage checkpoint
    if args.prev_ckpt and Path(args.prev_ckpt).exists():
        logger.info(f"Loading previous stage: {args.prev_ckpt}")
        load_checkpoint(args.prev_ckpt, model)

    # Freeze/unfreeze per stage config
    for m in stage["freeze"]:
        model.freeze_module(m)
    for m in stage["unfreeze"]:
        model.unfreeze_module(m)
    # CLIP and identity backbone are always frozen
    model.freeze_module("clip_alignment")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable/1e6:.1f}M")

    # Train
    trainer = TrainerV2(
        model, train_loader, val_loader, config,
        stage_name=stage["name"],
        use_adversarial=stage["adversarial"],
    )
    result = trainer.train(resume_from=args.resume)

    best_metrics = dict(result.get("best_val_metrics", {}))
    best_metrics["best_epoch"] = result.get("best_epoch")
    memory.record_training(
        step_name=f"training:{stage['name']}",
        checkpoint_path=result.get("best_checkpoint_path", f"checkpoints/{stage['name']}/best_model.pth"),
        dataset_info={
            "active_dataset": "multi_dataset",
            "data_root": args.data_root,
            "available_datasets": [name for name, present in {
                "faceforensics": (Path(args.data_root) / "FaceForensics++").exists(),
                "celebdf": (Path(args.data_root) / "CelebDF").exists(),
                "dfdc": (Path(args.data_root) / "DFDC").exists(),
            }.items() if present],
            "mode": stage["mode"],
            "image_size": config["data"]["image_size"],
            "num_frames": config["data"]["num_frames"],
        },
        training_parameters={
            "stage_name": stage["name"],
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
        notes=f"{stage['name']} training complete; best_val_auc={result['best_val_auc']:.4f}",
        extra={
            "model_architecture": {
                "active_variant": "DeepfakeDetectorV2",
                "summary_short": "EfficientNet-B0 spatial + DCT/FFT/Wavelet/Noise spectral + Video Swin-T + physiology + identity + CLIP + RAG + multimodal transformer + extended detection head",
                "active_modules": [
                    "spatial_encoder",
                    "frequency_encoder",
                    "spectral_combiner",
                    "temporal_model",
                    "physiology_encoder",
                    "identity_encoder",
                    "clip_alignment",
                    "rag_retrieval",
                    "fusion",
                    "detection_head",
                ],
                "inference_mode": stage["mode"],
                "config_path": args.model_config,
            }
        },
    )

    logger.info(f"Stage {args.stage} complete! Best val AUC: {result['best_val_auc']:.4f}")


if __name__ == "__main__":
    main()
