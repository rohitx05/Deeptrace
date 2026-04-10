"""
Evaluation CLI script.

Usage:
    python scripts/evaluate.py --dataset faceforensics --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --dataset celebdf --checkpoint checkpoints/best_model.pth --cross_dataset
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from torch.utils.data import DataLoader

from models.detector import DeepfakeDetector
from datasets.faceforensics import FaceForensicsDataset
from datasets.celebdf import CelebDFDataset
from datasets.dfdc import DFDCDataset
from evaluation.evaluator import Evaluator
from utils.logger import get_logger
from utils.checkpoint import load_checkpoint
from utils.project_memory import ProjectMemory

logger = get_logger("evaluate")

DATASET_MAP = {
    "faceforensics": FaceForensicsDataset,
    "celebdf": CelebDFDataset,
    "dfdc": DFDCDataset,
}


def main():
    memory = ProjectMemory()
    memory.load_primary_context(logger)

    parser = argparse.ArgumentParser(description="Evaluate Deepfake Detection Model")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_MAP.keys()))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"])
    parser.add_argument("--cross_dataset", action="store_true", help="Evaluate on all datasets")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    # Model
    model = DeepfakeDetector(config=model_config)
    load_checkpoint(args.checkpoint, model)

    evaluator = Evaluator(model, use_amp=config["hardware"]["mixed_precision"])

    datasets_to_eval = list(DATASET_MAP.keys()) if args.cross_dataset else [args.dataset]

    for ds_name in datasets_to_eval:
        ds_class = DATASET_MAP[ds_name]
        ds_path = Path(args.data_root)

        # Check if dataset directory exists
        check_dirs = {
            "faceforensics": ds_path / "FaceForensics++",
            "celebdf": ds_path / "CelebDF",
            "dfdc": ds_path / "DFDC",
        }
        if not check_dirs.get(ds_name, ds_path).exists():
            logger.warning(f"Skipping {ds_name}: directory not found")
            continue

        test_dataset = ds_class(
            root_dir=args.data_root, split="test",
            image_size=config["data"]["image_size"],
            num_frames=config["data"]["num_frames"],
            mode=args.mode,
        )

        test_loader = DataLoader(
            test_dataset, batch_size=config["training"]["batch_size"],
            shuffle=False, num_workers=config["hardware"]["num_workers"],
        )

        logger.info(f"\n{'='*50}\nEvaluating on {ds_name} ({len(test_dataset)} samples)\n{'='*50}")
        results = evaluator.evaluate_dataset(test_loader, ds_name)
        evaluator.generate_report(results, args.output)
        memory.record_testing(
            step_name="testing:evaluate",
            dataset_name=ds_name,
            metrics=results["metrics"],
            checkpoint_path=args.checkpoint,
            mode=args.mode,
            notes=f"evaluate.py completed on {ds_name}",
            extra={
                "dataset_info": {
                    "data_root": args.data_root,
                    "available_datasets": datasets_to_eval,
                    "image_size": config["data"]["image_size"],
                    "num_frames": config["data"]["num_frames"],
                }
            },
        )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
