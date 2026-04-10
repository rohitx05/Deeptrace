"""
CLI entry point for inference.

Usage:
    python scripts/run_inference.py --input path/to/image_or_video --checkpoint checkpoints/best_model.pth
    python scripts/run_inference.py --input path/to/folder/ --checkpoint checkpoints/best_model.pth --batch
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline import InferencePipeline
from utils.logger import get_logger
from utils.project_memory import ProjectMemory

logger = get_logger("inference")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}


def main():
    memory = ProjectMemory()
    memory.load_primary_context(logger)

    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to image, video, or directory")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--calibration", type=str, default=None)
    parser.add_argument("--temperature", type=str, default=None)
    parser.add_argument("--output", type=str, default="inference_output/")
    parser.add_argument("--batch", action="store_true", help="Process entire directory")
    parser.add_argument("--return_logits", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or memory.get_active_checkpoint("checkpoints/stage4_multitask/best_model.pth")
    calibration_path = args.calibration or args.temperature
    if calibration_path is None:
        stored_calibration = memory.get_temperature_path(None)
        if stored_calibration and Path(stored_calibration).exists():
            calibration_path = stored_calibration

    # Initialize pipeline
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        config_path=args.config,
        model_config_path=args.model_config,
        device=args.device,
        calibration_path=calibration_path,
    )

    input_path = Path(args.input)

    if args.batch or input_path.is_dir():
        # Batch inference
        files = [f for f in input_path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        logger.info(f"Found {len(files)} files to process")

        results = []
        for f in files:
            logger.info(f"Processing: {f}")
            result = pipeline.predict(str(f), save_dir=args.output, return_logits=args.return_logits)
            results.append(result)
            logger.info(f"  → {result['prediction']} (conf: {result.get('confidence', 0):.3f})")

        # Save batch results
        output_path = Path(args.output) / "batch_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Batch results saved to {output_path}")

    else:
        # Single file inference
        result = pipeline.predict(str(input_path), save_dir=args.output, return_logits=args.return_logits)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
