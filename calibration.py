"""
Temperature scaling calibration for deepfake inference.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from datasets.base_dataset import BaseDeepfakeDataset
from datasets.celebdf import CelebDFDataset
from datasets.dfdc import DFDCDataset
from datasets.faceforensics import FaceForensicsDataset
from datasets.kaggle_realfake import KaggleRealFakeDataset
from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint
from utils.device import get_device

logger = logging.getLogger(__name__)

SUPPORTED_MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
}

DATASET_MAP = {
    "faceforensics": FaceForensicsDataset,
    "celebdf": CelebDFDataset,
    "dfdc": DFDCDataset,
    "kaggle_realfake": KaggleRealFakeDataset,
}


class FolderRealFakeDataset(BaseDeepfakeDataset):
    """Simple folder dataset with `real/` and `fake/` subdirectories."""

    def _load_samples(self) -> list:
        samples = []
        for folder_name, label, manipulation_type in (
            ("real", 0, "real"),
            ("fake", 1, "Deepfakes"),
        ):
            folder = self.root_dir / folder_name
            if not folder.exists():
                continue

            for media_path in sorted(folder.rglob("*")):
                if media_path.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
                    continue
                samples.append({
                    "path": str(media_path),
                    "label": label,
                    "manipulation_type": manipulation_type,
                })

        return samples


class ModelWithTemperature(nn.Module):
    """Wrap a detector and calibrate binary logits with a learned temperature."""

    def __init__(self, model: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * self._validate_temperature(temperature))

    @staticmethod
    def _validate_temperature(value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Temperature must be positive and finite, got {value}")
        return value

    @staticmethod
    def _flatten_binary_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim > 1 and logits.size(-1) == 1:
            return logits.squeeze(-1)
        return logits

    @property
    def temperature_value(self) -> float:
        return float(self.temperature.detach().clamp_min(1e-6).item())

    def set_temperature_value(self, value: float):
        with torch.no_grad():
            self.temperature.fill_(self._validate_temperature(value))

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-6)

    def forward(self, *args, **kwargs) -> dict:
        predictions = self.model(*args, **kwargs)
        raw_logits = predictions["binary_logit"]
        calibrated_logits = self.temperature_scale(raw_logits)

        calibrated_predictions = dict(predictions)
        calibrated_predictions["binary_logit"] = raw_logits
        calibrated_predictions["scaled_binary_logit"] = calibrated_logits
        calibrated_predictions["binary_pred"] = torch.sigmoid(
            self._flatten_binary_logits(calibrated_logits)
        )
        return calibrated_predictions

    def collect_validation_logits(
        self,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.model.parameters()).device
        self.model.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = move_batch_to_device(batch, device)
                mode = "video" if "frames" in batch else "image"

                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    if mode == "video":
                        predictions = self.model(
                            frames=batch["frames"],
                            dct_frames=batch.get("dct_frames"),
                            mode="video",
                        )
                    else:
                        predictions = self.model(
                            images=batch["image"],
                            dct=batch.get("dct"),
                            mode="image",
                        )

                all_logits.append(self._flatten_binary_logits(predictions["binary_logit"]).float())
                all_labels.append(batch["label"].float())

        if not all_logits:
            raise ValueError("Validation dataloader produced no samples for calibration.")

        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

    def set_temperature(
        self,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        max_iter: int = 100,
    ) -> dict:
        device = device or next(self.model.parameters()).device
        self.to(device)
        self.model.eval()

        model_params = list(self.model.parameters())
        requires_grad = [param.requires_grad for param in model_params]
        for param in model_params:
            param.requires_grad_(False)

        logits, labels = self.collect_validation_logits(
            dataloader=dataloader,
            device=device,
            use_amp=use_amp,
        )

        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        before_nll = criterion(logits, labels).item()

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self._flatten_binary_logits(self.temperature_scale(logits)), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self.temperature.clamp_(min=1e-6)
            after_nll = criterion(
                self._flatten_binary_logits(self.temperature_scale(logits)),
                labels,
            ).item()
            # Compute calibrated probabilities then derive the optimal threshold
            calibrated_probs = torch.sigmoid(
                self._flatten_binary_logits(self.temperature_scale(logits))
            ).cpu().numpy()

        for param, original in zip(model_params, requires_grad):
            param.requires_grad_(original)

        optimal_threshold = compute_optimal_threshold(
            labels.cpu().numpy(), calibrated_probs
        )

        return {
            "temperature": self.temperature_value,
            "threshold": optimal_threshold,
            "before_nll": before_nll,
            "after_nll": after_nll,
            "num_samples": int(labels.numel()),
        }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor entries in a batch onto a device."""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_optimal_threshold(y_true, y_prob) -> float:
    """Return the threshold that maximises Youden's J (TPR - FPR) on the validation set."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[(tpr - fpr).argmax()])


def get_default_calibration_path(checkpoint_path: str) -> Path:
    """Return the default calibration.json sidecar location."""
    return Path(checkpoint_path).parent / "calibration.json"


def resolve_calibration_path(checkpoint_path: str, calibration_path: Optional[str] = None) -> Path:
    """Resolve an explicit calibration path or discover a default sidecar."""
    if calibration_path is not None:
        return Path(calibration_path)

    checkpoint = Path(checkpoint_path)
    candidates = [
        checkpoint.parent / "calibration.json",
        Path("calibration.json"),
        checkpoint.with_suffix(".temperature.json"),
        checkpoint.parent / "temperature.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_calibration(path: str) -> float:
    """Load temperature from a calibration JSON file (returns float for backward compat)."""
    return load_calibration_dict(path)["temperature"]


def load_calibration_dict(path: str) -> dict:
    """Load full calibration payload (temperature + optional threshold)."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = {"temperature": ModelWithTemperature._validate_temperature(payload["temperature"])}
    # Support both key names for backward compatibility
    threshold = payload.get("threshold") or payload.get("optimal_threshold")
    if threshold is not None:
        result["optimal_threshold"] = float(threshold)
    return result


def save_calibration(path: str, temperature: float, optimal_threshold: Optional[float] = None):
    """Save temperature and threshold to a calibration JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"temperature": ModelWithTemperature._validate_temperature(temperature)}
    if optimal_threshold is not None:
        payload["threshold"] = float(optimal_threshold)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def get_default_temperature_path(checkpoint_path: str) -> Path:
    """Backward-compatible alias for the default calibration path."""
    return get_default_calibration_path(checkpoint_path)


def resolve_temperature_path(checkpoint_path: str, temperature_path: Optional[str] = None) -> Path:
    """Backward-compatible alias for calibration path resolution."""
    return resolve_calibration_path(checkpoint_path, temperature_path)


def load_temperature(path: str) -> float:
    """Backward-compatible alias for loading calibration."""
    return load_calibration(path)


def save_temperature(path: str, temperature: float):
    """Backward-compatible alias for saving calibration."""
    save_calibration(path, temperature)


def infer_dataset_name(val_data_path: str) -> str:
    """Infer dataset layout from a validation path."""
    path = Path(val_data_path)
    lower_name = path.name.lower()

    if (path / "real").exists() and (path / "fake").exists():
        return "folder"
    if (path / "FaceForensics++").exists() or lower_name == "faceforensics++":
        return "faceforensics"
    if (path / "CelebDF").exists() or lower_name == "celebdf":
        return "celebdf"
    if (path / "DFDC").exists() or lower_name == "dfdc":
        return "dfdc"
    if (
        (path / "real_vs_fake" / "real-vs-fake").exists()
        or (path / "train").exists()
        or (path / "valid").exists()
        or (path / "val").exists()
        or lower_name == "real-vs-fake"
    ):
        return "kaggle_realfake"

    raise ValueError(f"Could not infer dataset type from validation path: {path}")


def _normalize_dataset_root(val_data_path: str, dataset_name: str) -> Path:
    path = Path(val_data_path)

    if dataset_name == "faceforensics" and path.name == "FaceForensics++":
        return path.parent
    if dataset_name == "celebdf" and path.name == "CelebDF":
        return path.parent
    if dataset_name == "dfdc" and path.name == "DFDC":
        return path.parent
    if dataset_name == "kaggle_realfake" and path.name == "real_vs_fake":
        candidate = path / "real-vs-fake"
        if candidate.exists():
            return candidate

    return path


def _resolve_folder_dataset_root(val_data_path: str, split: str) -> Path:
    path = Path(val_data_path)
    if (path / "real").exists() and (path / "fake").exists():
        return path

    for split_name in ("valid", "val", split, "test"):
        candidate = path / split_name
        if (candidate / "real").exists() and (candidate / "fake").exists():
            return candidate

    raise ValueError(f"Could not find real/fake subfolders under: {path}")


def build_validation_dataset(
    val_data_path: str,
    image_size: int,
    num_frames: int,
    mode: str = "image",
    split: str = "val",
    dataset_name: Optional[str] = None,
):
    """Build a validation dataset for calibration."""
    dataset_name = (dataset_name or infer_dataset_name(val_data_path)).lower()

    if dataset_name == "folder":
        dataset_root = _resolve_folder_dataset_root(val_data_path, split)
        return FolderRealFakeDataset(
            root_dir=str(dataset_root),
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            mode=mode,
        )

    dataset_root = _normalize_dataset_root(val_data_path, dataset_name)
    if dataset_name == "kaggle_realfake" and (dataset_root / "real").exists() and (dataset_root / "fake").exists():
        return FolderRealFakeDataset(
            root_dir=str(dataset_root),
            split=split,
            image_size=image_size,
            num_frames=num_frames,
            mode=mode,
        )

    dataset_cls = DATASET_MAP[dataset_name]
    return dataset_cls(
        root_dir=str(dataset_root),
        split=split,
        image_size=image_size,
        num_frames=num_frames,
        mode=mode,
    )


def build_validation_dataloader(
    val_data_path: str,
    image_size: int,
    num_frames: int,
    batch_size: int,
    mode: str = "image",
    split: str = "val",
    dataset_name: Optional[str] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a validation dataloader for calibration."""
    dataset = build_validation_dataset(
        val_data_path=val_data_path,
        image_size=image_size,
        num_frames=num_frames,
        mode=mode,
        split=split,
        dataset_name=dataset_name,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate deepfake inference probabilities")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--mode", type=str, default="image", choices=["image", "video"])
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(args.model_config, "r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)

    device = get_device() if args.device == "auto" else torch.device(args.device)

    dataloader = build_validation_dataloader(
        val_data_path=args.val_data,
        image_size=config["data"]["image_size"],
        num_frames=config["data"]["num_frames"],
        batch_size=args.batch_size or config["training"]["batch_size"],
        mode=args.mode,
        split=args.split,
        dataset_name=args.dataset,
        num_workers=args.num_workers if args.num_workers is not None else config["hardware"]["num_workers"],
        pin_memory=device.type == "cuda" and config["hardware"].get("pin_memory", True),
    )

    model = DeepfakeDetector(config=model_config)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)
    model.eval()

    calibrated_model = ModelWithTemperature(model)
    calibrated_model.to(device)

    metrics = calibrated_model.set_temperature(
        dataloader=dataloader,
        device=device,
        use_amp=device.type == "cuda" and config["hardware"].get("mixed_precision", True),
        max_iter=args.max_iter,
    )

    output_path = Path(args.output) if args.output else get_default_calibration_path(args.checkpoint)
    save_calibration(str(output_path), metrics["temperature"], metrics.get("threshold"))

    try:
        from utils.project_memory import ProjectMemory

        memory = ProjectMemory()
        memory.load_primary_context()
        memory.record_calibration(
            checkpoint_path=args.checkpoint,
            temperature_path=str(output_path),
            temperature=metrics["temperature"],
            before_nll=metrics["before_nll"],
            after_nll=metrics["after_nll"],
            optimal_threshold=memory.state.get("performance_metrics", {}).get("optimal_threshold"),
            notes="temperature scaling fit on validation dataloader",
            extra={
                "dataset_info": {
                    "data_root": args.val_data,
                    "mode": args.mode,
                    "image_size": config["data"]["image_size"],
                    "num_frames": config["data"]["num_frames"],
                }
            },
        )
    except Exception:
        logger.debug("Project memory update skipped", exc_info=True)

    print(json.dumps({
        "temperature": metrics["temperature"],
        "before_nll": metrics["before_nll"],
        "after_nll": metrics["after_nll"],
        "num_samples": metrics["num_samples"],
        "output_path": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
