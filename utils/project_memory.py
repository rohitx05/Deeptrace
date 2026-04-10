"""Persistent project memory and compact context management."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ProjectMemory:
    """Manage project state, compressed context, and runtime defaults."""

    def __init__(self, project_root: Optional[Path | str] = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent)
        self.system_config_path = self.project_root / "system_config.yaml"
        self.state_path = self.project_root / "project_state.json"
        self.summary_path = self.project_root / "context_summary.md"

        self.system_config = self._load_or_create_system_config()
        self.state = self._load_or_create_state()
        self._sync_system_config_from_state()
        self.summary = self._load_or_create_summary()

    def load_primary_context(self, log: Optional[logging.Logger] = None) -> tuple[dict, str]:
        """Reload state + summary from disk and optionally log a minimal summary."""
        self.system_config = self._load_or_create_system_config()
        self.state = self._load_or_create_state()
        self._sync_system_config_from_state()
        self.summary = self._load_or_create_summary()

        target_logger = log or logger
        if target_logger:
            target_logger.info("[project_memory] %s", self.minimal_summary())
        return self.state, self.summary

    def minimal_summary(self) -> str:
        """Return a compact one-line runtime summary."""
        performance = self.state.get("performance_metrics", {})
        calibration = self.state.get("calibration_status", {})
        checkpoint = self.state.get("checkpoint_paths", {}).get("active") or "none"
        step = self.state.get("last_completed_step", "unknown")
        dataset = self.state.get("dataset_info", {}).get("active_dataset", "unknown")

        accuracy = self._format_metric(performance.get("accuracy"))
        auc = self._format_metric(performance.get("auc"))
        threshold = self._format_metric(performance.get("optimal_threshold"), digits=4)
        temperature = self._format_metric(calibration.get("temperature"), digits=6)
        calibration_status = calibration.get("status", "unknown")

        return (
            f"step={step} | dataset={dataset} | acc={accuracy} | auc={auc} | "
            f"thr={threshold} | temp={temperature} ({calibration_status}) | ckpt={checkpoint}"
        )

    def update_state(
        self,
        updates: dict,
        notes: Optional[str] = None,
        last_step: Optional[str] = None,
        compress: bool = True,
    ) -> dict:
        """Deep-merge state updates and rewrite summary."""
        state = self.state if getattr(self, "state", None) else self._load_or_create_state()
        self._deep_merge(state, self._to_jsonable(updates))

        if notes is not None:
            state["notes"] = notes
        if last_step is not None:
            state["last_completed_step"] = last_step

        state["project"]["updated_at"] = self._now_iso()
        self.state = state
        self._save_json(self.state_path, self.state)
        self._sync_system_config_from_state()

        if compress:
            self.compress_context()
        else:
            self.summary = self.summary_path.read_text(encoding="utf-8") if self.summary_path.exists() else ""

        return self.state

    def compress_context(self) -> str:
        """Rewrite the compact markdown summary from current state."""
        self.state = self._load_or_create_state()
        self.summary = self.render_context_summary(self.state)
        self.summary_path.write_text(self.summary, encoding="utf-8")
        return self.summary

    def record_training(
        self,
        *,
        step_name: str,
        checkpoint_path: str,
        dataset_info: dict,
        training_parameters: dict,
        metrics: Optional[dict] = None,
        notes: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """Persist training state after a train/pretrain run."""
        metrics = metrics or {}
        checkpoint_rel = self._relpath(checkpoint_path)
        threshold = metrics.get("optimal_threshold", self.state.get("performance_metrics", {}).get("optimal_threshold"))

        updates = {
            "dataset_info": dataset_info,
            "training_parameters": training_parameters,
            "performance_metrics": {
                "source": step_name,
                "dataset": dataset_info.get("active_dataset"),
                "mode": dataset_info.get("mode"),
                "accuracy": metrics.get("accuracy"),
                "auc": metrics.get("roc_auc", metrics.get("auc")),
                "optimal_threshold": threshold,
                "loss": metrics.get("loss"),
                "best_epoch": metrics.get("best_epoch"),
            },
            "checkpoint_paths": {
                "active": checkpoint_rel,
            },
        }

        calibration = self.state.get("calibration_status", {})
        if threshold is not None:
            updates["calibration_status"] = {
                "optimal_threshold": threshold,
                "temperature": calibration.get("temperature", 1.0),
                "temperature_path": calibration.get("temperature_path"),
                "status": calibration.get("status", "pending"),
            }

        if extra:
            self._deep_merge(updates, extra)

        self._append_checkpoint(checkpoint_rel)
        return self.update_state(updates, notes=notes, last_step=step_name)

    def record_testing(
        self,
        *,
        step_name: str,
        dataset_name: str,
        metrics: dict,
        checkpoint_path: Optional[str] = None,
        mode: Optional[str] = None,
        notes: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """Persist testing/evaluation results."""
        checkpoint_rel = self._relpath(checkpoint_path) if checkpoint_path else self.state.get("checkpoint_paths", {}).get("active")
        threshold = metrics.get(
            "optimal_threshold",
            self.state.get("calibration_status", {}).get("optimal_threshold"),
        )

        updates = {
            "performance_metrics": {
                "source": step_name,
                "dataset": dataset_name,
                "mode": mode,
                "accuracy": metrics.get("accuracy"),
                "auc": metrics.get("roc_auc", metrics.get("auc")),
                "optimal_threshold": threshold,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "num_samples": metrics.get("num_samples"),
                "blur_accuracy": metrics.get("blur_accuracy"),
                "blur_auc": metrics.get("blur_auc"),
                "threshold_0_5_accuracy": metrics.get("threshold_0_5_accuracy"),
            },
            "dataset_info": {
                "active_dataset": dataset_name,
                "mode": mode or self.state.get("dataset_info", {}).get("mode"),
            },
            "checkpoint_paths": {
                "active": checkpoint_rel,
            },
            "calibration_status": {
                "optimal_threshold": threshold,
            },
        }

        if extra:
            self._deep_merge(updates, extra)

        if checkpoint_rel:
            self._append_checkpoint(checkpoint_rel)
        return self.update_state(updates, notes=notes, last_step=step_name)

    def record_calibration(
        self,
        *,
        checkpoint_path: str,
        temperature_path: str,
        temperature: float,
        before_nll: Optional[float] = None,
        after_nll: Optional[float] = None,
        optimal_threshold: Optional[float] = None,
        notes: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> dict:
        """Persist temperature scaling results."""
        checkpoint_rel = self._relpath(checkpoint_path)
        temperature_rel = self._relpath(temperature_path)

        updates = {
            "checkpoint_paths": {
                "active": checkpoint_rel,
            },
            "calibration_status": {
                "enabled": True,
                "status": "calibrated",
                "temperature": temperature,
                "temperature_path": temperature_rel,
                "optimal_threshold": optimal_threshold,
                "before_nll": before_nll,
                "after_nll": after_nll,
            },
        }

        if extra:
            self._deep_merge(updates, extra)

        self._append_checkpoint(checkpoint_rel)
        return self.update_state(updates, notes=notes, last_step="calibration")

    def suggest_next_step(self) -> str:
        """Return the next recommended technical action based on state."""
        performance = self.state.get("performance_metrics", {})
        calibration = self.state.get("calibration_status", {})
        checkpoint = self.state.get("checkpoint_paths", {}).get("active")
        auc = performance.get("auc")
        threshold = performance.get("optimal_threshold")

        if not checkpoint:
            return "Run training to produce an active checkpoint."
        if performance.get("auc") is None:
            return f"Run testing on the active checkpoint: {checkpoint}."
        if calibration.get("status") != "calibrated":
            return "Run temperature calibration on a validation split and save the sidecar temperature file."
        if threshold is not None and threshold < 0.05:
            return "Retest calibrated inference with the stored temperature and re-evaluate the decision threshold."
        if auc is not None and auc < 0.98:
            return "Improve false-negative recovery on hard fake samples, then rerun testing and calibration."
        return "Run a fresh held-out evaluation to verify current checkpoint stability."

    def get_active_checkpoint(self, fallback: Optional[str] = None) -> Optional[str]:
        """Get the active checkpoint path or a fallback."""
        checkpoint = self.state.get("checkpoint_paths", {}).get("active") or fallback
        if checkpoint is None:
            return None
        return str(self.project_root / checkpoint) if not Path(checkpoint).is_absolute() else checkpoint

    def get_temperature_path(self, fallback: Optional[str] = None) -> Optional[str]:
        """Get the stored temperature sidecar path or a fallback."""
        temperature_path = self.state.get("calibration_status", {}).get("temperature_path") or fallback
        if temperature_path is None:
            return None
        return str(self.project_root / temperature_path) if not Path(temperature_path).is_absolute() else temperature_path

    def render_context_summary(self, state: dict) -> str:
        """Generate the compressed markdown summary."""
        model = state.get("model_architecture", {})
        dataset = state.get("dataset_info", {})
        training = state.get("training_parameters", {})
        performance = state.get("performance_metrics", {})
        calibration = state.get("calibration_status", {})
        checkpoints = state.get("checkpoint_paths", {})
        known_issues = self._derive_known_issues(state)
        next_actions = self._derive_next_actions(state)

        lines = [
            "# Context Summary",
            "",
            f"- Updated: {state.get('project', {}).get('updated_at', 'unknown')}",
            f"- Last Step: {state.get('last_completed_step', 'unknown')}",
            "",
            "## Model Architecture",
            f"- Active Variant: {model.get('active_variant', 'unknown')}",
            f"- Summary: {model.get('summary_short', 'unknown')}",
            f"- Active Modules: {', '.join(model.get('active_modules', [])) or 'none'}",
            f"- Inference Mode: {model.get('inference_mode', 'unknown')}",
            f"- Config Path: {model.get('config_path', 'unknown')}",
            "",
            "## Training Setup",
            f"- Dataset: {dataset.get('active_dataset', 'unknown')}",
            f"- Data Root: {dataset.get('data_root', 'unknown')}",
            f"- Available Datasets: {', '.join(dataset.get('available_datasets', [])) or 'none'}",
            f"- Mode: {dataset.get('mode', 'unknown')}",
            f"- Image Size: {dataset.get('image_size', 'unknown')}",
            f"- Num Frames: {dataset.get('num_frames', 'unknown')}",
            f"- Batch Size: {training.get('batch_size', 'unknown')}",
            f"- Grad Accumulation: {training.get('gradient_accumulation_steps', 'unknown')}",
            f"- Learning Rate: {training.get('learning_rate', 'unknown')}",
            f"- Weight Decay: {training.get('weight_decay', 'unknown')}",
            f"- AMP: {training.get('use_amp', 'unknown')}",
            f"- Device: {training.get('device', 'unknown')}",
            "",
            "## Current Performance",
            f"- Metric Source: {performance.get('source', 'unknown')}",
            f"- Metric Dataset: {performance.get('dataset', 'unknown')}",
            f"- Accuracy: {self._format_metric(performance.get('accuracy'))}",
            f"- AUC: {self._format_metric(performance.get('auc'))}",
            f"- Optimal Threshold: {self._format_metric(performance.get('optimal_threshold'), digits=4)}",
            f"- Threshold@0.5 Accuracy: {self._format_metric(performance.get('threshold_0_5_accuracy'))}",
            f"- Blur Accuracy: {self._format_metric(performance.get('blur_accuracy'))}",
            f"- Blur AUC: {self._format_metric(performance.get('blur_auc'))}",
            f"- Temperature: {self._format_metric(calibration.get('temperature'), digits=6)}",
            f"- Calibration Status: {calibration.get('status', 'unknown')}",
            f"- Calibration File: {calibration.get('temperature_path', 'unknown')}",
            f"- Active Checkpoint: {checkpoints.get('active', 'none')}",
            "",
            "## Known Issues",
        ]

        lines.extend([f"- {issue}" for issue in known_issues])
        lines.extend([
            "",
            "## Next Actions",
        ])
        lines.extend([f"- {action}" for action in next_actions])

        notes = state.get("notes")
        if notes:
            lines.extend([
                "",
                "## Notes",
                f"- {notes}",
            ])

        return "\n".join(lines).strip() + "\n"

    def _load_or_create_system_config(self) -> dict:
        if self.system_config_path.exists():
            with open(self.system_config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}

        system_config = self._default_system_config()
        with open(self.system_config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(system_config, handle, sort_keys=False)
        return system_config

    def _load_or_create_state(self) -> dict:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        state = self._default_state()
        self._save_json(self.state_path, state)
        return state

    def _load_or_create_summary(self) -> str:
        if self.summary_path.exists():
            return self.summary_path.read_text(encoding="utf-8")

        summary = self.render_context_summary(self.state)
        self.summary_path.write_text(summary, encoding="utf-8")
        return summary

    def _sync_system_config_from_state(self):
        config = self._load_or_create_system_config()
        paths = config.setdefault("paths", {})
        runtime = config.setdefault("runtime", {})
        device_config = config.setdefault("device_config", {})
        flags = config.setdefault("flags", {})

        model = self.state.get("model_architecture", {})
        training = self.state.get("training_parameters", {})
        calibration = self.state.get("calibration_status", {})
        checkpoints = self.state.get("checkpoint_paths", {})

        if checkpoints.get("active"):
            paths["active_checkpoint"] = checkpoints["active"]
        if calibration.get("temperature_path"):
            paths["calibration_file"] = calibration["temperature_path"]
        if model.get("config_path"):
            paths["active_model_config"] = model["config_path"]
            if model.get("active_variant") == "DeepfakeDetectorV2":
                paths["model_config_v2"] = model["config_path"]
            else:
                paths["model_config"] = model["config_path"]

        if training.get("batch_size") is not None:
            runtime["batch_size"] = training["batch_size"]
        if self.state.get("dataset_info", {}).get("image_size") is not None:
            runtime["image_size"] = self.state["dataset_info"]["image_size"]
        if self.state.get("dataset_info", {}).get("num_frames") is not None:
            runtime["num_frames"] = self.state["dataset_info"]["num_frames"]

        if training.get("device") is not None:
            device_config["device"] = training["device"]
        if training.get("num_workers") is not None:
            device_config["num_workers"] = training["num_workers"]
        if training.get("use_amp") is not None:
            flags["use_amp"] = training["use_amp"]
        flags["use_calibration"] = calibration.get("enabled", flags.get("use_calibration", True))

        self.system_config = config
        with open(self.system_config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.system_config, handle, sort_keys=False)

    def _default_system_config(self) -> dict:
        config = self._safe_load_yaml(self.project_root / "configs" / "config.yaml")
        active_checkpoint = self._detect_active_checkpoint()
        calibration_path = self._default_temperature_path(active_checkpoint) if active_checkpoint else "checkpoints/calibration.json"

        return {
            "paths": {
                "root": ".",
                "state_file": "project_state.json",
                "summary_file": "context_summary.md",
                "base_config": "configs/config.yaml",
                "model_config": "configs/model_config.yaml",
                "model_config_v2": "configs/model_config_v2.yaml",
                "active_checkpoint": active_checkpoint,
                "calibration_file": calibration_path,
                "checkpoints_dir": "checkpoints",
                "logs_dir": "logs",
            },
            "device_config": {
                "device": "auto",
                "num_workers": config.get("hardware", {}).get("num_workers", 0),
                "pin_memory": config.get("hardware", {}).get("pin_memory", True),
            },
            "runtime": {
                "batch_size": config.get("training", {}).get("batch_size", 2),
                "image_size": config.get("data", {}).get("image_size", 160),
                "num_frames": config.get("data", {}).get("num_frames", 8),
            },
            "flags": {
                "use_calibration": True,
                "use_amp": config.get("hardware", {}).get("mixed_precision", True),
                "auto_load_project_memory": True,
                "auto_update_project_memory": True,
                "auto_compress_context": True,
            },
        }

    def _default_state(self) -> dict:
        base_config = self._safe_load_yaml(self.project_root / "configs" / "config.yaml")
        active_checkpoint = self.system_config.get("paths", {}).get("active_checkpoint") or self._detect_active_checkpoint()
        active_dataset = self._infer_dataset_from_checkpoint(active_checkpoint)

        return {
            "project": {
                "name": self.project_root.name,
                "root": ".",
                "updated_at": self._now_iso(),
            },
            "model_architecture": {
                "active_variant": "DeepfakeDetector",
                "summary_short": "EfficientNet-B0 spatial + DCT EfficientNet-B0 + Video Swin-T + BiLSTM physiology + CLIP + cross-attention fusion + detection head",
                "active_modules": [
                    "spatial_encoder",
                    "frequency_encoder",
                    "temporal_model",
                    "physiology_encoder",
                    "clip_alignment",
                    "fusion",
                    "detection_head",
                ],
                "inference_mode": "image",
                "config_path": "configs/model_config.yaml",
            },
            "dataset_info": {
                "active_dataset": active_dataset,
                "data_root": base_config.get("data", {}).get("root_dir", "data/"),
                "available_datasets": self._discover_available_datasets(),
                "mode": "image",
                "image_size": base_config.get("data", {}).get("image_size", 160),
                "num_frames": base_config.get("data", {}).get("num_frames", 8),
            },
            "training_parameters": {
                "batch_size": base_config.get("training", {}).get("batch_size", 2),
                "gradient_accumulation_steps": base_config.get("training", {}).get("gradient_accumulation_steps", 8),
                "learning_rate": base_config.get("training", {}).get("learning_rate", 1e-4),
                "weight_decay": base_config.get("training", {}).get("weight_decay", 1e-4),
                "num_epochs": base_config.get("training", {}).get("num_epochs", 50),
                "use_amp": base_config.get("hardware", {}).get("mixed_precision", True),
                "device": self.system_config.get("device_config", {}).get("device", "auto"),
                "num_workers": base_config.get("hardware", {}).get("num_workers", 0),
            },
            "performance_metrics": {
                "source": "bootstrap",
                "dataset": active_dataset,
                "mode": "image",
                "accuracy": None,
                "auc": None,
                "optimal_threshold": None,
                "threshold_0_5_accuracy": None,
            },
            "calibration_status": {
                "enabled": self.system_config.get("flags", {}).get("use_calibration", True),
                "status": "pending",
                "temperature": 1.0,
                "temperature_path": self.system_config.get("paths", {}).get("calibration_file"),
                "optimal_threshold": None,
            },
            "checkpoint_paths": {
                "active": active_checkpoint,
                "available": self._discover_checkpoints(),
            },
            "last_completed_step": "bootstrap",
            "notes": "state initialized from config defaults",
        }

    def _derive_known_issues(self, state: dict) -> list[str]:
        performance = state.get("performance_metrics", {})
        calibration = state.get("calibration_status", {})
        issues = []

        if calibration.get("status") != "calibrated":
            issues.append(f"Calibration pending: status={calibration.get('status', 'unknown')}")

        threshold = performance.get("optimal_threshold")
        if threshold is not None and threshold < 0.05:
            issues.append(f"Threshold drift: optimal_threshold={threshold:.4f}")

        accuracy = performance.get("accuracy")
        if accuracy is not None and accuracy < 0.9:
            issues.append(f"Accuracy below 0.90: {accuracy:.4f}")

        auc = performance.get("auc")
        if auc is not None and auc < 0.99:
            issues.append(f"AUC below 0.99: {auc:.4f}")

        if not issues:
            issues.append("No blocking issue recorded")

        return issues

    def _derive_next_actions(self, state: dict) -> list[str]:
        actions = [self.suggest_next_step()]

        performance = state.get("performance_metrics", {})
        calibration = state.get("calibration_status", {})

        if performance.get("optimal_threshold") is not None and calibration.get("status") != "calibrated":
            actions.append("Persist calibrated temperature and rerun held-out evaluation.")
        if performance.get("blur_auc") is not None:
            actions.append("Track robustness deltas after calibration or retraining.")
        if state.get("model_architecture", {}).get("active_variant") == "DeepfakeDetector":
            actions.append("Keep V2 migration gated until a trained V2 checkpoint exists.")

        deduped = []
        for action in actions:
            if action not in deduped:
                deduped.append(action)
        return deduped

    def _discover_available_datasets(self) -> list[str]:
        data_root = self.project_root / "data"
        available = []
        if (data_root / "FaceForensics++").exists():
            available.append("faceforensics")
        if (data_root / "CelebDF").exists():
            available.append("celebdf")
        if (data_root / "DFDC").exists():
            available.append("dfdc")
        if (data_root / "kaggle_realfake").exists():
            available.append("kaggle_realfake")
        return available

    def _discover_checkpoints(self) -> list[str]:
        checkpoint_root = self.project_root / "checkpoints"
        if not checkpoint_root.exists():
            return []

        files = sorted(
            checkpoint_root.rglob("*.pth"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        return [self._relpath(path) for path in files[:12]]

    def _detect_active_checkpoint(self) -> Optional[str]:
        available = self._discover_checkpoints()
        for candidate in available:
            if candidate.endswith("best_model.pth"):
                return candidate
        return available[0] if available else None

    def _infer_dataset_from_checkpoint(self, checkpoint_path: Optional[str]) -> str:
        checkpoint = (checkpoint_path or "").lower()
        if "kaggle" in checkpoint:
            return "kaggle_realfake"
        if "stage4" in checkpoint:
            return "multi_dataset"
        if "stage3" in checkpoint or "stage2" in checkpoint or "stage1" in checkpoint:
            return "faceforensics"
        return "unknown"

    def _default_temperature_path(self, checkpoint_path: Optional[str]) -> Optional[str]:
        if checkpoint_path is None:
            return None
        checkpoint = Path(checkpoint_path)
        return (checkpoint.parent / "calibration.json").as_posix()

    def _save_json(self, path: Path, payload: dict):
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _append_checkpoint(self, checkpoint_path: Optional[str]):
        if checkpoint_path is None:
            return
        checkpoints = self.state.get("checkpoint_paths", {}).get("available", [])
        if checkpoint_path in checkpoints:
            checkpoints.remove(checkpoint_path)
        checkpoints.insert(0, checkpoint_path)
        self.state.setdefault("checkpoint_paths", {})["available"] = checkpoints[:12]

    def _deep_merge(self, base: dict, updates: dict):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _to_jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, Path):
            return self._relpath(value)
        if hasattr(value, "item") and callable(value.item):
            try:
                return self._to_jsonable(value.item())
            except Exception:
                pass
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return float(value)
        if isinstance(value, (int, str, bool)) or value is None:
            return value
        return str(value)

    def _relpath(self, path: Optional[Path | str]) -> Optional[str]:
        if path is None:
            return None
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.relative_to(self.project_root).as_posix()
        except ValueError:
            return str(candidate)

    def _safe_load_yaml(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _format_metric(self, value: Any, digits: int = 4) -> str:
        if value is None:
            return "na"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return "na"
        return f"{numeric:.{digits}f}"


def parse_update_value(raw_value: str) -> Any:
    """Parse CLI update values as JSON when possible."""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        lowered = raw_value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        if re.fullmatch(r"-?\d+", raw_value):
            return int(raw_value)
        if re.fullmatch(r"-?\d+\.\d+", raw_value):
            return float(raw_value)
        return raw_value


def apply_dotted_update(payload: dict, dotted_key: str, value: Any):
    """Apply an update to a nested dict using dotted notation."""
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
