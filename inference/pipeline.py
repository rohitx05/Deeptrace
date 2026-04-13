"""
End-to-end inference pipeline.
Input: image or video file → Output: prediction JSON + heatmap + forensic report.
"""

import torch
import cv2
import numpy as np
import json
from pathlib import Path
import logging
import yaml

from models.detector import DeepfakeDetector
from datasets.transforms import get_val_transforms, apply_dct_transform
from explainability.gradcam import GradCAM
from explainability.forensic_report import generate_forensic_report
from calibration import ModelWithTemperature, load_calibration_dict, resolve_calibration_path
from utils.device import get_device
from utils.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)

MANIPULATION_LABELS = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


class InferencePipeline:
    """Full inference pipeline for deepfake detection."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/config.yaml",
        model_config_path: str = "configs/model_config.yaml",
        device: str = "auto",
        calibration_path: str = None,
        temperature_path: str = None,
    ):
        # Load configs
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        # Device
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)
        self.use_amp = self.device.type == "cuda" and self.config["hardware"].get("mixed_precision", True)

        # Image size
        self.image_size = self.config["data"]["image_size"]
        self.num_frames = self.config["data"]["num_frames"]
        self.transform = get_val_transforms(self.image_size)

        # Model
        self.base_model = DeepfakeDetector(config=model_config)
        load_checkpoint(checkpoint_path, self.base_model, device=self.device)
        self.base_model.to(self.device)
        self.base_model.eval()

        self.model = ModelWithTemperature(self.base_model)
        self.model.to(self.device)
        self.model.eval()

        # Decision threshold (calibrated) – default from config, overridden by sidecar
        self.threshold: float = self.config.get("evaluation", {}).get("threshold", 0.5)

        calibration_override = calibration_path or temperature_path
        self.calibration_path = resolve_calibration_path(checkpoint_path, calibration_override)
        self.calibration_loaded = False
        if self.calibration_path.exists():
            calib = load_calibration_dict(str(self.calibration_path))
            self.model.set_temperature_value(calib["temperature"])
            # Accept either key name (calibration.json uses "threshold", some old files use "optimal_threshold")
            thr = calib.get("threshold") or calib.get("optimal_threshold")
            if thr is not None:
                self.threshold = float(thr)
            self.calibration_loaded = True
            logger.info(
                "Loaded calibration: T=%.6f threshold=%.4f from %s",
                self.model.temperature_value,
                self.threshold,
                self.calibration_path,
            )
        else:
            logger.info("No calibration.json found, using T=1.0 threshold=%.4f", self.threshold)

        # GradCAM
        self.gradcam = GradCAM(self.base_model)

        logger.info(f"Inference pipeline ready (device={self.device})")

    def detect_media_type(self, path: str) -> str:
        """Detect if the file is an image or video."""
        ext = Path(path).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            return "image"
        elif ext in (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"):
            return "video"
        else:
            logger.warning(f"Unknown extension: {ext}, treating as image")
            return "image"

    def predict(self, file_path: str, save_dir: str = None, return_logits: bool = False) -> dict:
        """
        Run full inference on a single file.

        Args:
            file_path: Path to image or video
            save_dir: Directory to save heatmap and report (optional)

        Returns:
            dict: {prediction, confidence, manipulation_type, heatmap_path, forensic_explanation}
        """
        media_type = self.detect_media_type(file_path)

        if media_type == "image":
            result = self._predict_image(file_path, return_logits=return_logits)
        else:
            result = self._predict_video(file_path, return_logits=return_logits)

        # Generate heatmap
        heatmap_path = None
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # GradCAM heatmap
            heatmap = self._generate_heatmap(file_path, media_type)
            if heatmap is not None:
                heatmap_path = str(save_path / f"{Path(file_path).stem}_heatmap.png")
                cv2.imwrite(heatmap_path, heatmap)
                result["heatmap_path"] = heatmap_path

            # Forensic report
            report = generate_forensic_report(result)
            result["forensic_explanation"] = report
            report_path = save_path / f"{Path(file_path).stem}_report.json"
            with open(report_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    @torch.no_grad()
    def _predict_image(self, path: str, return_logits: bool = False) -> dict:
        """Predict on a single image with TTA (test-time augmentation)."""
        image = cv2.imread(str(path))
        if image is None:
            return {"error": f"Could not read image: {path}"}

        # ── Preprocess ─────────────────────────────────────────────────────────
        face = cv2.resize(image, (self.image_size, self.image_size))
        rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        dct  = apply_dct_transform(face)

        dct_norm   = (dct - dct.mean()) / (dct.std() + 1e-8)
        dct_tensor = torch.from_numpy(dct_norm).permute(2, 0, 1).float()
        dct_tensor = torch.nn.functional.interpolate(
            dct_tensor.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        # ── TTA: original + horizontal flip ────────────────────────────────────
        # Use AMP (matching training conditions so CLIP stays in FP16 as expected).
        # Immediately cast binary_pred to float32 to prevent NaN propagation.
        # ModelWithTemperature already returns binary_pred = sigmoid(logit / T)
        # so there is NO double-temperature application here.
        all_probs  = []
        last_preds = None

        for flip in (False, True):
            aug_rgb  = rgb[:, ::-1, :].copy() if flip else rgb
            aug_dct  = dct_tensor.flip(-1) if flip else dct_tensor

            augmented = self.transform(image=aug_rgb)
            tensor    = augmented["image"].unsqueeze(0).to(self.device)
            d_tensor  = aug_dct.unsqueeze(0).to(self.device)

            # Primary: run under AMP (matches training/calibration)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                preds = self.model(images=tensor, dct=d_tensor, mode="image")

            # Cast to float32 immediately, then check for NaN
            prob_fp32 = preds["binary_pred"].float()  # (1,)

            if torch.isnan(prob_fp32).any() or torch.isinf(prob_fp32).any():
                logger.warning(f"NaN/Inf in binary_pred (flip={flip}) — retrying in float32")
                with torch.amp.autocast("cuda", enabled=False):
                    preds = self.model(
                        images=tensor.float(), dct=d_tensor.float(), mode="image"
                    )
                prob_fp32 = preds["binary_pred"].float()

            all_probs.append(prob_fp32)
            last_preds = preds

        # ── Average TTA probabilities ───────────────────────────────────────────
        avg_prob = torch.stack(all_probs).mean(dim=0)
        prob_val = float(avg_prob.item())

        # ── Confidence: distance from threshold ────────────────────────────────
        thr = self.threshold
        if prob_val > thr:
            conf = (prob_val - thr) / max(1.0 - thr, 1e-8)
        else:
            conf = (thr - prob_val) / max(thr, 1e-8)
        conf = max(0.0, min(1.0, conf))

        # ── Manipulation class: argmax over multi-class head ───────────────────
        manip_out = last_preds["manipulation_pred"]
        if manip_out.ndim > 1 and manip_out.size(-1) > 1:
            manip_idx = int(manip_out.argmax(dim=-1).item())
        else:
            manip_idx = int(manip_out.round().clamp(0, len(MANIPULATION_LABELS) - 1).item())

        # ── Debug log (visible in terminal) ────────────────────────────────────
        verdict = "FAKE" if prob_val > thr else "REAL"
        raw_logit_val = float(last_preds["binary_logit"].item()) if "binary_logit" in last_preds else float("nan")
        logger.info(
            f"[INFERENCE] raw_logit={raw_logit_val:.4f} | prob={prob_val:.4f} | "
            f"thr={thr:.4f} | verdict={verdict} | conf={conf*100:.1f}% | manip={MANIPULATION_LABELS[manip_idx]}"
        )

        predictions = {
            "binary_logit":        last_preds.get("binary_logit", avg_prob),
            "scaled_binary_logit": last_preds.get("scaled_binary_logit", avg_prob),
            "binary_pred":         avg_prob,
            "confidence":          torch.tensor(conf),
            "manipulation_pred":   torch.tensor(manip_idx),
        }

        return self._format_result(predictions, path, return_logits=return_logits)

    @torch.no_grad()
    def _predict_video(self, path: str, return_logits: bool = False) -> dict:
        """Predict on a video (samples frames)."""
        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return {"error": f"Could not read video: {path}"}

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        dct_frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                face = cv2.resize(frame, (self.image_size, self.image_size))
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                dct = apply_dct_transform(face)

                augmented = self.transform(image=rgb)
                frames.append(augmented["image"])

                dct_normalized = (dct - dct.mean()) / (dct.std() + 1e-8)
                dct_tensor = torch.from_numpy(dct_normalized).permute(2, 0, 1).float()
                dct_tensor = torch.nn.functional.interpolate(
                    dct_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear"
                ).squeeze(0)
                dct_frames.append(dct_tensor)

        cap.release()

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
            dct_frames.append(dct_frames[-1])

        frames_tensor = torch.stack(frames[:self.num_frames]).unsqueeze(0).to(self.device)
        dct_tensor = torch.stack(dct_frames[:self.num_frames]).unsqueeze(0).to(self.device)

        # Predict
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predictions = self.model(frames=frames_tensor, dct_frames=dct_tensor, mode="video")

        return self._format_result(predictions, path, return_logits=return_logits)

    def _format_result(self, predictions: dict, path: str, return_logits: bool = False) -> dict:
        """Format model output into a human-readable result."""
        import math
        prob = float(predictions["binary_pred"].item())
        # Guard against NaN/Inf from unstable logits
        if math.isnan(prob) or math.isinf(prob):
            prob = 0.5
        prob = max(0.0, min(1.0, prob))  # clamp to [0, 1]

        manip_idx = predictions["manipulation_pred"].item()
        # Guard: manip_pred may be a float tensor (argmax not applied) → round and clamp
        if isinstance(manip_idx, float):
            manip_idx = int(round(manip_idx))
        manip_idx = max(0, min(manip_idx, len(MANIPULATION_LABELS) - 1))

        confidence = float(predictions["confidence"].item())
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = abs(prob - 0.5) * 2.0  # recompute from prob

        result = {
            "file": str(path),
            "prediction": "FAKE" if prob > self.threshold else "REAL",
            "fake_probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "manipulation_type": MANIPULATION_LABELS[manip_idx] if prob > self.threshold else "N/A",
            "threshold": round(self.threshold, 4),
        }

        if return_logits:
            result["binary_logit"] = round(predictions["binary_logit"].squeeze(-1).item(), 6)
            result["calibrated_logit"] = round(predictions["scaled_binary_logit"].squeeze(-1).item(), 6)
            result["temperature"] = round(self.model.temperature_value, 6)

        return result

    def _generate_heatmap(self, file_path: str, media_type: str) -> np.ndarray:
        """Generate GradCAM heatmap."""
        try:
            if media_type == "image":
                image = cv2.imread(str(file_path))
                face = cv2.resize(image, (self.image_size, self.image_size))
                rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                augmented = self.transform(image=rgb)
                tensor = augmented["image"].unsqueeze(0).to(self.device)
                heatmap = self.gradcam.generate(tensor)
                return self.gradcam.overlay_heatmap(heatmap, face)
            else:
                # Use middle frame for video heatmap
                cap = cv2.VideoCapture(str(file_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    face = cv2.resize(frame, (self.image_size, self.image_size))
                    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    augmented = self.transform(image=rgb)
                    tensor = augmented["image"].unsqueeze(0).to(self.device)
                    heatmap = self.gradcam.generate(tensor)
                    return self.gradcam.overlay_heatmap(heatmap, face)
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
        return None
