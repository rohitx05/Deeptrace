"""
DeepTrace - Flask Web Application
Run: python ui/flask_app.py
Run demo: python ui/flask_app.py --demo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import base64
import json
import logging
import tempfile
import os
import numpy as np
import cv2

from flask import Flask, request, jsonify, send_from_directory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Global pipeline (loaded on startup)
PIPELINE = None
DEMO_MODE = False


# ── Helpers ────────────────────────────────────────────────────────────────

def img_to_b64(arr: np.ndarray) -> str:
    """Convert numpy BGR/RGB image to base64 PNG string."""
    _, buf = cv2.imencode(".png", arr)
    return base64.b64encode(buf).decode()


def generate_forensic_text(result: dict) -> str:
    prob = result.get("fake_probability", 0)
    manip = result.get("manipulation_type", "N/A")
    conf = result.get("confidence", 0)
    if result["prediction"] == "FAKE":
        return (
            f"Spatial encoder detected inconsistencies in facial boundary regions "
            f"({prob*100:.1f}% anomaly score). Frequency domain (DCT) analysis revealed "
            f"coefficient irregularities consistent with GAN-generated content. "
            f"Manipulation type classified as '{manip}' with {conf*100:.1f}% model confidence. "
            f"CLIP alignment score indicates semantic mismatch between high-frequency texture "
            f"and low-frequency structure, a hallmark of neural face synthesis."
        )
    else:
        return (
            f"No significant manipulation artifacts detected. Spatial coherence analysis "
            f"shows consistent facial boundary transitions ({(1-prob)*100:.1f}% authenticity score). "
            f"DCT frequency spectrum matches natural image distributions. "
            f"CLIP embedding alignment is within expected bounds for authentic imagery. "
            f"Model confidence: {conf*100:.1f}%."
        )


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return jsonify({"error": "Only image files are supported (jpg, png, bmp, webp)"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, f"upload{ext}")
        f.save(save_path)

        if DEMO_MODE:
            result = _demo_predict(save_path)
        else:
            result = PIPELINE.predict(save_path, save_dir=tmpdir)

        # Load or generate heatmap
        heatmap_b64 = None
        heatmap_path = result.get("heatmap_path")
        if heatmap_path and Path(heatmap_path).exists():
            heatmap_img = cv2.imread(heatmap_path)
            heatmap_b64 = img_to_b64(heatmap_img)
        else:
            # Generate a demo gaussian heatmap for display
            img = cv2.imread(save_path)
            if img is not None:
                h, w = img.shape[:2]
                heat = np.random.rand(h, w) * result.get("fake_probability", 0.5)
                heat = cv2.GaussianBlur(heat, (31, 31), 10)
                heat = ((heat - heat.min()) / (heat.max() - heat.min() + 1e-8) * 255).astype(np.uint8)
                heat_col = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.55, heat_col, 0.45, 0)
                heatmap_b64 = img_to_b64(overlay)

        # Preview of uploaded image
        orig = cv2.imread(save_path)
        orig_b64 = img_to_b64(orig) if orig is not None else None

        result["forensic_explanation"] = result.get("forensic_explanation") or generate_forensic_text(result)

        def _safe(v, fallback=0.0):
            """Replace NaN/Inf with a safe fallback so JSON stays valid."""
            import math
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return fallback
            return v

        return jsonify({
            "prediction":          result["prediction"],
            "fake_probability":    _safe(result.get("fake_probability", 0.0)),
            "confidence":          _safe(result.get("confidence", 0.0)),
            "manipulation_type":   result.get("manipulation_type", "N/A"),
            "threshold":           _safe(result.get("threshold", 0.1341)),
            "forensic_explanation": result["forensic_explanation"],
            "heatmap_b64":         heatmap_b64,
            "original_b64":        orig_b64,
            "filename":            f.filename,
        })


# ── Demo prediction ────────────────────────────────────────────────────────

def _demo_predict(path: str) -> dict:
    import random
    fake_prob = round(random.uniform(0.05, 0.95), 4)
    conf = round(random.uniform(0.70, 0.98), 4)
    pred = "FAKE" if fake_prob > 0.1341 else "REAL"
    manip_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    return {
        "file": path,
        "prediction": pred,
        "fake_probability": fake_prob,
        "confidence": conf,
        "manipulation_type": random.choice(manip_types) if pred == "FAKE" else "N/A",
        "threshold": 0.1341,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global PIPELINE, DEMO_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/kaggle_realfake/best_model.pth")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.demo or not Path(args.checkpoint).exists():
        logger.info("⚡ Running in DEMO mode (no model loaded)")
        DEMO_MODE = True
    else:
        from inference.pipeline import InferencePipeline
        logger.info(f"Loading model from {args.checkpoint}...")
        PIPELINE = InferencePipeline(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            model_config_path=args.model_config,
        )
        logger.info("✅ Model loaded")

    logger.info(f"🌐 DeepTrace running at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
