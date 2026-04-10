"""
Gradio UI for Deepfake Detection.
Upload image/video → see prediction, confidence, heatmap, and forensic explanation.

Usage:
    python ui/app.py
    python ui/app.py --checkpoint checkpoints/stage4_multitask/best_model.pth
    python ui/app.py --demo  (runs without model for UI testing)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import gradio as gr
import numpy as np
import cv2
import tempfile
import json
import logging

logger = logging.getLogger(__name__)


def create_demo_app():
    """Create a demo Gradio app (no model, for UI testing)."""

    def predict_demo(file):
        """Demo prediction with random results."""
        if file is None:
            return None, "Please upload an image or video.", "{}"

        fake_prob = np.random.random()
        result = {
            "prediction": "FAKE" if fake_prob > 0.5 else "REAL",
            "fake_probability": round(fake_prob, 4),
            "confidence": round(np.random.random() * 0.3 + 0.7, 4),
            "manipulation_type": np.random.choice(["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]) if fake_prob > 0.5 else "N/A",
        }

        # Generate demo heatmap
        h, w = 160, 160
        heatmap = np.random.rand(h, w)
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 10)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        pred_str = f"{'🚨 FAKE' if result['prediction'] == 'FAKE' else '✅ REAL'}"
        summary = (
            f"## {pred_str}\n\n"
            f"**Confidence:** {result['confidence'] * 100:.1f}%\n\n"
            f"**Fake Probability:** {result['fake_probability'] * 100:.1f}%\n\n"
            f"**Manipulation Type:** {result['manipulation_type']}\n\n"
            f"*⚠️ Demo mode — using random predictions*"
        )

        return heatmap_rgb, summary, json.dumps(result, indent=2)

    return predict_demo


def create_model_app(checkpoint_path, config_path, model_config_path):
    """Create Gradio app with actual model."""
    from inference.pipeline import InferencePipeline
    from explainability.forensic_report import generate_forensic_report

    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        model_config_path=model_config_path,
    )

    def predict(file):
        if file is None:
            return None, "Please upload an image or video.", "{}"

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.predict(file, save_dir=tmpdir)

            # Load heatmap if generated
            heatmap_img = None
            if result.get("heatmap_path"):
                heatmap_img = cv2.imread(result["heatmap_path"])
                if heatmap_img is not None:
                    heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

            # Summary
            pred_str = f"{'🚨 FAKE' if result['prediction'] == 'FAKE' else '✅ REAL'}"
            summary = (
                f"## {pred_str}\n\n"
                f"**Confidence:** {result.get('confidence', 0) * 100:.1f}%\n\n"
                f"**Fake Probability:** {result.get('fake_probability', 0) * 100:.1f}%\n\n"
                f"**Manipulation Type:** {result.get('manipulation_type', 'N/A')}\n\n"
            )

            # Forensic explanation
            forensic = result.get("forensic_explanation", generate_forensic_report(result))
            summary += f"---\n\n```\n{forensic}\n```"

            return heatmap_img, summary, json.dumps(result, indent=2)

    return predict


def build_interface(predict_fn):
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Deepfake Detector",
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="blue",
        ),
    ) as app:
        gr.Markdown(
            """
            # 🔍 Multimodal Deepfake Detector
            ### Research-Grade Detection with Explainability

            Upload an **image** or **video** to analyze it for deepfake manipulation.
            The system uses spatial analysis, frequency domain analysis, temporal consistency checks,
            and CLIP embedding alignment for robust detection.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="📁 Upload Image or Video",
                    file_types=["image", "video"],
                    type="filepath",
                )
                analyze_btn = gr.Button("🔬 Analyze", variant="primary", size="lg")

            with gr.Column(scale=1):
                heatmap_output = gr.Image(label="🗺️ Artifact Heatmap", type="numpy")

        with gr.Row():
            with gr.Column(scale=2):
                result_md = gr.Markdown(label="📊 Analysis Results")
            with gr.Column(scale=1):
                raw_json = gr.Code(label="📋 Raw JSON", language="json")

        analyze_btn.click(
            fn=predict_fn,
            inputs=[input_file],
            outputs=[heatmap_output, result_md, raw_json],
        )

        gr.Markdown(
            """
            ---
            **Architecture:** EfficientNet-B0 (Spatial) + EfficientNet-B0 (Frequency/DCT) + Video Swin-T (Temporal) + CLIP ViT-B/32 (Alignment)

            **Explainability:** GradCAM heatmaps + Attention visualization + Forensic reports

            *Research prototype — not for production use*
            """
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection UI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage4_multitask/best_model.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config.yaml")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without model")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.demo or not Path(args.checkpoint).exists():
        logger.info("Running in DEMO mode (no model)")
        predict_fn = create_demo_app()
    else:
        predict_fn = create_model_app(args.checkpoint, args.config, args.model_config)

    app = build_interface(predict_fn)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
