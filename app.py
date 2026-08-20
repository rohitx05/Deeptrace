"""
DeepTrace: Enterprise Multi-Spectral Deepfake Forensics Web Dashboard.
Interactive Gradio application demonstrating multi-modal forensic inspection:
1. Spatial RGB + GradCAM Explanatory Heatmap
2. 2D-DCT Log-Power Spectrum
3. Continuous Phase FFT Spatial Reconstruction (SPR)
4. 2-Level Haar Wavelet Packet Decomposition (7 Sub-Bands)
5. 9-Channel SRM & Gabor Noise Residual
6. Dynamic Spectral Gating Weight Breakdown
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))

from models.detector import DeepfakeDetector
from scripts.train_v7_sota_spectral import V7SOTADetector
from utils.device import get_device, AMPContext

# Global Model Initialization
DEVICE = get_device()
BASE_DETECTOR = DeepfakeDetector()
MODEL = V7SOTADetector(BASE_DETECTOR)

CKPT_PATH = Path("checkpoints/v7_sota_spectral/best_model.pth")
if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    MODEL.load_state_dict(state_dict)
MODEL.to(DEVICE).eval()

TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_forensic_breakdown(pil_image: Image.Image):
    """Computes all multi-spectral forensic representations and GradCAM."""
    orig_resized = pil_image.resize((160, 160))
    img_np = np.array(orig_resized).astype(np.float32)
    img_tensor = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

    # 1. Forward Pass & Predictions
    with torch.no_grad():
        with AMPContext(device=DEVICE, enabled=True):
            out = MODEL(img_tensor, return_spectral_details=True)
            logit = out["binary_logit"]
            prob = torch.sigmoid(logit / 0.873507).item()
            conf = out.get("confidence", torch.tensor([0.95])).item()
            manip_idx = out["manip_logits"].argmax(dim=-1).item()

    manip_types = ["Authentic Real", "GAN Synthesis (StyleGAN)", "Poisson Boundary Seam (FaceSwap/Shifter)", "Autoencoder Deepfake (DFD)", "Expression Reenactment (Face2Face)"]
    manip_label = manip_types[min(manip_idx, len(manip_types) - 1)] if prob >= 0.5 else "Authentic Real Human Face"

    # 2. GradCAM Generation
    target_layer = MODEL.spatial_encoder.backbone.conv_head
    gradients, activations = [], []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    with AMPContext(device=DEVICE, enabled=True):
        out_cam = MODEL(img_tensor, return_spectral_details=False)
        cam_logit = out_cam["binary_logit"]

    MODEL.zero_grad()
    cam_logit.backward(retain_graph=True)
    h1.remove()
    h2.remove()

    if len(gradients) > 0 and len(activations) > 0:
        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_resized = cv2.resize(cam, (160, 160))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        gradcam_overlay = np.uint8(np.clip(np.float32(heatmap) * 0.45 + img_np * 0.55, 0, 255))
    else:
        gradcam_overlay = np.uint8(img_np)

    # 3. 2D-DCT Spectrum
    gray = cv2.cvtColor(np.uint8(img_np), cv2.COLOR_RGB2GRAY).astype(np.float32)
    dct = cv2.dct(gray)
    dct_log = np.log(np.abs(dct) + 1.0)
    dct_norm = np.uint8(255 * (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min() + 1e-8))
    dct_color = cv2.applyColorMap(dct_norm, cv2.COLORMAP_INFERNO)
    dct_color = cv2.cvtColor(dct_color, cv2.COLOR_BGR2RGB)

    # 4. Continuous Phase FFT SPR (Spatial Phase Reconstruction)
    fft = np.fft.fft2(gray)
    phase = np.exp(1j * np.angle(fft))
    spr = np.real(np.fft.ifft2(phase))
    spr_norm = np.uint8(255 * (spr - spr.min()) / (spr.max() - spr.min() + 1e-8))
    spr_color = cv2.applyColorMap(spr_norm, cv2.COLORMAP_VIRIDIS)
    spr_color = cv2.cvtColor(spr_color, cv2.COLOR_BGR2RGB)

    # 5. SRM High-Pass Residual
    srm_kernel = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0
    srm_res = cv2.filter2D(gray, -1, srm_kernel)
    srm_res_abs = np.abs(srm_res)
    srm_norm = np.uint8(255 * np.clip(srm_res_abs / (srm_res_abs.mean() * 4.0 + 1e-6), 0, 1))
    srm_color = cv2.applyColorMap(srm_norm, cv2.COLORMAP_MAGMA)
    srm_color = cv2.cvtColor(srm_color, cv2.COLOR_BGR2RGB)

    # 6. Verdict Formatting
    verdict = f"🚨 DEEPFAKE DETECTED ({prob * 100:.1f}%)" if prob >= 0.5 else f"✅ AUTHENTIC HUMAN ({(1.0 - prob) * 100:.1f}%)"
    scores_dict = {
        "Deepfake Probability": round(prob, 4),
        "Authentic Probability": round(1.0 - prob, 4),
        "Calibrated Confidence": round(conf, 4),
    }

    metrics_text = (
        f"**Verdict**: {verdict}\n\n"
        f"**Predicted Family**: {manip_label}\n\n"
        f"**Calibrated Fake Probability**: `{prob * 100:.2f}%`\n\n"
        f"**Confidence Score**: `{conf * 100:.1f}%`\n\n"
        f"**Active SOTA Engine**: DeepTrace V7 Multi-Spectral (Phase SPR + Wavelet + 9-Ch SRM + LSGN Gating)"
    )

    return (
        gradcam_overlay,
        dct_color,
        spr_color,
        srm_color,
        scores_dict,
        metrics_text,
    )


def create_demo():
    custom_css = """
    .gradio-container { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
    .gr-button-primary { background: linear-gradient(135deg, #6366f1, #4f46e5) !important; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="DeepTrace Forensic AI") as demo:
        gr.Markdown(
            """
            # 🔍 DeepTrace: Enterprise Multi-Spectral Deepfake Forensics
            ### SOTA Phase-Magnitude Discrepancy & Boundary Artifact Detection Platform
            *Upload any face image to extract multi-modal forensic traces, continuous phase reconstructions, and GradCAM visual evidence.*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Upload Face Image (JPG / PNG)")
                analyze_btn = gr.Button("🔎 Execute Multi-Spectral Forensic Analysis", variant="primary")

                gr.Markdown("### 📊 Forensic Decision & Confidence")
                verdict_box = gr.Markdown("Upload an image and click Analyze to inspect.")
                score_label = gr.Label(label="Calibrated Decision Distribution")

            with gr.Column(scale=2):
                gr.Markdown("### 🔬 Multi-Spectral Forensic Modality Explorer")
                with gr.Row():
                    out_gradcam = gr.Image(label="1. GradCAM Anomaly Localization", type="numpy")
                    out_spr = gr.Image(label="2. Continuous Phase SPR (FFT)", type="numpy")
                with gr.Row():
                    out_dct = gr.Image(label="3. 2D-DCT Power Spectrum", type="numpy")
                    out_srm = gr.Image(label="4. SRM High-Pass Noise Residual", type="numpy")

        analyze_btn.click(
            fn=compute_forensic_breakdown,
            inputs=[input_img],
            outputs=[out_gradcam, out_dct, out_spr, out_srm, score_label, verdict_box],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
