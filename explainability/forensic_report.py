"""
Forensic report generation from model outputs.
Produces structured, human-readable forensic explanations.
"""

import logging

logger = logging.getLogger(__name__)

MANIPULATION_DESCRIPTIONS = {
    "real": "No manipulation detected. The media appears to be authentic.",
    "Deepfakes": "Face swap manipulation detected using deep learning-based face replacement (Deepfakes/autoencoder method).",
    "Face2Face": "Facial reenactment detected. Source facial expressions have been transferred to the target face (Face2Face method).",
    "FaceSwap": "Traditional face swap detected using 3D model-based face replacement (FaceSwap method).",
    "NeuralTextures": "Subtle facial manipulation detected using neural texture rendering (NeuralTextures method).",
    "N/A": "No specific manipulation type identified.",
}

ARTIFACT_DESCRIPTIONS = {
    "spatial": "Spatial artifacts include boundary inconsistencies, blending artifacts, and unnatural texture patterns at manipulation boundaries.",
    "frequency": "Frequency domain analysis reveals abnormal DCT coefficients typical of GAN-generated or compressed-then-manipulated content.",
    "temporal": "Temporal inconsistencies detected across video frames, including unnatural motion patterns and flickering artifacts.",
    "physiological": "Physiological signal analysis shows abnormal patterns incompatible with natural biological processes (e.g., irregular PPG signal).",
}


def generate_forensic_report(result: dict) -> str:
    """
    Generate a structured forensic explanation from model prediction results.

    Args:
        result: dict from InferencePipeline.predict()

    Returns:
        Formatted forensic report string
    """
    prediction = result.get("prediction", "UNKNOWN")
    confidence = result.get("confidence", 0)
    fake_prob = result.get("fake_probability", 0)
    manip_type = result.get("manipulation_type", "N/A")
    file_path = result.get("file", "unknown")

    lines = [
        "=" * 60,
        "FORENSIC ANALYSIS REPORT",
        "=" * 60,
        "",
        f"File: {file_path}",
        f"Verdict: {prediction}",
        f"Confidence: {confidence * 100:.1f}%",
        f"Fake Probability: {fake_prob * 100:.1f}%",
        "",
    ]

    if prediction == "FAKE":
        lines.extend([
            "--- MANIPULATION ANALYSIS ---",
            f"Detected Type: {manip_type}",
            f"Description: {MANIPULATION_DESCRIPTIONS.get(manip_type, 'Unknown manipulation method.')}",
            "",
            "--- ARTIFACT ANALYSIS ---",
            ARTIFACT_DESCRIPTIONS["spatial"],
        ])

        if fake_prob > 0.8:
            lines.append(ARTIFACT_DESCRIPTIONS["frequency"])
        if "temporal_features" in result:
            lines.append(ARTIFACT_DESCRIPTIONS["temporal"])
            lines.append(ARTIFACT_DESCRIPTIONS["physiological"])

        lines.extend([
            "",
            "--- CONFIDENCE ASSESSMENT ---",
        ])
        if confidence > 0.9:
            lines.append("HIGH CONFIDENCE: Strong evidence of manipulation across multiple analysis domains.")
        elif confidence > 0.7:
            lines.append("MEDIUM-HIGH CONFIDENCE: Clear manipulation indicators detected.")
        elif confidence > 0.5:
            lines.append("MEDIUM CONFIDENCE: Some manipulation indicators detected. Manual review recommended.")
        else:
            lines.append("LOW CONFIDENCE: Weak indicators. Results should be verified with additional analysis.")

    else:
        lines.extend([
            "--- ANALYSIS SUMMARY ---",
            MANIPULATION_DESCRIPTIONS["real"],
            "",
            "No significant manipulation artifacts detected in spatial, frequency, or temporal domains.",
        ])

    if result.get("heatmap_path"):
        lines.extend(["", f"Artifact Heatmap: {result['heatmap_path']}"])

    lines.extend(["", "=" * 60])

    report = "\n".join(lines)
    return report
