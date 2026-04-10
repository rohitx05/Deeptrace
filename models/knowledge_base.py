"""
Deepfake Knowledge Base.
Manages structured knowledge about deepfake generators, manipulation techniques,
artifact types, and dataset references. Queried by the RAG system.
"""

import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ─── Built-in Knowledge ─────────────────────────────────────────────────────

GENERATOR_KNOWLEDGE = {
    "GAN": {
        "description": "Generative Adversarial Network based face synthesis",
        "subtypes": ["StyleGAN", "StyleGAN2", "StyleGAN3", "ProGAN", "StarGAN"],
        "artifacts": {
            "spectral": "High-frequency spectral peaks from upsampling layers, checkerboard patterns in FFT",
            "spatial": "Asymmetric facial features, inconsistent lighting, hair/background blending issues",
            "temporal": "Frame-to-frame identity flickering, inconsistent motion",
        },
        "fingerprint_signature": "Periodic peaks in frequency spectrum around Nyquist frequency",
    },
    "Diffusion": {
        "description": "Diffusion model based face synthesis",
        "subtypes": ["Stable Diffusion", "DALL-E", "Midjourney", "DeepFloyd IF"],
        "artifacts": {
            "spectral": "Smoother frequency roll-off than GANs, less high-frequency energy",
            "spatial": "Over-smooth textures, inconsistent reflections, unusual eye details",
            "temporal": "Inter-frame texture wobble, unstable fine details",
        },
        "fingerprint_signature": "Broadband frequency attenuation in high-frequency bands",
    },
    "FaceSwap": {
        "description": "Face swap methods (Deepfakes autoencoder, 3D morphable models)",
        "subtypes": ["Deepfakes (autoencoder)", "FaceSwap (3DMM)", "FSGAN"],
        "artifacts": {
            "spectral": "Boundary artifacts in DCT, compression mismatches at swap boundaries",
            "spatial": "Visible blending boundaries, skin color mismatch, misaligned features",
            "temporal": "Temporal boundary flickering, inconsistent head pose interpolation",
        },
        "fingerprint_signature": "Ring-like artifacts in frequency domain at face boundary",
    },
    "FaceReenactment": {
        "description": "Facial expression transfer methods",
        "subtypes": ["Face2Face", "NeuralTextures", "First Order Motion"],
        "artifacts": {
            "spectral": "Local frequency distortions around mouth/eyes",
            "spatial": "Unnatural expression transitions, texture inconsistencies around features",
            "temporal": "Expression desynchronization with audio, unnatural micro-movements",
        },
        "fingerprint_signature": "Localized spectral anomalies in facial feature regions",
    },
}

ARTIFACT_TAXONOMY = {
    "spectral_artifacts": [
        "high_freq_attenuation",
        "checkerboard_pattern",
        "spectral_peaks",
        "frequency_rolloff_mismatch",
        "dct_boundary_artifacts",
        "wavelet_coefficient_anomalies",
        "noise_residual_patterns",
    ],
    "spatial_artifacts": [
        "blending_boundary",
        "skin_color_mismatch",
        "asymmetric_features",
        "reflection_inconsistency",
        "hair_boundary_artifacts",
        "background_inconsistency",
        "texture_smoothing",
    ],
    "temporal_artifacts": [
        "identity_flickering",
        "expression_desync",
        "motion_artifacts",
        "frame_interpolation_artifacts",
        "ppg_signal_anomaly",
        "temporal_noise_pattern",
    ],
    "physiological_artifacts": [
        "irregular_ppg",
        "absent_micro_expressions",
        "unnatural_eye_blinking",
        "inconsistent_blood_flow",
    ],
}

DATASET_REFERENCES = {
    "FaceForensics++": {
        "generators": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        "compression": ["c23", "c40"],
        "size": "1000 original + 4000 manipulated videos",
        "key_characteristic": "Multiple manipulation types, controlled compression",
    },
    "CelebDF": {
        "generators": ["Improved autoencoder"],
        "compression": ["native"],
        "size": "590 real + 5639 synthesized videos",
        "key_characteristic": "Higher quality synthesis, harder detection",
    },
    "DFDC": {
        "generators": ["Multiple unknown methods"],
        "compression": ["mixed"],
        "size": "128,154 videos",
        "key_characteristic": "Large scale, diverse, real-world conditions",
    },
    "WildDeepfake": {
        "generators": ["Unknown internet sources"],
        "compression": ["mixed"],
        "size": "7314 face sequences",
        "key_characteristic": "In-the-wild deepfakes from the internet",
    },
}


class DeepfakeKnowledgeBase:
    """
    Structured knowledge base for deepfake detection.
    Provides context for the RAG retrieval system.
    """

    def __init__(self, custom_kb_path: str = None):
        self.generators = dict(GENERATOR_KNOWLEDGE)
        self.artifacts = dict(ARTIFACT_TAXONOMY)
        self.datasets = dict(DATASET_REFERENCES)
        self.custom_entries = []

        if custom_kb_path and Path(custom_kb_path).exists():
            self.load_custom(custom_kb_path)

    def query_generator(self, generator_type: str) -> dict:
        """Get knowledge about a specific generator type."""
        return self.generators.get(generator_type, {})

    def query_artifacts(self, artifact_category: str) -> list:
        """Get artifact types for a category."""
        return self.artifacts.get(artifact_category, [])

    def get_artifact_description(self, generator_type: str, domain: str) -> str:
        """Get artifact description for a generator in a specific domain."""
        gen = self.generators.get(generator_type, {})
        return gen.get("artifacts", {}).get(domain, "No information available")

    def classify_artifact_evidence(self, predictions: dict) -> dict:
        """
        Interpret model predictions using knowledge base.

        Args:
            predictions: model output dict

        Returns:
            Evidence dict with matched knowledge
        """
        evidence = {"detected_artifacts": [], "generator_analysis": {}, "confidence_factors": []}

        if predictions.get("fake_probability", 0) > 0.5:
            # Generator attribution
            gen_types = ["GAN", "Diffusion", "FaceSwap", "Unknown"]
            if "generator_pred" in predictions:
                pred_gen = gen_types[predictions["generator_pred"]]
                gen_info = self.query_generator(pred_gen)
                evidence["generator_analysis"] = {
                    "predicted_type": pred_gen,
                    "description": gen_info.get("description", ""),
                    "expected_artifacts": gen_info.get("artifacts", {}),
                    "fingerprint": gen_info.get("fingerprint_signature", ""),
                }

            if predictions.get("uncertainty", 0) < 0.1:
                evidence["confidence_factors"].append("Low uncertainty — high reliability")
            elif predictions.get("uncertainty", 0) > 0.3:
                evidence["confidence_factors"].append("High uncertainty — manual review recommended")

        return evidence

    def add_custom_entry(self, entry: dict):
        """Add a custom knowledge entry."""
        self.custom_entries.append(entry)

    def save_custom(self, path: str):
        """Save custom entries to disk."""
        with open(path, "w") as f:
            json.dump({
                "custom_entries": self.custom_entries,
                "generators": self.generators,
                "artifacts": self.artifacts,
            }, f, indent=2)

    def load_custom(self, path: str):
        """Load custom entries from disk."""
        with open(path, "r") as f:
            data = json.load(f)
            self.custom_entries = data.get("custom_entries", [])
            if "generators" in data:
                self.generators.update(data["generators"])
            logger.info(f"Loaded custom KB: {len(self.custom_entries)} entries")

    def get_full_context(self) -> str:
        """Get full knowledge base as formatted string for reports."""
        lines = ["=== DEEPFAKE KNOWLEDGE BASE ===\n"]
        for gen_type, info in self.generators.items():
            lines.append(f"\n[{gen_type}]")
            lines.append(f"  {info['description']}")
            lines.append(f"  Subtypes: {', '.join(info['subtypes'])}")
            lines.append(f"  Fingerprint: {info['fingerprint_signature']}")
        return "\n".join(lines)
