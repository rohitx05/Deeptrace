# Architecture Overview

A high-level overview of the DeepTrace multi-stream detection pipeline.

## Pipeline Diagram

```
┌─────────────┐  ┌──────────────────┐  ┌─────────────────┐
│  Spatial     │  │  Multi-Spectral  │  │  CLIP Alignment │
│  (EfficientNet)│ │ (FFT+Wavelet+   │  │  (ViT-L/14)     │
│              │  │  Noise+DCT)      │  │                 │
└──────┬───────┘  └───────┬──────────┘  └───────┬─────────┘
       │                  │                     │
       │   ┌──────────────┤                     │
       │   │   (video only)│                    │
       │   │  ┌────────────┴──┐ ┌────────────┐  │
       │   │  │  Temporal     │ │  Identity   │  │
       │   │  │  (Video Swin) │ │  (ArcFace)  │  │
       │   │  └──────┬────────┘ └─────┬───────┘  │
       │   │         │               │           │
       ▼   ▼         ▼               ▼           ▼
  ┌─────────────────────────────────────────────────┐
  │       Multimodal Transformer Fusion (8-token)   │
  └──────────────────────┬──────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Extended Detection    │
            │  Head                  │
            │  • binary (real/fake)  │
            │  • manipulation type   │
            │  • generator attrib.   │
            └────────────────────────┘
```

## Module Inventory

| Module                    | File                         | Input                     | Output Dim |
|---------------------------|------------------------------|---------------------------|------------|
| Spatial Encoder           | `models/spatial_encoder.py`  | RGB image (224×224)       | 1280       |
| Frequency Encoder         | `models/frequency_encoder.py`| DCT coefficients          | 512        |
| Multi-Spectral Combiner   | `models/spectral_branches.py`| FFT, Wavelet, Noise, DCT  | 512        |
| CLIP Alignment            | `models/clip_alignment.py`   | Image + text prompt       | 512        |
| Video Swin Transformer    | `models/temporal_model.py`   | Frame sequence (T×C×H×W)  | 768        |
| Physiology Encoder        | `models/physiology_encoder.py`| Facial landmarks         | 256        |
| Identity Encoder          | `models/identity_encoder.py` | Face crops                | 512        |
| RAG Retrieval             | `models/rag_retrieval.py`    | Feature query             | 128        |
| Transformer Fusion        | `models/fusion_transformer.py`| All stream tokens        | 512        |
| Detection Head V2         | `models/detection_head_v2.py`| Fused features            | 3 outputs  |
| MC-Dropout Uncertainty    | `models/uncertainty.py`      | Any model with Dropout    | stats      |

## Detector Versions

- **V1** (`models/detector.py`): Spatial + Frequency + CLIP + simple fusion head.
- **V2** (`models/detector_v2.py`): All modules above. Supports both `image` and `video` modes via a single `mode` argument.

## Evaluation Flow

1. Load checkpoint → instantiate `DeepfakeDetectorV2`.
2. Wrap with `MCDropoutWrapper` (optional) for uncertainty estimation.
3. Pass to `Evaluator` which runs inference, computes metrics, and generates plots.
4. Results are saved as JSON in `results/` for traceability.
