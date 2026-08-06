# DeepTrace Architecture Specification — V1.5 Active System

**Document Version:** 1.5 (Post Phase 2 CLIP Unfreeze)  
**Target Hardware:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM)  
**Primary Framework:** PyTorch 2.6.0+cu124  

---

## 1. Executive Summary & Active Status

DeepTrace V1.5 is a multimodal deepfake detector operating on spatial visual cues, discrete cosine transform (DCT) frequency artifacts, and fine-tuned visual-semantic embeddings from CLIP ViT-B/32.

```
Overall Status: TRAINED & OPERATIONAL
- Baseline Val Accuracy: 99.68%
- Peak Val AUC: 0.9999
- Temperature Calibration: T = 4.396576
- Decision Threshold: 0.1341
```

## 2. Visual Pipeline Flowchart

```
                          ┌───────────────────────────┐
                          │     Input Media File      │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 [ Image Path ]                 [ Video Path ]
                         │                             │
                MTCNN Face Detector           Uniform Frame Sampler
                         │                             │
             ┌───────────┴───────────┐                 │
             ▼                       ▼                 │
     RGB Tensor (160x160)   2D-DCT Tensor (160x160)    │ (Frame Averaging)
             │                       │                 │
             ├───────────────────────┼─────────────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │ Spatial Encoder │     │Frequency Encoder│
    │ EfficientNet-B0 │     │EfficientNet-B0  │
    └────────┬────────┘     └────────┬────────┘
             │ (1280d)               │ (1280d)
             │                       │
             ├───────────┬───────────┤
             │           │           │
             ▼           ▼           │
     ┌─────────────────────────┐     │
     │   CLIP Alignment Module │     │
     │   (ViT-B/32 - Unfrozen) │     │
     └───────────┬─────────────┘     │
                 │ (256d)            │
                 │                   │
                 ▼                   ▼
    ┌──────────────────────────────────────────────────┐
    │          Multimodal Cross-Attention Fusion       │
    │         Stage 1: Spatial ↔ Frequency             │
    │         Stage 2: Concatenate CLIP (256d)         │
    └────────────────────────┬─────────────────────────┘
                             │ (512d)
                             ▼
    ┌──────────────────────────────────────────────────┐
    │                 Detection Head                   │
    │  ├─ Binary Head (BCE + Temp Scaling T=4.396)     │
    │  ├─ Manipulation Type Head (5 classes)           │
    │  └─ Generator Attribution Head (4 classes)       │
    └────────────────────────┬─────────────────────────┘
                             │
                             ▼
                [ Calibrated Output JSON ]
                - FAKE / REAL Verdict
                - Confidence Score (%)
                - Manipulation Type
```

---

## 3. Module State Matrix (Active vs Frozen)

| Module | Backbone / Method | Input Dim | Output Dim | Status | Trainable Params |
|---|---|---|---|---|---|
| **Spatial Encoder** | EfficientNet-B0 (ra_in1k) | (B, 3, 160, 160) | 1280d | **TRAINED** | 5.3M |
| **Frequency Encoder** | EfficientNet-B0 on 2D-DCT | (B, 3, 160, 160) | 1280d | **TRAINED** | 5.3M |
| **CLIP Alignment** | ViT-B/32 (openai) | (B, 3, 224, 224) | 256d | **PARTIALLY UNFROZEN (Phase 2)** | 15.16M (Blocks 10-11, ln_post, proj, linear projection) |
| **Multimodal Fusion** | 2-Layer Cross-Attention | (B, 1280+1280+256) | 512d | **TRAINED** | 8.4M |
| **Detection Head** | Multi-task MLP | 512d | 1 (binary) + 5 (type) | **TRAINED** | 0.3M |
| **Generator Head** | 2-Layer MLP | 512d | 4 classes | **TRAINED (Phase 1)** | 0.1M |
| **Temporal Model** | Video Swin Transformer Tiny | (B, T, 3, H, W) | 768d | *FROZEN / UNTRAINED* | 0M (28.3M params inactive) |
| **Physiology Encoder**| BiLSTM PPG Extractor | (B, T, 3, H, W) | 64d | *FROZEN / UNTRAINED* | 0M (0.2M params inactive) |

---

## 3. Data Processing & Inference Pipeline

```
[Input File] 
     │
     ├── Image (.jpg, .png, etc.) ──► MTCNN Crop ──► Resize 160x160 ──► RGB Tensor (3, 160, 160)
     │                                                              └─► 2D-DCT Block Transform (3, 160, 160)
     │
     └── Video (.mp4, .avi, etc.)  ──► Frame Uniform Sampler (N frames) ──► Per-Frame Crop/Resize
                                                                         └── Per-Frame Average Aggregation
```

### Inference Flow (Image Mode)
1. **Spatial Branch:** RGB tensor passed through EfficientNet-B0 → 1280d feature vector.
2. **Frequency Branch:** 2D-DCT log-magnitude spectrum passed through EfficientNet-B0 → 1280d feature vector.
3. **CLIP Branch:** Image resized to 224×224 bilinear → CLIP ViT-B/32 visual backbone → 256d projected alignment vector.
4. **Cross-Attention Fusion:**
   - Spatial features attend to Frequency features (LayerNorm + Multi-Head Self-Attention + FFN).
   - Frequency features attend to Spatial features.
   - Outputs averaged, concatenated with 256d CLIP projection and 64d zero-padded PPG vector → Linear projection → 512d fused vector.
5. **Detection Head Output:**
   - Binary Logit → Temperature Scaled (`/ 4.396576`) → Sigmoid → Calibrated Probability (`fake_prob`).
   - Decision threshold applied: `fake_prob > 0.1341` ? FAKE : REAL.
   - Multi-class Manipulation Head: Argmax over `[real, Deepfakes, Face2Face, FaceSwap, NeuralTextures]`.

---

## 4. Loss Function Structure

Phase 2 joint optimization loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}, y) + 0.3 \cdot \mathcal{L}_{\text{CLIP}}(\mathbf{z}_{\text{spatial}}, \mathbf{z}_{\text{CLIP}})$$

Where $\mathcal{L}_{\text{CLIP}} = 1.0 - \cos(\mathbf{z}_{\text{spatial}}, \mathbf{z}_{\text{CLIP}})$.

---

## 5. Technical Limitations & Gaps (For Review)

1. **Video Temporal Awareness:** The Video Swin Transformer and BiLSTM PPG encoders exist in code but are untrained. Video inference relies on frame-by-frame averaging.
2. **Cross-Dataset Generalisation:** High performance verified on Kaggle RealFake / StyleGAN (in-distribution), but unverified on FaceForensics++, Celeb-DF v2, or DFDC.
3. **Single Frequency Stream:** Uses 2D-DCT only; lacks spatial-domain high-pass noise residuals or Wavelet decomposition.
