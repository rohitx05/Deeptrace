# DeepTrace Architecture Specification — V2 Proposed System

**Document Version:** 2.0 Blueprint  
**Target Hardware:** NVIDIA RTX 4050 Laptop GPU (6GB VRAM)  
**Status:** Coded (`models/detector_v2.py`, `models/spectral_branches.py`, `models/rag_retrieval.py`), Untrained  

---

## 1. System Overview

DeepTrace V2 expands the 51.7M parameter baseline into a 150M total parameter multi-spectral, retrieval-augmented deepfake detection system with calibrated epistemic uncertainty.

```
Key Blueprint Additions:
1. Multi-Spectral Frequency Analysis (DCT + FFT + Wavelet + Noise Residual)
2. Identity Consistency Encoder (ArcFace ResNet-18)
3. Retrieval-Augmented Generation (RAG) artifact search via FAISS
4. 8-Token Multimodal Transformer Encoder Fusion
5. Epistemic Uncertainty via Monte Carlo (MC) Dropout
```

## 2. Visual Pipeline Flowchart

```
                                  ┌───────────────────────────┐
                                  │     Input Media File      │
                                  └─────────────┬─────────────┘
                                                │
                                       MTCNN Face Extractor
                                                │
       ┌──────────────┬──────────────┬──────────┴───┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼              ▼              ▼
 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
 │ Spatial  │   │ Spectral │   │ Identity │   │ CLIP ViT │   │ VideoSwin│   │ Physiology│
 │ Encoder  │   │ Combiner │   │ Encoder  │   │ Alignment│   │ Temporal │   │ Encoder  │
 │EffNet-B0 │   │(4-Branch)│   │ArcFaceR18│   │  B-32    │   │ Tiny     │   │ BiLSTM   │
 └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
      │ (1280d)      │ (1280d)      │ (128d)       │ (256d)       │ (768d)       │ (64d)
      │              │              │              │              │              │
      ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┘
      │              │
      ▼              ▼
 ┌─────────────────────────┐
 │ RAG Artifact Subsystem  │
 │  FAISS Top-8 Query      │
 └───────────┬─────────────┘
             │ (256d Context Token)
             │
             ▼
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │                 8-TOKEN MULTIMODAL TRANSFORMER ENCODER FUSION                  │
 │ Tokens: [CLS], Spatial, Spectral, Temporal, Physio, CLIP, Identity, RAG        │
 │ Modality Gating: σ(g_i) per token for graceful missing-modality degradation   │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Fused [CLS] Token (512d)
                                         ▼
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │                     EXTENDED 5-BRANCH DETECTION HEAD                           │
 │                                                                                │
 │ ├─ Binary Logit Head (Real vs Fake)                                            │
 │ ├─ Manipulation Type Head (5 Classes)                                          │
 │ ├─ Generator Attribution Head (StyleGAN, Diffusion, FaceSwap, Unknown)        │
 │ ├─ Confidence Calibration Head                                                 │
 │ └─ MC Dropout Engine (N=10 Stochastic Passes → Epistemic Uncertainty σ)        │
 └───────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
                             [ Rich Forensic Output ]
                             - Probability & Binary Verdict
                             - Epistemic Uncertainty (std)
                             - Generator Attribution & Type
                             - RAG Evidence / Retrieved Artifacts
```

---

## 3. Expanded Component Matrix

```
[Input Image/Video]
       │
       ├──► Spatial Encoder (EfficientNet-B0) ──────────────────────────► 1280d Token
       │
       ├──► Multi-Spectral Combiner ────────────────────────────────────► 1280d Token
       │      ├─ 2D-DCT (EfficientNet-B0) -> 1280d
       │      ├─ 2D-FFT Magnitude+Phase (4-Layer CNN) -> 256d
       │      ├─ DWT Wavelet LL,LH,HL,HH (4-Layer CNN) -> 256d
       │      └─ SRM High-Pass Noise Residual (4-Layer CNN) -> 256d
       │
       ├──► Identity Consistency Encoder (ArcFace R18 Frozen) ──────────► 128d Token
       │
       ├──► RAG FAISS Index Search (Top-k=8 Artifact Retrieval) ─────────► 256d Token
       │
       ├──► CLIP ViT-B/32 Semantic Projection ──────────────────────────► 256d Token
       │
       └──► [Temporal VideoSwin] + [BiLSTM PPG] ───────────────────────► 768d + 64d Tokens
```

---

## 3. Multimodal Transformer Fusion (8-Token Attention)

Replaces simple cross-attention concatenation with a 4-Layer 8-Head Transformer Encoder (`hidden_dim=512`).

Tokens fed into Transformer:
1. `[CLS]` Token (512d) → Fused output representation
2. `Spatial Token` (1280d → 512d)
3. `Spectral Token` (1280d → 512d)
4. `Temporal Token` (768d → 512d)
5. `Physiology Token` (64d → 512d)
6. `CLIP Token` (256d → 512d)
7. `Identity Token` (128d → 512d)
8. `RAG Context Token` (256d → 512d)

*Modality Gating:* Each token passes through a learnable scalar gate $\sigma(g_i) \in [0, 1]$ to allow graceful degradation when modalities are missing (e.g. static image mode missing temporal tokens).

---

## 4. RAG Artifact Retrieval Subsystem

1. **Database Construction:** Extract 512d fused embeddings from training samples, store in CPU-side `FAISS IndexFlatIP` with JSON metadata:
   - `label`: real/fake
   - `generator_type`: StyleGAN / Diffusion / FaceSwap / Unknown
   - `artifact_type`: boundary_blur, spectral_spike, etc.
2. **Inference Query:** Query embedding retrieved against FAISS index → Top-8 matches aggregated via cosine-similarity weighted mean → 256d context token.

---

## 5. Extended 5-Branch Detection Head & Uncertainty

The output CLS token (512d) drives five parallel prediction heads:

1. **Binary Head:** Real vs Fake classification logit.
2. **Manipulation Head:** 5-class classification (`[real, Deepfakes, Face2Face, FaceSwap, NeuralTextures]`).
3. **Generator Attribution Head:** 4-class classification on spectral token (`[GAN, Diffusion, FaceSwap, Unknown]`).
4. **Calibrated Confidence Head:** Predicts expected error bound.
5. **MC Dropout Uncertainty Engine:** Executes $N=10$ stochastic forward passes at inference with dropout enabled.
   $$\sigma_{\text{epistemic}} = \sqrt{\frac{1}{N}\sum_{i=1}^N (p_i - \bar{p})^2}$$

---

## 6. Target Multi-Task Loss Formulation

$$\mathcal{L}_{\text{V2}} = \mathcal{L}_{\text{BCE}} + 0.4\mathcal{L}_{\text{manip}} + 0.4\mathcal{L}_{\text{generator}} + 0.3\mathcal{L}_{\text{CLIP}} + 0.2\mathcal{L}_{\text{identity}} + 0.15\mathcal{L}_{\text{RAG}}$$

---

## 7. Comparative Benchmark Plan (What is needed for publication)

To evaluate this V2 architecture against published SOTA models, the following cross-dataset benchmark matrix is planned:

| Model Architecture | In-Distribution (Kaggle) | Cross-Dataset (FaceForensics++) | Cross-Dataset (Celeb-DF v2) |
|---|---|---|---|
| FaceXRay (HRNet) | Baseline | Target > 85.0% AUC | Target > 78.0% AUC |
| UniversalFakeDetect (CLIP-Probe) | Baseline | Target > 88.0% AUC | Target > 82.0% AUC |
| **DeepTrace V1.5 (Current)** | **0.9999 AUC** | *Pending Evaluation* | *Pending Evaluation* |
| **DeepTrace V2 (Planned)** | **Target 0.9999 AUC** | **Target > 92.0% AUC** | **Target > 88.0% AUC** |
