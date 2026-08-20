---
tags:
  - roadmap
  - planning
  - deeptrace
updated: 2026-08-06
---

# 🛣️ DeepTrace — Training & Research Roadmap

← Back to [[HOME]]

---

## Phase Overview

```
Phase 1   GAN Finetune          ✅ COMPLETE
Phase 2   CLIP Partial Unfreeze 🟡 IN PROGRESS (3/10 epochs)
Phase 3A  Multi-Spectral V2     ⬜ NEXT
Phase 3B  ArcFace Identity      ⬜ PLANNED
Phase 4   FAISS RAG             ⬜ PLANNED
Phase 5   Cross-Dataset Eval    ⬜ PLANNED
Phase 6   Paper Submission      ⬜ TARGET
```

---

## ✅ Phase 1 — GAN Finetune (COMPLETE)

**Goal**: Teach V1 model to distinguish GAN generator types (StyleGAN vs Diffusion vs Real)

**What changed**:
- Added `GeneratorHead` (Linear 512→128→4) on top of fused features
- Loss: `BCE_binary + 0.4 × CE_generator` (reals ignored via `ignore_index=-1`)
- 10k StyleGAN images from held-out test split (zero overlap with train)

**Result**:
| Metric | Value |
|---|---|
| Epochs | 15/15 ✅ |
| Val Accuracy | 99.68% |
| Val AUC | 0.9998 |
| Best AUC | 0.9999 (Epoch 3) |
| Checkpoint | `checkpoints/v1_gan_finetune/best_model.pth` (534 MB) |

---

## 🟡 Phase 2 — CLIP Partial Unfreeze (RUNNING)

**Goal**: Improve generalization to unseen generators by fine-tuning the top CLIP ViT blocks

**What's unfrozen**:
- CLIP ViT-B/32 Blocks **10–11** (of 12)
- `ln_post` (post-attention layer norm)
- `proj` (output projection weight)
- Total new trainable params: **~14.6M**

**Config**:
```
epochs        : 10
batch_size    : 16  (effective: 64 with accum=4)
main_lr       : 1e-5
clip_lr       : 5e-6  (separate param group)
scheduler     : CosineAnnealingLR (eta_min=1e-7)
VRAM estimate : ~5.5 GB
```

**Current Progress** (as of 2026-08-06):
| Metric | Value |
|---|---|
| Epochs done | 3/10 |
| Epochs left | 7 (~14h) |
| Best AUC | 0.9999 |
| Val Accuracy | 99.655% |
| Checkpoint | `checkpoints/v2_clip_finetune/best_model.pth` |

**Resume**:
```powershell
.venv\Scripts\python.exe scripts/train_clip_finetune.py --epochs 10 --batch_size 16 --accumulation 4 --num_workers 2 --resume checkpoints/v2_clip_finetune/last.pth
```

---

## ⬜ Phase 3A — Multi-Spectral Frequency (NEXT PRIORITY)

**Goal**: Replace single DCT branch with 3 parallel frequency branches for richer forgery signals

**New branches** (coded in `models/spectral_branches.py`):

| Branch | Signal Caught |
|---|---|
| **FFT Magnitude** | Global periodic aliasing, upsampling artifacts |
| **Wavelet (DWT)** | Multi-scale texture + edge anomalies |
| **SRM Noise Residual** | Generator-specific noise floor fingerprints |

**Why this matters**: StyleGAN and Diffusion models leave different frequency signatures. SRM residuals expose the noise floor unique to each generator — a spatial CNN completely misses this.

**Status**: Code complete (`spectral_branches.py`, `detector_v2.py`). Needs:
- [ ] Fix `train_v2.py` to support `kaggle_realfake` dataset
- [ ] Stage 1 training: multi-spectral only (freeze everything else)
- [ ] Merge with Phase 2 CLIP checkpoint

---

## ⬜ Phase 3B — ArcFace Identity Consistency

**Goal**: Catch faceswap/inpainting deepfakes by verifying identity consistency across facial regions

**How it works**: ArcFace-R18 (frozen) extracts a 128-dim identity embedding. Inconsistencies between the inner face and surrounding regions expose boundary artifacts that spatial CNNs miss.

**Status**: Coded (`models/identity_encoder.py`). Activates in Phase 3+ alongside spectral branches.

---

## ⬜ Phase 4 — FAISS RAG Artifact Retrieval

**Goal**: At inference time, match the face's spectral fingerprint against a database of known generator signatures for zero-shot generalisation + explainability

**Architecture**:
```
Query → Multi-spectral features → 2048-dim vector
                                        ↓
                              FAISS.search(k=8)
                                        ↓
                      Top-8 nearest known generator samples
                                        ↓
              Weighted vote + fusion with detector → attribution
```

**Why this is novel**:
- New generators: just add their fingerprints to FAISS index (no retraining)
- Explainable: "matches StyleGAN2 signature (sim=0.89)"
- Forensically auditable: chain-of-evidence output

**Status**: Coded (`models/rag_retrieval.py`, `scripts/build_rag_db.py`). Needs Phase 3 checkpoint first.

---

## ⬜ Phase 5 — Cross-Dataset Evaluation

**Goal**: Prove generalization for publication

**Datasets needed**:
- FaceForensics++ (C23 + C40 compression)
- Celeb-DF v2
- DFDC (DeepFake Detection Challenge)
- DiffusionDB faces / FLUX-generated faces

**Metrics to report**:
- Accuracy, ROC AUC, EER, ECE (Expected Calibration Error)
- Cross-dataset transfer table
- Robustness: JPEG Q={100,80,60,40}, Gaussian blur σ={1,2,3,5}

---

## ⬜ Phase 6 — Paper Submission

**Target venues**:
1. **IEEE T-IFS** (Transactions on Information Forensics & Security) — primary
2. **ACM Multimedia 2026** — secondary
3. **WACV 2027** — fallback

**Key contributions to highlight**:
1. Multi-spectral fusion (FFT + Wavelet + SRM + DCT) — unified frequency analysis
2. FAISS RAG for zero-shot generator attribution — novel application
3. Calibrated uncertainty (MC Dropout + temperature scaling) — forensic reliability
4. CLIP semantic alignment for cross-generator generalization
