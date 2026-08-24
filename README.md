# DeepTrace

**Multimodal deepfake detection research combining spatial, frequency, and vision-language representations — with rigorous cross-dataset evaluation.**

DeepTrace investigates whether a multi-stream neural pipeline trained on GAN-synthesised face images can generalise to video-domain forgeries (face swaps, reenactments). The project tracks a clear and substantial benchmark-to-generalisation gap and documents every confound honestly.

> **Research status:** Active. All numbers reported here are drawn from verified JSON result files in `results/`. No claimed result exceeds what the JSON evidence supports.

---

## Key Results

### In-Domain Held-Out Benchmark (V2 · Kaggle Real-vs-Fake)

Trained and calibrated on the Kaggle Real-vs-Fake dataset (GAN/StyleGAN faces). Evaluated on a strictly held-out 20 000-sample test split.

| Metric | Value |
|---|---|
| Accuracy (calibrated, t = 0.63) | **99.80%** |
| ROC-AUC | **0.99995** |
| F1-Score | **~99.80%** |
| Brier Score | **0.00193** |

> **Scope caveat.** These numbers apply only to the Kaggle Real-vs-Fake test distribution (GAN-synthesised still images). They do **not** represent general deepfake-detection performance.

---

### Cross-Dataset / FaceForensics++ Generalisation

All FF++ evaluations use compression level c23. Results are from strictly held-out, video-disjoint splits — no training video sequence appears in any test cohort.

#### V2 True Zero-Shot (Kaggle-trained -> FF++ without fine-tuning)

| FF++ Cohort | Accuracy | ROC-AUC |
|---|---|---|
| Deepfakes | 50.00% | 0.5059 |
| FaceSwap | 50.00% | 0.5012 |
| Face2Face | 50.00% | 0.5015 |
| NeuralTextures | 50.00% | 0.5022 |
| FaceShifter | 50.00% | 0.5024 |
| DeepFakeDetection | 50.00% | 0.5084 |
| **Overall (N = 14 000)** | **14.29%** | **0.5036** |

The 14.29% overall accuracy is an artefact of 1:6 class imbalance (2 000 real, 12 000 fake); the model predicts nearly everything as real, reflected in the near-chance AUC.

#### V7 Multi-Source Fine-Tuned (actor-disjoint, 5 genuine cohorts)

V7 was trained on Kaggle + FF++ c23. The actor-disjoint split assigns 600 actors to training and 400 disjoint actors to testing (seed 42, DFD excluded due to studio/YouTube domain confound).

| FF++ Cohort | ROC-AUC | Calibrated Acc |
|---|---|---|
| Deepfakes | 0.6405 | 61.06% |
| FaceSwap | 0.5630 | 55.56% |
| FaceShifter | 0.5695 | 55.74% |
| Face2Face | 0.5623 | 55.35% |
| NeuralTextures | 0.5693 | 55.63% |
| **Macro Average (5 cohorts)** | **0.5809** | — |

#### V9 Clean Actor-Disjoint (100% leak-free, per-cohort calibrated thresholds)

| FF++ Cohort | ROC-AUC | Calibrated Acc |
|---|---|---|
| Deepfakes | 0.6289 | 60.00% |
| FaceSwap | 0.5815 | 57.25% |
| FaceShifter | 0.6171 | 58.98% |
| Face2Face | 0.5733 | 56.02% |
| NeuralTextures | 0.5647 | 55.49% |
| **Macro Average (5 cohorts)** | **0.5931** | **57.55%** |

#### Benchmark-to-Generalisation Gap (summary)

| Setting | ROC-AUC |
|---|---|
| In-domain Kaggle held-out (V2) | 0.99995 |
| Zero-shot FF++ (V2) | 0.5036 |
| Actor-disjoint FF++ (V7) | 0.5809 |
| Actor-disjoint FF++ (V9) | 0.5931 |

The gap is large and consistent. In-domain results reflect near-perfect discrimination of StyleGAN faces from FFHQ photographs. Actor-disjoint results reflect genuine but modest cross-manipulation transfer (~0.59 macro AUC vs. 0.50 chance).

---

## Architecture

DeepTrace V2 assembles multiple specialist modules into a multimodal transformer fusion pipeline. The actively-trained path for image-mode inference:

`
Input Image (160 x 160)
    |-> Spatial Encoder (EfficientNet-B0, 1280d)            -|
    |-> Frequency Encoder (EfficientNet-B0 on 2D-DCT, 1280d) -|
    |-> Spectral Combiner (FFT + Wavelet + SRM/Gabor, 1280d)  |--> MultimodalTransformerFusion (8-token, 512d)
    |-> CLIP Alignment (OpenCLIP ViT-B/32, 256d projected)   -|         |
    |-> RAG Retrieval (FAISS artifact DB, 256d)              -|         v
                                                                ExtendedDetectionHead
                                                                |-- Binary logit (real / fake)
                                                                |-- Manipulation type (5 classes)
                                                                |-- Generator attribution (4 classes)
                                                                         |
                                                                Temperature Scaling Calibration
                                                                         |
                                                                Calibrated Probability + Verdict
`

`mermaid
flowchart TD
    IN["Input Image 160x160"]
    SE["Spatial Encoder\nEfficientNet-B0 -> 1280d"]
    FE["Frequency Encoder\nEfficientNet-B0 on 2D-DCT -> 1280d"]
    SC["Spectral Combiner\nFFT · Wavelet · SRM/Gabor -> 1280d"]
    CL["CLIP Alignment\nOpenCLIP ViT-B/32 -> 256d"]
    RAG["RAG Retrieval\nFAISS Artifact DB -> 256d"]
    FUS["Multimodal Transformer Fusion\n8-token · 4-layer · 8-head -> 512d"]
    HEAD["Extended Detection Head\nBinary · Manipulation Type · Generator"]
    CAL["Temperature Scaling Calibration"]
    OUT["Calibrated Verdict + Confidence + GradCAM"]

    IN --> SE
    IN --> FE
    IN --> SC
    IN --> CL
    SE --> RAG
    SE --> FUS
    FE --> SC
    SC --> FUS
    CL --> FUS
    RAG --> FUS
    FUS --> HEAD
    HEAD --> CAL
    CAL --> OUT
`

### Module Table

| Module | Method | Output Dim | Status |
|---|---|---|---|
| **Spatial Encoder** | EfficientNet-B0 (ra_in1k) | 1280d | Trained |
| **Frequency Encoder** | EfficientNet-B0 on 2D-DCT spectrum | 1280d | Trained |
| **Spectral Combiner** | FFT + 2-level Wavelet + 9-ch SRM/Gabor + LSGN gating | 1280d | Trained (V7) |
| **CLIP Alignment** | OpenCLIP ViT-B/32, top 2 blocks partially unfrozen | 256d | Trained (Phase 2) |
| **RAG Retrieval** | FAISS-indexed feature DB, learned query/output projections | 256d | Trained |
| **Fusion** | 4-layer multi-head transformer, 8 modality tokens | 512d | Trained |
| **Detection Head** | MLP: binary + 5-class type + 4-class generator | — | Trained |
| **Calibration** | Temperature scaling (T fitted on validation split) | — | Fitted |
| **Temporal Model** | Video Swin Transformer Tiny | 768d | **Implemented, not trained** |
| **Physiology Encoder** | BiLSTM PPG extractor | 64d | **Implemented, not trained** |
| **Identity Encoder** | ArcFace-based | 128d | **Implemented, not trained** |

The Temporal, Physiology, and Identity modules are present in the codebase but not trained. Video inference relies on per-frame processing and frame-average aggregation.

**Spectral Combiner branches** (models/spectral_branches.py):
- **FFT branch** — 4-channel continuous phase and spatial phase reconstruction (SPR).
- **Wavelet branch** — 2-level Haar wavelet packet decomposition (7 sub-bands).
- **Noise residual branch** — 9-channel forensic filterbank (5 SRM high-pass + 4 directional Gabor).
- **LSGN router** — dual-pooling (GAP + GMP) gating network that weights the three branches adaptively.

---

## Evaluation

### Held-Out Benchmark (Kaggle Real-vs-Fake)

The 99.80% / AUC 0.99995 result is from a 20 000-sample test split held out from the Kaggle Real-vs-Fake dataset. Training data: FFHQ real photographs vs. StyleGAN-synthesised faces. This is a single-source, same-domain evaluation and **should not be used as a proxy for general deepfake detection**.

A JPEG compression shortcut test (esults/benchmark_eval_v7/reconciliation_and_calibration_report.json) confirmed that the model's real-face logits shift upward under JPEG re-compression, indicating sensitivity to compression artefacts.

### Cross-Dataset / FF++ Evaluation

Two models were evaluated on FaceForensics++ c23:

- **V2 zero-shot** — trained on Kaggle only, applied directly to FF++ with no fine-tuning. AUC ~0.50 across all cohorts; no meaningful transfer.
- **V7 multi-source** — trained on Kaggle + FF++ c23 (training-actor slices only). Evaluated on strictly video-disjoint held-out splits. Macro AUC ~0.58–0.63 depending on cohort.

The DFD cohort produced an anomalous AUC of 1.0 at video level (near-chance accuracy). Forensic audit (esults/benchmark_eval_v7/leakage_audit_and_heldout_verification.json) traced this to a studio-vs-YouTube recording environment shortcut — not facial forgery cues. **DFD is excluded from all actor-disjoint macro averages from V9 onwards.**

### Actor-Disjoint Evaluation (V7 / V9)

**Why actor-disjoint?** Standard video-level splits can leak actor identity — the same person appearing in training as "real" and in test as "fake" allows the model to learn identity rather than forgery signals. Actor-disjoint splits assign every FF++ video subject entirely to either train or test.

**Protocol:**
- 1 000 unique actors split 60/40 by actor identity (seed 42).
- Train actors: 600 · Test actors: 400.
- Balanced 50/50 real/fake sampling per cohort: N = 2 840 per cohort.
- DFD excluded (unresolvable domain confound).
- Calibrated accuracy uses per-cohort Youden-optimal threshold.

Result files: esults/benchmark_eval_v7/actor_disjoint_leak_free_eval.json, esults/benchmark_eval_v7/v9_clean_actor_disjoint_eval.json

---

## Explainability

GradCAM is implemented in explainability/gradcam.py, targeting the last convolutional block of the spatial encoder (EfficientNet-B0 locks[-1]). The implementation uses standard gradient-weighted class activation mapping with global average pooling of gradients over the spatial dimensions.

Saved heatmaps in esults/gradcam_v6/ and esults/gradcam_v7/:

| Sample | P(Fake) | Observed Pattern |
|---|---|---|
| Authentic face (V7) | 0.421 | Diffuse, low-magnitude activation |
| StyleGAN synthesis (V7) | 0.985 | Concentrated activation on facial landmarks |
| FF++ FaceSwap blending (V6) | 0.824 | Activation concentrated on jawline contour |
| FF++ DeepFakeDetection (V6) | 1.000 | Activation over face-swap lighting boundary |

These observations are qualitative. GradCAM localises regions influencing the classifier's output; it does not provide causal evidence of which signal — pixel statistics, compression, identity, or boundary seam — drives the prediction.

Generation:
`ash
python scripts/generate_v6_gradcam_artifacts.py
python scripts/generate_v7_gradcam_artifacts.py
`

---

## Project Structure

`
deeptrace/
|-- models/
|   |-- detector.py              # V1 baseline detector
|   |-- detector_v2.py           # V2 extended multimodal detector
|   |-- spatial_encoder.py       # EfficientNet-B0 spatial stream
|   |-- frequency_encoder.py     # EfficientNet-B0 on DCT
|   |-- spectral_branches.py     # FFT / Wavelet / SRM-Gabor + LSGN
|   |-- clip_alignment.py        # OpenCLIP ViT-B/32 alignment
|   |-- fusion_transformer.py    # Multimodal transformer fusion
|   |-- detection_head_v2.py     # Extended detection head
|   |-- rag_retrieval.py         # FAISS-based artifact retrieval
|   |-- identity_encoder.py      # ArcFace identity (untrained)
|   -- temporal_model.py        # Video Swin Transformer (untrained)
|-- scripts/
|   |-- run_inference.py
|   |-- train_v7_sota_spectral.py
|   |-- train_v9_actor_disjoint.py
|   |-- evaluate_v9_actor_disjoint.py
|   |-- evaluate_ffpp_zeroshot.py
|   -- generate_v7_gradcam_artifacts.py
|-- explainability/
|   |-- gradcam.py
|   -- forensic_report.py
|-- results/
|   -- benchmark_eval_v7/       # Latest verified benchmark JSONs
|-- configs/
|   |-- config.yaml
|   |-- model_config.yaml
|   -- model_config_v2.yaml
|-- calibration.py
|-- training/
|   |-- trainer.py
|   -- trainer_v2.py
|-- utils/
|   |-- actor_splits.py          # Actor-disjoint split utility
|   |-- metrics.py
|   -- checkpoint.py
|-- checkpoints/                 # Trained checkpoints (not in git)
|-- app.py                       # Gradio forensics dashboard
-- requirements.txt
`

---

## Installation

**Requirements:** Python >= 3.9, CUDA-capable GPU recommended (tested on NVIDIA RTX 4050 6 GB).

`ash
# 1. Clone the repository
git clone https://github.com/<your-org>/deeptrace.git
cd deeptrace

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
`

> **mmcv / mmaction2:** These packages are required by the Video Swin Transformer and are Linux-only (equirements.txt excludes them on Windows). The Temporal module is currently untrained; all current evaluations run in image mode.

> **FAISS index:** The RAG retrieval module requires a populated FAISS index. Build it with python scripts/build_rag_db.py before using V2 in full-pipeline mode.

### Checkpoints

Checkpoints are not tracked in git. SHA-256 hashes are listed in CHECKPOINT_MANIFEST.yaml.

| Checkpoint | Path | Notes |
|---|---|---|
| V2 CLIP Finetune | checkpoints/v2_clip_finetune/best_model.pth | Kaggle in-domain best (99.80% Acc) |
| V3 Multi-Source | checkpoints/v3_e2e_multisource/best_model.pth | FF++ multi-source |
| V5 SRM Residual | checkpoints/v5_srm_residual/best_model.pth | SRM steganalysis stream |
| V7 SOTA Spectral | checkpoints/v7_sota_spectral/best_model.pth | Multi-spectral, actor-disjoint model |

---

## Usage

### Inference (CLI)

`ash
# Single image
python scripts/run_inference.py \
  --input path/to/face.jpg \
  --checkpoint checkpoints/v7_sota_spectral/best_model.pth \
  --config configs/config.yaml

# Batch directory
python scripts/run_inference.py \
  --input path/to/folder/ \
  --checkpoint checkpoints/v7_sota_spectral/best_model.pth \
  --batch \
  --output inference_output/
`

### Gradio Forensics Dashboard

Launches an interactive dashboard showing spatial RGB + GradCAM, DCT spectrum, FFT phase reconstruction, wavelet sub-bands, SRM/Gabor residuals, and spectral gating weights.

`ash
python app.py
`

Requires checkpoints/v7_sota_spectral/best_model.pth.

### Evaluation

`ash
# V9 actor-disjoint evaluation (FF++ c23, 5 cohorts, balanced 50/50)
python scripts/evaluate_v9_actor_disjoint.py

# V7 multi-spectral evaluation
python scripts/evaluate_v7_sota_spectral.py

# V2 zero-shot FF++ evaluation
python scripts/evaluate_ffpp_zeroshot.py

# GradCAM artifact generation
python scripts/generate_v7_gradcam_artifacts.py
`

### Training

Training is compute-intensive and logged to logs/. Checkpoints saved to checkpoints/.

`ash
# V7 multi-spectral training (Kaggle + FF++ multi-source, on-the-fly SBI augmentation)
python scripts/train_v7_sota_spectral.py

# V9 actor-disjoint training
python scripts/train_v9_actor_disjoint.py

# Temperature scaling calibration (run after training)
python calibration.py \
  --checkpoint checkpoints/v7_sota_spectral/best_model.pth \
  --dataset kaggle_realfake \
  --split val
`

### Dataset Setup

See DATASET_SETUP.md for full instructions. FaceForensics++ requires a Google Form license agreement.

`
data/
|-- FaceForensics++/
|   |-- original_sequences/youtube/c23/videos/
|   -- manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/videos/
-- kaggle_realfake/
    |-- real/
    -- fake/
`

---

## Research / Reproducibility

- **Random seed:** Actor-disjoint splits use seed=42 (set in utils/actor_splits.py).
- **Evaluation balance:** All FF++ evaluations use balanced 50/50 sampling (N = 2 840 per cohort).
- **Calibration:** Temperature scaling fitted on a validation split disjoint from the test split. V9 uses per-cohort Youden-optimal thresholds.
- **Result files:** All headline numbers are traceable to JSON files in esults/benchmark_eval_v7/. Full file index in EMPIRICAL_EVALUATION_METRICS_AND_PROOFS.md.
- **Known audit findings:** The DFD AUC of 1.0 observed in early V7 runs was traced to evaluation over the training data slice. Corrected in the actor-disjoint protocol; DFD subsequently excluded. See esults/benchmark_eval_v7/leakage_audit_and_heldout_verification.json.
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6 GB GDDR6). Apple Silicon M4 MPS support documented in cross_platform_apple_silicon_m4.md.

---

## Limitations

- **Severe cross-dataset degradation.** Zero-shot transfer from Kaggle (GAN images) to FF++ (video forgeries) yields near-chance AUC. Multi-source fine-tuning with actor-disjoint evaluation plateaus around 0.59 macro AUC.
- **Domain shift.** The model is sensitive to JPEG compression and image resolution — confirmed by the compression shortcut test. In-the-wild performance may differ substantially.
- **Manipulation-specific variation.** AUC ranges from ~0.63 (Deepfakes cohort) to ~0.56 (NeuralTextures). Generalisation is not uniform across forgery types.
- **No temporal modelling.** The Video Swin Transformer and BiLSTM physiology encoder are implemented but untrained. Video inference uses per-frame averaging.
- **Benchmark dependence.** The 99.80% headline metric reflects the specific StyleGAN-vs-FFHQ distribution. It does not imply robustness to diffusion models, adversarial deepfakes, or unseen generation methods.
- **RAG retrieval.** The FAISS artifact database requires manual population; its marginal contribution to generalisation has not been separately ablated.

---

## Research Status

| Component | Status |
|---|---|
| Spatial encoder (EfficientNet-B0) | Trained |
| Frequency encoder (2D-DCT) | Trained |
| Multi-spectral combiner (FFT + Wavelet + SRM/Gabor + LSGN) | Trained (V7) |
| CLIP alignment (OpenCLIP ViT-B/32, partial unfreeze) | Trained (Phase 2) |
| Multimodal transformer fusion (4-layer, 8-head) | Trained |
| Extended detection head (binary + type + generator) | Trained |
| Temperature scaling calibration | Fitted |
| GradCAM explainability | Implemented |
| Actor-disjoint evaluation protocol | Verified (V9) |
| Video Swin Transformer (temporal) | Implemented, not trained |
| BiLSTM physiology encoder | Implemented, not trained |
| Identity encoder (ArcFace) | Implemented, not trained |
| Diffusion / DDPM face evaluation | Not yet evaluated |
| Celeb-DF / DFDC cross-dataset evaluation | Not yet evaluated |

---

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{srivastava2026deeptrace,
  title   = {DeepTrace: A Multimodal Deepfake Detection System Using a Hybrid Spatial–Frequency Approach},
  author  = {Srivastava, Amogh and Rohit and Gaba, Udit},
  journal = {Journal of Deep Learning and Computer Vision},
  volume  = {1},
  number  = {1},
  year    = {2026},
  month   = {May}
}
```

---

## License

No license has been specified in this repository. All rights reserved unless otherwise stated.

<!-- Status: Active -->
