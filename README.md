# 🔍 Multimodal Deepfake Detection System

Research-grade deepfake detection system with explainability, optimized for RTX 4050 (6GB VRAM).

## Architecture

| Component | Implementation |
|---|---|
| Spatial Encoder | EfficientNet-B0 (pretrained, gradient checkpointing) |
| Frequency Encoder | EfficientNet-B0 on DCT-transformed input |
| Temporal Model | Video Swin Transformer Tiny (shifted-window attention) |
| CLIP Alignment | Frozen CLIP ViT-B/32 for cross-generator generalization |
| Physiology Encoder | BiLSTM on green-channel PPG signal |
| Fusion | Cross-attention (spatial↔frequency, then fused↔temporal) |
| Detection Head | Multi-task: binary + manipulation type + confidence |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `einops` is required for the temporal model: `pip install einops`

### 2. Set Up Datasets

See [DATASET_SETUP.md](DATASET_SETUP.md) for detailed instructions. Place datasets in `data/`:

```
data/
├── FaceForensics++/
│   ├── original_sequences/youtube/c23/videos/
│   └── manipulated_sequences/{method}/c23/videos/
├── CelebDF/
│   ├── Celeb-real/
│   ├── Celeb-synthesis/
│   └── List_of_testing_videos.txt
└── DFDC/
    ├── dfdc_train_part_0/
    └── metadata.json (per part)
```

### 3. Train (4 Stages)

```bash
# Stage 1: Spatial encoder only
python scripts/train_stage1.py --data_root data/

# Stage 2: Spatial + Frequency + CLIP
python scripts/train_stage2.py --stage1_ckpt checkpoints/stage1_spatial/best_model.pth

# Stage 3: Add temporal model (video mode)
python scripts/train_stage3.py --stage2_ckpt checkpoints/stage2_spatial_frequency/best_model.pth

# Stage 4: Full multi-task fine-tuning
python scripts/train_stage4.py --stage3_ckpt checkpoints/stage3_temporal/best_model.pth
```

### 4. Evaluate

```bash
# Per-dataset
python scripts/evaluate.py --dataset faceforensics --checkpoint checkpoints/stage4_multitask/best_model.pth

# Cross-dataset
python scripts/evaluate.py --dataset faceforensics --checkpoint checkpoints/stage4_multitask/best_model.pth --cross_dataset
```

### 5. Run Inference

```bash
# Single file
python scripts/run_inference.py --input path/to/image.jpg --checkpoint checkpoints/stage4_multitask/best_model.pth

# Batch
python scripts/run_inference.py --input path/to/folder/ --batch --checkpoint checkpoints/stage4_multitask/best_model.pth
```

### 6. Launch Demo UI

```bash
# With model
python ui/app.py --checkpoint checkpoints/stage4_multitask/best_model.pth

# Demo mode (no model needed)
python ui/app.py --demo
```

## Repository Structure

```
deepfake1/
├── configs/
│   ├── config.yaml           # Training/inference settings
│   └── model_config.yaml     # Model architecture parameters
├── datasets/
│   ├── base_dataset.py       # Abstract base with face extraction + DCT
│   ├── faceforensics.py      # FaceForensics++ loader
│   ├── celebdf.py            # CelebDF v2 loader
│   ├── dfdc.py               # DFDC loader
│   └── transforms.py         # Augmentations + DCT transform
├── models/
│   ├── spatial_encoder.py    # EfficientNet-B0 spatial
│   ├── frequency_encoder.py  # EfficientNet-B0 on DCT
│   ├── temporal_model.py     # Video Swin Transformer Tiny
│   ├── clip_alignment.py     # Frozen CLIP + alignment head
│   ├── physiology_encoder.py # PPG-based BiLSTM
│   ├── fusion.py             # Cross-attention fusion
│   ├── detection_head.py     # Multi-task prediction head
│   └── detector.py           # Full model assembler
├── training/
│   ├── losses.py             # Multi-task loss functions
│   └── trainer.py            # Training loop + AMP + checkpointing
├── inference/
│   └── pipeline.py           # End-to-end inference
├── evaluation/
│   └── evaluator.py          # Metrics + report generation
├── explainability/
│   ├── gradcam.py            # GradCAM heatmaps
│   ├── attention_viz.py      # Attention weight visualization
│   └── forensic_report.py    # Structured forensic reports
├── ui/
│   └── app.py                # Gradio web interface
├── scripts/
│   ├── train_stage1.py       # Stage 1: Spatial only
│   ├── train_stage2.py       # Stage 2: Spatial + Frequency
│   ├── train_stage3.py       # Stage 3: + Temporal (video)
│   ├── train_stage4.py       # Stage 4: Full multi-task
│   ├── evaluate.py           # Evaluation CLI
│   └── run_inference.py      # Inference CLI
├── utils/
│   ├── device.py             # GPU/AMP utilities
│   ├── logger.py             # Structured logging
│   ├── checkpoint.py         # Checkpoint management
│   └── metrics.py            # Metric computation
├── requirements.txt
├── README.md
└── DATASET_SETUP.md
```

## Hardware Requirements

- **GPU:** RTX 4050 (6GB VRAM) or higher
- **Settings:** Resolution=160, Frames=8, Batch=2, AMP enabled
- **Estimated VRAM:** ~5.4GB with gradient checkpointing

## Performance Targets

| Dataset | Target Accuracy |
|---|---|
| FaceForensics++ | 95% |
| CelebDF | 90% |
| Cross-dataset | 80% |
