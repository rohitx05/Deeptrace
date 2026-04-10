# Context Summary

- Updated: 2026-04-10T18:37:35+00:00
- Last Step: testing:test_generalization

## Model Architecture
- Active Variant: DeepfakeDetector
- Summary: EfficientNet-B0 spatial + DCT EfficientNet-B0 + Video Swin-T + BiLSTM physiology + CLIP + cross-attention fusion + detection head
- Active Modules: spatial_encoder, frequency_encoder, temporal_model, physiology_encoder, clip_alignment, fusion, detection_head
- Inference Mode: image
- Config Path: configs/model_config.yaml

## Training Setup
- Dataset: test_data
- Data Root: test_data
- Available Datasets: kaggle_realfake
- Mode: image
- Image Size: 160
- Num Frames: 8
- Batch Size: 2
- Grad Accumulation: 8
- Learning Rate: 0.0001
- Weight Decay: 0.0001
- AMP: True
- Device: auto

## Current Performance
- Metric Source: testing:test_generalization
- Metric Dataset: test_data
- Accuracy: 0.9200
- AUC: 0.9820
- Optimal Threshold: 0.1341
- Threshold@0.5 Accuracy: 0.9200
- Blur Accuracy: 0.9200
- Blur AUC: 0.9748
- Temperature: 4.396576
- Calibration Status: done
- Calibration File: checkpoints/kaggle_realfake/calibration.json
- Active Checkpoint: checkpoints/kaggle_realfake/best_model.pth

## Known Issues
- Calibration pending: status=done
- AUC below 0.99: 0.9820

## Next Actions
- Run temperature calibration on a validation split and save the sidecar temperature file.
- Persist calibrated temperature and rerun held-out evaluation.
- Track robustness deltas after calibration or retraining.
- Keep V2 migration gated until a trained V2 checkpoint exists.

## Notes
- test_generalization.py refreshed held-out metrics and blur robustness deltas
