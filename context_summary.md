# Context Summary

- Updated: 2026-07-03
- Last Step: fix:phase1_gan_auc_and_threshold

## Model Architecture
- Active Variant: DeepfakeDetector
- Summary: EfficientNet-B0 spatial + DCT EfficientNet-B0 + Video Swin-T + BiLSTM physiology + CLIP + cross-attention fusion + detection head
- Active Modules: spatial_encoder, frequency_encoder, temporal_model, physiology_encoder, clip_alignment, fusion, detection_head
- Inference Mode: image
- Config Path: configs/model_config.yaml

## Training Setup
- Dataset: kaggle_realfake
- Data Root: data/kaggle_realfake
- Available Datasets: kaggle_realfake
- Mode: image
- Image Size: 160
- Num Frames: 8
- Batch Size: 16
- Grad Accumulation: 2
- Learning Rate: 0.0001
- Weight Decay: 0.0001
- AMP: True
- Device: cuda

## Current Performance
- Metric Source: training:kaggle_realfake
- Metric Dataset: kaggle_realfake
- Accuracy: 0.9939
- AUC: 0.9998
- Optimal Threshold: 0.1341
- Threshold@0.5 Accuracy: 0.9200
- Blur Accuracy: 0.9200
- Blur AUC: 0.9748
- Temperature: 4.396576
- Calibration Status: done
- Calibration File: checkpoints/kaggle_realfake/calibration.json
- Active Checkpoint: checkpoints/kaggle_realfake/best_model.pth

## Known Issues
- Phase 1 GAN finetune full best checkpoint is not confirmed yet; `last.pth` exists but `best_model.pth` is not present.

## Next Actions
- Run full Phase 1 GAN finetune with `.venv\Scripts\python.exe scripts/train_gan_finetune.py --epochs 15 --batch_size 16`.
- Confirm `checkpoints/v1_gan_finetune/best_model.pth` is created.
- After Phase 1, continue CLIP partial unfreeze and keep V2 migration gated until a trained V2 checkpoint exists.

## Notes
- kaggle_realfake training complete; best_val_auc=0.9998
