# Verified Research Results

## Actor-Disjoint Evaluation Details
- V7: 600 train actors / 400 test actors, seed 42, DFD excluded.
- V7 macro AUC: 0.5809
- V9 macro AUC: 0.5931
- V9 macro calibrated accuracy: 57.55%
- Evaluation is explicitly described as 100% leak-free actor-disjoint.

## Calibration Results
- Temperature scaling: T = 1.703706
- Test Brier score: 0.001980 -> 0.001934 after calibration.
- Validation Brier score: 0.002792 -> 0.002637 after calibration.
- Test threshold after calibration: 0.63.
- ROC-AUC remained 0.999952 because temperature scaling preserves score ordering.

## Cross-Dataset Generalization
- V2 zero-shot FF++: 14.29% accuracy, ROC-AUC 0.5275, F1 0.0, Brier 0.8571.
- V3 multisource FF++: 82.96% accuracy, ROC-AUC 0.5231, Brier 0.2088.
- V7 actor-disjoint macro AUC: 0.5809.
- V9 clean actor-disjoint macro AUC: 0.5931.
- V9 macro calibrated accuracy: 57.55%.
