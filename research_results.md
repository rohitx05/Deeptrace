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

## Baseline Comparison
- MesoNet-4: 84.16% test accuracy, 0.9204 ROC-AUC, 0.8406 F1, 0.1147 Brier.
- Xception: 98.34% test accuracy, 0.9998 ROC-AUC, 0.9837 F1, 0.0123 Brier.
- MesoNet was trained for 15 epochs; Xception was reported at 2 epochs, so the comparison is not a perfectly matched training protocol.

## GradCAM Explainability Audit
- Authentic test image: real/00001.jpg.
- Manipulated test image: fake/00276TOPP4.jpg.
- Authentic manipulated probability: 1.18e-06.
- StyleGAN manipulated probability: 0.9999988.
- Generated qualitative GradCAM overlays for both samples.
- No quantitative GradCAM metric is claimed.

## GradCAM Explainability Audit
- Authentic test image: real/00001.jpg.
- Manipulated test image: fake/00276TOPP4.jpg.
- Authentic manipulated probability: 1.18e-06.
- StyleGAN manipulated probability: 0.9999988.
- Generated qualitative GradCAM overlays for both samples.
- No quantitative GradCAM metric is claimed.

## Evaluation Notes
- Held-out benchmark performance and actor-disjoint generalization measure different evaluation settings and should not be compared as equivalent tests.
- The strong held-out benchmark result does not establish strong cross-dataset generalization.
- The V2 zero-shot FF++ evaluation showed severe domain-shift failure.
- The actor-disjoint V9 evaluation provides a stricter assessment of generalization.
