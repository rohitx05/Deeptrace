# DeepTrace Research Results Summary

## Main V2 Benchmark
- Accuracy: 99.80%
- ROC-AUC: 0.999952
- F1: 0.9980
- Brier score: 0.001934
- Calibration temperature: 1.703706

## Baselines
- MesoNet-4: 84.16% accuracy, 0.9204 AUC, 0.8406 F1, 0.1147 Brier
- Xception: 98.34% accuracy, 0.9998 AUC, 0.9837 F1, 0.0123 Brier

## Generalization
- V2 zero-shot FF++: 14.29% accuracy, 0.5275 AUC
- V7 actor-disjoint: 0.5809 macro AUC
- V9 clean actor-disjoint: 0.5931 macro AUC
- V9 calibrated accuracy: 57.55% macro

## Interpretation
Held-out benchmark performance and actor-disjoint/cross-dataset performance represent different evaluation settings. The strong held-out result does not establish equivalent cross-dataset generalization.
