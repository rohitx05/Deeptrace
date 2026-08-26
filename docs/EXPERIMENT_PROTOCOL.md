# Experiment Protocol

## Held-Out Benchmark
The V2 benchmark reports performance on the project's held-out Real-vs-Fake evaluation using the v2_clip_finetune checkpoint.

## Cross-Dataset Evaluation
FaceForensics++ evaluations measure transfer beyond the primary held-out benchmark and must be reported separately.

## Actor-Disjoint Evaluation
The V7/V9 evaluations use an actor-disjoint split to reduce identity leakage between training and evaluation. V9 is designated as the clean, leak-free actor-disjoint evaluation.

## Calibration Methodology

All reported accuracies use **calibrated thresholds** rather than the default 0.5 cutoff:

1. **Temperature scaling** is applied post-hoc using `calibration.py`. The calibration temperature is fit on a held-out validation split disjoint from the test set.
2. **Per-cohort thresholds** are optimised independently for each FF++ manipulation type using Youden's J statistic (`utils/metrics.find_optimal_threshold`).
3. The calibration temperature and optimal threshold must be logged alongside every accuracy number. Raw (uncalibrated) AUC should also be reported since AUC is threshold-invariant.

## Uncertainty Reporting

When MC-Dropout uncertainty is available:

- Report the **fraction of high-uncertainty predictions** (`frac_uncertain`, std > 0.2) alongside accuracy.
- Flag any cohort where > 30% of predictions are high-uncertainty — this indicates the model is operating outside its reliable regime.
- Epistemic uncertainty should be computed over N ≥ 10 stochastic forward passes.

## Reporting Rule
Results from these protocols must not be combined into a single accuracy or treated as equivalent tests.

