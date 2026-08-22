# Experiment Protocol

## Held-Out Benchmark
The V2 benchmark reports performance on the project's held-out Real-vs-Fake evaluation using the v2_clip_finetune checkpoint.

## Cross-Dataset Evaluation
FaceForensics++ evaluations measure transfer beyond the primary held-out benchmark and must be reported separately.

## Actor-Disjoint Evaluation
The V7/V9 evaluations use an actor-disjoint split to reduce identity leakage between training and evaluation. V9 is designated as the clean, leak-free actor-disjoint evaluation.

## Reporting Rule
Results from these protocols must not be combined into a single accuracy or treated as equivalent tests.
