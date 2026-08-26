# Contributing to DeepTrace

Thank you for your interest in contributing to the DeepTrace deepfake detection project. This document outlines the guidelines for contributing.

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```
3. Create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Project Structure

| Directory     | Purpose                                       |
|---------------|-----------------------------------------------|
| `models/`     | Neural network modules (encoders, heads, etc) |
| `evaluation/` | Evaluation loop and report generation         |
| `scripts/`    | Training, evaluation, and utility scripts     |
| `utils/`      | Shared helpers (metrics, device, checkpoints) |
| `configs/`    | YAML configuration files                      |
| `results/`    | Auto-generated evaluation JSONs and plots     |
| `docs/`       | Project documentation                         |

## Commit Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(scope): description` — new features
- `fix(scope): description` — bug fixes
- `docs: description` — documentation only
- `refactor(scope): description` — code restructuring without behaviour change
- `test(scope): description` — adding or updating tests

## Evaluation Integrity Rules

DeepTrace is a research project. To maintain scientific rigour:

1. **Never mix evaluation protocols.** Held-out benchmark, zero-shot, and actor-disjoint results are separate evaluation settings and must not be combined.
2. **All claimed metrics must be backed by JSON evidence** in `results/`. If a result file doesn't exist, don't claim the number.
3. **Actor-disjoint splits must remain leak-free.** No training actor may appear in any test cohort. Use `utils/actor_splits.py` for split generation.
4. **Report compression level.** All FF++ results assume c23 unless explicitly stated otherwise.

## Adding a New Model Module

1. Create the module in `models/` (e.g. `models/my_encoder.py`).
2. Wire it into `models/detector_v2.py` if it's a new stream.
3. Update `configs/model_config_v2.yaml` with default hyperparameters.
4. Add at least one evaluation script in `scripts/` that produces `results/*_metrics.json`.

## Code Style

- Python 3.10+
- Type hints on all public function signatures.
- Docstrings for every class and public method (Google style).
- Use `logging` (never bare `print()`) for runtime output.

## Questions?

Open an issue or reach out via the repository discussions.
