import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.pipeline import InferencePipeline
from utils.project_memory import ProjectMemory

# Disable overly verbose logging
logging.getLogger("inference").setLevel(logging.WARNING)

def evaluate_dataset(pipeline, data_dir, apply_blur=False, threshold=None):
    real_dir = Path(data_dir) / "real"
    fake_dir = Path(data_dir) / "fake"
    
    real_paths = list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))
    fake_paths = list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))
    
    all_paths = real_paths + fake_paths
    true_labels = [0] * len(real_paths) + [1] * len(fake_paths)
    
    if len(all_paths) == 0:
        print("No test images found!")
        return None
        
    probabilities = []
    predictions = []
    sample_preds = []
    
    for i, path in enumerate(all_paths):
        # Override cv2 imread in pipeline if blur is applied
        original_imread = None
        if apply_blur:
            original_imread = cv2.imread
            def blurred_imread(p):
                img = original_imread(str(p))
                if img is not None:
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                return img
            cv2.imread = blurred_imread

        try:
            result = pipeline.predict(str(path), save_dir=None)
        finally:
            if apply_blur and original_imread is not None:
                cv2.imread = original_imread
            
        prob = result.get('fake_probability', 0.0)
        thr = threshold if threshold is not None else 0.5  # load from calibration.json
        probabilities.append(prob)
        predictions.append(1 if prob > thr else 0)
        
        if len(sample_preds) < 5 and i % max(1, len(all_paths)//5) == 0:
            sample_preds.append({
                'file': path.name,
                'true': "FAKE" if true_labels[i] == 1 else "REAL",
                'pred': "FAKE" if prob > thr else "REAL",
                'prob': prob
            })
            
    # Default metrics (threshold = 0.5)
    acc = accuracy_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, probabilities)

    # Compute optimal threshold
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]

    # Recompute predictions using optimal threshold
    optimal_predictions = [1 if p > optimal_threshold else 0 for p in probabilities]
    optimal_acc = accuracy_score(true_labels, optimal_predictions)

    prob_array = np.array(probabilities, dtype=np.float32)

    return {
        "accuracy": acc,
        "auc": auc,
        "optimal_accuracy": optimal_acc,
        "optimal_threshold": optimal_threshold,
        "num_samples": len(all_paths),
        "sample_predictions": sample_preds,
        "prob_min": float(prob_array.min()),
        "prob_max": float(prob_array.max()),
        "prob_mean": float(prob_array.mean()),
    }

if __name__ == "__main__":
    memory = ProjectMemory()
    memory.load_primary_context()

    checkpoint_path = "checkpoints/kaggle_realfake/best_model.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/best.pth" # fallback if user moved it
        
    print(f"Loading model from {checkpoint_path}...")
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        config_path="configs/config.yaml",
        model_config_path="configs/model_config.yaml",
        device="auto"
    )

    calibration_file = getattr(pipeline, "calibration_path", None)
    if calibration_file is not None:
        print(f"Calibration file: {calibration_file}")
    print(f"Calibration loaded: {getattr(pipeline, 'calibration_loaded', False)}")
    print(f"Calibration temperature: {pipeline.model.temperature_value:.6f}")
    
    calibrated_threshold = getattr(pipeline, 'threshold', 0.5)  # load from calibration.json
    print(f"Decision threshold: {calibrated_threshold:.4f} (from calibration.json)")
    print("\n--- Evaluating on Original Test Data ---")
    results = evaluate_dataset(pipeline, "test_data", apply_blur=False, threshold=calibrated_threshold)
    if results is None:
        raise SystemExit(1)
    
    print(f"Total images tested: {results['num_samples']}")
    print(f"Accuracy (threshold={calibrated_threshold:.4f}): {results['accuracy']:.4f}")
    print(f"ROC AUC Score: {results['auc']:.4f}")
    print(f"Optimal threshold: {results['optimal_threshold']:.4f}")
    print(f"Accuracy (optimal threshold): {results['optimal_accuracy']:.4f}")
    print(
        f"Probability stats: min={results['prob_min']:.4f}, "
        f"max={results['prob_max']:.4f}, mean={results['prob_mean']:.4f}"
    )
    
    print("\n5 Sample Predictions:")
    for s in results["sample_predictions"]:
        print(f"  File: {s['file']}, True: {s['true']}, Pred: {s['pred']} (Prob: {s['prob']:.4f})")
        
    print("\n--- Robustness Test: Gaussian Blur (5x5) ---")
    blur_results = evaluate_dataset(pipeline, "test_data", apply_blur=True, threshold=calibrated_threshold)
    print(
        f"Accuracy (Blurred, threshold={calibrated_threshold:.4f}): {blur_results['accuracy']:.4f} "
        f"(Drop: {results['accuracy'] - blur_results['accuracy']:.4f})"
    )
    print(f"ROC AUC  (Blurred): {blur_results['auc']:.4f} (Drop: {results['auc'] - blur_results['auc']:.4f})")
    print(f"Optimal threshold (Blurred): {blur_results['optimal_threshold']:.4f}")
    print(
        f"Probability stats (Blurred): min={blur_results['prob_min']:.4f}, "
        f"max={blur_results['prob_max']:.4f}, mean={blur_results['prob_mean']:.4f}"
    )

    memory.record_testing(
        step_name="testing:test_generalization",
        dataset_name="test_data",
        metrics={
            "accuracy": results["accuracy"],
            "auc": results["auc"],
            "optimal_threshold": results["optimal_threshold"],
            "threshold_0_5_accuracy": results["accuracy"],
            "optimal_accuracy": results["optimal_accuracy"],
            "blur_accuracy": blur_results["accuracy"],
            "blur_auc": blur_results["auc"],
            "blur_optimal_threshold": blur_results["optimal_threshold"],
            "num_samples": results["num_samples"],
        },
        checkpoint_path=checkpoint_path,
        mode="image",
        notes="test_generalization.py refreshed held-out metrics and blur robustness deltas",
        extra={
            "dataset_info": {
                "data_root": "test_data",
                "active_dataset": "test_data",
                "available_datasets": memory.state.get("dataset_info", {}).get("available_datasets", []),
                "image_size": memory.state.get("dataset_info", {}).get("image_size"),
                "num_frames": memory.state.get("dataset_info", {}).get("num_frames"),
            }
        },
    )
