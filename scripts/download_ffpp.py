"""
Download a small FF++ C23 subset for zero-shot evaluation.
Uses HuggingFace bitmind/FaceForensicsC23 dataset.
Saves to D:\datasets\ffpp_c23\

Structure after download:
  D:\datasets\ffpp_c23\
    Original/
    Deepfakes/
    Face2Face/
    FaceSwap/
    NeuralTextures/
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_tree, HfApi
except ImportError:
    print("pip install huggingface_hub first")
    sys.exit(1)


def try_hf_download():
    """Try downloading from HuggingFace."""
    target = Path("D:/datasets/ffpp_c23")
    target.mkdir(parents=True, exist_ok=True)
    
    # Try bitmind/FaceForensicsC23
    repo_id = "bitmind/FaceForensicsC23"
    print(f"Attempting to download from {repo_id}...")
    
    try:
        api = HfApi()
        # List what's available
        files = list(api.list_repo_tree(repo_id, repo_type="dataset"))
        print(f"Found {len(files)} items in repo")
        for f in files[:20]:
            print(f"  {f}")
        
        # Download
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["*.jpg", "*.png", "*.jpeg"],
        )
        print(f"Downloaded to: {path}")
        return True
    except Exception as e:
        print(f"HF download failed: {e}")
        return False


def try_alternative_download():
    """Alternative: download from a different source."""
    # Try ondyari/FaceForensics or similar
    target = Path("D:/datasets/ffpp_c23")
    target.mkdir(parents=True, exist_ok=True)
    
    repos = [
        "bitmind/FaceForensicsC23",
        "FatimaIrshad/faceforensics-extracted-dataset-c23",
    ]
    
    for repo_id in repos:
        try:
            api = HfApi()
            info = api.dataset_info(repo_id)
            print(f"\nRepo: {repo_id}")
            print(f"  Size: {info.siblings}")
            
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target),
            )
            print(f"Downloaded to: {path}")
            return True
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    return False


if __name__ == "__main__":
    if not try_hf_download():
        if not try_alternative_download():
            print("\n" + "="*60)
            print("MANUAL DOWNLOAD NEEDED")
            print("="*60)
            print("Option 1: Go to kaggle.com and search 'faceforensics c23 extracted'")
            print("Option 2: Go to huggingface.co/datasets and search 'FaceForensicsC23'")
            print(f"Save extracted faces to: D:\\datasets\\ffpp_c23\\")
            print("With subdirectories: Original/, Deepfakes/, Face2Face/, FaceSwap/, NeuralTextures/")
