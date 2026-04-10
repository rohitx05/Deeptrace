"""
Build RAG Artifact Database.
Run AFTER training to populate the FAISS index with training set embeddings.

Usage:
    python scripts/build_rag_db.py --checkpoint checkpoints/v2_stage4/best_model.pth --data_root data/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging

from models.detector_v2 import DeepfakeDetectorV2
from models.generator_head import GeneratorFingerprintHead, GENERATOR_TYPES
from datasets.faceforensics import FaceForensicsDataset
from utils.device import get_device
from utils.checkpoint import load_checkpoint
from utils.logger import get_logger

logger = get_logger("build_rag_db")

MANIPULATION_LABELS = ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def main():
    parser = argparse.ArgumentParser(description="Build RAG Artifact Database")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_config_v2.yaml")
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--output", type=str, default="rag_database/")
    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)

    device = get_device()

    # Model
    model = DeepfakeDetectorV2(config=model_config)
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)
    model.eval()

    # Dataset
    dataset = FaceForensicsDataset(
        root_dir=args.data_root, split="train",
        image_size=config["data"]["image_size"], mode="image",
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    logger.info(f"Building RAG database from {len(dataset)} samples (max {args.max_samples})")

    all_embeddings = []
    all_metadata = []
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            if count >= args.max_samples:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Extract spatial features
            spatial_feat = model.spatial_encoder(batch["image"])

            # Project to RAG query space
            query = model.rag_retrieval.query_proj(spatial_feat)
            query = torch.nn.functional.normalize(query, dim=-1)

            embeddings_np = query.cpu().numpy()
            all_embeddings.append(embeddings_np)

            # Build metadata
            labels = batch["label"].cpu().numpy()
            manip_types = batch["manipulation_type"].cpu().numpy()

            for i in range(len(labels)):
                meta = {
                    "label": "fake" if labels[i] > 0.5 else "real",
                    "manipulation_type": MANIPULATION_LABELS[int(manip_types[i])],
                    "generator_type": GENERATOR_TYPES[
                        GeneratorFingerprintHead.map_manipulation_to_generator(int(manip_types[i]))
                    ],
                    "dataset": "FaceForensics++",
                }
                all_metadata.append(meta)

            count += len(labels)

    all_embeddings = np.concatenate(all_embeddings, axis=0)[:args.max_samples]
    all_metadata = all_metadata[:args.max_samples]

    logger.info(f"Extracted {len(all_embeddings)} embeddings, dim={all_embeddings.shape[1]}")

    # Build FAISS index
    model.rag_retrieval.build_index(all_embeddings.astype(np.float32), all_metadata)
    model.rag_retrieval.save_index(args.output)

    logger.info(f"RAG database saved to {args.output}")
    logger.info(f"  Total entries: {len(all_metadata)}")
    logger.info(f"  Fake: {sum(1 for m in all_metadata if m['label'] == 'fake')}")
    logger.info(f"  Real: {sum(1 for m in all_metadata if m['label'] == 'real')}")


if __name__ == "__main__":
    main()
