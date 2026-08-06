import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import yaml
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from models.frequency_encoder import FrequencyEncoder
from models.clip_alignment import CLIPAlignmentModule
from datasets.transforms import get_val_transforms, apply_dct_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("precompute")


class RawImageDataset(Dataset):
    """Simple dataset that loads raw images from disk — no augmentation."""

    def __init__(self, image_paths: list, image_size: int = 160):
        self.image_paths = image_paths
        self.image_size = image_size
        self.transform = get_val_transforms(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(str(path))
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        face = cv2.resize(image, (self.image_size, self.image_size))

        # DCT for frequency encoder
        dct = apply_dct_transform(face)
        dct_normalized = (dct - dct.mean()) / (dct.std() + 1e-8)
        dct_tensor = torch.from_numpy(dct_normalized).permute(2, 0, 1).float()
        dct_tensor = torch.nn.functional.interpolate(
            dct_tensor.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        # RGB for CLIP
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=rgb)
        rgb_tensor = aug["image"]

        return {"path": str(path), "dct": dct_tensor, "rgb": rgb_tensor}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/kaggle_realfake")
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    # Cache dir
    cache_dir = Path(args.data_root) / "feature_cache"
    cache_dir.mkdir(exist_ok=True)
    logger.info(f"Cache directory: {cache_dir}")

    # Load frozen encoders
    logger.info("Loading FrequencyEncoder...")
    fc = model_config.get("frequency_encoder", {})
    freq_encoder = FrequencyEncoder(
        pretrained=fc.get("pretrained", True),
        feature_dim=fc.get("feature_dim", 1280),
        dct_channels=fc.get("dct_channels", 3),
        drop_rate=0.0,  # no dropout during inference
        gradient_checkpointing=False,
    ).to(device).eval()

    logger.info("Loading CLIPAlignmentModule...")
    cc = model_config.get("clip_alignment", {})
    sc = model_config.get("spatial_encoder", {})
    clip_module = CLIPAlignmentModule(
        model_name=cc.get("model_name", "ViT-B-32"),
        pretrained_dataset=cc.get("pretrained_dataset", "openai"),
        feature_dim=cc.get("feature_dim", 512),
        projection_dim=cc.get("projection_dim", 256),
        spatial_feature_dim=sc.get("feature_dim", 1280),
        freeze_clip=True,
    ).to(device).eval()

    # Collect all image paths
    img_root = Path(args.data_root) / "real_vs_fake" / "real-vs-fake"
    all_paths = []
    for split in ["train", "valid", "test"]:
        for label in ["real", "fake"]:
            folder = img_root / split / label
            if folder.exists():
                paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                all_paths.extend(paths)
                logger.info(f"  {split}/{label}: {len(paths)} images")

    logger.info(f"Total images to cache: {len(all_paths)}")

    # Skip already cached
    pending = [p for p in all_paths if not (cache_dir / f"{p.stem}_freq.pt").exists()]
    logger.info(f"Pending (not yet cached): {len(pending)}")

    if not pending:
        logger.info("All features already cached!")
        return

    dataset = RawImageDataset(pending, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    logger.info("Starting precomputation...")
    errors = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in tqdm(loader, desc="Caching features"):
            dct = batch["dct"].to(device)
            rgb = batch["rgb"].to(device)
            paths = batch["path"]

            # Frequency features: (B, 1280)
            freq_feats = freq_encoder(dct).float().cpu().numpy()

            # CLIP visual features
            dummy_spatial = torch.zeros(len(paths), sc.get("feature_dim", 1280), device=device)
            clip_result = clip_module(dummy_spatial, rgb)
            clip_feats = clip_result["spatial_projected"].float().cpu().numpy()  # (B, 256)

            # Save per-image as .npy (simple binary, no zip overhead)
            for i, path in enumerate(paths):
                stem = Path(path).stem
                try:
                    np.save(cache_dir / f"{stem}_freq.npy", freq_feats[i])
                    np.save(cache_dir / f"{stem}_clip.npy", clip_feats[i])
                except Exception as e:
                    errors += 1
                    logger.warning(f"Failed to save {stem}: {e}")

    logger.info(f"Done! Cached {len(pending) - errors} images ({errors} errors) to {cache_dir}")


if __name__ == "__main__":
    main()
