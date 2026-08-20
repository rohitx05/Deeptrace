"""
Actor-Disjoint Split Utility for FaceForensics++.

Single source of truth for train/test partitioning used by BOTH training
and evaluation scripts. Guarantees zero actor identity overlap.

Design decisions:
- Shuffled actor assignment (seed=42) to avoid FF++ pairing locality bias
- 60/40 actor split: 600 train actors, 400 test actors
- Fake video src_tgt assigned to TRAIN only if both src AND tgt are train actors
- Fake video src_tgt assigned to TEST only if both src AND tgt are test actors
- Mixed pairs (one train, one test actor) are DISCARDED from both sets
- DFD uses separate actor pool (01-28), excluded from cross-manipulation benchmarks

Frame budget (verified):
- Train: 600 actors × 10 real frames = 6,000 real + 342 fake videos × 10 = 3,420 fakes/cohort
- Test: 400 actors × 10 real frames = 4,000 real + 142 fake videos × 10 = 1,420 fakes/cohort
- Balanced eval: 700 real + 700 fake = 1,400 per cohort
"""

import random
import logging
from pathlib import Path
from typing import Dict, Set, Tuple, List

import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42
TRAIN_RATIO = 0.60  # 600 train actors, 400 test actors
N_ACTORS = 1000


def _extract_video_id(filepath: str) -> str:
    """Extract video sequence ID from frame filepath."""
    return Path(filepath).stem.split("_frame_")[0]


def _extract_actor_ids(video_id: str) -> Tuple[str, str]:
    """
    Extract source and target actor IDs from a fake video ID.
    E.g., '003_456' -> ('003', '456')
    Returns (video_id, video_id) for real videos (single actor).
    """
    parts = video_id.split("_")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return parts[0], parts[1]
    return video_id, video_id


def get_actor_split() -> Tuple[Set[str], Set[str]]:
    """
    Returns (train_actors, test_actors) as sets of zero-padded 3-digit strings.

    Uses a deterministic shuffled split (seed=42) to avoid locality bias
    in FF++'s source-target pairing (57% of pairs have |src-tgt| < 50).
    """
    rng = random.Random(SEED)
    actors = list(range(N_ACTORS))
    rng.shuffle(actors)

    split_point = int(N_ACTORS * TRAIN_RATIO)
    train_actors = set(f"{a:03d}" for a in actors[:split_point])
    test_actors = set(f"{a:03d}" for a in actors[split_point:])

    assert len(train_actors & test_actors) == 0, "Actor overlap detected!"
    assert len(train_actors) + len(test_actors) == N_ACTORS

    return train_actors, test_actors


def get_actor_disjoint_split(
    manifest_df: pd.DataFrame,
    filepath_col: str = "filepath",
) -> Dict:
    """
    Partitions the manifest into actor-disjoint train and test sets.

    Returns dict with:
        train_actors, test_actors: sets of actor ID strings
        train_reals, test_reals: DataFrames of real frames
        train_fakes, test_fakes: dicts of {manipulation_type: DataFrame}
        discarded_fakes: dicts of {manipulation_type: DataFrame} (mixed pairs)
        dfd_excluded: DataFrame of all DFD frames (excluded from benchmarks)
    """
    train_actors, test_actors = get_actor_split()

    # Add video_id column
    df = manifest_df.copy()
    df["video_id"] = df[filepath_col].apply(_extract_video_id)

    # --- Partition reals ---
    df_real = df[df["manipulation_type"] == "real"]
    train_reals = df_real[df_real["video_id"].isin(train_actors)]
    test_reals = df_real[df_real["video_id"].isin(test_actors)]

    # --- Partition fakes (excluding DFD) ---
    standard_manips = ["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"]
    train_fakes = {}
    test_fakes = {}
    discarded_fakes = {}

    for manip in standard_manips:
        df_m = df[df["manipulation_type"] == manip].copy()
        df_m["src_actor"], df_m["tgt_actor"] = zip(
            *df_m["video_id"].apply(_extract_actor_ids)
        )

        both_train = df_m[
            df_m["src_actor"].isin(train_actors) & df_m["tgt_actor"].isin(train_actors)
        ]
        both_test = df_m[
            df_m["src_actor"].isin(test_actors) & df_m["tgt_actor"].isin(test_actors)
        ]
        mixed = df_m[
            ~df_m.index.isin(both_train.index) & ~df_m.index.isin(both_test.index)
        ]

        train_fakes[manip] = both_train
        test_fakes[manip] = both_test
        discarded_fakes[manip] = mixed

        logger.info(
            f"{manip}: train={len(both_train)} frames ({both_train['video_id'].nunique()} vids) | "
            f"test={len(both_test)} frames ({both_test['video_id'].nunique()} vids) | "
            f"discarded={len(mixed)} frames ({mixed['video_id'].nunique()} vids)"
        )

    # --- DFD: separate actor pool, excluded ---
    dfd_excluded = df[df["manipulation_type"] == "DeepFakeDetection"]

    result = {
        "train_actors": train_actors,
        "test_actors": test_actors,
        "train_reals": train_reals,
        "test_reals": test_reals,
        "train_fakes": train_fakes,
        "test_fakes": test_fakes,
        "discarded_fakes": discarded_fakes,
        "dfd_excluded": dfd_excluded,
    }

    # Log summary
    logger.info(
        f"Actor-Disjoint Split Summary:\n"
        f"  Train actors: {len(train_actors)} | Test actors: {len(test_actors)}\n"
        f"  Train reals: {len(train_reals)} frames | Test reals: {len(test_reals)} frames\n"
        f"  DFD: {len(dfd_excluded)} frames EXCLUDED (separate actor pool, no matching reals)"
    )

    return result
