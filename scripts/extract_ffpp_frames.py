"""
High-Speed Multi-Threaded FaceForensics++ Frame Extractor.
Extracts 10 representative frames per video from the 7,000 FF++ C23 videos on D: drive.

Structure on D: Drive:
  D:/datasets/ffpp_c23/
    FaceForensics++_C23.zip (source archive)
    frames/
      real/ (10,000 frames)
      Deepfakes/ (10,000 frames)
      FaceSwap/ (10,000 frames)
      Face2Face/ (10,000 frames)
      NeuralTextures/ (10,000 frames)
      FaceShifter/ (10,000 frames)
      DeepFakeDetection/ (10,000 frames)
"""

import os
import sys
import zipfile
import tempfile
import concurrent.futures
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

ZIP_PATH = Path("D:/datasets/ffpp_c23/FaceForensics++_C23.zip")
OUTPUT_FRAMES_DIR = Path("D:/datasets/ffpp_c23/frames")
MANIFEST_PATH = Path("manifests/ffpp_c23_manifest.csv")
FRAMES_PER_VIDEO = 10


def extract_frames_from_video_bytes(video_bytes, output_dir, video_id, method_name, num_frames=10):
    """Write bytes to temporary file, extract frames, and delete temp file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(temp_fd, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        # Evenly spaced frame indices
        step = max(1, total_frames // (num_frames + 1))
        frame_indices = [i * step for i in range(1, num_frames + 1)]

        extracted = []
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                out_name = f"{video_id}_frame_{f_idx:04d}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted.append((str(out_path), 0 if method_name == "real" else 1, method_name))
        cap.release()
        return extracted
    except Exception as e:
        return []
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def process_archive(max_workers=8):
    print("Starting FaceForensics++ frame extraction directly from zip archive...")
    OUTPUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        all_files = [p for p in zf.namelist() if p.endswith(".mp4")]
        print(f"Found {len(all_files):,} video files in archive.")

    records = []

    def task(v_path):
        with zipfile.ZipFile(ZIP_PATH, "r") as local_zf:
            v_bytes = local_zf.read(v_path)

        parts = v_path.split("/")
        # e.g., FaceForensics++_C23/real/001.mp4 or FaceForensics++_C23/fake/FaceSwap/001_002.mp4
        if "/real/" in v_path:
            method = "real"
            vid_name = Path(v_path).stem
            out_sub = OUTPUT_FRAMES_DIR / "real"
        else:
            method = parts[2]
            vid_name = Path(v_path).stem
            out_sub = OUTPUT_FRAMES_DIR / method

        return extract_frames_from_video_bytes(v_bytes, out_sub, vid_name, method, num_frames=FRAMES_PER_VIDEO)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(task, p): p for p in all_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting FF++ Videos"):
            try:
                res = future.result()
                records.extend(res)
            except Exception:
                pass

    print(f"\nExtraction complete! Total extracted face frames: {len(records):,}")
    
    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=["filepath", "label", "manipulation_type"])
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print(df["manipulation_type"].value_counts())
    return df


if __name__ == "__main__":
    process_archive(max_workers=8)
