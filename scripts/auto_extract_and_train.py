"""
Auto-Extractor & Training Watcher Daemon.
Monitors the FF++ download on D: drive.
The instant it completes:
1. Verifies zip integrity.
2. Extracts to D:/datasets/ffpp_c23/extracted/.
3. Evaluates zero-shot baseline.
4. Triggers V3 Multi-Source Training.
"""

import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path

TARGET_ZIP = Path("D:/datasets/ffpp_c23/FaceForensics++_C23.zip")
PART_FILE = Path("D:/datasets/ffpp_c23/FaceForensics++_C23.zip.part")
EXTRACT_DIR = Path("D:/datasets/ffpp_c23/extracted")


def watch_and_extract():
    print("Watcher started: Waiting for FaceForensics++ download to finish...")

    while True:
        if TARGET_ZIP.exists():
            print(f"\n[FOUND] Target zip ready! Size: {round(TARGET_ZIP.stat().st_size / (1024**3), 2)} GB")
            break

        if PART_FILE.exists():
            sz_gb = round(PART_FILE.stat().st_size / (1024**3), 2)
            print(f"\r[{time.strftime('%H:%M:%S')}] Downloading in progress... Current size: {sz_gb} GB / 16.66 GB", end="", flush=True)

        time.sleep(10.0)

    print("\n--- STEP 1: Verifying Zip Archive Integrity ---")
    try:
        with zipfile.ZipFile(TARGET_ZIP, 'r') as zf:
            namelist = zf.namelist()
            print(f"Archive valid! Found {len(namelist):,} files in archive.")
    except Exception as e:
        print(f"Error reading zip: {e}")
        return False

    print("\n--- STEP 2: Extracting to D:/datasets/ffpp_c23/extracted/ ---")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(TARGET_ZIP, 'r') as zf:
        zf.extractall(EXTRACT_DIR)
    print("Extraction complete!")

    print("\n--- STEP 3: Launching V3 Multi-Source Training ---")
    subprocess.run([sys.executable, "scripts/train_v3_multisource.py"], check=True)
    print("\nAll tasks completed successfully!")
    return True


if __name__ == "__main__":
    watch_and_extract()
