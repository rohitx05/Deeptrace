"""
Robust Resumable Downloader for FaceForensics++ C23 Dataset.
Downloads the 16.66 GB zip archive to D:/datasets/ffpp_c23/
Supports automatic retries, HTTP Range resume, and progress tracking.
"""

import os
import sys
import time
import requests
from pathlib import Path

URL = "https://huggingface.co/datasets/bitmind/FaceForensicsC23/resolve/main/FaceForensics%2B%2B_C23.zip"
TARGET_DIR = Path("D:/datasets/ffpp_c23")
TARGET_ZIP = TARGET_DIR / "FaceForensics++_C23.zip"
PART_FILE = TARGET_DIR / "FaceForensics++_C23.zip.part"


def get_remote_file_size(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.status_code == 200:
            return int(r.headers.get("content-length", 0))
    except Exception as e:
        print(f"Error getting remote size: {e}")
    return 17883848075  # Known size in bytes


def download():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_ZIP.exists():
        print(f"Target zip already exists: {TARGET_ZIP} ({round(TARGET_ZIP.stat().st_size / (1024**3), 2)} GB)")
        return True

    total_size = get_remote_file_size(URL)
    print(f"Total file size to download: {total_size} bytes ({round(total_size / (1024**3), 2)} GB)")

    # Resume from existing .part file if available
    downloaded = PART_FILE.stat().st_size if PART_FILE.exists() else 0
    print(f"Starting / Resuming from byte {downloaded} ({round(downloaded / (1024**3), 2)} GB)...")

    chunk_size = 4 * 1024 * 1024  # 4MB chunks
    max_retries = 50
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            downloaded = PART_FILE.stat().st_size if PART_FILE.exists() else 0
            if downloaded >= total_size:
                print("Download complete!")
                break

            headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
            print(f"[Attempt {attempt}] Connecting with Range: {downloaded}...")

            with requests.get(URL, headers=headers, stream=True, timeout=(15, 60), allow_redirects=True) as response:
                if response.status_code not in (200, 206):
                    print(f"Unexpected HTTP status {response.status_code}")
                    time.sleep(retry_delay)
                    continue

                mode = "ab" if downloaded > 0 else "wb"
                last_log = time.time()

                with open(PART_FILE, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            now = time.time()
                            if now - last_log >= 10.0:  # Log every 10 seconds
                                pct = (downloaded / total_size) * 100
                                mb = downloaded / (1024 * 1024)
                                gb = downloaded / (1024**3)
                                print(f"[{time.strftime('%H:%M:%S')}] {gb:.2f} GB / {total_size/(1024**3):.2f} GB ({pct:.1f}%) downloaded", flush=True)
                                last_log = now

            downloaded = PART_FILE.stat().st_size if PART_FILE.exists() else 0
            if downloaded >= total_size:
                print("Finished full file download!")
                break

        except (requests.exceptions.RequestException, TimeoutError, ConnectionError) as e:
            print(f"[Attempt {attempt}] Network interruption: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    if PART_FILE.exists() and PART_FILE.stat().st_size >= total_size:
        PART_FILE.rename(TARGET_ZIP)
        print(f"Successfully finalized archive: {TARGET_ZIP}")
        return True
    else:
        print(f"Download in progress or incomplete: {downloaded}/{total_size} bytes")
        return False


if __name__ == "__main__":
    download()
