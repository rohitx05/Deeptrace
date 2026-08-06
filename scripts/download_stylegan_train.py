"""
Download StyleGAN (TPDNE) images directly into the Kaggle training folder.
Run ONCE before Stage 2 training. Safe to interrupt and resume.

Usage:
    python scripts/download_stylegan_train.py
    python scripts/download_stylegan_train.py --count 10000
"""

import os
import time
import argparse
import requests
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000,
                        help="Total number of StyleGAN images to have in folder")
    parser.add_argument("--out_dir", default="data/kaggle_realfake/real_vs_fake/real-vs-fake/train/fake",
                        help="Target folder (mixed into training automatically)")
    parser.add_argument("--delay", type=float, default=1.1,
                        help="Seconds between requests (respect rate limit)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries per image on failure")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find highest existing index for resume support
    existing = sorted(out_dir.glob("stylegan_*.jpg"))
    start_idx = len(existing)
    remaining = args.count - start_idx

    print(f"Target folder  : {out_dir}")
    print(f"Already have   : {start_idx} stylegan_*.jpg files")
    print(f"Need to fetch  : {remaining} more")
    eta_min = remaining * args.delay / 60
    print(f"Estimated time : {eta_min:.0f} min ({eta_min/60:.1f} hrs)\n")

    if remaining <= 0:
        print("Already have enough images. Done!")
        return

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    ok = 0
    fail = 0

    for i in range(remaining):
        idx = start_idx + i
        save_path = out_dir / f"stylegan_{idx:05d}.jpg"

        if save_path.exists():
            ok += 1
            continue

        # Retry loop
        success = False
        for attempt in range(args.retries):
            try:
                resp = requests.get(
                    "https://thispersondoesnotexist.com",
                    headers=headers,
                    timeout=15,
                )
                if resp.status_code == 200 and len(resp.content) > 10_000:
                    save_path.write_bytes(resp.content)
                    ok += 1
                    success = True
                    break
                else:
                    time.sleep(2)  # brief pause before retry
            except Exception as e:
                if attempt == args.retries - 1:
                    print(f"  FAIL #{idx}: {e}")
                else:
                    time.sleep(3)

        if not success:
            fail += 1

        # Progress every 100 images
        done = ok + fail
        if done % 100 == 0 and done > 0:
            pct = done / remaining * 100
            eta = (remaining - done) * args.delay / 60
            print(f"  [{pct:5.1f}%] {done}/{remaining} done | {ok} ok, {fail} failed | ETA ~{eta:.0f} min")

        time.sleep(args.delay)

    total = len(list(out_dir.glob("stylegan_*.jpg")))
    print(f"\nFinished! {ok} downloaded, {fail} failed")
    print(f"Total stylegan_*.jpg in training folder: {total}")


if __name__ == "__main__":
    main()
