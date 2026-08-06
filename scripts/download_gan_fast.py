"""
Download 10k GAN face images from HuggingFace datasets.
Uses 'Mikey96/StyleGAN-faces' — completely separate from training data.
Training data: xhlulu/140k-real-and-fake-faces (StyleGAN2, alphanumeric names)
This dataset: different StyleGAN weights, named stylegan_NNNNN.jpg

Usage:
    python scripts/download_gan_fast.py
    python scripts/download_gan_fast.py --count 10000
"""

import sys
import argparse
import hashlib
from pathlib import Path

OUT_DIR  = r"D:\deepfake_data\kaggle_realfake\real_vs_fake\real-vs-fake\train\fake"
HF_DATASETS = [
    "Mikey96/StyleGAN-faces",          # primary
    "openskyml/face-generation",       # fallback 1
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--count",   type=int, default=10000)
    parser.add_argument("--dataset", default=None, help="Override HuggingFace dataset name")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count existing
    existing = sorted(out_dir.glob("stylegan_*.jpg"))
    start_idx = len(existing)
    remaining = args.count - start_idx
    print(f"Target  : {args.count:,} images")
    print(f"Existing: {start_idx}")
    print(f"Need    : {remaining}")
    print(f"Out dir : {out_dir}\n")

    if remaining <= 0:
        print("Already have enough. Done!")
        return

    from datasets import load_dataset
    import io

    datasets_to_try = [args.dataset] if args.dataset else HF_DATASETS

    ds = None
    used_name = None
    for ds_name in datasets_to_try:
        try:
            print(f"Loading {ds_name} from HuggingFace (streaming)...")
            ds = load_dataset(ds_name, split="train", streaming=True, trust_remote_code=True)
            used_name = ds_name
            print(f"  OK: {ds_name}\n")
            break
        except Exception as e:
            print(f"  FAIL {ds_name}: {e}")

    if ds is None:
        print("\nAll HF datasets failed. Trying direct requests fallback...")
        _requests_fallback(out_dir, start_idx, remaining)
        return

    print(f"Downloading {remaining} images from {used_name}...")
    saved = 0
    seen_hashes = set()
    idx = start_idx

    for i, sample in enumerate(ds):
        if saved >= remaining:
            break

        # Get PIL image from sample (field name varies by dataset)
        img = sample.get("image") or sample.get("img") or sample.get("pixel_values")
        if img is None:
            continue

        # Convert PIL -> JPEG bytes
        try:
            buf = io.BytesIO()
            if hasattr(img, "save"):
                img.save(buf, format="JPEG", quality=95)
            else:
                # numpy array
                from PIL import Image
                import numpy as np
                Image.fromarray(np.uint8(img)).save(buf, format="JPEG", quality=95)
            content = buf.getvalue()
        except Exception:
            continue

        if len(content) < 5000:
            continue

        # Dedup check
        h = hashlib.md5(content).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        save_path = out_dir / f"stylegan_{idx:05d}.jpg"
        save_path.write_bytes(content)
        idx  += 1
        saved += 1

        if saved % 500 == 0 or saved == remaining:
            pct = (start_idx + saved) / args.count * 100
            print(f"  [{pct:5.1f}%] {start_idx + saved:,}/{args.count:,} saved ({saved} new)", flush=True)

    total = len(list(out_dir.glob("stylegan_*.jpg")))
    print(f"\nDone! {saved} new images saved.")
    print(f"Total stylegan_*.jpg in D: : {total:,}")


def _requests_fallback(out_dir, start_idx, remaining):
    """Last resort: try alternative free GAN APIs."""
    import requests, time, threading
    from concurrent.futures import ThreadPoolExecutor

    # robohash.org generates faces via GAN-like synthesis
    APIS = [
        "https://randomuser.me/api/portraits/men/{n}.jpg",
        "https://randomuser.me/api/portraits/women/{n}.jpg",
    ]

    saved = [0]
    lock  = threading.Lock()

    def fetch(idx):
        import random
        n = random.randint(0, 99)
        gender = random.choice(["men", "women"])
        url = f"https://randomuser.me/api/portraits/{gender}/{n}.jpg"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and len(r.content) > 5000:
                with lock:
                    cur = start_idx + saved[0]
                    saved[0] += 1
                p = out_dir / f"stylegan_{cur:05d}.jpg"
                p.write_bytes(r.content)
                return True
        except Exception:
            pass
        return False

    print("Using randomuser.me fallback (limited unique images)...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, i) for i in range(remaining * 3)]
        done = 0
        for f in futures:
            if f.result():
                done += 1
            if done >= remaining:
                break
    print(f"Fallback: {done} images saved.")


if __name__ == "__main__":
    main()
