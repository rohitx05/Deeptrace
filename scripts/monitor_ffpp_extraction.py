"""
Live FF++ Frame Extraction Monitor.
Calculates extraction speed, frame count, category breakdowns, and ETA.
"""

import os
import json
import time
from pathlib import Path

LOG_PATH = Path(r"C:\Users\Udit\.gemini\antigravity\brain\d3cd5a5d-0cab-4202-846b-8c312f2f0cb5\.system_generated\tasks\task-1907.log")
FRAMES_DIR = Path("D:/datasets/ffpp_c23/frames")
TOTAL_VIDEOS = 7000


def get_stats():
    processed_videos = 0
    eta_str = "Calculating..."
    rate_str = "Calculating..."

    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "Extracting FF++ Videos:" in line and "/" in line:
                    # e.g., Extracting FF++ Videos:   7%|6         | 468/7000 [05:32<1:35:40,  1.14it/s]
                    parts = line.split("|")
                    if len(parts) >= 3:
                        sub = parts[2].strip().split()[0] # 468/7000
                        num = int(sub.split("/")[0])
                        processed_videos = num
                        if "<" in line:
                            eta_part = line.split("<")[1].split(",")[0].strip()
                            eta_str = eta_part
                        if "it/s" in line or "s/it" in line:
                            rate_part = line.split(",")[-1].strip().replace("]", "")
                            rate_str = rate_part
                        break
        except Exception:
            pass

    # Count categories in frames dir
    cat_counts = {}
    total_frames = 0
    if FRAMES_DIR.exists():
        for sub in FRAMES_DIR.iterdir():
            if sub.is_dir():
                cnt = len(list(sub.glob("*.jpg")))
                cat_counts[sub.name] = cnt
                total_frames += cnt

    pct = round((processed_videos / TOTAL_VIDEOS) * 100, 1)
    
    return {
        "processed_videos": processed_videos,
        "total_videos": TOTAL_VIDEOS,
        "percentage": pct,
        "total_frames_extracted": total_frames,
        "speed": rate_str,
        "eta": eta_str,
        "category_breakdown": cat_counts,
    }


if __name__ == "__main__":
    stats = get_stats()
    print(json.dumps(stats, indent=2))
