"""
Track download speed, progress, and ETA for FF++ dataset.
"""

import os
import time
import json
from pathlib import Path

STATE_FILE = Path("D:/datasets/ffpp_c23/download_state.json")
PART_FILE = Path("D:/datasets/ffpp_c23/FaceForensics++_C23.zip.part")
ZIP_FILE = Path("D:/datasets/ffpp_c23/FaceForensics++_C23.zip")
TOTAL_SIZE = 17883848075  # 16.66 GB


def get_status():
    if ZIP_FILE.exists():
        return {
            "status": "COMPLETE",
            "downloaded_gb": round(ZIP_FILE.stat().st_size / (1024**3), 3),
            "total_gb": round(TOTAL_SIZE / (1024**3), 2),
            "percentage": 100.0,
            "speed_mb_s": 0.0,
            "eta_formatted": "Completed",
        }

    if not PART_FILE.exists():
        return {"status": "NOT_FOUND"}

    now = time.time()
    current_size = PART_FILE.stat().st_size
    prev_time = None
    prev_size = None

    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                prev_time = data.get("timestamp")
                prev_size = data.get("size")
        except Exception:
            pass

    # Update state file
    with open(STATE_FILE, "w") as f:
        json.dump({"timestamp": now, "size": current_size}, f)

    if prev_time and prev_size and (now - prev_time) > 1.0:
        dt = now - prev_time
        d_bytes = max(0, current_size - prev_size)
        speed_bps = d_bytes / dt
    else:
        # Measure delta over 6 seconds
        time.sleep(6.0)
        now2 = time.time()
        current_size2 = PART_FILE.stat().st_size
        speed_bps = (current_size2 - current_size) / (now2 - now)
        current_size = current_size2

    speed_mb_s = speed_bps / (1024 * 1024)
    remaining_bytes = max(0, TOTAL_SIZE - current_size)

    if speed_bps > 1000:
        eta_sec = remaining_bytes / speed_bps
        eta_min = eta_sec / 60
        eta_hours = eta_min / 60
        if eta_hours >= 1:
            eta_str = f"{int(eta_hours)}h {int(eta_min % 60)}m"
        else:
            eta_str = f"{int(eta_min)}m"
    else:
        eta_str = "Calculating / Waiting for chunk..."

    pct = (current_size / TOTAL_SIZE) * 100

    return {
        "status": "DOWNLOADING",
        "downloaded_gb": round(current_size / (1024**3), 3),
        "total_gb": round(TOTAL_SIZE / (1024**3), 2),
        "percentage": round(pct, 2),
        "speed_mb_s": round(speed_mb_s, 2),
        "eta_formatted": eta_str,
    }


if __name__ == "__main__":
    st = get_status()
    print(json.dumps(st, indent=2))
