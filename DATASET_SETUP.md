# Dataset Setup Guide

This guide explains how to download and structure datasets for the deepfake detection system.

## Directory Structure

Create a `data/` directory in the project root:

```
data/
├── FaceForensics++/
├── CelebDF/
└── DFDC/
```

---

## 1. FaceForensics++

**Download:** [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)

> Requires a Google Form license agreement. You'll receive a download script.

### Structure

```
data/FaceForensics++/
├── original_sequences/
│   └── youtube/
│       └── c23/              # c23 = light compression (recommended)
│           └── videos/
│               ├── 000.mp4
│               ├── 001.mp4
│               └── ...
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/videos/*.mp4
    ├── Face2Face/
    │   └── c23/videos/*.mp4
    ├── FaceSwap/
    │   └── c23/videos/*.mp4
    └── NeuralTextures/
        └── c23/videos/*.mp4
```

### Download Commands (after getting the script)

```bash
python download_faceforensics.py data/FaceForensics++ -d original -c c23 -t videos
python download_faceforensics.py data/FaceForensics++ -d Deepfakes -c c23 -t videos
python download_faceforensics.py data/FaceForensics++ -d Face2Face -c c23 -t videos
python download_faceforensics.py data/FaceForensics++ -d FaceSwap -c c23 -t videos
python download_faceforensics.py data/FaceForensics++ -d NeuralTextures -c c23 -t videos
```

---

## 2. CelebDF v2

**Download:** [CelebDF GitHub](https://github.com/yuezunli/celeb-deepfakeforensics)

> Request access via Google Form. ~14GB.

### Structure

```
data/CelebDF/
├── Celeb-real/
│   ├── id0_0000.mp4
│   ├── id0_0001.mp4
│   └── ...
├── Celeb-synthesis/
│   ├── id0_id1_0000.mp4
│   └── ...
├── YouTube-real/         # optional
│   └── ...
└── List_of_testing_videos.txt
```

The `List_of_testing_videos.txt` file defines the official test split. Each line has:
```
1 id0_id1_0000.mp4
0 id0_0000.mp4
```
(1 = fake, 0 = real, followed by filename)

---

## 3. DFDC (DeepFake Detection Challenge)

**Download:** [Kaggle DFDC](https://www.kaggle.com/c/deepfake-detection-challenge/data)

> ~470GB total. Download select parts if space is limited.

### Structure

```
data/DFDC/
├── dfdc_train_part_0/
│   ├── aaqaifqrwn.mp4
│   ├── afoovlsmtx.mp4
│   ├── ...
│   └── metadata.json
├── dfdc_train_part_1/
│   ├── ...
│   └── metadata.json
└── ... (up to dfdc_train_part_49)
```

### metadata.json Format
```json
{
  "aaqaifqrwn.mp4": {
    "label": "FAKE",
    "split": "train",
    "original": "vudstoqkul.mp4"
  },
  "afoovlsmtx.mp4": {
    "label": "REAL",
    "split": "train"
  }
}
```

**Tip:** For initial experiments, download just parts 0–4 (~50GB). Configure in the training script:
```python
DFDCDataset(root_dir="data/", parts=[0, 1, 2, 3, 4])
```

---

## Verification

After placing datasets, verify the structure:

```bash
python -c "
from datasets.faceforensics import FaceForensicsDataset
ds = FaceForensicsDataset(root_dir='data/', split='train', mode='image')
print(f'FaceForensics++ train: {len(ds)} samples')
"
```

If you see `Loaded X train samples`, the dataset is configured correctly.
