# PROJECT_MISTAKES.md
> Every bug, wrong assumption, failed attempt — logged with fix.
> Use this for project writeup / explanation later.
> Format: [DATE] MISTAKE → FIX

---

## [2026-07-03] GAN fine-tune AUC key mismatch
- **Mistake:** `train_gan_finetune.py` used `metrics.get("auc", 0.0)`, but `compute_metrics()` returns `roc_auc`
- **Fix:** Read `roc_auc` with `auc` fallback and initialize `best_auc=-1.0` so the first valid epoch can save `best_model.pth`

## [2026-07-03] Calibration threshold missing from sidecar
- **Mistake:** `checkpoints/kaggle_realfake/calibration.json` only stored temperature, so inference used the config fallback threshold
- **Fix:** Added `threshold: 0.1341` to the sidecar and changed the config fallback threshold to `0.1341`

## [2026-06-25] StyleGAN download — wrong URL
- **Mistake:** Used `thispersondoesnotexist.com` directly → returns 4KB HTML not image
- **Fix:** Found real API at `this-person-does-not-exist.com/new` returning JSON with image path

## [2026-06-25] StyleGAN download — tiny image pool
- **Mistake:** Site only has ~500 unique images, 83% of requests were duplicates → would take 1000+ min for 10k
- **Fix:** Switched to copying 10k images from the dataset's own `test/fake` split (held-out, never trained on, zero overlap)

## [2026-06-25] HuggingFace datasets — deprecated param
- **Mistake:** Used `trust_remote_code=True` → error: not supported in newer `datasets` version
- **Fix:** Removed param — but all target HF datasets also didn't exist on Hub

## [2026-06-25] Unicode arrow in print → Windows cp1252 crash
- **Mistake:** Used `→` character in f-string → `UnicodeEncodeError` on Windows terminal
- **Fix:** Replaced with ASCII `->` 

## [2026-06-25] Old stage scripts left after cleanup
- **Mistake:** `train_stage1/2/3/4.py` + `train_kaggle.py` still existed alongside `train_v2.py` → confusion which to use
- **Fix:** Deleted all old stage scripts, `train_v2.py` is single entry point

## [2026-06-25] `inference/calibration.py` duplicate shim
- **Mistake:** `inference/calibration.py` just did `from calibration import *` — redundant, confusing
- **Fix:** Deleted it. `inference/pipeline.py` imports from root `calibration.py` directly

## [2026-06-25] `calibration.py` status contradiction in context_summary.md
- **Mistake:** `context_summary.md` said "Calibration: pending" but `project_state.json` said "done"
- **Fix:** Updated both files to reflect calibration is complete (T=4.396, threshold=0.1341)

## [2026-06-25] Epoch checkpoint bloat
- **Mistake:** epoch_5/10/15/20.pth kept on disk after training — wasted 2.5GB
- **Fix:** Deleted all intermediate checkpoints, kept only `best_model.pth`

## [2026-06-25] `project_state.json` listed deleted checkpoints
- **Mistake:** `available_checkpoints` still listed epoch_5.pth after deletion
- **Fix:** Updated list to only contain `best_model.pth`
