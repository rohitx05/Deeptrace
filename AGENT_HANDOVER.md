# AGENT_HANDOVER.md
> Complete context for any AI agent / IDE picking up this project.
> Read this + PROJECT_MEMORY.yaml before doing anything.
> Updated every session. Newest entries at top of each section.

---

## HOW TO START (any new agent)
1. Read `PROJECT_MEMORY.yaml` — full state, architecture, roadmap
2. Read `AGENT_HANDOVER.md` (this file) — decisions, mistakes, rules
3. Do NOT scan the full repo — waste of tokens
4. Check `project.status` in MEMORY — tells you exactly where we are
5. Continue from `roadmap` section — find first `pending` item

> [!IMPORTANT]
> **ALWAYS use `.venv\Scripts\python.exe` for training — NOT system `python`**
> System python = `2.10.0+cpu` (no GPU). `.venv` = `torch 2.6.0+cu124` + RTX 4050.
> `venv` also works: `torch 2.11.0+cu128` + RTX 4050.
> Run all scripts as: `.venv\Scripts\python.exe scripts/train_gan_finetune.py ...`


---

## AGENT RULES
```
on_mistake    → append to AGENT_HANDOVER.md § Mistakes & Fixes
on_arch_change→ append to AGENT_HANDOVER.md § Architecture Changes
on_freeze     → append to AGENT_HANDOVER.md § Freeze/Unfreeze Log
on_method     → append to AGENT_HANDOVER.md § Method Changes
on_session_end→ update PROJECT_MEMORY.yaml status field
language      → short, clear, low-token (caveman-style ok)
```

---

## CURRENT STATUS (2026-08-18)
- **Active Production Model**: DeepTrace V6 Dual-Stream Ensemble (`checkpoints/v3_e2e_multisource/best_model.pth` + `checkpoints/v5_srm_residual/best_model.pth`).
- **In-Domain Kaggle Retention ($N=20,000$)**: **99.69% Accuracy · 0.99991 ROC-AUC · 0.9969 F1 · 0.0024 Brier** (Dominates MesoNet $84.16\%$ & XceptionNet $98.34\%$).
- **FaceForensics++ DeepFakeDetection ($N=4,000$)**: **83.12% Accuracy · 0.9999 ROC-AUC · 0.8556 F1 · P(Fake)=1.0000**.
- **FaceForensics++ Macro ($N=14,000$)**: **85.72% Macro Accuracy · 0.9231 Macro F1 · 0.6321 ROC-AUC**.
- **FaceSwap Poisson Boundary Detection**: Jumped from $P(\text{Fake})=0.0000 \longrightarrow 0.5276 \longrightarrow \mathbf{0.8239}$ via 3-channel SRM high-pass residual filtering.
- **Uncertainty Calibration**: Fitted $T^* = \mathbf{0.873507}$ strictly on validation set ($N=20,000$) $\longrightarrow$ Test Brier score = **$0.004000$**.
- **Cross-Platform Auto-Detection**: [`utils/device.py`](file:///c:/Users/Udit/Desktop/deepfake1/utils/device.py) supports NVIDIA CUDA (Windows RTX 4050) & Apple Silicon MPS (macOS M4).
- **All Checkpoints Isolated & Intact**: `v1_gan_finetune`, `v2_clip_finetune`, `v3_e2e_multisource`, `v4_seam_hardmining`, `v5_srm_residual`, `mesonet_baseline_seed42`, `xception_baseline_seed42`.
- **Next Target (Phase 7)**: SOTA Multi-Spectral V7 (Continuous Phase FFT, 2-Level Wavelet Packet, 9-ch SRM/Gabor + Dynamic Self-Blended SBI Synthesis).

---

## ARCHITECTURE CHANGES (newest first)

### [2026-07-03] models/clip_alignment.py + detector.py — optional CLIP alignment loss
- Added `compute_alignment_loss` / `compute_clip_alignment_loss` flag
- Default remains `True` for normal training/inference
- `train_gan_finetune.py` passes `False` because GAN finetune loss does not use `clip_alignment_loss`
- Saves the frozen CLIP ViT forward pass during Phase 1 without changing binary/gen losses

### [2026-06-25] datasets/kaggle_realfake.py — generator_type added
- Added `GENERATOR_TYPES` map: real=-1, stylegan=0, diffusion=1, other_gan=2, unknown=3
- Auto-infers type from filename stem at load time
- Added `extra_fake_dirs` param → scans D: drive GAN folder transparently
- `__getitem__` now returns `generator_type` tensor

### [2026-06-25] scripts/train_gan_finetune.py — CREATED
- Phase 1 GAN finetune script
- Loads V1 `best_model.pth` with `strict=False`
- Adds inline `GeneratorHead` (Linear 512→128→4) on top of `fused_features`
- Loss: `binary_BCE + 0.4 * generator_CE (ignore_index=-1 for reals)`
- Config: LR=1e-5, epochs=15, batch=16, accumulation=4

### [2026-06-25] models/clip_alignment.py — partial_unfreeze() PLANNED
- Not yet implemented — Phase 2
- Will unfreeze last 2 ViT transformer blocks + ln_post + proj
- LR for CLIP blocks = 5e-6 (separate param group)

### [2026-06-25] scripts/pretrain_mae.py — CREATED
- MAE self-supervised pretraining for Video Swin-T
- 75% patch masking, 2-layer transformer decoder
- Status: created, never run

### [2026-06-21] V2 architecture — CODED (16 files, untrained)
- New modules: spectral_combiner (FFT+Wavelet+Noise), identity_encoder (ArcFace-R18),
  rag_retrieval (FAISS), fusion_transformer (4L×8H), detection_head_v2
- Entry: `scripts/train_v2.py` stages 1→5
- Status: import smoke tests pass, no checkpoint

---

## FREEZE / UNFREEZE LOG (newest first)

### [2026-06-25] Phase 1 GAN finetune freeze config
```
FROZEN:   temporal_model, physiology_encoder, clip_visual (backbone only)
TRAINED:  spatial_encoder, frequency_encoder, fusion, detection_head, GeneratorHead (new)
REASON:   GAN fingerprints live in spatial+frequency domain; temporal unused in image mode
VRAM est: ~4.5 GB
```

### [2026-06-25] Phase 2 CLIP partial unfreeze (PLANNED)
```
FROZEN:   temporal_model, physiology_encoder, CLIP blocks 0..9
TRAINED:  spatial+freq encoders, fusion, detection_head, CLIP blocks 10-11 + ln_post + proj
REASON:   Last 2 blocks contain highest-level semantic features; full unfreeze = VRAM OOM
LR:       1e-5 main | 5e-6 CLIP blocks (separate param group)
VRAM est: ~5.5 GB (tight — reduce batch to 8 if OOM)
```

### [2026-06-21] V1 training freeze config (completed)
```
FROZEN:   clip_vit_b32 (88M params), temporal_swin_t, physiology_bilstm
TRAINED:  spatial_encoder, frequency_encoder, clip_projection, fusion, detection_head
RESULT:   99.39% acc, 0.9998 AUC, T=4.396, threshold=0.1341
```

---

## METHOD CHANGES (newest first)

### [2026-07-03] Phase 1 training speed optimization
- Skips unused frozen CLIP visual pass during GAN finetune
- Enables CUDA `cudnn.benchmark`, TF32, and non-blocking H2D transfers
- Defaults changed to `batch_size=16`, `accumulation=4`, `num_workers=2`
- Fused AdamW is opt-in only with `--fused_optimizer`; default disabled because it can crash with AMP on this PyTorch/CUDA build
- If Windows pagefile/spawn errors occur, rerun with `--num_workers 0`

### [2026-07-03] Fused AdamW + AMP crash
- Error: `AssertionError` in `torch.optim.adamw._multi_tensor_adamw` during `scaler.step(optimizer)`
- Cause: unsafe fused/foreach AdamW path with AMP on local PyTorch/CUDA build
- Fix: default optimizer is standard AdamW; fused optimizer is now opt-in via `--fused_optimizer`

### [2026-06-25] GAN data source — 3 attempts before correct solution
1. `thispersondoesnotexist.com` → wrong URL (returns HTML not image)
2. `this-person-does-not-exist.com/new` API → only ~500 unique images, 83% dup rate
3. HuggingFace datasets → all target datasets missing from Hub
4. ✅ **FINAL:** copied `test/fake` split (10k, held-out, zero overlap) with `stylegan_` prefix

### [2026-06-25] Training script deletion — old vs new
- Deleted: `train_stage1/2/3/4.py`, `train_kaggle.py` (V1 era, superseded)
- Current: `train_v2.py` (all stages), `train_gan_finetune.py` (Phase 1), `train_clip_finetune.py` (Phase 2, TBD)

### [2026-06-21] Calibration method
- Post-hoc temperature scaling (not trained into model)
- Calibrated on val set after training: T=4.396576
- Threshold tuned separately: 0.1341 (not 0.5 — see DECISIONS)

---

## MISTAKES & FIXES (newest first)

### [2026-07-03] CRITICAL — train_gan_finetune.py read wrong AUC metric key
- `compute_metrics()` returns `roc_auc`, but `train_gan_finetune.py` read `metrics.get("auc", 0.0)`
- Result: validation AUC stayed 0.0 and `best_model.pth` might never be saved
- Fix: read `roc_auc` with `auc` fallback and initialize `best_auc=-1.0` so first epoch can save

### [2026-07-03] Calibration threshold sidecar missing
- `calibration.json` had only `temperature`, so inference fell back to `configs/config.yaml`
- `configs/config.yaml` had stale threshold 0.162 while docs/project state use 0.1341
- Fix: added `threshold: 0.1341` to `calibration.json` and set config fallback to 0.1341

### [2026-06-25] CRITICAL — Wrong Python binary used for training
- `python` in PATH = `2.10.0+cpu` — NO CUDA, runs on CPU only
- Fix: use `.venv\Scripts\python.exe` (cu124) or `venv\Scripts\python.exe` (cu128)

### [2026-06-25] train_gan_finetune.py — 9 API bugs (util signatures not checked)
- `get_device(use_cuda=)` → `prefer_cuda=`
- `load_checkpoint()` has no `strict` param → replaced with direct `torch.load`
- `get_grad_scaler(use_amp=)` → `enabled=`
- `torch.autocast(device_type="cuda")` → use `device.type` for CPU safety
- `compute_metrics(list, list)` → needs `(np.array, np.array, np.array)`
- DataLoader collate crashes on `None` (cache miss) → custom `collate_fn`
- `num_workers=4` on Windows → causes multiprocessing spawn crash → set to `0`
- `pin_memory=True` on CPU → warning + waste → `pin = device.type=="cuda"`
- `model(dct_input=)` → actual kwarg is `dct=`

### [2026-06-25] Unicode `->` in Python print → cp1252 crash on Windows
- `UnicodeEncodeError` in `download_gan_fast.py`
- Fix: replaced all `→` with ASCII `->`

### [2026-06-25] `trust_remote_code=True` deprecated in newer `datasets`
- Fix: removed param — but datasets didn't exist on Hub anyway

### [2026-06-25] Wrong download URL for StyleGAN site
- `thispersondoesnotexist.com` returns HTML page (4KB), not image
- Fix: found real endpoint at `this-person-does-not-exist.com/new` (JSON API)
- Then found site pool too small (500 uniq / 10k needed) — abandoned

### [2026-06-25] Duplicate checkpoint confusion
- `project_state.json` listed deleted epoch checkpoints as available
- Fix: updated to only list `best_model.pth`

### [2026-06-25] Calibration status contradiction
- `context_summary.md` said pending, `project_state.json` said done
- Fix: both updated to reflect calibration complete

### [2026-06-25] `inference/calibration.py` was dead shim
- Just re-exported from root `calibration.py` — no value
- Fix: deleted, `pipeline.py` imports root directly

---

## NEXT ACTIONS (in order)
1. `.venv\Scripts\python.exe scripts/train_gan_finetune.py --epochs 15 --batch_size 16`
2. After: add `partial_unfreeze()` to `models/clip_alignment.py`
3. Create `scripts/train_clip_finetune.py`
4. Run CLIP finetune
5. Fix `train_v2.py` → add `--v1_ckpt` + kaggle_realfake support
6. Run V2 stages 1, 2, 4, 5
7. Recalibrate + build RAG DB + evaluate

---

## KEY NUMBERS (don't recompute)
| Metric | Value |
|--------|-------|
| V1 accuracy | 99.39% |
| V1 AUC | 0.9998 |
| Calibration T | 4.396576 |
| Optimal threshold | 0.1341 |
| Total params | 51.7M |
| Trained params | 19.8M |
| CLIP params (frozen) | 88M |
| image_size | 160×160 (fixed — VRAM constraint) |
| GPU | RTX 4050 6GB VRAM |
| Max safe VRAM | ~5.8 GB |
