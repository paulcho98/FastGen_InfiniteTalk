# Stage 1 Verification + Lazy Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify DF and KD training work with corrected preprocessing (448x896 resolution + zero-padded T5), then implement Option 3 lazy caching in the dataloader for scaling beyond the initial 3000 samples.

**Architecture:** Tasks 1-2 are verification-only (no code changes, just re-run with correct data). Task 3 adds on-the-fly VAE/CLIP/audio encoding to the dataloader when cached `.pt` files are missing (T5 must be pre-computed). Task 4 is documentation.

**Tech Stack:** PyTorch (bf16), FastGen training infrastructure, WanVAE, CLIP ViT-H/14, wav2vec2, flash_attn 2.7.3, single A100-80GB per training run.

---

## Prerequisites

All data fixes from 2026-03-26 must be committed:
- Aspect-ratio bucket resolution (448x896 for 16:9 videos)
- Zero-padded T5 text embeddings
- motion_frame.pt (single-image VAE encode)
- flash_attn for audio cross-attention
- PIL BILINEAR resize for reference frame

Precomputed test data must be regenerated at correct resolution before Tasks 1-2.

---

## File Structure

### Files to modify:
```
fastgen/datasets/infinitetalk_dataloader.py    # Task 3: add lazy encoding
fastgen/configs/experiments/InfiniteTalk/
    config_df_test.py                           # Task 1: verify input_shape + data paths
    config_kd_test.py                           # Task 2: new test config for KD
docs/BUGS_AND_FIXES.md                         # Task 4: already created, update if needed
docs/IMPLEMENTATION_PROGRESS.md                # Task 4: update with final status
```

### Files to create:
```
fastgen/configs/experiments/InfiniteTalk/
    config_kd_test.py                           # Task 2: KD test config (20 iterations)
```

### Test data to regenerate:
```
data/test_precomputed/                          # Must be at 448x896, not 640x640
```

---

## Task 1: DF Training Verification at Correct Resolution

**Goal:** Confirm Diffusion Forcing training works with 448x896 latents + zero-padded T5.

**Execution:** GPU 0

**Files:**
- Modify: `data/test_precomputed/` (regenerate at 448x896)
- Verify: `fastgen/configs/experiments/InfiniteTalk/config_df_test.py` (input_shape=[16,21,56,112])

- [ ] **Step 1: Regenerate test data at correct resolution**

Delete old test data and re-precompute 3 samples at 448x896 with zero-padded T5:
```bash
rm -rf data/test_precomputed/data_*/
python scripts/precompute_infinitetalk_data.py \
  --csv_path data/test_precomputed/test_one_sample.csv \
  --data_root /data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/ \
  --output_dir data/test_precomputed/ \
  --weights_dir /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P \
  --wav2vec_dir /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base \
  --num_samples 3 \
  --device cuda:0
```

- [ ] **Step 2: Verify test data shapes**

```python
import torch
lat = torch.load("data/test_precomputed/data_-0F1owya2oo_-0F1owya2oo_189664_194706/vae_latents.pt")
assert lat.shape == torch.Size([16, 21, 56, 112]), f"Wrong shape: {lat.shape}"
text = torch.load("data/test_precomputed/data_-0F1owya2oo_-0F1owya2oo_189664_194706/text_embeds.pt")
assert text[0, 113:].abs().max() == 0, "T5 padding not zero!"
mf = torch.load("data/test_precomputed/data_-0F1owya2oo_-0F1owya2oo_189664_194706/motion_frame.pt")
assert mf.shape[1] == 1 and mf.shape[2] == 56 and mf.shape[3] == 112
print("All shapes correct")
```

- [ ] **Step 3: Verify config_df_test.py has correct input_shape**

`config_df_test.py` inherits from `config_df.py` which has `input_shape = [16, 21, 56, 112]`. Verify this propagates correctly.

- [ ] **Step 4: Delete old checkpoint and run fresh DF training**

```bash
rm -rf FASTGEN_OUTPUT/fastgen/infinitetalk_df/debug/
CUDA_VISIBLE_DEVICES=0 \
INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
INFINITETALK_CKPT=/.../infinitetalk.safetensors \
python train.py --config fastgen/configs/experiments/InfiniteTalk/config_df_test.py
```

- [ ] **Step 5: Verify results**

Expected:
- 20 iterations complete without OOM
- Loss printed at each iteration (0.01-0.20 range)
- Loss trending downward
- ~45s/iter on single A100-80GB
- Checkpoints saved at iter 10 and 20

---

## Task 2: KD Training Verification (ODE Initialization)

**Goal:** Confirm KD training works with ODE trajectories at correct resolution.

**Execution:** GPU 1 (parallel with Task 1)

**Depends on:** ODE trajectories must exist for test samples. Generate them first.

**Files:**
- Create: `fastgen/configs/experiments/InfiniteTalk/config_kd_test.py`
- Verify: `fastgen/methods/infinitetalk_kd.py`

- [ ] **Step 1: Generate ODE trajectories for test samples**

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/generate_infinitetalk_ode_pairs.py \
  --data_list_path data/test_precomputed/sample_list.txt \
  --neg_text_emb_path data/test_precomputed/neg_text_embeds.pt \
  --weights_dir /.../Wan2.1-I2V-14B-480P \
  --infinitetalk_ckpt /.../infinitetalk.safetensors \
  --num_steps 40 --batch_size 1 --max_samples 1 --seed 42
```

Verify: `ode_path.pt` shape is `[4, 16, 21, 56, 112]` (4 noisy states at correct resolution).

- [ ] **Step 2: Create config_kd_test.py**

```python
# fastgen/configs/experiments/InfiniteTalk/config_kd_test.py
import os
from fastgen.configs.experiments.InfiniteTalk.config_kd import create_config as create_base_config
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.utils import LazyCall as L
from fastgen.callbacks.callback import Callback

class StdoutLoggerCallback(Callback):
    def on_training_step_end(self, model=None, loss_dict=None, iteration=None, **kwargs):
        if loss_dict:
            parts = []
            for k, v in loss_dict.items():
                try: parts.append(f"{k}={float(v):.6f}")
                except: pass
            print(f"  [iter {iteration}] {', '.join(parts)}")

def create_config():
    config = create_base_config()
    config.trainer.max_iter = 20
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 10
    config.trainer.validation_iter = 999999
    config.trainer.callbacks = {"stdout": L(StdoutLoggerCallback)()}
    config.trainer.fsdp = False
    config.trainer.ddp = False
    DATA_LIST = os.environ.get("INFINITETALK_DATA_LIST", "data/test_precomputed/sample_list.txt")
    NEG_TEXT_EMB = os.environ.get("INFINITETALK_NEG_TEXT_EMB", "data/test_precomputed/neg_text_embeds.pt")
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=DATA_LIST,
        neg_text_emb_path=NEG_TEXT_EMB,
        batch_size=1,
        load_ode_path=True,  # KD needs ODE trajectories
        expected_latent_shape=config.model.input_shape,
        num_workers=0,
    )
    config.trainer.grad_accum_rounds = 1
    return config
```

- [ ] **Step 3: Run KD training**

```bash
rm -rf FASTGEN_OUTPUT/fastgen/infinitetalk_kd/debug/
CUDA_VISIBLE_DEVICES=1 \
INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
INFINITETALK_CKPT=/.../infinitetalk.safetensors \
python train.py --config fastgen/configs/experiments/InfiniteTalk/config_kd_test.py
```

- [ ] **Step 4: Verify results**

Expected:
- 20 iterations complete without OOM
- Loss printed at each iteration
- Loss in reasonable range
- Checkpoints saved

- [ ] **Step 5: Commit**

---

## Task 3: Implement Option 3 — Lazy Caching in Dataloader

**Goal:** When precomputed `.pt` files are missing but raw video/audio exist, encode on-the-fly using VAE/CLIP/wav2vec2 (loaded on the training GPU), save to disk, and return. T5 must be pre-computed (too large to fit alongside training model).

**Execution:** Code implementation while Tasks 1-2 run on GPUs.

**Files:**
- Modify: `fastgen/datasets/infinitetalk_dataloader.py`

**Design:**

```
InfiniteTalkDataset.__init__():
    - New params: raw_data_root, csv_path, weights_dir, wav2vec_dir
    - If raw_data_root is set, enable lazy mode
    - Load VAE + CLIP + wav2vec2 to self._encoders (shared across __getitem__ calls)
    - Build mapping: sample_name -> (video_path, audio_path, text)
    - Require num_workers=0 (GPU encoders can't be forked)

InfiniteTalkDataset.__getitem__():
    - Try loading cached .pt files (existing behavior)
    - If any file missing AND lazy mode enabled:
        - Check text_embeds.pt exists (required, T5 too large)
        - Encode missing modalities on GPU (VAE, CLIP, audio)
        - Save .pt files to disk (cache for next time)
        - Return the encoded data
    - If file missing AND lazy mode disabled: return None (existing behavior)
```

**Key constraints:**
- `num_workers=0` enforced when lazy mode is on (GPU tensors can't cross process boundaries)
- T5 is NOT loaded — `text_embeds.pt` must pre-exist
- Encoders loaded once in `__init__`, reused across all samples
- Thread-safe file writing (use atomic rename to prevent partial files)
- Memory: VAE ~1GB + CLIP ~2GB + wav2vec2 ~0.4GB = ~3.4GB extra on training GPU

- [ ] **Step 1: Add encoder loading to `__init__`**

Add optional `raw_data_root`, `csv_path`, `weights_dir`, `wav2vec_dir` params.
When set, load encoders to `self._device` and build the raw data mapping.

- [ ] **Step 2: Add `_encode_sample()` method**

Implements on-the-fly encoding matching `precompute_infinitetalk_data.py` exactly:
- Read video frames with av
- PIL resize first frame + torch resize all frames (aspect-ratio bucket)
- VAE encode (latents, first_frame_cond, motion_frame)
- CLIP encode
- wav2vec2 audio encode
- Save all `.pt` files atomically (write to `.tmp`, rename)

- [ ] **Step 3: Modify `__getitem__` to fallback to lazy encoding**

If cached files missing and `self._encoders` is set, call `_encode_sample()`.
If `text_embeds.pt` doesn't exist, skip (return None) — T5 must be pre-computed.

- [ ] **Step 4: Enforce `num_workers=0` in `InfiniteTalkDataLoader`**

When `raw_data_root` is set, override `num_workers=0` with a warning.

- [ ] **Step 5: Test with a mix of cached and uncached samples**

Create a test with 3 precomputed + 1 new (only text_embeds.pt). Verify the new sample
gets encoded, cached, and loaded correctly on the second access.

- [ ] **Step 6: Commit**

---

## Task 4: Documentation Update

**Files:**
- Modify: `docs/IMPLEMENTATION_PROGRESS.md`
- Modify: `docs/BUGS_AND_FIXES.md` (already up to date from earlier)
- Modify: `scripts/RUN_INSTRUCTIONS.txt` (already up to date)
- Modify: memory file

- [ ] **Step 1: Update IMPLEMENTATION_PROGRESS.md**

Add final status for all tasks including today's fixes:
- Resolution fix (448x896)
- T5 zero-padding fix
- ODE visual verification (clear output matching original pipeline)
- DF training verification (loss logging working)
- KD training verification (results from Task 2)
- Lazy caching implementation (Task 3)

- [ ] **Step 2: Update memory file**

Update `project_infinitetalk_implementation.md` with current status.

- [ ] **Step 3: Commit all documentation**

---

## Execution Order

```
Time 0:     Regenerate test data at 448x896 (both GPUs free, ~2 min)
            |
Time 2min:  GPU 0: Task 1 (DF training, ~20 min)
            GPU 1: Task 2 Step 1 (ODE extraction, ~25 min)
            Code:  Task 3 (lazy caching implementation)
            |
Time 25min: GPU 1: Task 2 Steps 2-4 (KD training, ~20 min)
            Code:  Task 3 continues / Task 4
            |
Time 45min: All verification complete. Commit everything.
```

---

## Success Criteria

- [ ] DF training: 20 iterations, loss 0.01-0.20, no OOM at [16,21,56,112]
- [ ] KD training: 20 iterations, loss reasonable, ODE path loaded correctly
- [ ] Lazy caching: new sample encoded on first access, cached `.pt` files created, second access loads from cache
- [ ] All documentation updated
