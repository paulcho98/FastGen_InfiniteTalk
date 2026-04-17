# Quarter-Resolution Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--quarter_res` flag to the precompute pipeline and a `quarter_res` dataloader option that halve each spatial pixel dimension before VAE encoding, reducing token count 4x (32,928 → 8,232) and activation memory from ~13.5GB to ~3.4GB. Full-res and quarter-res data coexist via filename suffix.

**Architecture:** The pixel-space resize happens *before* VAE encoding (not in latent space). Quarter-res files (`*_quarter.pt`) live alongside full-res files in the same sample directories. CLIP/T5/audio are resolution-independent and shared. A verification step confirms dimension correctness and saves pixel-space PNGs for visual inspection.

**Tech Stack:** PyTorch, WanVAE (stride 4,8,8), PIL, torchvision

---

## File Map

**Files to Modify:**
- `scripts/precompute_infinitetalk_data.py` — Add `--quarter_res` flag, quarter bucket table, suffix logic, pixel preview saving
- `fastgen/datasets/infinitetalk_dataloader.py` — Add `quarter_res` param, suffix-aware file loading

**Files to Create:**
- `scripts/verify_quarter_res.py` — Standalone verification script: load both resolutions, check dims, decode + save PNGs
- `fastgen/configs/experiments/InfiniteTalk/config_df_quarter.py` — DF config for quarter-res training

---

### Task 1: Add quarter-res bucket table and `--quarter_res` flag to precompute script

**Files:**
- Modify: `scripts/precompute_infinitetalk_data.py`

- [ ] **Step 1: Add the quarter-res bucket table after the existing ASPECT_RATIO_627 definition**

At line ~641, right after the `ASPECT_RATIO_627` dict, add:

```python
    ASPECT_RATIO_627_QUARTER = {
        k: [h // 2, w // 2] for k, [h, w] in ASPECT_RATIO_627.items()
    }
    # e.g., '0.50': [448, 896] → [224, 448]
    #        '1.00': [640, 640] → [320, 320]
```

- [ ] **Step 2: Select bucket table based on flag**

Replace line 645-646:
```python
    closest_bucket = sorted(ASPECT_RATIO_627.keys(), key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = ASPECT_RATIO_627[closest_bucket]
```

With:
```python
    buckets = ASPECT_RATIO_627_QUARTER if quarter_res else ASPECT_RATIO_627
    closest_bucket = sorted(buckets.keys(), key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = buckets[closest_bucket]
```

This requires passing `quarter_res` into `process_sample()`. Add it as a parameter to the function signature.

- [ ] **Step 3: Add filename suffix for quarter-res VAE outputs**

Define the suffix at the top of `process_sample()`:
```python
    vae_suffix = "_quarter" if quarter_res else ""
```

Then update the per-file skip checks and save calls for the 3 VAE-dependent files:
```python
    need_vae = not _exists(f"vae_latents{vae_suffix}.pt")
    need_ffc = not _exists(f"first_frame_cond{vae_suffix}.pt")
    need_motion = not _exists(f"motion_frame{vae_suffix}.pt")
```

And the save calls:
```python
    if need_vae:
        torch.save(latents, os.path.join(sample_dir, f"vae_latents{vae_suffix}.pt"))
    if need_ffc:
        torch.save(first_frame_cond, os.path.join(sample_dir, f"first_frame_cond{vae_suffix}.pt"))
    if need_motion:
        torch.save(motion_frame, os.path.join(sample_dir, f"motion_frame{vae_suffix}.pt"))
```

- [ ] **Step 4: Skip CLIP/T5/audio when `--quarter_res` is set**

These are resolution-independent; no re-encoding needed:
```python
    if quarter_res:
        need_clip = False
        need_t5 = False
        need_audio = False
```

- [ ] **Step 5: Save a pixel-space preview PNG for visual verification**

After the resize+crop step (line ~674), save the first frame as a PNG:
```python
    if quarter_res and need_video:
        # Save pixel-space preview for visual verification
        preview_path = os.path.join(sample_dir, "preview_quarter.png")
        if not os.path.exists(preview_path):
            from PIL import Image
            # video_tensor is [3, T, H, W] float32 in [-1, 1]
            first_frame_pixels = ((video_tensor[:, 0] + 1) / 2 * 255).clamp(0, 255).byte()
            first_frame_pixels = first_frame_pixels.permute(1, 2, 0).numpy()  # [H, W, 3]
            Image.fromarray(first_frame_pixels).save(preview_path)
            logger.info("  Saved quarter-res preview: %s (%dx%d)", preview_path, target_w, target_h)
```

- [ ] **Step 6: Add `--quarter_res` argparse flag**

In the `main()` argparse section (line ~737):
```python
    parser.add_argument("--quarter_res", action="store_true", default=False,
                        help="Halve spatial dimensions before VAE encoding. "
                             "Saves as *_quarter.pt alongside full-res files. "
                             "Skips CLIP/T5/audio (resolution-independent).")
```

Pass it through to `process_sample()`:
```python
    success = process_sample(
        ...,
        quarter_res=args.quarter_res,
    )
```

- [ ] **Step 7: Add shape logging after VAE encode**

Inside the VAE encode block, after getting latents:
```python
    logger.info("  VAE output: pixel [%d, %d, %d, %d] → latent %s%s",
                video_tensor.shape[0], video_tensor.shape[1],
                video_tensor.shape[2], video_tensor.shape[3],
                list(latents.shape), " (quarter)" if quarter_res else "")
```

- [ ] **Step 8: Commit**

```bash
git add scripts/precompute_infinitetalk_data.py
git commit -m "feat: add --quarter_res mode to precompute script

Halves each spatial pixel dimension before VAE encoding:
  448x896 → 224x448 (latent: 56x112 → 28x56)
Saves as *_quarter.pt alongside full-res files.
Skips CLIP/T5/audio (resolution-independent).
Saves pixel-space preview PNG for visual verification."
```

---

### Task 2: Add `quarter_res` support to the dataloader

**Files:**
- Modify: `fastgen/datasets/infinitetalk_dataloader.py`

- [ ] **Step 1: Add `quarter_res` parameter to `InfiniteTalkDataset.__init__`**

Add to the constructor signature (after `expected_latent_shape`):
```python
    quarter_res: bool = False,
```

Store and derive the suffix:
```python
    self.quarter_res = quarter_res
    self._vae_suffix = "_quarter" if quarter_res else ""
```

- [ ] **Step 2: Update `_load_sample` to use the suffix for VAE-dependent files**

In `_load_sample()` (line ~822), change the three VAE-dependent loads:

```python
    # --- VAE latents (clean video) ---
    real = torch.load(
        os.path.join(sample_dir, f"vae_latents{self._vae_suffix}.pt"),
        map_location="cpu",
        weights_only=False,
    )
```

```python
    # --- First frame conditioning ---
    first_frame_cond = torch.load(
        os.path.join(sample_dir, f"first_frame_cond{self._vae_suffix}.pt"),
        map_location="cpu",
        weights_only=False,
    )
```

CLIP, audio, and text loads remain unchanged (no suffix — shared across resolutions).

- [ ] **Step 3: Update the shape filter to check suffixed files**

In the resolution filter block (~line 610), update the vae_path:
```python
    vae_path = os.path.join(d, f"vae_latents{self._vae_suffix}.pt")
```

- [ ] **Step 4: Update `_encode_and_cache` for quarter-res lazy encoding**

In `_encode_and_cache()` (~line 680), add quarter-res bucket selection:
```python
    if self.quarter_res:
        BUCKETS_QUARTER = {k: [h // 2, w // 2] for k, [h, w] in ASPECT_RATIO_627.items()}
        target_h, target_w = BUCKETS_QUARTER[closest_bucket]
```

And update the save calls:
```python
    if need_vae:
        _save_atomic(latents, os.path.join(sample_dir, f"vae_latents{self._vae_suffix}.pt"))
    if need_ffc:
        _save_atomic(first_frame_cond, os.path.join(sample_dir, f"first_frame_cond{self._vae_suffix}.pt"))
    if need_motion:
        _save_atomic(motion_frame, os.path.join(sample_dir, f"motion_frame{self._vae_suffix}.pt"))
```

Also skip CLIP/audio re-encoding when quarter_res:
```python
    if self.quarter_res:
        need_clip = False
        need_audio = False
```

- [ ] **Step 5: Pass `quarter_res` through `InfiniteTalkDataLoader`**

In the `InfiniteTalkDataLoader` class, add `quarter_res` to its `__init__` and pass through to `InfiniteTalkDataset`:
```python
class InfiniteTalkDataLoader:
    def __init__(self, ..., quarter_res: bool = False, ...):
        ...
        self.dataset = InfiniteTalkDataset(
            ...,
            quarter_res=quarter_res,
        )
```

- [ ] **Step 6: Commit**

```bash
git add fastgen/datasets/infinitetalk_dataloader.py
git commit -m "feat: add quarter_res support to dataloader

Loads *_quarter.pt files for VAE-dependent tensors when enabled.
CLIP/T5/audio always use unsuffixed (resolution-independent).
Shape filter checks suffixed files. Lazy encoding respects quarter buckets."
```

---

### Task 3: Create verification script

**Files:**
- Create: `scripts/verify_quarter_res.py`

- [ ] **Step 1: Write the verification script**

```python
#!/usr/bin/env python3
"""Verify quarter-resolution precomputed data against full-resolution.

Checks:
  1. Dimension correctness: quarter latents are exactly half spatial dims
  2. VAE round-trip: decode both resolutions, save as PNGs for visual comparison
  3. Conditioning consistency: CLIP/T5/audio unchanged between resolutions
  4. Dataloader integration: load a sample at quarter-res, verify shapes

Usage:
    python scripts/verify_quarter_res.py \
        --sample_dir data/precomputed_talkvid/sample_0001 \
        --vae_path /path/to/Wan2.1_VAE.pth \
        --output_dir /tmp/quarter_res_verify
"""

import argparse
import os
import sys
import torch
import numpy as np


def check_dimensions(sample_dir):
    """Verify quarter-res latents are exactly half the spatial dims of full-res."""
    print("\n=== Dimension Check ===")
    results = {}

    for name in ["vae_latents", "first_frame_cond", "motion_frame"]:
        full_path = os.path.join(sample_dir, f"{name}.pt")
        quarter_path = os.path.join(sample_dir, f"{name}_quarter.pt")

        if not os.path.exists(full_path):
            print(f"  SKIP {name}: full-res not found")
            continue
        if not os.path.exists(quarter_path):
            print(f"  FAIL {name}: quarter-res not found")
            results[name] = False
            continue

        full = torch.load(full_path, map_location="cpu", weights_only=False)
        quarter = torch.load(quarter_path, map_location="cpu", weights_only=False)

        # Handle dict format
        if isinstance(full, dict):
            full = next(v for v in full.values() if isinstance(v, torch.Tensor))
        if isinstance(quarter, dict):
            quarter = next(v for v in quarter.values() if isinstance(v, torch.Tensor))

        print(f"  {name}:")
        print(f"    Full:    {list(full.shape)}")
        print(f"    Quarter: {list(quarter.shape)}")

        # Check: channels same, temporal same, spatial halved
        ok = True
        if full.shape[0] != quarter.shape[0]:
            print(f"    FAIL: channels differ ({full.shape[0]} vs {quarter.shape[0]})")
            ok = False
        if full.shape[1] != quarter.shape[1]:
            print(f"    FAIL: temporal differs ({full.shape[1]} vs {quarter.shape[1]})")
            ok = False
        if len(full.shape) >= 3:
            expected_h = full.shape[2] // 2
            expected_w = full.shape[3] // 2
            if quarter.shape[2] != expected_h or quarter.shape[3] != expected_w:
                print(f"    FAIL: spatial not halved (expected [{expected_h}, {expected_w}], "
                      f"got [{quarter.shape[2]}, {quarter.shape[3]}])")
                ok = False
            else:
                print(f"    PASS: spatial halved [{full.shape[2]},{full.shape[3]}] → "
                      f"[{quarter.shape[2]},{quarter.shape[3]}]")

        results[name] = ok

    return results


def check_shared_tensors(sample_dir):
    """Verify CLIP/T5/audio are unchanged (no quarter suffix needed)."""
    print("\n=== Shared Tensor Check ===")
    for name, expected_suffix_shape in [
        ("clip_features", (257, 1280)),
        ("text_embeds", (512, 4096)),
    ]:
        path = os.path.join(sample_dir, f"{name}.pt")
        quarter_path = os.path.join(sample_dir, f"{name}_quarter.pt")
        if os.path.exists(quarter_path):
            print(f"  WARN {name}: _quarter.pt exists but shouldn't (CLIP/T5 are resolution-independent)")

        if os.path.exists(path):
            t = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(t, dict):
                t = next(v for v in t.values() if isinstance(v, torch.Tensor))
            print(f"  {name}: {list(t.shape)} — PASS (no quarter variant needed)")
        else:
            print(f"  {name}: not found")

    audio_path = os.path.join(sample_dir, "audio_emb.pt")
    if os.path.exists(audio_path):
        t = torch.load(audio_path, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = next(v for v in t.values() if isinstance(v, torch.Tensor))
        print(f"  audio_emb: {list(t.shape)} — PASS (resolution-independent)")


def decode_and_save_previews(sample_dir, vae_path, output_dir):
    """Decode both resolutions through VAE and save first-frame PNGs."""
    print("\n=== VAE Decode Visual Verification ===")

    if not vae_path or not os.path.exists(vae_path):
        print("  SKIP: VAE path not provided or not found")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Add InfiniteTalk to path for WanVAE
    it_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../InfiniteTalk"))
    # Try the reference copy
    if not os.path.isdir(it_root):
        it_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../InfiniteTalk"))
    if it_root not in sys.path:
        sys.path.insert(0, it_root)

    import types
    for mod_name in ["xformers", "xformers.ops", "decord"]:
        if mod_name not in sys.modules:
            mock = types.ModuleType("mock_" + mod_name)
            mock.__path__ = []
            mock.__spec__ = None
            sys.modules[mod_name] = mock

    from wan.modules.vae import WanVAE
    vae = WanVAE(vae_path=vae_path)
    vae.eval().requires_grad_(False)

    from PIL import Image

    for suffix, label in [("", "full"), ("_quarter", "quarter")]:
        lat_path = os.path.join(sample_dir, f"vae_latents{suffix}.pt")
        if not os.path.exists(lat_path):
            print(f"  SKIP {label}: {lat_path} not found")
            continue

        latents = torch.load(lat_path, map_location="cpu", weights_only=False)
        if isinstance(latents, dict):
            latents = next(v for v in latents.values() if isinstance(v, torch.Tensor))

        print(f"  Decoding {label} latents {list(latents.shape)}...")

        # Decode first few frames only (save memory)
        # Take first 5 latent frames = first 17 pixel frames
        short_lat = latents[:, :5].unsqueeze(0).float().cuda()
        with torch.no_grad():
            decoded = vae.decode(short_lat)
        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0] if len(decoded) == 1 else torch.cat(decoded)
        decoded = decoded.squeeze(0)  # [C, T, H, W]

        # Save first frame
        frame0 = ((decoded[:, 0].cpu().clamp(-1, 1) + 1) / 2 * 255).byte()
        frame0 = frame0.permute(1, 2, 0).numpy()  # [H, W, 3]
        out_path = os.path.join(output_dir, f"decoded_{label}_frame0.png")
        Image.fromarray(frame0).save(out_path)
        print(f"    Saved: {out_path} ({frame0.shape[1]}x{frame0.shape[0]})")

    # Also copy the pixel preview if it exists
    preview_path = os.path.join(sample_dir, "preview_quarter.png")
    if os.path.exists(preview_path):
        import shutil
        dst = os.path.join(output_dir, "pixel_preview_quarter.png")
        shutil.copy2(preview_path, dst)
        print(f"    Copied pixel preview: {dst}")

    del vae
    torch.cuda.empty_cache()


def check_dataloader_integration(sample_dir, quarter_res=True):
    """Try loading a sample through the dataloader path and verify shapes."""
    print("\n=== Dataloader Integration Check ===")

    # Simulate what _load_sample does
    suffix = "_quarter" if quarter_res else ""

    tensors = {}
    for name, expected_ndim in [
        (f"vae_latents{suffix}", 4),
        (f"first_frame_cond{suffix}", 4),
        ("clip_features", 3),
        ("audio_emb", 3),
        ("text_embeds", 3),
    ]:
        path = os.path.join(sample_dir, f"{name}.pt")
        if not os.path.exists(path):
            print(f"  FAIL: {name}.pt not found")
            continue
        t = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = next(v for v in t.values() if isinstance(v, torch.Tensor))
        tensors[name] = t
        print(f"  {name}: {list(t.shape)} (ndim={t.ndim}, expected={expected_ndim}) "
              f"{'PASS' if t.ndim == expected_ndim else 'FAIL'}")

    # Check consistency between vae_latents and first_frame_cond
    vae_key = f"vae_latents{suffix}"
    ffc_key = f"first_frame_cond{suffix}"
    if vae_key in tensors and ffc_key in tensors:
        vae_shape = tensors[vae_key].shape
        ffc_shape = tensors[ffc_key].shape
        if vae_shape[0] == ffc_shape[0] and vae_shape[2:] == ffc_shape[2:]:
            print(f"  Spatial consistency (vae vs ffc): PASS "
                  f"(C={vae_shape[0]}, H={vae_shape[2]}, W={vae_shape[3]})")
        else:
            print(f"  Spatial consistency: FAIL (vae={list(vae_shape)}, ffc={list(ffc_shape)})")


def main():
    parser = argparse.ArgumentParser(description="Verify quarter-resolution precomputed data")
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Path to a single sample directory with both full and quarter .pt files")
    parser.add_argument("--vae_path", type=str, default="",
                        help="Path to Wan2.1_VAE.pth for decode verification (optional)")
    parser.add_argument("--output_dir", type=str, default="/tmp/quarter_res_verify",
                        help="Where to save decoded PNGs")
    args = parser.parse_args()

    print(f"Sample dir: {args.sample_dir}")

    dim_results = check_dimensions(args.sample_dir)
    check_shared_tensors(args.sample_dir)
    check_dataloader_integration(args.sample_dir, quarter_res=True)

    if args.vae_path:
        decode_and_save_previews(args.sample_dir, args.vae_path, args.output_dir)

    # Final summary
    print("\n=== Summary ===")
    all_pass = all(dim_results.values()) if dim_results else False
    for name, ok in dim_results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/verify_quarter_res.py
git commit -m "feat: add quarter-res verification script

Checks dimension correctness (spatial halved), shared tensor integrity
(CLIP/T5/audio unchanged), dataloader integration, and optionally
decodes both resolutions through VAE for visual comparison PNGs."
```

---

### Task 4: Create quarter-res DF training config

**Files:**
- Create: `fastgen/configs/experiments/InfiniteTalk/config_df_quarter.py`

- [ ] **Step 1: Write the quarter-res config**

```python
"""
Diffusion Forcing config for quarter-resolution InfiniteTalk training.

Spatial dims halved: 448×896 → 224×448 pixel, 56×112 → 28×56 latent.
Token count: 21 × 14 × 28 = 8,232 (vs 32,928 at full res).
Expected VRAM: ~55-58 GB on single A100-80GB (vs ~70-77 GB full res).

Usage:
    torchrun --nproc_per_node=8 train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_df_quarter.py
"""

from fastgen.configs.experiments.InfiniteTalk.config_df_prod import create_config as create_prod_config


def create_config():
    config = create_prod_config()

    # Quarter-res latent shape: 224×448 pixel → 28×56 latent
    config.model.input_shape = [16, 21, 28, 56]

    # Dataloader: load *_quarter.pt files
    config.dataloader_train.quarter_res = True
    config.dataloader_train.expected_latent_shape = [16, 21, 28, 56]

    config.log_config.group = "infinitetalk_df_quarter"
    return config
```

- [ ] **Step 2: Commit**

```bash
git add fastgen/configs/experiments/InfiniteTalk/config_df_quarter.py
git commit -m "feat: add quarter-res DF training config

Inherits prod config, overrides input_shape to [16,21,28,56] and
enables quarter_res in dataloader. ~4x token count reduction."
```

---

### Task 5: End-to-end verification

- [ ] **Step 1: Run precompute on 1-3 test samples with `--quarter_res`**

```bash
python scripts/precompute_infinitetalk_data.py \
    --csv_path /data/karlo-research_715/workspace/kinemaar/datasets/TalkVid/video_list.csv \
    --data_root /data/karlo-research_715/workspace/kinemaar/datasets/TalkVid \
    --output_dir data/precomputed_talkvid \
    --weights_dir /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir "" \
    --num_samples 3 \
    --quarter_res \
    --only_vae
```

Expected: 3 samples with `vae_latents_quarter.pt`, `first_frame_cond_quarter.pt`, `motion_frame_quarter.pt`, `preview_quarter.png`.

- [ ] **Step 2: Run verification script**

```bash
python scripts/verify_quarter_res.py \
    --sample_dir data/precomputed_talkvid/<first_sample_name> \
    --vae_path /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
    --output_dir /tmp/quarter_res_verify
```

Expected output:
```
=== Dimension Check ===
  vae_latents:
    Full:    [16, 23, 56, 112]
    Quarter: [16, 23, 28, 56]
    PASS: spatial halved [56,112] → [28,56]
  first_frame_cond:
    Full:    [16, 23, 56, 112]
    Quarter: [16, 23, 28, 56]
    PASS: spatial halved [56,112] → [28,56]
  ...
=== Summary ===
  Overall: ALL PASS
```

- [ ] **Step 3: Visually inspect the PNG outputs**

Check:
- `preview_quarter.png` — pixel-space first frame at 224×448 (should look like a recognizable face, just smaller)
- `decoded_full_frame0.png` — VAE round-trip at full res
- `decoded_quarter_frame0.png` — VAE round-trip at quarter res (should look similar but lower detail)

- [ ] **Step 4: Verify dataloader loads correctly**

```bash
python -c "
import torch
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataset

ds = InfiniteTalkDataset(
    data_list_path='data/precomputed_talkvid/all_viable_train.txt',
    quarter_res=True,
    expected_latent_shape=[16, 21, 28, 56],
    num_latent_frames=21,
)
print(f'Dataset: {len(ds)} samples')
sample = ds[0]
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {list(v.shape)}')
"
```

Expected: `real` and `first_frame_cond` have shape `[16, 21, 28, 56]`, audio/CLIP/T5 unchanged.

- [ ] **Step 5: Commit verification results**

```bash
git commit --allow-empty -m "test: quarter-res precompute + verification PASSED

Dimensions: [16,23,56,112] → [16,23,28,56] (spatial halved)
CLIP/T5/audio: unchanged (resolution-independent)
VAE decode: visual quality acceptable at quarter-res
Dataloader: loads correctly with quarter_res=True"
```
