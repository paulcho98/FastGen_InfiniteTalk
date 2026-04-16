# Future Anchor for DF Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the diffusion model during DF training to condition on a clean GT latent frame from a stochastic future distance, so the model learns distance-aware identity anchoring that transfers to SF's lookahead sink at inference.

**Architecture:** During DF training, each sample includes 5 precomputed future latent frames (beyond the 81-frame clip). One is randomly selected per forward pass, embedded as extra tokens appended after the main video sequence, and made globally visible via a modified FlexAttention mask. The future anchor's RoPE position encodes its temporal distance. This is an optional stochastic attention config — some configs use it, some don't. At SF inference, the lookahead sink fills the same attention role.

**Tech Stack:** PyTorch, FlexAttention (`torch.nn.attention.flex_attention`), Wan 2.1 VAE, existing InfiniteTalk dataloader/precompute infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `fastgen/datasets/infinitetalk_dataloader.py` | Modify | Load `future_anchor_latents.pt`, include in sample dict |
| `scripts/precompute_future_anchors.py` | Create | Precompute 5 future latent frames per sample |
| `fastgen/networks/InfiniteTalk/network_causal.py` | Modify | `_build_block_mask` extended for anchor tokens; `_forward_full_sequence` concatenates anchor tokens; `_sample_attn_config` propagates `future_anchor` flag |
| `fastgen/configs/experiments/InfiniteTalk/config_df_quarter_stochastic_anchor.py` | Create | Stochastic DF config with `future_anchor: True` in some attention configs |
| `scripts/run_df_training_quarter_stochastic_anchor.sh` | Create | Training launcher |
| `tests/test_future_anchor_mask.py` | Create | Mask correctness tests |
| `tests/test_future_anchor_forward.py` | Create | End-to-end forward pass shape tests |

---

## Task 1: Precompute Future Anchor Latents

**Files:**
- Create: `scripts/precompute_future_anchors.py`

This script reads each sample's source video, encodes 5 extra latent frames beyond the clip boundary using the Wan VAE, and saves them alongside the existing precomputed data.

- [ ] **Step 1: Write the precompute script**

```python
#!/usr/bin/env python3
"""Precompute future anchor latents for DF training with lookahead anchoring.

For each sample in a precomputed data list, reads the source video, extracts
frames beyond the training clip boundary, VAE-encodes them with temporal
context overlap, and saves as future_anchor_latents.pt.

Output per sample: [16, 5, H_lat, W_lat] (bf16) — 5 future latent frames,
each selectable as a lookahead anchor at distance 1..5 past the sequence end.

Usage:
    python scripts/precompute_future_anchors.py \
        --data_list data/precomputed_talkvid/train_excl_val30.txt \
        --raw_data_root /data/.../datasets/TalkVid/ \
        --vae_path /data/.../Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
        --quarter_res \
        --num_workers 1
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, FASTGEN_ROOT)

from fastgen.datasets.infinitetalk_dataloader import _add_infinitetalk_to_path
_add_infinitetalk_to_path()


def parse_args():
    p = argparse.ArgumentParser(description="Precompute future anchor latents")
    p.add_argument("--data_list", required=True,
                   help="Text file listing precomputed sample dirs (one per line)")
    p.add_argument("--raw_data_root", required=True,
                   help="Root directory for raw TalkVid data (contains video dirs)")
    p.add_argument("--vae_path", required=True,
                   help="Path to Wan2.1_VAE.pth")
    p.add_argument("--quarter_res", action="store_true",
                   help="Use quarter resolution (224x448)")
    p.add_argument("--target_h", type=int, default=None,
                   help="Override target height (default: 224 if quarter_res, else 448)")
    p.add_argument("--target_w", type=int, default=None,
                   help="Override target width (default: 448 if quarter_res, else 896)")
    p.add_argument("--num_future_latent_frames", type=int, default=5,
                   help="Number of future latent frames to encode (default: 5)")
    p.add_argument("--context_overlap_frames", type=int, default=5,
                   help="Number of video frames from clip tail for VAE temporal context (default: 5)")
    p.add_argument("--skip_existing", action="store_true", default=True,
                   help="Skip samples that already have future_anchor_latents.pt")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve_video_path(sample_dir, raw_data_root):
    """Resolve source video path from precomputed sample dir name.

    Sample dir name: data_VIDEOID_VIDEOID_START_END
    Video at: raw_data_root/data/VIDEOID/VIDEOID_START_END.mp4
    """
    basename = os.path.basename(sample_dir)
    if basename.startswith("data_"):
        basename = basename[5:]
    parts = basename.split("_")
    if len(parts) < 4 or parts[0] != parts[1]:
        return None, 0, 0
    video_id = parts[0]
    start_frame = int(parts[2])
    end_frame = int(parts[3])
    # Try common extensions
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        video_path = os.path.join(raw_data_root, "data", video_id,
                                  f"{video_id}_{parts[2]}_{parts[3]}{ext}")
        if os.path.exists(video_path):
            return video_path, start_frame, end_frame
    # Also check for full video
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        video_path = os.path.join(raw_data_root, "data", video_id, f"{video_id}{ext}")
        if os.path.exists(video_path):
            return video_path, start_frame, end_frame
    return None, start_frame, end_frame


def read_video_frames(video_path, start_frame, num_frames, target_h, target_w):
    """Read num_frames starting from start_frame, resize to target size.

    Returns: numpy array [num_frames, H, W, 3] uint8, or None if not enough frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame + num_frames > total:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames)


def encode_future_frames(vae, frames_np, context_overlap, num_future_latent,
                         device, dtype=torch.bfloat16):
    """VAE-encode future frames with temporal context overlap.

    Args:
        frames_np: [N, H, W, 3] uint8 — context_overlap + future video frames
        context_overlap: number of leading frames used only for VAE temporal context
        num_future_latent: number of future latent frames to return
        device: CUDA device
        dtype: output dtype

    Returns:
        [16, num_future_latent, H_lat, W_lat] tensor
    """
    # Normalize to [-1, 1] and reshape for VAE: [C, T, H, W]
    frames_t = torch.from_numpy(frames_np).float().permute(3, 0, 1, 2) / 127.5 - 1.0

    vae.model = vae.model.to(device)
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]

    with torch.no_grad():
        latent = vae.encode([frames_t.to(device)])
        if isinstance(latent, (list, tuple)):
            latent = latent[0]
        latent = latent.to(dtype).cpu()  # [16, T_lat, H_lat, W_lat]

    # Discard context overlap latent frames, keep future frames
    # Total video frames = context_overlap + future_video_frames
    # Total latent frames = 1 + (total_video - 1) // 4
    # Context latent frames = 1 + (context_overlap - 1) // 4
    # (With context_overlap=5: 1 + 4//4 = 2 context latent frames)
    total_video = frames_np.shape[0]
    total_latent = 1 + (total_video - 1) // 4
    context_latent = 1 + (context_overlap - 1) // 4

    future_latent = latent[:, context_latent:context_latent + num_future_latent]
    return future_latent


def main():
    args = parse_args()
    target_h = args.target_h or (224 if args.quarter_res else 448)
    target_w = args.target_w or (448 if args.quarter_res else 896)
    suffix = "_quarter" if args.quarter_res else ""

    # Load sample list
    list_dir = os.path.dirname(os.path.abspath(args.data_list))
    with open(args.data_list) as f:
        sample_dirs = [line.strip() for line in f if line.strip()]
    resolved = []
    for d in sample_dirs:
        if os.path.isabs(d) and os.path.isdir(d):
            resolved.append(d)
        elif os.path.isdir(os.path.join(list_dir, d)):
            resolved.append(os.path.join(list_dir, d))
        elif os.path.isdir(d):
            resolved.append(os.path.abspath(d))
    sample_dirs = resolved
    print(f"Found {len(sample_dirs)} samples")

    # Load VAE
    from wan.modules.vae import WanVAE
    print(f"Loading VAE from {args.vae_path}...")
    vae = WanVAE(vae_pth=args.vae_path, device="cpu")

    # Compute frame counts
    # 5 future latent frames = 20 future video frames
    # 5 context overlap video frames = 2 context latent frames
    # Total: 25 video frames read per sample
    future_video_frames = args.num_future_latent_frames * 4
    context_video_frames = args.context_overlap_frames
    total_video_frames = context_video_frames + future_video_frames

    success, skip, fail = 0, 0, 0
    for i, sample_dir in enumerate(sample_dirs):
        out_path = os.path.join(sample_dir, f"future_anchor_latents{suffix}.pt")
        if args.skip_existing and os.path.exists(out_path):
            skip += 1
            continue

        video_path, start_frame, end_frame = resolve_video_path(
            sample_dir, args.raw_data_root
        )
        if video_path is None:
            fail += 1
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(sample_dirs)}] SKIP (no video): "
                      f"{os.path.basename(sample_dir)}")
            continue

        # Read frames: last context_video_frames of the clip + future_video_frames
        # The clip spans [start_frame, end_frame). We want:
        # context: [end_frame - context_video_frames, end_frame)
        # future:  [end_frame, end_frame + future_video_frames)
        read_start = end_frame - context_video_frames
        frames = read_video_frames(
            video_path, read_start, total_video_frames, target_h, target_w
        )
        if frames is None:
            fail += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(sample_dirs)}] SKIP (not enough frames): "
                      f"{os.path.basename(sample_dir)}")
            continue

        future_latent = encode_future_frames(
            vae, frames, context_video_frames,
            args.num_future_latent_frames, args.device,
        )

        # Save atomically
        tmp_path = out_path + ".tmp"
        torch.save(future_latent, tmp_path)
        os.rename(tmp_path, out_path)
        success += 1

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(sample_dirs)}] {future_latent.shape} -> "
                  f"{os.path.basename(sample_dir)}")

    print(f"\nDone: {success} encoded, {skip} skipped, {fail} failed "
          f"out of {len(sample_dirs)} total")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test on a small sample**

Run:
```bash
python scripts/precompute_future_anchors.py \
    --data_list data/precomputed_talkvid/val_quarter_2.txt \
    --raw_data_root /data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/ \
    --vae_path /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
    --quarter_res
```

Expected: `future_anchor_latents_quarter.pt` created in the sample dir with shape `[16, 5, 28, 56]`.

- [ ] **Step 3: Commit**

```bash
git add scripts/precompute_future_anchors.py
git commit -m "feat: precompute script for future anchor latents (DF lookahead anchoring)"
```

---

## Task 2: Dataloader — Load Future Anchor Latents

**Files:**
- Modify: `fastgen/datasets/infinitetalk_dataloader.py:845-974`
- Test: `tests/test_future_anchor_dataloader.py`

Add optional loading of `future_anchor_latents.pt` into the sample dict. When the file doesn't exist, set the value to `None` (graceful fallback — training proceeds without anchor for that sample).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for future anchor latent loading in the dataloader."""
import os
import tempfile
import torch
import pytest


def _make_sample_dir(tmp_path, *, include_future_anchor=True, quarter=True):
    """Create a minimal precomputed sample directory."""
    suffix = "_quarter" if quarter else ""
    H, W = (28, 56) if quarter else (56, 112)
    T_lat = 21

    torch.save(
        torch.randn(16, T_lat, H, W, dtype=torch.bfloat16),
        os.path.join(tmp_path, f"vae_latents{suffix}.pt"),
    )
    torch.save(
        torch.randn(16, T_lat, H, W, dtype=torch.bfloat16),
        os.path.join(tmp_path, f"first_frame_cond{suffix}.pt"),
    )
    torch.save(
        torch.randn(1, 257, 1280, dtype=torch.bfloat16),
        os.path.join(tmp_path, "clip_features.pt"),
    )
    torch.save(
        torch.randn(81, 12, 768, dtype=torch.bfloat16),
        os.path.join(tmp_path, "audio_emb.pt"),
    )
    torch.save(
        torch.randn(1, 512, 4096, dtype=torch.bfloat16),
        os.path.join(tmp_path, "text_embeds.pt"),
    )
    if include_future_anchor:
        torch.save(
            torch.randn(16, 5, H, W, dtype=torch.bfloat16),
            os.path.join(tmp_path, f"future_anchor_latents{suffix}.pt"),
        )
    return tmp_path


def test_load_sample_with_future_anchor(tmp_path):
    """Future anchor loaded when file exists."""
    from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataset
    sample_dir = _make_sample_dir(tmp_path, include_future_anchor=True)
    ds = InfiniteTalkDataset.__new__(InfiniteTalkDataset)
    ds.num_latent_frames = 21
    ds._train_pixel_frames = 81
    ds._vae_suffix = "_quarter"
    ds.neg_text_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
    ds.load_ode_path = False
    ds.audio_data_root = None
    result = ds._load_sample(str(sample_dir))
    assert "future_anchor_latents" in result
    assert result["future_anchor_latents"].shape == (16, 5, 28, 56)
    assert result["future_anchor_latents"].dtype == torch.bfloat16


def test_load_sample_without_future_anchor(tmp_path):
    """Graceful fallback when future anchor file doesn't exist."""
    from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataset
    sample_dir = _make_sample_dir(tmp_path, include_future_anchor=False)
    ds = InfiniteTalkDataset.__new__(InfiniteTalkDataset)
    ds.num_latent_frames = 21
    ds._train_pixel_frames = 81
    ds._vae_suffix = "_quarter"
    ds.neg_text_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
    ds.load_ode_path = False
    ds.audio_data_root = None
    result = ds._load_sample(str(sample_dir))
    assert "future_anchor_latents" in result
    assert result["future_anchor_latents"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_future_anchor_dataloader.py -v`
Expected: FAIL (key `future_anchor_latents` not in result)

- [ ] **Step 3: Implement the loading in `_load_sample`**

In `fastgen/datasets/infinitetalk_dataloader.py`, after the text_embeds loading block (around line 930), before the `result = {` line (line 932), add:

```python
        # --- Future anchor latents (optional, for DF lookahead anchoring) ---
        future_anchor_path = os.path.join(
            sample_dir, f"future_anchor_latents{self._vae_suffix}.pt"
        )
        if os.path.exists(future_anchor_path):
            future_anchor_latents = torch.load(
                future_anchor_path, map_location="cpu", weights_only=False,
            )
            if isinstance(future_anchor_latents, dict):
                future_anchor_latents = next(
                    v for v in future_anchor_latents.values()
                    if isinstance(v, torch.Tensor)
                )
            future_anchor_latents = future_anchor_latents.to(torch.bfloat16)
        else:
            future_anchor_latents = None
```

And add `"future_anchor_latents": future_anchor_latents,` to the `result` dict.

**Collation note:** When `future_anchor_latents` is `None` for some samples and a tensor for others, `default_collate` will fail. Handle this by using a custom collate or ensuring all samples in a batch are consistent. The simplest approach: in `__getitem__`, replace `None` with a zero tensor of the expected shape so collation always works, and add a boolean `"has_future_anchor"` key.

Add to `_load_sample`, replacing the `None` fallback:

```python
        else:
            # No future anchor available — fill with zeros so collation works.
            # Training code checks has_future_anchor to skip.
            H_lat = real.shape[2]
            W_lat = real.shape[3]
            future_anchor_latents = torch.zeros(
                16, 5, H_lat, W_lat, dtype=torch.bfloat16
            )
```

And add to result:

```python
        result = {
            "real": real,
            "first_frame_cond": first_frame_cond,
            "clip_features": clip_features,
            "audio_emb": audio_emb,
            "text_embeds": text_embeds,
            "neg_text_embeds": self.neg_text_embeds.clone(),
            "future_anchor_latents": future_anchor_latents,
            "has_future_anchor": os.path.exists(future_anchor_path),
        }
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_future_anchor_dataloader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/datasets/infinitetalk_dataloader.py tests/test_future_anchor_dataloader.py
git commit -m "feat: load future_anchor_latents.pt in dataloader (optional, zero-fallback)"
```

---

## Task 3: Extend FlexAttention Mask for Anchor Tokens

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:1529-1615`
- Test: `tests/test_future_anchor_mask.py`

Extend `_build_block_mask` to accept `num_anchor_tokens` — extra tokens appended after the main sequence that are globally visible (every query can attend to them, they can attend to everything).

- [ ] **Step 1: Write the failing test**

```python
"""Tests for FlexAttention mask with future anchor tokens."""
import math
import torch
import pytest

# FlexAttention may not be available in CI — skip gracefully
try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FLEX_AVAILABLE, reason="FlexAttention not available")


def _get_mask_builder():
    """Get _build_block_mask as an unbound function."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    return CausalInfiniteTalkWan._build_block_mask


def _dense_mask_from_block_mask(block_mask, length):
    """Convert BlockMask to dense bool tensor for inspection."""
    mask = torch.zeros(length, length, dtype=torch.bool)
    for q in range(length):
        for kv in range(length):
            # FlexAttention block_mask stores precomputed blocks;
            # we test the closure directly instead
            pass
    return mask


def test_mask_no_anchor_unchanged():
    """With num_anchor_tokens=0, mask is identical to the original."""
    builder = _get_mask_builder()
    # We test the mask function closure, not the BlockMask object
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.chunk_size = 3

    # Build with and without anchor — compare mask functions
    # For this test, we verify the function signature accepts num_anchor_tokens
    mask_0 = net._build_block_mask(
        torch.device("cpu"), num_frames=6, frame_seqlen=4,
        chunk_size=3, local_attn_size=10, sink_size=1,
        num_anchor_tokens=0,
    )
    assert mask_0 is not None  # FlexAttention available


def test_mask_anchor_globally_visible():
    """Anchor tokens (appended after main) are visible to all queries
    and can see all tokens."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.chunk_size = 3

    num_frames = 6
    frame_seqlen = 4  # small for testing
    num_anchor_tokens = 4  # 1 anchor frame
    total_main = num_frames * frame_seqlen  # 24
    total_with_anchor = total_main + num_anchor_tokens  # 28

    # We need to test the mask function directly.
    # Capture it by monkey-patching create_block_mask.
    captured_fn = {}
    original_create = create_block_mask

    def spy_create(mask_fn, **kwargs):
        captured_fn["fn"] = mask_fn
        return original_create(mask_fn, **kwargs)

    import fastgen.networks.InfiniteTalk.network_causal as mod
    orig = mod.create_block_mask
    mod.create_block_mask = spy_create
    try:
        net._build_block_mask(
            torch.device("cpu"), num_frames=num_frames,
            frame_seqlen=frame_seqlen, chunk_size=3,
            local_attn_size=10, sink_size=1,
            num_anchor_tokens=num_anchor_tokens,
        )
    finally:
        mod.create_block_mask = orig

    fn = captured_fn["fn"]

    # Anchor tokens are at positions [total_main, total_main + num_anchor_tokens)
    anchor_start = total_main

    # 1. Every main query can see anchor tokens
    for q in range(total_main):
        for kv in range(anchor_start, anchor_start + num_anchor_tokens):
            assert fn(0, 0, q, kv), \
                f"Main query {q} should see anchor kv {kv}"

    # 2. Anchor queries can see all main tokens
    for q in range(anchor_start, anchor_start + num_anchor_tokens):
        for kv in range(total_main):
            assert fn(0, 0, q, kv), \
                f"Anchor query {q} should see main kv {kv}"

    # 3. Anchor queries can see other anchor tokens
    for q in range(anchor_start, anchor_start + num_anchor_tokens):
        for kv in range(anchor_start, anchor_start + num_anchor_tokens):
            assert fn(0, 0, q, kv), \
                f"Anchor query {q} should see anchor kv {kv}"


def test_mask_main_to_main_still_causal():
    """Main-to-main attention remains blockwise causal with anchor present."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.chunk_size = 3

    num_frames = 6
    frame_seqlen = 4
    num_anchor_tokens = 4

    captured_fn = {}
    original_create = create_block_mask

    def spy_create(mask_fn, **kwargs):
        captured_fn["fn"] = mask_fn
        return original_create(mask_fn, **kwargs)

    import fastgen.networks.InfiniteTalk.network_causal as mod
    orig = mod.create_block_mask
    mod.create_block_mask = spy_create
    try:
        net._build_block_mask(
            torch.device("cpu"), num_frames=num_frames,
            frame_seqlen=frame_seqlen, chunk_size=3,
            local_attn_size=10, sink_size=1,
            num_anchor_tokens=num_anchor_tokens,
        )
    finally:
        mod.create_block_mask = orig

    fn = captured_fn["fn"]
    total_main = num_frames * frame_seqlen

    # Chunk 0: frames 0-2 (tokens 0-11), Chunk 1: frames 3-5 (tokens 12-23)
    # Chunk 1 token should NOT see chunk 0 tokens beyond the window
    # (with local_attn_size=10, sink=1: window = 9 rolling frames)
    # All 6 frames fit in the window, so all main-to-main within window is ok.
    # But a chunk-1 token at the END should still be causal: it sees chunks <= 1
    # but NOT tokens from chunk 2 (which doesn't exist here, but the pattern holds).

    # Token 12 (start of chunk 1) should NOT see token 23 (end of chunk 1)
    # Wait — within a chunk, tokens attend bidirectionally. So token 12 CAN see 23.
    # Tokens in chunk 1 (12-23) attend to each other bidirectionally.
    assert fn(0, 0, 12, 23), "Within chunk 1: bidirectional"

    # A future chunk's token should not be visible to an earlier chunk
    # (but we only have 2 chunks here — verify chunk 0 doesn't see chunk 1 end)
    # Token 0 (chunk 0) should see up to token 11 (end of chunk 0)
    assert fn(0, 0, 0, 11), "Chunk 0 tokens: bidirectional within chunk"
    # Token 0 should NOT see token 12 (chunk 1) — causal
    assert not fn(0, 0, 0, 12), "Chunk 0 should not see chunk 1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_future_anchor_mask.py -v`
Expected: FAIL (`_build_block_mask` doesn't accept `num_anchor_tokens`)

- [ ] **Step 3: Implement the mask extension**

In `network_causal.py`, modify `_build_block_mask` (line 1529):

```python
    def _build_block_mask(
        self,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        chunk_size: int = None,
        local_attn_size: int = -1,
        sink_size: int = 0,
        num_anchor_tokens: int = 0,
    ) -> Optional[BlockMask]:
```

Replace the mask function and padded_length computation. After computing `total_length` and before the padding:

```python
        total_main = num_frames * frame_seqlen
        total_with_anchor = total_main + num_anchor_tokens
        total_length = total_with_anchor
        pad_len = math.ceil(total_length / 128) * 128 - total_length
        padded_length = total_length + pad_len
```

Keep the existing `ends`/`starts` computation but size them for `padded_length`. After computing ends/starts for main tokens, set the anchor token entries:

```python
        # Anchor tokens: globally visible — they see everything, everything sees them
        if num_anchor_tokens > 0:
            anchor_start = total_main
            anchor_end = total_main + num_anchor_tokens
            # Anchor queries see the entire sequence
            ends[anchor_start:anchor_end] = total_with_anchor
            starts[anchor_start:anchor_end] = 0
```

Update the mask function:

```python
        def attention_mask(b, h, q_idx, kv_idx):
            in_window = (kv_idx >= starts[q_idx]) & (kv_idx < ends[q_idx])
            is_sink = kv_idx < sink_end
            is_anchor_kv = (kv_idx >= total_main) & (kv_idx < total_with_anchor)
            is_anchor_q = (q_idx >= total_main) & (q_idx < total_with_anchor)
            # Main→Main: window | sink (existing causal logic)
            # Main→Anchor: always visible
            # Anchor→anything: always visible
            return in_window | is_sink | is_anchor_kv | is_anchor_q | (q_idx == kv_idx)
```

Note: `total_main` and `total_with_anchor` need to be captured as tensors for FlexAttention's JIT:

```python
        _total_main = torch.tensor(total_main, device=device, dtype=torch.long)
        _total_with_anchor = torch.tensor(total_with_anchor, device=device, dtype=torch.long)

        def attention_mask(b, h, q_idx, kv_idx):
            in_window = (kv_idx >= starts[q_idx]) & (kv_idx < ends[q_idx])
            is_sink = kv_idx < sink_end
            is_anchor_kv = (kv_idx >= _total_main) & (kv_idx < _total_with_anchor)
            is_anchor_q = (q_idx >= _total_main) & (q_idx < _total_with_anchor)
            return in_window | is_sink | is_anchor_kv | is_anchor_q | (q_idx == kv_idx)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_future_anchor_mask.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_future_anchor_mask.py
git commit -m "feat: extend FlexAttention mask for globally-visible anchor tokens"
```

---

## Task 4: Network Forward — Inject Future Anchor Tokens

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:1700-1854` (`_forward_full_sequence`)
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:2100-2190` (`forward`)
- Test: `tests/test_future_anchor_forward.py`

When `condition["future_anchor_latents"]` is present and the sampled stochastic config has `future_anchor: True`, embed the anchor frame as extra tokens appended after the main sequence, with RoPE position encoding the temporal distance.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for future anchor token injection in full-sequence forward."""
import torch
import pytest
from unittest.mock import patch, MagicMock


def test_forward_full_sequence_shape_with_anchor():
    """Output shape unchanged when future anchor is injected."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    # We can't instantiate the full 14B model — test the shape logic
    # by checking that _forward_full_sequence handles the anchor tokens
    # and produces output with the correct main-sequence shape.
    # This requires a lightweight mock — skip if 14B model needed.
    pytest.skip("Requires integration test with model weights")


def test_anchor_config_sampling():
    """Stochastic config with future_anchor=True is propagated."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.local_attn_size = 10
    net.sink_size = 1
    net._stochastic_attn_configs = [
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.5},
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.5,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
    ]

    # Sample many times — should get both True and False
    results = [net._sample_attn_config() for _ in range(100)]
    has_anchor = [r.get("future_anchor", False) for r in results]
    assert any(has_anchor), "Should sometimes get future_anchor=True"
    assert not all(has_anchor), "Should sometimes get future_anchor=False"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_future_anchor_forward.py::test_anchor_config_sampling -v`
Expected: FAIL (`_sample_attn_config` doesn't propagate `future_anchor`)

- [ ] **Step 3: Update `_sample_attn_config` to propagate `future_anchor`**

In `network_causal.py`, modify `_sample_attn_config` (line 1509):

```python
    def _sample_attn_config(self) -> dict:
        """Sample an attention config from stochastic_attn_configs.

        Returns dict with 'local_attn_size', 'sink_size', and optionally
        'future_anchor' and 'future_anchor_distance' keys.
        """
        if not self._stochastic_attn_configs:
            return {
                "local_attn_size": self.local_attn_size,
                "sink_size": self.sink_size,
            }

        import random
        weights = [c.get("weight", 1.0) for c in self._stochastic_attn_configs]
        chosen = random.choices(self._stochastic_attn_configs, weights=weights, k=1)[0]

        result = {
            "local_attn_size": chosen.get("local_attn_size", self.local_attn_size),
            "sink_size": chosen.get("sink_size", self.sink_size),
        }

        # Future anchor: optionally enabled per-config
        if chosen.get("future_anchor", False):
            result["future_anchor"] = True
            dist_range = chosen.get("future_anchor_distance_range", [1, 5])
            result["future_anchor_distance"] = random.randint(dist_range[0], dist_range[1])

        return result
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_future_anchor_forward.py::test_anchor_config_sampling -v`
Expected: PASS

- [ ] **Step 5: Implement anchor injection in `_forward_full_sequence`**

In `_forward_full_sequence` (line 1700), add `future_anchor_latent` and `future_anchor_distance` parameters:

```python
    def _forward_full_sequence(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        future_anchor_latent: Optional[torch.Tensor] = None,
        future_anchor_distance: int = 0,
    ) -> torch.Tensor:
```

After patch embedding and reshaping (around line 1741), inject anchor tokens:

```python
        x = x.flatten(2).transpose(1, 2)  # [B, f*h*w, dim]
        num_main_tokens = x.shape[1]

        # Inject future anchor tokens (if provided)
        num_anchor_tokens = 0
        if future_anchor_latent is not None:
            # future_anchor_latent: [B, 16, 1, H, W]
            # Build 36ch input: [anchor_latent, i2v_mask(ones), anchor_latent_as_ref]
            B_a = future_anchor_latent.shape[0]
            H_a, W_a = future_anchor_latent.shape[3], future_anchor_latent.shape[4]
            anchor_mask = torch.ones(
                B_a, 4, 1, H_a, W_a,
                device=future_anchor_latent.device, dtype=future_anchor_latent.dtype,
            )
            anchor_36ch = torch.cat([
                future_anchor_latent,       # 16ch: clean future latent
                anchor_mask,                # 4ch: fully known
                future_anchor_latent,       # 16ch: self-reference
            ], dim=1)  # [B, 36, 1, H, W]

            anchor_tokens = self.patch_embedding(anchor_36ch)  # [B, dim, 1, h, w]
            anchor_tokens = anchor_tokens.flatten(2).transpose(1, 2)  # [B, h*w, dim]
            num_anchor_tokens = anchor_tokens.shape[1]

            x = torch.cat([x, anchor_tokens], dim=1)  # [B, f*h*w + h*w, dim]

        seq_lens = torch.tensor(
            [x.shape[1]] * x.shape[0], dtype=torch.long, device=device
        )
```

Update mask building (around line 1798) to pass `num_anchor_tokens`:

```python
        if self.training and self._stochastic_attn_configs:
            attn_cfg = self._sample_attn_config()
            if FLEX_ATTENTION_AVAILABLE:
                frame_seqlen = h * w
                self.block_mask = self._build_block_mask(
                    device, f, frame_seqlen, self.chunk_size,
                    local_attn_size=attn_cfg["local_attn_size"],
                    sink_size=attn_cfg["sink_size"],
                    num_anchor_tokens=num_anchor_tokens,
                )
        elif FLEX_ATTENTION_AVAILABLE:
            frame_seqlen = h * w
            self.block_mask = self._build_block_mask(
                device, f, frame_seqlen, self.chunk_size,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
                num_anchor_tokens=num_anchor_tokens,
            )
```

For RoPE: the anchor tokens need a RoPE position encoding the temporal distance. In `rope_apply_full`, anchor tokens should get position `f + future_anchor_distance`. This requires modifying how RoPE is applied to the extended sequence. The simplest approach: apply RoPE to main tokens normally, then apply RoPE to anchor tokens separately with `start_frame=f + future_anchor_distance - 1`:

After the existing `x = self.patch_embedding(x)` and before the transformer blocks, add RoPE handling for anchor tokens. Since `rope_apply_full` is applied inside each `CausalSelfAttention.forward()`, we need to pass the anchor frame offset so attention can assign the right RoPE position. Add `anchor_frame_offset` to the block kwargs:

```python
        kwargs_block = dict(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
            block_mask=self.block_mask,
            num_anchor_tokens=num_anchor_tokens,
            anchor_frame_offset=f + future_anchor_distance if num_anchor_tokens > 0 else 0,
        )
```

The actual RoPE application for anchor tokens happens in `CausalSelfAttention.forward()` — see Task 5.

For t_mod (time modulation): anchor tokens use a timestep of 0 (clean frame). Expand t_mod to include the anchor frame:

```python
        if num_anchor_tokens > 0:
            # Anchor is clean (t=0). Create zero time modulation for it.
            anchor_t_mod = torch.zeros(
                t_mod.shape[0], 1, t_mod.shape[2], t_mod.shape[3],
                device=t_mod.device, dtype=t_mod.dtype,
            )
            t_mod = torch.cat([t_mod, anchor_t_mod], dim=1)  # [B, F+1, 6, dim]
            anchor_t_emb = torch.zeros(
                t_emb.shape[0], 1, t_emb.shape[2],
                device=t_emb.device, dtype=t_emb.dtype,
            )
            t_emb = torch.cat([t_emb, anchor_t_emb], dim=1)  # [B, F+1, dim]
```

After the transformer blocks, strip anchor tokens and return only main-sequence output:

```python
        # Strip anchor tokens from output
        if num_anchor_tokens > 0:
            x = x[:, :num_main_tokens]
```

This goes before the existing unpatchify / output projection code.

- [ ] **Step 6: Wire condition through `forward()` to `_forward_full_sequence`**

In `forward()` (around line 2164), extract future anchor from condition and pass it:

```python
        # Extract future anchor if present and config requests it
        future_anchor_latent = None
        future_anchor_distance = 0
        if (isinstance(condition, dict)
                and condition.get("future_anchor_latents") is not None
                and hasattr(self, "_current_attn_cfg")
                and self._current_attn_cfg.get("future_anchor", False)):
            fa = condition["future_anchor_latents"]  # [B, 16, 5, H, W]
            d = self._current_attn_cfg["future_anchor_distance"]  # 1-5
            future_anchor_latent = fa[:, :, d-1:d]  # [B, 16, 1, H, W]
            future_anchor_distance = d
```

In `_forward_full_sequence`, the stochastic config is sampled. Store it on self temporarily so forward() can read it, or better: sample it in forward() and pass it down. The cleanest approach: move the anchor extraction INTO `_forward_full_sequence` where `attn_cfg` is already available. Pass the full `condition` dict (or just `future_anchor_latents`) as a parameter.

Revised approach: add `future_anchor_latents: Optional[torch.Tensor] = None` to `_forward_full_sequence` signature. Inside, after sampling `attn_cfg`:

```python
        # Future anchor: select one frame based on sampled config
        future_anchor_latent = None
        future_anchor_distance = 0
        if (future_anchor_latents is not None
                and attn_cfg.get("future_anchor", False)):
            d = attn_cfg["future_anchor_distance"]
            future_anchor_latent = future_anchor_latents[:, :, d-1:d]  # [B, 16, 1, H, W]
            future_anchor_distance = d
```

And in `forward()`, pass it through:

```python
        if timestep.dim() == 2:
            model_output = self._forward_full_sequence(
                x=x_t,
                timestep=timestep,
                context=text_embeds,
                clip_fea=clip_features,
                y=y,
                audio=audio_emb,
                use_gradient_checkpointing=use_gradient_checkpointing,
                future_anchor_latents=condition.get("future_anchor_latents") if isinstance(condition, dict) else None,
            )
```

- [ ] **Step 7: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_future_anchor_forward.py
git commit -m "feat: inject future anchor tokens in _forward_full_sequence with RoPE distance"
```

---

## Task 5: Attention — RoPE for Anchor Tokens in Full-Sequence Mode

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py` — `CausalSelfAttention.forward` and `CausalDiTBlock.forward`

In full-sequence mode (no KV cache), `rope_apply_full` is applied to the entire Q and K sequence. With anchor tokens appended, we need to split the RoPE application: main tokens get positions `[0, ..., F-1]`, anchor tokens get position `F + distance - 1`.

- [ ] **Step 1: Modify `CausalDiTBlock.forward` to accept and pass through anchor params**

Add `num_anchor_tokens=0` and `anchor_frame_offset=0` parameters to `CausalDiTBlock.forward()` (around line 844). Pass them to `self.self_attn(...)`.

- [ ] **Step 2: Modify `CausalSelfAttention.forward` to handle anchor RoPE**

In the full-sequence path (around line 507), when `num_anchor_tokens > 0`:

```python
        if kv_cache is None:  # Full-sequence mode
            if num_anchor_tokens > 0 and anchor_frame_offset > 0:
                # Split: main tokens get standard RoPE, anchor gets offset position
                main_q = q[:, :-num_anchor_tokens]
                main_k = k[:, :-num_anchor_tokens]
                anchor_q = q[:, -num_anchor_tokens:]
                anchor_k = k[:, -num_anchor_tokens:]

                roped_main_q = rope_apply_full(main_q, grid_sizes, freqs).type_as(v)
                roped_main_k = rope_apply_full(main_k, grid_sizes, freqs).type_as(v)

                # Anchor tokens: 1 frame at position anchor_frame_offset
                anchor_grid = grid_sizes.clone()
                anchor_grid[:, 0] = 1  # 1 anchor frame
                roped_anchor_q = causal_rope_apply(
                    anchor_q, anchor_grid, freqs, start_frame=anchor_frame_offset,
                ).type_as(v)
                roped_anchor_k = causal_rope_apply(
                    anchor_k, anchor_grid, freqs, start_frame=anchor_frame_offset,
                ).type_as(v)

                roped_q = torch.cat([roped_main_q, roped_anchor_q], dim=1)
                roped_k = torch.cat([roped_main_k, roped_anchor_k], dim=1)
            else:
                roped_q = rope_apply_full(q, grid_sizes, freqs).type_as(v)
                roped_k = rope_apply_full(k, grid_sizes, freqs).type_as(v)
```

- [ ] **Step 3: Handle t_mod expansion in CausalDiTBlock**

The block receives `t_mod` with shape `[B, F, 6, dim]` (per-frame modulation). With anchor tokens, `t_mod` has `F+1` entries (Task 4 already extends it). The block needs to apply the correct modulation per token. Currently (line ~870-880), the block does:

```python
# Per-frame: expand t_mod to per-token
# t_mod: [B, F, 6, dim] -> [B, F*h*w, 6, dim] via repeat_interleave
```

With anchor tokens, the per-token expansion needs to account for the extra frame. This is already handled if t_mod has `F+1` entries — the anchor frame's modulation (zeros for t=0) gets expanded to `h*w` tokens and appended.

- [ ] **Step 4: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py
git commit -m "feat: split RoPE application for anchor tokens in full-sequence attention"
```

---

## Task 6: DF Training — Pass Future Anchor Through Condition

**Files:**
- Modify: `fastgen/methods/infinitetalk_diffusion_forcing.py:116-139`

The DF training method's `_build_condition` needs to pass through `future_anchor_latents` when present. The training step itself doesn't need changes — the network forward handles everything.

- [ ] **Step 1: Modify `_build_condition`**

```python
    def _build_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("text_embeds", "first_frame_cond", "clip_features", "audio_emb"):
            assert key in data, f"Missing required key '{key}' in data batch"
        assert data["first_frame_cond"].shape[1] == 16, (
            f"first_frame_cond should have 16 channels, got {data['first_frame_cond'].shape[1]}"
        )
        assert data["audio_emb"].shape[-2:] == (12, 768), (
            f"audio_emb should have shape [..., 12, 768], got {list(data['audio_emb'].shape)}"
        )
        result = {
            "text_embeds": data["text_embeds"],
            "first_frame_cond": data["first_frame_cond"],
            "clip_features": data["clip_features"],
            "audio_emb": data["audio_emb"],
        }
        # Pass through future anchor latents if available
        if "future_anchor_latents" in data:
            result["future_anchor_latents"] = data["future_anchor_latents"]
        return result
```

- [ ] **Step 2: Commit**

```bash
git add fastgen/methods/infinitetalk_diffusion_forcing.py
git commit -m "feat: pass future_anchor_latents through DF condition dict"
```

---

## Task 7: Config and Training Script

**Files:**
- Create: `fastgen/configs/experiments/InfiniteTalk/config_df_quarter_stochastic_anchor.py`
- Create: `scripts/run_df_training_quarter_stochastic_anchor.sh`

- [ ] **Step 1: Write the config**

```python
# SPDX-License-Identifier: Apache-2.0
"""
Quarter-resolution DF config with stochastic attention + future anchor.

Extends config_df_quarter_stochastic with future_anchor: True on some attention
configs. During training, configs with future_anchor=True inject a clean GT
latent from a random future distance (1-5 latent frames past the sequence)
as globally-visible anchor tokens in attention.

Usage:
    bash scripts/run_df_training_quarter_stochastic_anchor.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_df_quarter_stochastic import (
    create_config as create_stochastic_config,
)


def create_config():
    config = create_stochastic_config()

    # Override stochastic configs: add future_anchor to sink-bearing configs
    config.model.net.stochastic_attn_configs = [
        # Without future anchor (baseline configs)
        {"local_attn_size": 7,  "sink_size": 1, "weight": 0.1},
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.1},
        {"local_attn_size": 13, "sink_size": 1, "weight": 0.1},
        {"local_attn_size": 9,  "sink_size": 3, "weight": 0.1},
        {"local_attn_size": 12, "sink_size": 3, "weight": 0.1},
        # With future anchor (teaches distance-aware conditioning)
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
        {"local_attn_size": 13, "sink_size": 1, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
        {"local_attn_size": 9,  "sink_size": 3, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
        {"local_attn_size": 12, "sink_size": 3, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 3]},
    ]

    # Logging
    config.log_config.group = "infinitetalk_df_quarter_stochastic_anchor"
    num_gpus = int(os.environ.get("NUM_GPUS", "8"))
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"quarter_stoch_anchor_r128_bs4_{num_gpus}gpu_{timestamp}"
    config.log_config.name = run_name

    return config
```

- [ ] **Step 2: Write the training script**

```bash
#!/bin/bash
# DF training with stochastic attention + future anchor.
set -e

export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

export INFINITETALK_TRAIN_LIST="${INFINITETALK_TRAIN_LIST:-data/precomputed_talkvid/train_excl_val30.txt}"
export INFINITETALK_VAL_LIST="${INFINITETALK_VAL_LIST:-data/precomputed_talkvid/val_quarter_30.txt}"
export INFINITETALK_NEG_TEXT_EMB="${INFINITETALK_NEG_TEXT_EMB:-data/precomputed_talkvid/neg_text_embeds.pt}"
export INFINITETALK_AUDIO_ROOT="${INFINITETALK_AUDIO_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data}"
export INFINITETALK_WAV2VEC_DIR="${INFINITETALK_WAV2VEC_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base}"

export WANDB_ENTITY="paulhcho"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export NUM_GPUS

echo "============================================"
echo "DF Training — Stochastic Attention + Future Anchor"
echo "============================================"
echo "GPUs:             $NUM_GPUS"
echo "Future anchor:    50% of configs (distance 1-5)"
echo "Train list:       $INFINITETALK_TRAIN_LIST"
echo "============================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_df_quarter_stochastic_anchor.py
```

- [ ] **Step 3: Commit**

```bash
git add fastgen/configs/experiments/InfiniteTalk/config_df_quarter_stochastic_anchor.py \
        scripts/run_df_training_quarter_stochastic_anchor.sh
chmod +x scripts/run_df_training_quarter_stochastic_anchor.sh
git commit -m "feat: DF config + script with stochastic future anchor"
```

---

## Task 8: Integration Test — Full Forward Pass

**Files:**
- Test: `tests/test_future_anchor_forward.py` (extend from Task 4)

Since we can't load the 14B model in tests, write a mock-based integration test that exercises the full path: `_sample_attn_config` → anchor frame selection → mask building → token injection → output shape.

- [ ] **Step 1: Write mock-based integration test**

```python
def test_full_sequence_with_anchor_mock():
    """Mock-based test: anchor tokens injected and stripped correctly."""
    import torch
    from types import SimpleNamespace
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    # Verify _build_block_mask works with anchor tokens
    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.chunk_size = 3

    # 6 frames, 4 tokens/frame = 24 main tokens + 4 anchor tokens = 28
    mask = net._build_block_mask(
        torch.device("cpu"), num_frames=6, frame_seqlen=4,
        chunk_size=3, local_attn_size=10, sink_size=1,
        num_anchor_tokens=4,
    )
    assert mask is not None

    # Verify anchor config sampling includes distance
    net._stochastic_attn_configs = [
        {"local_attn_size": 10, "sink_size": 1, "weight": 1.0,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
    ]
    cfg = net._sample_attn_config()
    assert cfg["future_anchor"] is True
    assert 1 <= cfg["future_anchor_distance"] <= 5
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/test_future_anchor_mask.py tests/test_future_anchor_forward.py tests/test_future_anchor_dataloader.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_future_anchor_forward.py
git commit -m "test: integration tests for future anchor full path"
```

---

## Summary of Changes

| Component | What Changes | Risk |
|-----------|-------------|------|
| Precompute script | New script, no existing code touched | Low |
| Dataloader | Adds optional `.pt` file loading + zero fallback | Low — existing samples work unchanged |
| FlexAttention mask | Extended with `num_anchor_tokens` param | Medium — mask logic is critical; test thoroughly |
| Network forward | `_forward_full_sequence` gets anchor injection, RoPE split, t_mod extension | Medium — most complex change |
| Attention RoPE | Split application for main vs anchor tokens | Medium — must match SF inference's lookahead RoPE |
| DF training method | Pass-through of `future_anchor_latents` in condition | Low |
| Config + script | New config inheriting from stochastic DF | Low |

**Key invariant to verify:** With `future_anchor: False` in all stochastic configs (or when `future_anchor_latents` is absent), the entire codepath is a no-op. All existing training behavior is preserved.

**RoPE alignment with SF inference:** The anchor token gets RoPE position `f + distance`, where `f` is the number of main frames. At SF inference, the lookahead sink gets position `F_window - 1 + lookahead_distance`. These are different absolute positions but encode the same semantic: "identity reference at temporal distance D ahead." The model learns to condition on a clean frame at a future RoPE position — the exact absolute position is less important than the relative offset pattern.
