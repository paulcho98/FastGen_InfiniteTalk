# InfiniteTalk FastGen Adaptation — Bugs Found and Fixed

This document records all bugs discovered during implementation and verification,
their root causes, and fixes. **Essential reading before adapting FastGen for other
baselines (SoulX-LiveAct, LiveAvatar, etc.).**

---

## Critical Bugs (Affected Output Quality)

### 1. Wrong Resolution — Hardcoded 640x640 Instead of Aspect-Ratio Buckets

**Symptom:** ODE denoising produced blurry later frames despite clear frame 0.

**Root cause:** Precompute script used `--resolution 640` which forces 640x640 (square)
on all videos. The original InfiniteTalk pipeline selects from `ASPECT_RATIO_627` bucket
config based on source video aspect ratio. For 16:9 videos (99.3% of TalkVid), the
correct bucket is [448, 896].

**Impact:** The model never saw 640x640 during training — spatial attention patterns
and audio conditioning were all wrong for this resolution. Frame 0 appeared clear only
because it was anchored to the clean reference latent.

**Fix:** `precompute_infinitetalk_data.py` now reads `height`/`width` from CSV and selects
the closest bucket from `ASPECT_RATIO_627`. Training configs updated to `input_shape=[16, 21, 56, 112]`.

**Lesson for other baselines:** Always match the original pipeline's resolution selection
logic. Never hardcode a square resolution unless the model was trained on squares.

---

### 2. T5 Text Embeddings — Non-Zero Padding Instead of Zeros

**Symptom:** ODE output blurry on later frames (compounded with resolution bug).

**Root cause:** We saved the raw T5 transformer output for all 512 positions.
T5 produces non-zero values (up to 0.84 magnitude) at padding positions beyond
the actual token length. The original pipeline uses `T5EncoderModel.__call__()` which
returns **trimmed** embeddings, then `WanModel.forward()` zero-pads internally.

**Impact:** Every forward pass, text cross-attention in all 40 transformer blocks
saw different conditioning. Over 40 ODE steps, this accumulated into blurriness.

**Fix:** `encode_t5()` now calls `T5EncoderModel.__call__()` for trimmed output,
then explicitly zero-pads to 512 tokens.

**Lesson for other baselines:** Always check how the original pipeline handles text
padding. Don't assume raw encoder output == what the model expects. The model's
internal padding convention matters.

---

### 3. SDPA Instead of flash_attn for Audio Cross-Attention

**Symptom:** Numerical divergence in audio-conditioned outputs.

**Root cause:** Audio cross-attention (`SingleStreamAttention`) used
`torch.nn.functional.scaled_dot_product_attention` instead of `flash_attn_func`.
The original uses `xformers.ops.memory_efficient_attention` which is numerically
closer to `flash_attn` than SDPA.

**Impact:** Different attention kernels produce slightly different outputs. Over 40
transformer blocks, this accumulates.

**Fix:** Replaced SDPA with `flash_attn.flash_attn_func` (same [B,M,H,K] layout
as xformers). Also removed spurious dropout_p parameter not in the original.

**Lesson:** Use the same attention kernel as the original. If xformers is unavailable,
flash_attn is the closest substitute. SDPA should be fallback-only.

---

### 4. SDPA Instead of flash_attn for CLIP Encoding

**Symptom:** CLIP features had 4.29 max diff vs original (fp16 amplification).

**Root cause:** Precompute script monkey-patched `flash_attention` with an SDPA
fallback for CLIP encoding. But flash_attn 2.7.3 was available all along.

**Fix:** Removed the monkey-patch. CLIP now uses the original `flash_attention`
function which calls `flash_attn_varlen_func` directly. CLIP features now 0.0 diff.

---

### 5. Image Preprocessing — torch bilinear Instead of PIL BILINEAR

**Symptom:** first_frame_cond and CLIP features had small but nonzero diff vs original.

**Root cause:** Precompute used `F.interpolate(mode='bilinear', align_corners=False)`
for resize. Original uses `PIL.Image.resize(BILINEAR)` → numpy → torch. These produce
different pixel values (max diff 0.008 = 1 uint8 unit).

**Fix:** Added `resize_and_centercrop_pil()` matching the original's PIL path for the
reference frame. Video frames still use torch bilinear (only affects VAE training
target, not conditioning).

---

### 6. ODE Subsample Indexing — `trajectory[:-1]` Bug

**Symptom:** ODE extraction saved 3 states instead of 4 (default t_list).

**Root cause:** `extract_ode_trajectory()` already filters out t=0.0 (returns only
non-negative indices). But the save logic did `trajectory[:-1]` assuming the clean
state was the last element — which it wasn't (already filtered). This dropped the
last valid noisy state.

**Fix:** Removed `[:-1]` slice.

---

### 7. Missing Motion Frame Anchoring in ODE Extraction

**Symptom:** ODE output drifted from reference frame.

**Root cause:** Original pipeline forces `latent[:, :1] = latent_motion_frames` at
every ODE step (before and after model forward). ODE extraction script didn't do this.

**Fix:** Added anchoring before and after each Euler step. Also added `motion_frame.pt`
precomputation (single-image VAE encode without temporal context).

---

### 8. FastGenModel.build_model() Unfreezes All Parameters

**Symptom:** OOM during backward pass (78.4GB).

**Root cause:** `FastGenModel.build_model()` line 260 calls
`self.net.train().requires_grad_(True)` on ALL 19.1B params, overriding our
`freeze_base()` call in the constructor.

**Fix:** Override `build_model()` in all three method classes (DF, KD, SF) to
re-apply `freeze_base()` after the base class call.

**Lesson:** Always check if the training framework's base class modifies
requires_grad after your initialization.

---

### 9. in_dim=16 Instead of 36 in Wrapper Config

**Symptom:** DF training ran with randomly initialized patch_embedding for 20 extra
channels (mask + first_frame_cond).

**Root cause:** `_COMMON_CFG` in both `network.py` and `network_causal.py` had
`in_dim=16` (noise channels only). Should be 36 (16 noise + 4 mask + 16 VAE ref).

**Fix:** Changed to `in_dim=36`.

---

### 10. Python 3.12 ArgSpec in Multiprocessing Workers

**Symptom:** T5 precompute workers crashed with `ImportError: ArgSpec`.

**Root cause:** `mp.spawn` creates fresh processes that don't inherit the parent's
`inspect.ArgSpec = inspect.FullArgSpec` patch.

**Fix:** Added the patch inside the worker function.

---

## Non-Bugs (Verified as Correct)

- `torch.amp.autocast('cuda', dtype=torch.float32)` — verified identical to
  `torch.cuda.amp.autocast(dtype=torch.float32)` in PyTorch 2.8
- WanSelfAttention returning 1 value (not 2) — correct for single-speaker,
  `x_ref_attn_map` only used by `SingleStreamMutiAttention`
- Motion frame diff (0.008) between `vae.encode(single_image)` and
  `first_frame_cond[:, :1]` — negligible, VAE temporal convs barely affect first frame

---

## Verification Results

| Test | Result |
|------|--------|
| Stage 0: Weight loading (1633 params) | 0.0 diff |
| Stage 0: Forward pass (3 samples) | 0.0 diff |
| ODE: Our wrapper vs original model (40 steps) | corr=0.999932 |
| Preprocessing: CLIP features | 0.0 diff (after flash_attn fix) |
| Preprocessing: first_frame_cond | 0.0 diff (after PIL resize fix) |
| Preprocessing: T5 text_embeds | 0.0 diff (after zero-padding fix) |
| Preprocessing: audio_emb | 0.01 max diff (float precision, corr=1.0) |
| DF training: 20 iterations | Loss 0.017-0.163, no OOM, 45s/iter |
| Original pipeline: End-to-end | Crystal clear output at 448x896 |

---

## Key Architectural Decisions

- **flash_attn for audio cross-attention** (replacing xformers which can't be installed
  without breaking torch 2.8). Same [B,M,H,K] layout, closest available kernel.
- **Single-speaker only** — stripped multi-speaker (`SingleStreamMutiAttention`,
  `ref_target_masks`, `human_num`). TalkVid is single-speaker.
- **No TeaCache, sageattn, xfuser, FA3** — inference optimizations not needed for training.
- **LoRA on all attention** — base DiT Q/K/V/O/FFN + audio_cross_attn q/kv/proj.
  294M trainable out of 19.1B (1.54%).
