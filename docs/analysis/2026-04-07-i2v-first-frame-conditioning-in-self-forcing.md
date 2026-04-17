# First-Frame I2V Conditioning in Self-Forcing Distillation

**Date:** 2026-04-07
**Scope:** How Wan 2.1 I2V and Wan 2.2 TI2V handle first-frame conditioning during Self-Forcing training, and how this informs our InfiniteTalk implementation.

---

## 1. Background: Why First-Frame Conditioning Matters in Self-Forcing

Standard diffusion training (denoising score matching) operates on a **single forward pass**: the model sees a noisy version of the entire video and predicts the clean target. Each training step is independent. The first frame is trivially recoverable from the I2V conditioning signal, and errors do not compound.

Self-Forcing distillation is fundamentally different. The student performs a **multi-step autoregressive rollout**:

```
Block 0 (frames 0-2):
  step 1: x0_pred = student(noise, t=0.999, condition)
  step 2: noisy_input = add_noise(x0_pred, eps, t=0.875)
           x0_pred = student(noisy_input, t=0.875, condition)
  ...
  step N: x0_pred = student(noisy_input, t_exit, condition)  -- gradient flows here
  cache: store x0_pred in KV cache

Block 1 (frames 3-5):
  ...attends to block 0 via KV cache...
```

If the x0 prediction for frame 0 is imperfect at any step:
- The imperfect frame 0 gets re-noised and fed back as input
- The KV cache stores an imperfect representation of block 0
- All downstream blocks attend to corrupted context
- The VSD loss receives biased teacher/fake_score signals

This compounding error is why first-frame handling requires careful design in Self-Forcing.

---

## 2. Wan 2.1 I2V: Channel Concatenation Approach

### Architecture

Wan 2.1 I2V uses `concat_mask=True`. The model input is formed by concatenating three tensors along the channel dimension:

| Component | Shape | Description |
|-----------|-------|-------------|
| `x_t` (noisy latent) | `[B, 16, T, H, W]` | Standard noisy input |
| `mask` | `[B, 4, T, H, W]` | Binary mask: 1 for frame 0 (reference), 0 for others |
| `first_frame_cond` | `[B, 16, T, H, W]` | VAE-encoded reference frame at position 0, zeros elsewhere |

Total input: **36 channels** = 16 + 4 + 16, fed to `patch_embedding`.

The mask handles the VAE's temporal stride of 4: the first pixel frame is replicated 4x before reshaping into latent temporal dimensions.

### How It Works in Each Role

**Teacher (bidirectional, `WanI2V.forward()`):**
- `_compute_i2v_inputs()` builds the 36-channel concatenated input
- Timestep is **NOT masked** for frame 0 (`mask=None` passed to `_compute_timestep_inputs`)
- Output: `_replace_first_frame` is **NOT called** (only applies when `not self.concat_mask`)
- The model learns to predict clean frame 0 from the soft concatenated signal alone

**Student (causal, `CausalWanI2V.forward()`):**
- Same 36-channel concatenation, but `first_frame_cond` is **sliced** to the current chunk: `first_frame_cond[:, :, cur_start_frame:cur_start_frame + x_t.shape[2]]`
- For block 0: the chunk contains the reference frame + zeros
- For blocks 1+: the chunk is all zeros (no reference signal in the concatenated channels)
- Timestep: **NOT masked** for frame 0
- Output: `_replace_first_frame` is **NOT called**

**Fake Score (same architecture as teacher):**
- Identical behavior to teacher

**Key insight for Wan 2.1:** The conditioning is entirely via channel concatenation. There is no hard anchoring of frame 0 in the output. The model must learn to reconstruct it from the soft signal. The `preserve_conditioning` method is a **no-op** for `concat_mask=True`.

### Self-Forcing Rollout Behavior

During `rollout_with_gradient()`:
1. Each `self.net()` call passes the concatenated I2V signal
2. The x0 prediction for frame 0 depends entirely on the model's learned behavior
3. Between steps, x0 is re-noised — frame 0 becomes noisy, but the next call's concatenated conditioning still provides the clean reference
4. No `preserve_conditioning` is called (the method is a no-op anyway)
5. KV cache stores the model's prediction (which may be slightly imperfect for frame 0)

### Summary for Wan 2.1

The approach relies fully on the model's capacity to learn "when the mask says reference, copy from the conditioning channel." No external enforcement. This works because Wan 2.1 was trained this way from scratch — the model has strong learned behavior for respecting the mask.

---

## 3. Wan 2.2 TI2V: First-Frame Replacement Approach

### Architecture

Wan 2.2 TI2V uses `concat_mask=False`. Instead of channel concatenation, it uses **direct frame replacement**:

| Mechanism | Location | Description |
|-----------|----------|-------------|
| Input replacement | `_compute_i2v_inputs()` | `x_t[:, :, 0]` replaced with clean `first_frame_cond[:, :, 0]` |
| Timestep masking | `_compute_timestep_inputs()` | Timestep for frame 0 set to 0 (tells model "this frame is clean") |
| Output replacement | `forward()` after `convert_model_output` | `out[:, :, 0]` replaced with clean `first_frame_cond[:, :, 0]` |

The model input is the standard 16 channels — no extra mask or conditioning channels.

### How It Works in Each Role

**Teacher (bidirectional, `WanI2V.forward()`):**
```python
# Input: replace frame 0 with clean latent
latent_model_input, first_frame_mask = self._replace_first_frame(first_frame_cond, x_t, return_mask=True)

# Timestep: zero out for frame 0
timestep_mask = first_frame_mask[:, 0]  # mask with 0 at frame 0, 1 elsewhere
timestep = self._compute_timestep_inputs(timestep, timestep_mask)
# Result: timestep = mask * timestep → frame 0 gets t=0

# Output: replace frame 0 again
out = self.noise_scheduler.convert_model_output(...)
if not self.concat_mask:
    out = self._replace_first_frame(first_frame_cond, out)
```

The teacher **always** returns perfect frame 0, regardless of what the model predicts.

**Student (causal, `CausalWanI2V.forward()`):**
```python
# Input: replace frame 0 ONLY when processing the first chunk
if cur_start_frame == 0:
    latent_model_input, first_frame_mask = self._replace_first_frame(first_frame_cond, x_t, return_mask=True)
    if not mask_all_frames:
        timestep_mask = first_frame_mask[:, 0]
timestep = self._compute_timestep_inputs(timestep, timestep_mask)

# Output: replace frame 0 ONLY when processing the first chunk
if cur_start_frame == 0 and not self.concat_mask:
    out = self._replace_first_frame(first_frame_cond, out)
```

For blocks 1+ (`cur_start_frame > 0`): no replacement, no timestep masking. These blocks rely on KV cache from block 0.

**Fake Score (same architecture as teacher):**
- Identical behavior to teacher — always replaces frame 0 in input and output.

### Self-Forcing Rollout Behavior

During `rollout_with_gradient()`:
1. Block 0, each denoising step:
   - `self.net(noisy_input, t, ..., cur_start_frame=0)` is called
   - Inside `forward()`: input frame 0 is replaced with clean reference
   - Inside `forward()`: output frame 0 is replaced with clean reference
   - x0_pred has **perfect** frame 0
   - Re-noising: `forward_process(x0_pred, eps, t_next)` makes frame 0 noisy
   - Next step: the noisy frame 0 goes in, but `forward()` replaces it again before the transformer sees it
2. KV cache store: `self.net(x0_pred, t=0, ..., store_kv=True, cur_start_frame=0)`
   - `forward()` still replaces frame 0 (clean reference stored in cache)
3. Blocks 1+: attend to perfect frame 0 in KV cache

**No `preserve_conditioning` is called during rollout** — but it's not needed because the replacement happens inside `forward()`.

### Inference Behavior (`_student_sample_loop`)

The generic `_student_sample_loop` in `FastGenModel` adds an extra safety layer:
```python
has_preserve_hook = hasattr(net, "preserve_conditioning")
# After x0 prediction:
if has_preserve_hook:
    x_pred = net.preserve_conditioning(x_pred, condition)
# After re-noising:
if has_preserve_hook:
    x = net.preserve_conditioning(x, condition)
```

For Wan 2.2: `preserve_conditioning` overwrites `x[:, :, 0] = first_frame_cond[:, :, 0]`.
For Wan 2.1: `preserve_conditioning` is a no-op (returns x unchanged).

This provides **belt-and-suspenders** protection: even if some code path bypasses the network-level replacement, the sample loop enforces it.

### Summary for Wan 2.2

Three-layer defense for frame 0:
1. **Input replacement** in `_compute_i2v_inputs()` — model never sees noisy frame 0
2. **Output replacement** in `forward()` — model output always has clean frame 0
3. **Sample loop replacement** via `preserve_conditioning` — inference loop enforces it

Plus **timestep masking** (t=0 for frame 0) gives the model an explicit "this frame is clean" signal.

---

## 4. Side-by-Side Comparison

| Aspect | Wan 2.1 I2V (`concat_mask=True`) | Wan 2.2 TI2V (`concat_mask=False`) |
|--------|----------------------------------|-------------------------------------|
| **Input channels** | 36 (16 noise + 4 mask + 16 ref) | 16 (standard, frame 0 replaced) |
| **Conditioning signal** | Soft (mask + channel concat) | Hard (frame replacement) |
| **Timestep masking** | No (all frames same timestep) | Yes (frame 0 gets t=0) |
| **Output replacement** | No | Yes (`_replace_first_frame` in `forward()`) |
| **`preserve_conditioning`** | No-op | Overwrites frame 0 |
| **Student rollout frame 0** | Model-predicted (may be imperfect) | Always clean (replaced in `forward()`) |
| **Teacher frame 0** | Model-predicted | Always clean (replaced in `forward()`) |
| **KV cache frame 0** | Model-predicted | Always clean |
| **Error compounding risk** | Higher (soft signal only) | None (hard anchor at every step) |

---

## 5. Our InfiniteTalk Implementation

### Model Architecture

InfiniteTalk uses **Wan 2.1-style channel concatenation** (`in_dim=36`):
- 16 channels: noisy latent
- 4 channels: temporal mask (frame 0 = 1, rest = 0)
- 16 channels: VAE-encoded reference frame (zero-padded after frame 0)

This is architecturally identical to Wan 2.1 I2V in terms of how the first frame enters the model.

### What the Original InfiniteTalk Inference Does

The original InfiniteTalk inference (`wan/multitalk.py`) applies **hard first-frame overwrite** at every denoising step:

```python
# Before model forward (line 711):
latent[:, :cur_motion_frames_latent_num] = latent_motion_frames

# After Euler step + re-noising (lines 766-773):
latent[:, :T_m] = add_latent  # re-noised motion frames
latent[:, :cur_motion_frames_latent_num] = latent_motion_frames  # clean overwrite
```

This is an **external** enforcement — the model itself does not have internal replacement logic. It is an inference-time technique applied outside the model.

### The Problem: Our Initial SF Implementation Had No Overwrite

Our initial Self-Forcing implementation had:
- No `_replace_first_frame` in `forward()` (neither student nor teacher)
- No `preserve_conditioning` hook
- No timestep masking for frame 0
- No external overwrite in the rollout

This meant the student, teacher, and fake score all relied purely on the soft I2V conditioning (mask + channel concat) to handle frame 0. During the multi-step AR rollout, imperfect frame 0 predictions compounded through re-noising, corrupted the KV cache, and degraded all downstream blocks.

### Our Fix: Network-Level Output Replacement

We added hard first-frame replacement in the network `forward()` methods:

**Causal student (`CausalInfiniteTalkWan.forward()`, network_causal.py):**
```python
out = self.noise_scheduler.convert_model_output(...)

# Hard-anchor frame 0 to clean reference when processing the first chunk.
if cur_start_frame == 0 and "first_frame_cond" in condition:
    first_frame_cond = condition["first_frame_cond"]
    out = out.clone()
    out[:, :, 0:1] = first_frame_cond[:, :, 0:1]
```

**Bidirectional teacher/fake_score (`InfiniteTalkWan.forward()`, network.py):**
```python
out = self.noise_scheduler.convert_model_output(...)

# Hard-anchor frame 0 to clean reference.
if isinstance(condition, dict) and "first_frame_cond" in condition:
    first_frame_cond = condition["first_frame_cond"]
    out = out.clone()
    out[:, :, 0:1] = first_frame_cond[:, :, 0:1]
```

**Inference script (`inference_causal.py`):**
Additional input-side overwrite for belt-and-suspenders safety:
```python
# Before model forward:
if anchor_first_frame and cur_start_frame == 0:
    noisy_input[:, :, 0:1] = first_frame_latent

# After model forward:
if anchor_first_frame and cur_start_frame == 0:
    x0_pred[:, :, 0:1] = first_frame_latent
```

### Design Rationale

We chose **output-level replacement in the network `forward()`** rather than:

1. **Input-level replacement only** — Would not guarantee the output is clean, and the VSD loss would receive imperfect teacher predictions.

2. **External rollout-level replacement** — Would only fix the student side. The teacher and fake_score (which also call `forward()`) would still return imperfect frame 0, biasing the distillation signal.

3. **Timestep masking** — Would be a useful addition but is not strictly necessary with output replacement. The original InfiniteTalk model was not trained with timestep masking, so adding it could create a train/inference mismatch. We opted not to add this to minimize deviation from the pretrained model's expected inputs.

By placing the fix in `forward()`:
- **Student rollout**: every denoising step produces perfect frame 0
- **KV cache**: stores perfect frame 0 for downstream blocks
- **Teacher VSD scoring**: returns perfect frame 0 in the distillation target
- **Fake score**: returns perfect frame 0 (consistent with teacher)
- **Inference**: automatic, no extra code needed
- **Validation**: automatic via `_student_sample_loop`

### What We Did NOT Add (and Why)

| Mechanism | Added? | Rationale |
|-----------|--------|-----------|
| Output replacement in `forward()` | **Yes** | Core fix: prevents error compounding in rollout, gives clean VSD signal |
| Input replacement in `forward()` | **No** | Not needed: the model already receives clean frame via I2V concat channels. Output replacement is sufficient. |
| Timestep masking (t=0 for frame 0) | **No** | The pretrained model was not trained with this. Adding it would change the input distribution. The concat mask channel serves the same purpose. |
| `preserve_conditioning` hook | **No** | Not needed: the `CausVidModel._student_sample_loop` (our AR sample loop) calls `net()` with `cur_start_frame`, and our `forward()` handles replacement. Adding it would be redundant. |
| Inference input-side overwrite | **Yes** | Extra safety: ensures noisy input to model also has clean frame 0 (belt-and-suspenders, on by default, `--no_anchor_first_frame` to disable) |

### Comparison with Wan 2.1 FastGen (Hypothetical)

If FastGen had a Self-Forcing config for Wan 2.1 I2V (which it doesn't — only Wan 2.2 TI2V and T2V exist), it would face the same compounding problem we faced, because `concat_mask=True` models have no internal `_replace_first_frame`. The fact that FastGen only ships SF configs for T2V and VACE V2V (which have their own conditioning paths) and Wan 2.2 I2V (which has hard replacement built in) suggests this is a known limitation of the concat-mask approach for distillation.

Our fix bridges this gap: we add the hard replacement that Wan 2.2's architecture provides natively, but apply it externally to our Wan 2.1-style model. This gives us the best of both worlds — the pretrained model's learned behavior from concat-mask training, plus the error prevention of hard anchoring during distillation.

---

## 6. Summary

The fundamental insight is that **bidirectional training does not need hard first-frame anchoring** (single forward pass, no error compounding), but **autoregressive distillation (Self-Forcing) does** (multi-step rollout where errors compound through re-noising and KV caching).

FastGen's Wan 2.2 TI2V handles this naturally via `_replace_first_frame` in the network `forward()`. Our InfiniteTalk model, based on Wan 2.1's concat-mask architecture, did not have this protection. Adding output-level replacement in `forward()` for both the student and teacher/fake_score networks resolves the issue while maintaining compatibility with the pretrained model and all code paths (training, validation, inference).
