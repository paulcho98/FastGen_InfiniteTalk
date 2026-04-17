# FastGen I2V Self-Forcing: Wan 2.1 vs Wan 2.2 vs InfiniteTalk

**Date:** 2026-04-07
**Purpose:** Comprehensive comparison of how FastGen handles first-frame I2V conditioning across different model variants and training stages, and where our InfiniteTalk implementation stands.

---

## 1. Model Variants

| Property | Wan 2.1 I2V (14B) | Wan 2.2 TI2V (5B) | InfiniteTalk (14B) |
|----------|-------------------|--------------------|--------------------|
| `concat_mask` | True | False | True (36ch input) |
| Input channels | 36 (16 noise + 4 mask + 16 ref) | 16 (noise only, frame 0 replaced) | 36 (16 noise + 4 mask + 16 ref) |
| Timestep masking | No (all frames same t) | Yes (frame 0 gets t=0) | No (all frames same t) |
| Output replacement | No | Yes (`_replace_first_frame`) | Yes (our addition) |
| `preserve_conditioning` | No-op | Active (replaces frame 0) | Not implemented |
| Image encoder (CLIP) | Yes | No | Yes |

**InfiniteTalk is architecturally Wan 2.1-style** (36-channel concat, no timestep masking, CLIP features), but we added Wan 2.2-style output replacement.

---

## 2. FastGen Available Configs

### I2V-specific configs (WanI2V directory):
- **SFT** (bidirectional): `config_sft_14b.py`, `config_sft_wan22_5b.py`
- **Causal SFT / DF** (causal, inhomogeneous timesteps): `config_sft_causal_14b.py`, `config_sft_causal_wan22_5b.py`
- **DMD2** (bidirectional student + teacher + fake_score): `config_dmd2_14b.py`, `config_dmd2_wan22_5b.py`
- **Self-Forcing**: **NONE** — no SF config exists for I2V

### SF configs exist only for:
- **WanT2V**: `config_sf.py`, `config_sf_14b.py` (text-to-video, no I2V conditioning)
- **WanV2V**: `config_sf.py` (VACE video-to-video, different conditioning)

**Key implication**: FastGen's authors have not shipped Self-Forcing for I2V. The I2V distillation pipeline uses only DMD2 (bidirectional, non-AR student). Our InfiniteTalk SF implementation is novel in applying SF's causal AR rollout to an I2V model.

---

## 3. Frame 0 Handling in Forward Pass

### 3A. Bidirectional Teacher / Fake Score

| Step | Wan 2.1 (`concat_mask=True`) | Wan 2.2 (`concat_mask=False`) | InfiniteTalk (ours) |
|------|-----|------|------|
| **Input** | Concat `[x_t, mask, first_frame_cond]` → 36ch. Frame 0 of `x_t` is noisy. | `_replace_first_frame`: frame 0 of `x_t` → clean ref. 16ch input. | Concat `[x_t, y]` where `y = [mask, first_frame_cond]` → 36ch. Frame 0 of `x_t` is noisy. Same as Wan 2.1. |
| **Timestep** | All frames get same t | Frame 0 gets t=0 (masked) | All frames get same t. Same as Wan 2.1. |
| **Output** | Raw prediction kept (no replacement) | `_replace_first_frame` on output: frame 0 → clean ref | **Output replacement added**: frame 0 → clean ref. Diverges from Wan 2.1, matches Wan 2.2. |

### 3B. Causal Student

| Step | Wan 2.1 (`concat_mask=True`) | Wan 2.2 (`concat_mask=False`) | InfiniteTalk (ours) |
|------|-----|------|------|
| **Input (chunk 0)** | Concat `[x_t, mask_chunk, ffc_chunk]` → 36ch. Frame 0 noisy. | `_replace_first_frame`: frame 0 → clean ref. 16ch. | Concat `[x_t, y_chunk]` → 36ch. Frame 0 noisy. Same as Wan 2.1. |
| **Input (chunk 1+)** | Concat `[x_t, zeros, zeros]` → 36ch. No ref signal. | Plain `x_t`. No replacement. | Concat `[x_t, y_chunk]` where mask=0 and ffc≈0. Same as Wan 2.1. |
| **Timestep (chunk 0)** | All frames same t | Frame 0 gets t=0 | All frames same t. Same as Wan 2.1. |
| **Timestep (chunk 1+)** | All frames same t | All frames same t | All frames same t. Same as all. |
| **Output (chunk 0)** | Raw prediction (no replacement) | `_replace_first_frame`: frame 0 → clean ref | **Output replacement**: frame 0 → clean ref. Diverges from Wan 2.1, matches Wan 2.2. |
| **Output (chunk 1+)** | Raw prediction | Raw prediction | Raw prediction. Same as all. |

---

## 4. Training Stage Implementations

### 4A. Diffusion Forcing / Causal SFT (Stage 1)

**FastGen** (`CausalSFTModel`):
- Single forward pass with inhomogeneous per-frame timesteps `[B, T]`
- Auto-routes to full-sequence mode (no chunking, no KV cache)
- DSM loss: `||model_output - target||²`
- Frame 0 handling: via `_compute_i2v_inputs` (concat for 2.1, replace for 2.2)
- No multi-step rollout, no error compounding

**InfiniteTalk** (`InfiniteTalkDiffusionForcingModel`):
- Same: single forward pass with inhomogeneous timesteps
- Auto-routes to full-sequence mode via `t.dim() == 2` check
- DSM loss
- Frame 0: via `_build_y` (36ch concat). **No output replacement during DF training** (no anchor existed when DF was trained)

### 4B. DMD2 Distillation (FastGen I2V only)

**FastGen** (`DMD2Model`):
- **All three models are bidirectional** (student, teacher, fake_score)
- Student generates via `FastGenModel._student_sample_loop` (non-causal, multi-step)
- `_student_sample_loop` calls `preserve_conditioning` (active for Wan 2.2, no-op for Wan 2.1)
- VSD loss: teacher vs fake_score scoring of student output
- No AR rollout, no KV cache, no chunking

**InfiniteTalk**: N/A (we don't use DMD2, we go directly to Self-Forcing)

### 4C. Self-Forcing Distillation (Stage 2)

**FastGen** (T2V/V2V only, NOT I2V):
- Student: causal `CausalWan` (1.3B)
- Teacher: bidirectional `Wan` (14B, frozen)
- Fake score: bidirectional `Wan` (1.3B, trainable)
- `rollout_with_gradient`: chunk-by-chunk AR with KV cache
- **Does NOT call `preserve_conditioning`**
- Relies on `CausalWan.forward()` to handle conditioning internally
- For T2V/V2V: no I2V conditioning at all, so frame 0 question is moot

**InfiniteTalk** (`InfiniteTalkSelfForcingModel`):
- Student: causal `CausalInfiniteTalkWan` (14B LoRA)
- Teacher: bidirectional `InfiniteTalkWan` (14B, frozen)
- Fake score: bidirectional `InfiniteTalkWan` (14B LoRA, trainable)
- `rollout_with_gradient`: same chunk-by-chunk AR (inherited from `SelfForcingModel`)
- Does NOT call `preserve_conditioning` (same as FastGen)
- **Output anchor added to `forward()`**: frame 0 of x0 prediction → clean ref when `cur_start_frame == 0`
- This fires at every denoising step within the rollout, keeping frame 0 clean
- Teacher and fake_score also have the output anchor (in `InfiniteTalkWan.forward()`)

---

## 5. Inference Implementations

### 5A. CausVid `_student_sample_loop` (Causal AR)

Used by: SF inference, SF validation, DF validation (via `generator_fn`)

**FastGen**:
- Chunk-by-chunk denoising with KV cache
- **Does NOT call `preserve_conditioning`**
- Relies on `forward()` internal I2V handling
- For Wan 2.2: `forward` replaces frame 0 in input and output → clean throughout
- For Wan 2.1: `forward` only provides concat mask signal → model must predict frame 0 correctly

**InfiniteTalk**:
- Same loop (inherited from `CausVidModel._student_sample_loop`)
- Our `forward()` has the output anchor → frame 0 always clean in x0 predictions
- Between denoising steps, `forward_process` noises all frames including frame 0
- Next `forward()` call reproduces clean frame 0 via the anchor

### 5B. Base `_student_sample_loop` (Non-causal)

Used by: DMD2 inference (bidirectional student)

- **DOES call `preserve_conditioning`** on both x0 prediction and noised latents
- For Wan 2.2: frame 0 is preserved after every step
- For Wan 2.1: no-op
- Not relevant to our InfiniteTalk (we use the causal version)

### 5C. Inference Script (`inference_causal.py`)

Our custom AR loop (not using `_student_sample_loop`):
- **No pre-forward input anchor** (removed — was causing static video)
- **Post-forward output anchor** (redundant with network's `forward()` anchor, kept as safety)
- Network's `forward()` handles frame 0 via output replacement

---

## 6. Where We Stand

### What we match:
- **Wan 2.1 architecture**: 36-channel concat conditioning (mask + first_frame_cond) ✓
- **Wan 2.1 timestep handling**: all frames get same t (no masking) ✓
- **Wan 2.1 input handling**: noisy frame 0 in `x_t`, clean ref via concat channels ✓
- **FastGen SF rollout**: `rollout_with_gradient` with no `preserve_conditioning` ✓
- **FastGen CausVid inference**: chunk-by-chunk AR, relies on `forward()` internals ✓

### What we added beyond Wan 2.1:
- **Output replacement in `forward()`** for student, teacher, AND fake_score
  - Wan 2.1 does NOT do this (raw model prediction kept)
  - Wan 2.2 DOES do this (via `_replace_first_frame`)
  - Rationale: prevents error compounding in SF rollout (a problem that doesn't exist in Wan 2.1's DMD2 pipeline because DMD2 uses a bidirectional student with no AR rollout)

### What we do NOT have (unlike Wan 2.2):
- **Input replacement** (frame 0 of `x_t` → clean ref before transformer) — not needed with concat conditioning
- **Timestep masking** (frame 0 gets t=0) — not needed with concat conditioning; model was not trained with this
- **`preserve_conditioning` hook** — not needed because CausVid `_student_sample_loop` doesn't use it, and our `forward()` handles output replacement

### Why our approach is justified:
1. InfiniteTalk is Wan 2.1-style (concat mask), so input replacement and timestep masking would be distribution mismatches
2. FastGen has no SF config for I2V — we are the first to apply SF to this architecture
3. The output anchor is a minimal, safe addition: it doesn't change what the transformer processes, only fixes frame 0 in the output
4. The anchor prevents the specific error-compounding problem that SF's multi-step AR rollout introduces (which Wan 2.1's existing DMD2 pipeline doesn't face)

---

## 7. Open Questions

1. **Is the output anchor necessary?** The soft conditioning (Wan 2.1 style) SF run at iter 400 can be compared against the I2V (with anchor) run at iter 400 to measure the impact.

2. **Should the anchor apply during inference?** Currently unconditional. Could make it training-only via `self.training` guard, but since SF was trained with it, inference should match.

3. **Teacher output anchor**: Currently the bidirectional teacher also anchors frame 0. This means the VSD target always has perfect frame 0. For Wan 2.1 (which does NOT anchor teacher output), the teacher's frame 0 prediction might be slightly imperfect, providing a different gradient signal. Our choice to anchor the teacher is a conservative decision that may reduce gradient noise at frame 0.
