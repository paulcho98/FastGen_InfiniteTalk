# Deviations from Original FastGen Implementation

**Date:** 2026-03-28
**Scope:** InfiniteTalk adaptation vs original FastGen CausalWanI2V pipeline

This documents all non-trivial differences between our `CausalInfiniteTalkWan` and
the original FastGen `CausalWanI2V`, **excluding** audio modules and InfiniteTalk-specific
conditioning (those are expected additions, not deviations).

---

## 1. `is_ar` Default: `True` vs `False`

**Original** `CausalWanI2V.forward()`: `is_ar: bool = False`
**Ours** `CausalInfiniteTalkWan.forward()`: `is_ar: bool = True`

**Impact:** Any code calling `net(x, t_scalar, condition=...)` without explicit `is_ar`
will dispatch to AR mode (ours) vs full-sequence mode (original). During DF/KD training
this doesn't matter (per-frame timesteps always route to full-sequence). During
visualization via `generator_fn`, the AR override of `_student_sample_loop` makes
this moot. But other callers could behave differently.

## 2. No `_replace_first_frame` Post-Processing

**Original:** After the model forward, replaces output frame 0 with clean `first_frame_cond`.
Hard constraint that the network's prediction for the reference frame is always clean.

**Ours:** Returns raw model output for all frames including frame 0.

**Impact:** Model learns to predict frame 0 (wastes some capacity, but frame 0 is easy).
During AR generation, frame 0 won't be hard-anchored to the reference. The I2V mask
conditioning (which marks frame 0 as the reference) provides a soft signal instead.

## 3. No `preserve_conditioning` Hook

**Original** `WanI2V`: `preserve_conditioning(x, condition)` replaces `x[:,:,0]` with
`first_frame_cond[:,:,0]`. Called by `FastGenModel._student_sample_loop` after each
denoising step and after noise re-injection.

**Ours:** No such method. The `_student_sample_loop` (overridden to AR) does its own
chunk management instead.

**Impact:** During non-AR multi-step generation (if ever called), the first frame can
drift across denoising steps without re-anchoring.

## 4. Standalone DiT vs Diffusers Wrapper

**Original:** Wraps `diffusers.WanTransformer3DModel` with custom attention processors
(`_wan_set_attn_processor`). Transformer internals come from diffusers.

**Ours:** Standalone `WanModel` ported from InfiniteTalk's `multitalk_model.py`.
Own implementations of all blocks, attention, RoPE.

**Impact:** Functionally equivalent (verified 0.0 diff in Stage 0). But different
code paths and attention implementations.

## 5. LoRA Wrappers on All Linear Layers

**Original:** Direct `nn.Linear` layers. Full fine-tuning or FSDP sharding of all params.

**Ours:** `LoRALinear` wrappers on Q/K/V/O/FFN + audio cross-attention linears.
Forward: `F.linear(x, base.weight) + F.linear(F.linear(x, lora_A), lora_B) * scaling`.
294M trainable (rank 32) or 954M trainable (rank 128) out of 19B total.

## 6. `build_model` Re-Freeze Workaround

**Original:** `FastGenModel.build_model()` sets `requires_grad_(True)` on all params.
Fine for full fine-tuning.

**Ours:** Override `build_model()` → call `super()` → re-apply `freeze_base()` to
lock all non-LoRA params. Without this, all 19B params are trainable → OOM.

## 7. No Timestep Masking for Frame 0

**Original** `CausalWanI2V`: When processing the first chunk, sets `timestep=0` for
frame 0 via `_compute_timestep_inputs(timestep, timestep_mask)`. Tells the model
"frame 0 is clean" through the timestep signal.

**Ours:** Passes the raw sampled timestep for all frames including frame 0. The model
sees the same noise level for frame 0 as other frames, relying on the I2V mask
conditioning to differentiate.

## 8. Audio Caching: Training vs Inference

**Original:** No audio modules (diffusers-based, no InfiniteTalk audio).

**Ours:** `_forward_ar` caches AudioProjModel output across chunks during **inference**
(avoids redundant recomputation). During **training** (`self.training == True`),
recomputes fresh each call so gradients flow correctly through AudioProjModel
on Self-Forcing exit steps.

## 9. AR Visualization via `_student_sample_loop` Override

**Original** `CausalKDModel._get_outputs`: Uses `CausVidModel.generator_fn` which
resolves to `FastGenModel.generator_fn` → `FastGenModel._student_sample_loop`
(non-AR, full-sequence per step).

**Ours:** `_student_sample_loop = CausVidModel._student_sample_loop` (class-level
override). Visualization runs proper chunk-by-chunk AR generation with KV cache,
reflecting actual inference behavior.

---

## Summary Table

| Aspect | Original FastGen | Ours | Risk |
|--------|-----------------|------|------|
| `is_ar` default | `False` | `True` | Low (training unaffected) |
| First-frame replacement | Yes (hard anchor) | No | Medium (quality monitor) |
| `preserve_conditioning` | Yes | No | Low (AR vis has own logic) |
| Model backend | diffusers | standalone port | None (verified 0.0 diff) |
| LoRA | No | Yes (rank 32-128) | None (standard technique) |
| build_model freeze | N/A | Re-freeze after super() | None (workaround for base class) |
| Timestep masking frame 0 | Yes (t=0) | No | Low (I2V mask compensates) |
| Audio caching | N/A | Train: fresh, Infer: cached | None |
| Visualization loop | Non-AR | AR (chunk+KV cache) | None (more accurate) |
