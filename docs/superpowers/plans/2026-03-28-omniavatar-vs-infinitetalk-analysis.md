# OmniAvatar vs InfiniteTalk FastGen Discrepancy Analysis

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Systematically compare the OmniAvatar FastGen implementation against our verified InfiniteTalk FastGen implementation to identify major discrepancies, potential bugs, and design decisions that need validation.

**Architecture:** Both implementations adapt the FastGen self-forcing framework to audio-driven talking-head models (Wan2.1 14B backbone). InfiniteTalk is assumed correct (Stage 1 DF verified). OmniAvatar was used as reference but never fully debugged.

**Tech Stack:** PyTorch, FlexAttention, flash_attn, safetensors, FSDP

---

## File Map

**Our InfiniteTalk (REFERENCE — assumed correct):**
- `FastGen_InfiniteTalk/fastgen/networks/InfiniteTalk/network_causal.py` (1864 lines)
- `FastGen_InfiniteTalk/fastgen/networks/InfiniteTalk/wan_model.py` (788 lines)
- `FastGen_InfiniteTalk/fastgen/networks/InfiniteTalk/audio_modules.py` (210 lines)
- `FastGen_InfiniteTalk/fastgen/networks/InfiniteTalk/network.py` (591 lines)
- `FastGen_InfiniteTalk/fastgen/methods/infinitetalk_diffusion_forcing.py` (185 lines)
- `FastGen_InfiniteTalk/fastgen/methods/infinitetalk_self_forcing.py` (211 lines)
- `FastGen_InfiniteTalk/fastgen/configs/experiments/InfiniteTalk/config_df.py`
- `FastGen_InfiniteTalk/fastgen/configs/experiments/InfiniteTalk/config_sf.py`

**OmniAvatar (UNDER ANALYSIS):**
- `reference_FastGen_OmniAvatar/FastGen/fastgen/networks/OmniAvatar/network_causal.py` (1783 lines)
- `reference_FastGen_OmniAvatar/FastGen/fastgen/networks/OmniAvatar/network.py`
- `reference_FastGen_OmniAvatar/FastGen/fastgen/methods/omniavatar_diffusion_forcing.py` (129 lines)
- `reference_FastGen_OmniAvatar/FastGen/fastgen/methods/omniavatar_self_forcing.py` (105 lines)
- `reference_FastGen_OmniAvatar/FastGen/fastgen/configs/experiments/OmniAvatar/config_df.py`
- `reference_FastGen_OmniAvatar/FastGen/fastgen/configs/experiments/OmniAvatar/config_sf.py`

**Shared Base (both inherit from):**
- `fastgen/methods/distribution_matching/self_forcing.py` — rollout_with_gradient
- `fastgen/methods/distribution_matching/causvid.py` — _generate_noise_and_time
- `fastgen/methods/distribution_matching/dmd2.py` — VSD loss, training step
- `fastgen/networks/network.py` — CausalFastGenNetwork base class
- `fastgen/networks/noise_schedule.py` — sample_t_inhom, forward_process

**Output:** `docs/analysis/2026-03-28-omniavatar-discrepancy-report.md`

---

### Task 1: Causal Attention Mask Construction

**Files to compare:**
- InfiniteTalk: `network_causal.py:1169-1222` (`_build_block_mask`)
- OmniAvatar: `network_causal.py:1096-1149` (`_build_block_mask`)

- [ ] **Step 1: Read both `_build_block_mask` implementations**

Dispatch an agent to read both files at the exact line ranges and produce a diff.

- [ ] **Step 2: Verify mask semantics are identical**

Both should implement: "tokens within the same chunk attend bidirectionally; tokens can attend to all previous chunks but not future chunks." The mask function is:
```python
def attention_mask(b, h, q_idx, kv_idx):
    return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
```

Confirm both use identical chunk boundary calculation:
- `num_chunks = num_frames // chunk_size`
- `remaining_size = num_frames % chunk_size`
- First chunk gets `chunk_size + remaining_size` frames
- Subsequent chunks get `chunk_size` frames each

- [ ] **Step 3: Document finding**

**Pre-analysis finding:** The block mask construction is **IDENTICAL** between both implementations. No discrepancy. Both use the same FlexAttention `create_block_mask` with 128-byte padding and identical `attention_mask` lambda.

---

### Task 2: KV Cache Architecture

**Files to compare:**
- InfiniteTalk: `network_causal.py:1228-1265` (`_init_caches`, `clear_caches`)
- OmniAvatar: `network_causal.py:1155-1206` (`_init_caches`, `clear_caches`)

- [ ] **Step 1: Compare cache initialization**

InfiniteTalk allocates:
```python
self._kv_caches = [{
    "k": zeros(B, total_tokens, n, d),
    "v": zeros(B, total_tokens, n, d),
    "global_end_index": tensor(0),
    "local_end_index": tensor(0),
}]  # one per block
```

OmniAvatar allocates:
```python
self._kv_caches = [...]   # same structure
self._crossattn_caches = [{"is_init": False}]  # ADDITIONAL: one per block
```
Plus conditional sizing: `cache_tokens = self.local_attn_size * frame_seqlen` if `local_attn_size > 0`.

- [ ] **Step 2: Compare cache clearing**

InfiniteTalk clears: `_kv_caches`, `block_mask`, `_cached_audio`.
OmniAvatar additionally clears: `_crossattn_caches` (pops k/v, resets `is_init`).

- [ ] **Step 3: Document discrepancies**

**DISCREPANCY 1 — Cross-attention caching:** OmniAvatar caches cross-attention K/V (text conditioning) to avoid recomputation per chunk during AR generation. InfiniteTalk does NOT cache cross-attention. This means InfiniteTalk recomputes text cross-attention K/V projections for every chunk during AR — functionally correct but less efficient. **Severity: LOW (performance only, not correctness).**

**DISCREPANCY 2 — Rolling window eviction:** OmniAvatar supports `local_attn_size > 0` with sink tokens for long-sequence generation beyond cache capacity. InfiniteTalk hardcodes "21 frames, cache fits" with no eviction. **Severity: NONE for current use (21 frames fits). Would matter if extending to longer sequences.**

---

### Task 3: KV Cache Update & Read Logic (Self-Attention)

**Files to compare:**
- InfiniteTalk: `network_causal.py:430-487` (CausalSelfAttention AR mode)
- OmniAvatar: `network_causal.py:401-528` (CausalSelfAttention AR mode)

- [ ] **Step 1: Read both CausalSelfAttention.forward AR branches in full**

Dispatch agent to read both files at the AR-mode code paths.

- [ ] **Step 2: Compare the 8 substeps**

| Substep | InfiniteTalk | OmniAvatar | Match? |
|---------|-------------|-----------|--------|
| 1. RoPE application | Always applies RoPE to Q and K before caching | Conditional: `use_dynamic_rope` flag controls whether K is cached with or without RoPE | DIFFERENT |
| 2. Cache metadata read | Uses `cache_local_end_override` for gradient-checkpoint safety | Identical pattern | MATCH |
| 3. Eviction check | No eviction (comment: "21 frames fits") | Full rolling eviction with sink tokens | DIFFERENT (by design) |
| 4. Cache write | `kv_cache["k"][:, start:end] = k_to_cache.detach()` | Identical | MATCH |
| 5. Window construction | Simple contiguous: `cat([k_past, k_to_cache])` | Complex: sink + rolling window, non-contiguous | DIFFERENT (by design) |
| 6. RoPE mode | N/A (already applied) | Dynamic mode: `causal_rope_apply` on full window with relative positions | DIFFERENT |
| 7. Flash attention call | `_flash_attention(roped_q, k_win, v_win)` | Same | MATCH |
| 8. Metadata update | `global_end_index`, `local_end_index` | Same | MATCH |

- [ ] **Step 3: Assess impact of dynamic RoPE**

**DISCREPANCY 3 — Dynamic RoPE:** OmniAvatar has a `use_dynamic_rope` flag that, when True, stores raw (un-RoPE'd) keys in the cache and applies RoPE dynamically using window-local positions. InfiniteTalk always stores RoPE-applied keys (standard approach).

The dynamic RoPE approach enables the rolling-window eviction to work correctly — when tokens are evicted, the remaining tokens can be re-RoPE'd relative to the window. With pre-applied RoPE and eviction, position encodings would become inconsistent.

**For InfiniteTalk's 21-frame case (no eviction), this difference has zero impact.** Both approaches produce identical attention outputs when the full sequence fits in cache.

**Severity: NONE for current config. Critical if local_attn_size were enabled.**

---

### Task 4: RoPE with Frame Offset

**Files to compare:**
- InfiniteTalk: `network_causal.py:188-230` (`causal_rope_apply`)
- OmniAvatar: `network_causal.py:179-223` (`causal_rope_apply`)

- [ ] **Step 1: Diff the two implementations**

- [ ] **Step 2: Document differences**

**Pre-analysis finding:** Nearly identical. Two minor differences:

1. OmniAvatar computes `split_sizes` and `freq_parts` tuple (lines 198-199) but doesn't use `split_sizes` — appears to be dead code from an earlier refactor. No functional impact.

2. InfiniteTalk includes `.to(device=x_i.device)` on the frequency tensor (line 223); OmniAvatar omits this. If freqs are already on the correct device (which they should be from model initialization), this is a no-op. But it's a safety measure.

**Severity: NONE. Functionally identical.**

---

### Task 5: Diffusion Forcing Training Method

**Files to compare:**
- InfiniteTalk: `infinitetalk_diffusion_forcing.py` (185 lines)
- OmniAvatar: `omniavatar_diffusion_forcing.py` (129 lines)
- Shared: `noise_schedule.py:sample_t_inhom` (both use same base)

- [ ] **Step 1: Compare single_train_step implementations**

Both follow the same pattern:
```
1. Build condition from data batch
2. Sample t_inhom via noise_scheduler.sample_t_inhom()
3. Add noise: noisy_data = forward_process(real_data, eps, t_inhom)
4. Student denoise: gen_data = gen_data_from_net(noisy_data, t_inhom, condition)
5. L2 loss: mse_loss(gen_data, real_data)
```

- [ ] **Step 2: Compare condition building**

InfiniteTalk `_build_condition()`:
```python
condition = {
    "text_embeds": data["text_embeds"],
    "first_frame_cond": data["first_frame_cond"],
    "clip_features": data["clip_features"],
    "audio_emb": data["audio_emb"],
}
```

OmniAvatar `_build_condition()`:
```python
ref_latent = real_data[:, :, :1, :, :]  # extract from clean data
condition = {
    "text_embeds": data["text_embeds"],
    "audio_emb": data["audio_emb"],
    "ref_latent": ref_latent,
    "mask": data["mask"],
    "masked_video": data["masked_video"],
}
if "ref_sequence" in data:
    condition["ref_sequence"] = data["ref_sequence"]
```

- [ ] **Step 3: Document key differences in conditioning**

**DISCREPANCY 4 — Conditioning format:** This is a fundamental architectural difference, not a bug:

| Feature | InfiniteTalk | OmniAvatar |
|---------|-------------|-----------|
| Reference frame | Separate `first_frame_cond` (mask+VAE pre-computed, 20ch concatenated with noise=16ch → in_dim=36) | `ref_latent` extracted from first frame of clean data + `mask` + `masked_video` (in_dim=65: 16 noise + 16 ref + 1 mask + 16 masked + 16 ref_seq) |
| CLIP features | Explicit `clip_features` (257 tokens) passed through `img_emb` MLP | None — OmniAvatar doesn't use CLIP image features |
| Audio format | `audio_emb` [81, 12, 768] — raw wav2vec2 hidden states | `audio_emb` [B, 81, 10752] — flattened AudioPack format |
| Spatial mask | No spatial mask (full-frame conditioning) | Binary mask [H, W] indicating mouth/generation region |
| V2V conditioning | N/A | `masked_video` provides inpainting-style conditioning |

**Severity: BY DESIGN. These are model-specific adaptations, not bugs.**

- [ ] **Step 4: Verify noise sampling is identical**

Both call `self.net.noise_scheduler.sample_t_inhom()` with same args:
```python
t_inhom, _ = self.net.noise_scheduler.sample_t_inhom(
    batch_size, num_frames, chunk_size,
    sample_steps=self.config.student_sample_steps,
    t_list=self.config.sample_t_cfg.t_list,
    device=self.device, dtype=real_data.dtype,
)
```

The `sample_t_inhom` implementation is in `noise_schedule.py` which is shared between both. **Identical behavior.**

---

### Task 6: Self-Forcing Training Method & CFG

**Files to compare:**
- InfiniteTalk: `infinitetalk_self_forcing.py` (211 lines)
- OmniAvatar: `omniavatar_self_forcing.py` (105 lines)
- Shared: `self_forcing.py:rollout_with_gradient` (257 lines)

- [ ] **Step 1: Compare CFG strategy**

**DISCREPANCY 5 — Major: 3-call vs 2-call CFG:**

InfiniteTalk overrides `_apply_classifier_free_guidance()` for 3-call CFG:
```python
# 3 teacher evaluations per step:
output = uncond + text_scale * (cond - drop_text) + audio_scale * (drop_text - uncond)
```
This separates text and audio guidance with independent scales (text=5.0, audio=4.0).

OmniAvatar uses the base SelfForcingModel's 2-call CFG:
```python
# 2 teacher evaluations per step:
output = uncond + guidance_scale * (cond - uncond)
```
This provides a single combined guidance scale (4.5) for all conditioning.

**Severity: BY DESIGN. InfiniteTalk has separate text and audio control; OmniAvatar treats all conditioning as one signal.**

- [ ] **Step 2: Compare negative condition construction**

InfiniteTalk negative condition for 3-call CFG:
```python
# drop_text condition: keep audio, replace text with neg_text
neg_text_condition = {**self._current_condition}
neg_text_condition["text_embeds"] = neg_text_embeds

# uncond condition: zero audio, replace text with neg_text
neg_condition = {**self._current_condition}
neg_condition["text_embeds"] = neg_text_embeds
neg_condition["audio_emb"] = torch.zeros_like(audio_emb)
```

OmniAvatar negative condition for 2-call CFG:
```python
neg_condition = {
    "text_embeds": data.get("neg_text_embeds", ...),
    "audio_emb": torch.zeros_like(data["audio_emb"]),
    "ref_latent": ref_latent,     # keep spatial
    "mask": data["mask"],          # keep spatial
    "masked_video": data["masked_video"],  # keep spatial
}
```

**Key observation:** OmniAvatar zeros the audio in the negative condition. InfiniteTalk creates TWO negative conditions (drop-text-only and full-uncond).

- [ ] **Step 3: Verify rollout_with_gradient is shared and unmodified**

Both InfiniteTalk and OmniAvatar inherit `rollout_with_gradient()` from the shared `SelfForcingModel` base class. Confirm neither overrides it.

InfiniteTalk: inherits from `SelfForcingModel` → no override of `rollout_with_gradient`.
OmniAvatar: inherits from `SelfForcingModel` → no override of `rollout_with_gradient`.

**MATCH. The core self-forcing rollout loop is identical.**

---

### Task 7: Forward Pass Architecture (full_sequence vs AR dispatch)

**Files to compare:**
- InfiniteTalk: `network_causal.py:1596-1752` (`forward`)
- OmniAvatar: `network_causal.py:1472-1601` (`forward`)

- [ ] **Step 1: Compare dispatch logic**

Both use the same auto-dispatch:
```python
if t.dim() == 2:  # per-frame timesteps from DF/KD
    use full-sequence mode
elif is_ar:
    use AR mode with KV cache
else:
    use full-sequence mode
```

- [ ] **Step 2: Compare condition unpacking**

InfiniteTalk forward unpacks:
```python
text_embeds = condition.get("text_embeds")
clip_features = condition.get("clip_features")
audio_emb = condition.get("audio_emb")
first_frame_cond = condition.get("first_frame_cond")
# Builds y = first_frame_cond (mask+VAE, already 20ch)
# Passes clip_fea=clip_features explicitly to _forward_full_sequence/_forward_ar
```

OmniAvatar forward unpacks:
```python
text_embeds = condition.get("text_embeds")
audio_emb = condition.get("audio_emb")
ref_latent = condition.get("ref_latent")
mask = condition.get("mask")
masked_video = condition.get("masked_video")
ref_sequence = condition.get("ref_sequence")
# Builds y via _build_y(): cat([ref_latent, mask, masked_video, ref_sequence])
# No separate clip_fea parameter
```

- [ ] **Step 3: Check audio preprocessing in forward**

**DISCREPANCY 6 — Audio sliding window in forward:**

InfiniteTalk applies a 5-frame sliding window to audio embeddings in the `forward()` method if audio is 4D:
```python
if audio_emb is not None and audio_emb.dim() == 4:
    # audio_emb is [B, num_frames, 12, 768] — apply 5-frame window
    indices = (torch.arange(5) - 2)  # [-2, -1, 0, 1, 2]
    center_indices = torch.arange(num_frames).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = center_indices.clamp(0, audio_emb.shape[1] - 1)
    audio_emb = audio_emb[:, center_indices]  # [B, T, 5, 12, 768]
```

OmniAvatar does NOT have this logic — audio embeddings arrive pre-windowed or in a different format (flattened AudioPack [B, 81, 10752]).

**Severity: BY DESIGN. Different audio preprocessing pipelines.**

- [ ] **Step 4: Check timestep rescaling**

Both rescale timesteps from FastGen convention [0, 1) to model convention [0, 1000):
```python
timestep = t * 1000  # InfiniteTalk: explicit multiplication
# OR
timestep = self.noise_scheduler.rescale_t(t)  # OmniAvatar: method call
```

Verify both produce the same result. **Expected: MATCH.**

---

### Task 8: Config Parameter Comparison

**Files to compare:**
- InfiniteTalk: `config_df.py`, `config_sf.py`
- OmniAvatar: `config_df.py`, `config_sf.py`

- [ ] **Step 1: Build comparison table**

| Parameter | IT DF | IT SF | OA DF | OA SF | Notes |
|-----------|-------|-------|-------|-------|-------|
| chunk_size | 3 | 3 | 3 | 3 | Match |
| total_num_frames | 21 | 21 | 21 | 21 | Match |
| input_shape | [16,21,56,112] | [16,21,56,112] | [16,21,64,64] | [16,21,64,64] | Different resolution |
| shift | 7.0 | 7.0 | N/A | 3.0 | **POTENTIAL ISSUE** |
| t_list | [0.999,0.937,0.833,0.624,0.0] | same | same | same | Match |
| student_sample_steps | 4 | 4 | 4 | 4 | Match |
| precision | bf16 | bf16 | bf16 | bf16 | Match |
| precision_fsdp | **bf16** | f32 | f32 | f32 | **DISCREPANCY** |
| guidance_scale | N/A | 5.0 | N/A | 4.5 | Different |
| text_guide_scale | N/A | 5.0 | N/A | N/A | IT-only |
| audio_guide_scale | N/A | 4.0 | N/A | N/A | IT-only |
| teacher size | N/A | 14B | N/A | 14B | Match |
| student size | 14B+LoRA | 14B+LoRA | 1.3B | 1.3B | Different |
| fake_score size | N/A | 14B+LoRA | N/A | 1.3B | Different |
| gan_loss | N/A | 0 | N/A | 0 | Match (disabled) |
| context_noise | N/A | 0.0 | N/A | 0.0 | Match |

- [ ] **Step 2: Assess shift parameter**

**DISCREPANCY 7 — Timestep shift:**
- InfiniteTalk uses shift=7.0 (for both DF and SF)
- OmniAvatar DF doesn't specify shift (will use noise_schedule default)
- OmniAvatar SF uses shift=3.0

The original InfiniteTalk model uses shift=7 for 480p and shift=11 for 720p. OmniAvatar's shift=3.0 is the standard Wan I2V default.

**This is model-specific and correct for each model.** InfiniteTalk at 480p needs higher shift. Not a bug.

- [ ] **Step 3: Assess FSDP precision mismatch**

**DISCREPANCY 8 — FSDP precision in InfiniteTalk DF:**
InfiniteTalk DF config sets `precision_fsdp = "bfloat16"` while all other configs (IT SF, OA DF, OA SF) use `"float32"`.

When FSDP reduces gradients, using bfloat16 can cause gradient underflow/overflow in accumulated values. The OmniAvatar configs consistently use float32 for FSDP communication.

**Severity: MEDIUM. Our DF training may have slightly less stable gradient accumulation. Worth checking if this was intentional or a copy-paste oversight.**

---

### Task 9: Weight Loading & Architecture Initialization

**Files to compare:**
- InfiniteTalk: `network_causal.py` (`_load_weights` or `__init__`)
- OmniAvatar: `network_causal.py:1607-1782` (`_load_weights`)

- [ ] **Step 1: Compare weight loading strategies**

InfiniteTalk loads:
1. Base Wan I2V 14B (7 safetensor shards)
2. InfiniteTalk-specific weights (audio modules, etc.)
3. Optional LoRA for teacher merging

OmniAvatar loads:
1. Base Wan 14B or 1.3B
2. OmniAvatar-specific weights (audio_pack, patch_embedding expansion)
3. Handles LoRA key conversion and patch_embedding channel expansion

- [ ] **Step 2: Check in_dim handling**

InfiniteTalk: `in_dim=36` (16 noise + 4 mask + 16 VAE reference)
OmniAvatar: `in_dim=65` (16 noise + 16 ref + 1 mask + 16 masked_video + 16 ref_sequence)

OmniAvatar's `_load_weights` has special handling to expand `patch_embedding.weight` from the base 16-channel to 65-channel by zero-initializing new channels. Verify InfiniteTalk does the same for its 36-channel expansion.

**Severity: Must verify. Incorrect patch_embedding expansion would corrupt all model outputs.**

---

### Task 10: Audio Processing Pipeline

**Files to compare:**
- InfiniteTalk: `audio_modules.py` + `network_causal.py:_process_audio`
- OmniAvatar: `network_causal.py:_process_audio_embeddings` + `_inject_audio_at_layer`

- [ ] **Step 1: Compare audio projection architecture**

InfiniteTalk uses `AudioProjModel`:
- Input: 5-frame sliding window × 12 wav2vec2 layers × 768 dim
- Two separate projections: first-frame (window=5) and latter-frames (window=variable)
- Output: 32 context tokens per frame at dim=768
- Fed to `SingleStreamAttention` (cross-attention from visual → audio tokens)

OmniAvatar uses `AudioPack` + per-layer linear projections:
- Input: pre-flattened [B, 81, 10752] audio features
- Projects at specific transformer layers (not all blocks)
- Additive injection into hidden states (not cross-attention)

- [ ] **Step 2: Document structural difference**

**DISCREPANCY 9 — Audio conditioning mechanism:**

| Aspect | InfiniteTalk | OmniAvatar |
|--------|-------------|-----------|
| **Architecture** | Cross-attention (Q=visual, K/V=audio) | Additive projection at specific layers |
| **Audio injection** | Every transformer block via `audio_cross_attn` | Specific layers only via `_inject_audio_at_layer` |
| **Audio format** | Per-frame windowed: [B, T, 5, 12, 768] | Flattened: [B, T, 10752] |
| **Projection** | AudioProjModel → 32 tokens/frame → cross-attention | Linear → additive to hidden states |
| **Multi-person** | Supported via spatial RoPE separation | Single-person only |

**Severity: BY DESIGN. Completely different audio conditioning architectures.**

---

### Task 11: Compile Final Analysis Report

- [ ] **Step 1: Create analysis output document**

Write `docs/analysis/2026-03-28-omniavatar-discrepancy-report.md` with:
1. Executive summary of all 9 discrepancies
2. Classification: Bug / By Design / Performance Only
3. Severity assessment for each
4. Actionable recommendations

- [ ] **Step 2: Create discrepancy severity matrix**

| # | Discrepancy | Type | Severity | Action Needed |
|---|------------|------|----------|---------------|
| 1 | No cross-attention caching in IT | Performance | LOW | Optional optimization |
| 2 | No rolling-window eviction in IT | By Design | NONE (21 frames fits) | Only if extending to longer sequences |
| 3 | No dynamic RoPE in IT | By Design | NONE (no eviction) | Paired with #2 |
| 4 | Different conditioning format | By Design | NONE | Model-specific |
| 5 | 3-call vs 2-call CFG | By Design | NONE | IT needs separate text/audio control |
| 6 | Audio sliding window in forward | By Design | NONE | Different audio pipelines |
| 7 | Timestep shift (7.0 vs 3.0) | By Design | NONE | Model-specific |
| 8 | FSDP precision bf16 in IT DF | **Potential Bug** | MEDIUM | Verify intentional or fix to f32 |
| 9 | Different audio conditioning | By Design | NONE | Fundamentally different architectures |

- [ ] **Step 3: Write actionable recommendations**

**Immediate action items:**
1. **Check FSDP precision** in `config_df.py` — should it be `float32` like all other configs?
2. **Consider cross-attention caching** for inference speedup (low priority)

**No action needed:**
- All other discrepancies are intentional design differences reflecting the different model architectures (InfiniteTalk's multi-person cross-attention audio vs OmniAvatar's additive audio injection, I2V conditioning vs V2V inpainting)

---

## Summary

The two implementations share identical core FastGen self-forcing mechanics:
- Block-wise causal attention mask: **IDENTICAL**
- KV cache update/read pattern: **IDENTICAL** (modulo OmniAvatar's extra eviction/dynamic-RoPE for longer sequences)
- Self-forcing rollout_with_gradient: **IDENTICAL** (shared base class, neither overrides)
- Noise sampling (sample_t_inhom): **IDENTICAL** (shared noise_schedule.py)
- Diffusion forcing training step: **IDENTICAL** structure

All differences fall into two categories:
1. **Model-specific adaptations** (conditioning, audio, CFG, input channels) — correct and expected
2. **One potential config issue** (FSDP precision in IT DF config) — worth checking
