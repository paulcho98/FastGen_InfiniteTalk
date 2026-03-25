# InfiniteTalk FastGen Adaptation — Implementation Progress

**Started:** 2026-03-25
**Plan:** `/home/bc-user/.claude/plans/woolly-roaming-lecun.md`
**Working directory:** `reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/`
**Branch:** `feat/infinitetalk-adaptation`
**GitHub:** `https://github.com/paulcho98/FastGen_InfiniteTalk`

---

## Status Summary (updated 2026-03-26 end of session)

| Task | Status | Notes |
|------|--------|-------|
| Task 0: Environment Setup | DONE | triton 3.2.0, no xformers |
| Task 1: Port Standalone DiT | DONE | wan_model.py (788 lines) |
| Task 2: Port Audio Modules | DONE | audio_modules.py (210 lines) |
| Task 3: LoRA Utilities | DONE | lora.py (463 lines), LoRA on base DiT + audio_cross_attn |
| Task 4: Bidirectional Wrapper | DONE | network.py (591 lines) |
| Task 5: Stage 0 Verification | DONE | BIT-EXACT on 3 real samples (max_diff=0.0) |
| Task 5b: Precompute Data | DONE | 3 TalkVid samples preprocessed |
| Task 6: Causal Wrapper | DONE | network_causal.py (1864 lines) |
| Task 7: Dataset Adapter | DONE | infinitetalk_dataloader.py (305 lines) |
| Task 9: Method Subclasses | DONE | DF, KD, SF methods |
| Task 10: Config Files | DONE | Method + experiment configs |
| Task 11: Base FastGen Mods | DONE | fake_score_net, dtype, registered classes |
| Task 12: DF E2E Test | DONE | 20 iters, 47s/iter, 67.5GB peak, single A100 |
| Task 5c: ODE Script | DONE | 3 samples extracted (~23 min/sample). Early stopping added. |

---

## Current Blocker: OOM During Training Backward Pass

### What works
- Full 14B model loads successfully with LoRA (294M trainable params)
- Forward pass through 40-block causal DiT completes on single A100
- FlexAttention compiles and runs correctly
- FSDP wrapping works (2-GPU)
- All data shapes are correct

### What doesn't work
- **Backward pass OOMs** on both single-GPU and 2-GPU FSDP
- Single A100: 78.4GB used during backward (80GB total)
- 2-GPU FSDP: 78.5GB per GPU during backward (even after sharding)

### Root cause analysis
The 14B model with 80x80x21 latents (33,600 token sequence after patching) generates massive intermediate activations during the forward pass. Even with gradient checkpointing enabled, the memory is dominated by:
1. **FSDP unsharded params** during forward (each block unsharted to full size during its forward)
2. **FlexAttention intermediate states** (attention maps for 33,600-token sequences)
3. **Audio cross-attention per block** (cross-attend to 32 context tokens per frame)

### Attempted solutions
1. expandable_segments=True: no effect
2. bf16 precision (instead of fp32 FSDP): reduced from OOM to OOM at same threshold
3. 2-GPU FSDP: sharding not effective enough (each block unsharted during compute)

### Recommended next steps (for Paul)
1. **Reduce resolution**: Try 480x480 (60x60 latent → 30x30 after patching) instead of 640x640 (80x80 → 40x40). This cuts token count by 2.25x and memory by ~4x.
2. **FSDP activation checkpointing**: Use `apply_fsdp_checkpointing()` instead of regular gradient checkpointing — this is what the built-in Wan network does.
3. **4-GPU FSDP**: Use 4 GPUs to shard params more aggressively.
4. **CPU offloading**: FSDP `cpu_offload=True` to keep optimizer states on CPU.

---

## Deviations from Plan

1. **LoRA on audio_cross_attn**: Plan said "LoRA adapters + audio modules trainable". Original implementation made all 2.4B audio_cross_attn params trainable (full fine-tune). Changed to LoRA on audio_cross_attn linear layers (40M extra LoRA params) + full fine-tune AudioProjModel (74M). Total trainable: 294M.

2. **2-GPU FSDP required**: Plan estimated Stage 1 would fit on single GPU (~23-27GB/GPU with FSDP). Actual memory usage is much higher due to sequence length (33,600 tokens) and activation storage.

3. **Audio windowing location**: Plan said windowing happens inside WanModel.forward(). Actually, 5-frame sliding window extraction happens in the inference loop BEFORE model.forward(). Our causal wrapper now applies windowing when audio is not pre-windowed (4D → 5D).

4. **Shape mismatches**: Dataset provides [B, 1, seq, dim] for text_embeds and clip_features (extra dim from .pt storage). Added squeezes in the causal forward to handle both shapes.

5. **triton version**: Downgraded from 3.6.0 to 3.2.0 for FlexAttention compatibility.

---

## Files Created/Modified (summary)

### New files (19 total, ~6000+ lines)
```
fastgen/networks/InfiniteTalk/
    __init__.py, wan_model.py, audio_modules.py, network.py, network_causal.py, lora.py
fastgen/datasets/infinitetalk_dataloader.py
fastgen/methods/infinitetalk_diffusion_forcing.py, infinitetalk_kd.py, infinitetalk_self_forcing.py
fastgen/configs/methods/config_infinitetalk_df.py, config_infinitetalk_kd.py, config_infinitetalk_sf.py
fastgen/configs/experiments/InfiniteTalk/__init__.py, config_df.py, config_kd.py, config_sf.py, config_df_test.py
scripts/precompute_infinitetalk_data.py, verify_infinitetalk_equivalence.py
```

### Modified files (5 base FastGen files)
```
fastgen/configs/config.py              — fake_score_net field
fastgen/methods/distribution_matching/dmd2.py  — fake_score_net check
fastgen/methods/__init__.py            — registered InfiniteTalk classes
fastgen/networks/noise_schedule.py     — dtype param
.gitignore                             — test data exclusion
```

### Commits (7 total on feat/infinitetalk-adaptation)
```
5efa38a fix: FSDP fully_shard for ABC compatibility + bf16 precision
4d04743 fix: LoRA on audio_cross_attn + fully_shard for FSDP
dd69b2e fix: resolve shape mismatches for DF training
281621b feat: add causal wrapper, training infrastructure, configs
5989a0d feat: Stage 0 verification PASSED — bit-exact equivalence
af50dbb feat: add data preprocessing script for TalkVid
ceaabd1 feat: add InfiniteTalkWan bidirectional wrapper
3ddbcf4 feat: add LoRA utilities for parameter-efficient training
a7e4efa feat: port InfiniteTalk DiT model and audio modules
```
