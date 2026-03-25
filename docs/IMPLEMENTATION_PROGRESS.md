# InfiniteTalk FastGen Adaptation — Implementation Progress

**Started:** 2026-03-25
**Plan:** `/home/bc-user/.claude/plans/woolly-roaming-lecun.md`
**Working directory:** `reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/`
**Branch:** `feat/infinitetalk-adaptation`
**GitHub:** `https://github.com/paulcho98/FastGen_InfiniteTalk`

---

## Status Summary (updated 2026-03-26)

| Task | Status | Notes |
|------|--------|-------|
| Task 0: Environment Setup | DONE | Installed deps. xformers breaks torch; using SDPA. triton 3.2.0 for FlexAttention. |
| Task 1: Port Standalone DiT | DONE | 788 lines `wan_model.py` |
| Task 2: Port Audio Modules | DONE | 210 lines `audio_modules.py` |
| Task 3: LoRA Utilities | DONE | 463 lines `lora.py` — functional tests pass |
| Task 4: Bidirectional Wrapper | DONE | 591 lines `network.py` — InfiniteTalkWan(FastGenNetwork) |
| Task 5: Stage 0 Verification | DONE | Level 1: PASS (0 diff), Level 2: PASS (0 diff on 3 real samples) |
| Task 5b: Precompute TalkVid Data | DONE | 3 test samples precomputed (~10s/sample) |
| Task 5c: ODE Trajectory Script | Not started | |
| Task 6: Causal Wrapper | DONE | 1864 lines `network_causal.py` — FlexAttention, KV cache, LoRA |
| Task 7: Dataset Adapter | DONE | 305 lines `infinitetalk_dataloader.py` |
| Task 9: Method Subclasses | DONE | DF, KD, SF methods |
| Task 10: Config Files | DONE | Method + experiment configs |
| Task 11: Base FastGen Mods | DONE | fake_score_net, dtype, registered classes |
| Task 12: DF E2E Test | **IN PROGRESS** | Forward pass works. OOM on backward (single GPU). Testing with expandable_segments. |

---

## Detailed Log

### 2026-03-25 — Session 1

**Completed:** Tasks 0-4, 5, 5b (environment, DiT port, audio modules, LoRA, bidirectional wrapper, Stage 0 verification, data precomputation)

**Key results:**
- Stage 0 verification: BIT-EXACT equivalence (max_diff = 0.0) on all 3 real TalkVid samples
- Data precomputation: 3 TalkVid samples preprocessed with correct shapes

### 2026-03-26 — Session 2 (current)

**Completed:** Tasks 6, 7, 9, 10, 11 (causal wrapper, dataset, methods, configs, base mods)

**Task 12 (DF E2E): In progress — multiple fixes applied:**

1. **Dataloader fix:** `create_infinitetalk_dataloader()` needs `data_list_path` not `data_root`
2. **text_embeds shape:** Dataset returns `[B, 1, 512, 4096]` after collation → squeezed to `[B, 512, 4096]`
3. **clip_features shape:** Dataset returns `[B, 1, 257, 1280]` after collation → squeezed to `[B, 257, 1280]`
4. **Audio windowing:** Dataset provides raw `[B, 81, 12, 768]`; added 5-frame sliding window in causal forward to get `[B, 81, 5, 12, 768]`
5. **Audio reshape:** `SingleStreamAttention` now handles 4D `[B, N_t, N_a, C]` by reshaping to `[B*N_t, N_a, C]`
6. **triton version:** Installed triton 3.2.0 (from 3.6.0) to fix `triton_key` import error with FlexAttention
7. **Memory:** Forward pass uses ~73GB on A100-80GB. Backward OOMs at ~79GB. Testing `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Deviation from plan:** The plan assumed DF training would fit on a single A100-80GB. The 14B model with LoRA uses ~73GB for forward alone, leaving insufficient headroom for backward. Options:
- `expandable_segments=True` (testing now)
- FSDP across 2 GPUs
- Reduce resolution (smaller latents)
- Reduce sequence length (fewer frames)

---

## Where We Are When Session Ended

**All code is written and committed.** The full pipeline compiles and runs:
- Model loads 14B weights + LoRA adapters ✓
- Dataset loads real precomputed TalkVid data ✓
- Forward pass through 40-block causal DiT with FlexAttention ✓
- FlexAttention triton kernel compilation succeeds ✓
- Backward pass OOMs on single A100-80GB — needs FSDP or memory optimization

**Next steps for you:**
1. Check if the `expandable_segments` run succeeded (or switch to 2-GPU FSDP)
2. If FSDP needed, enable with `config.trainer.fsdp = True` in the test config
3. Once DF training completes 20 iters, check loss values
4. Proceed to ODE extraction script and Self-Forcing Stage 2
