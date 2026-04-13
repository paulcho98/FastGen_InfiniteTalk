# Session Notes: Lookahead Sink + Anchor-Free SF Training

**Date:** 2026-04-13
**Branch:** `feat/infinitetalk-adaptation`
**Final commit:** `70b6c12`
**Plan:** `docs/superpowers/plans/2026-04-13-lookahead-sink-and-cache-toggles.md`

---

## Summary

Implemented four new toggleable features on the InfiniteTalk causal student (SF stage-2 distillation):

| | Flag | Semantics |
|---|---|---|
| **F1** | `lookahead_sink_enabled`, `lookahead_distance` | Sink K/V rotated at RoPE position `F_window - 1 + lookahead_distance` (future-positioned) instead of position 0 (past-positioned). Sink physically stays at the front of the cache but is interpreted as being "ahead" of the current block. |
| **F1b** | `lookahead_distance_min`, `lookahead_distance_max` | Training-time: sample a fresh distance uniformly from `[min, max]` per forward call. Gated on `self.training`. Eval/inference uses the fixed `lookahead_distance` for reproducibility. |
| **F2** | `model_sink_cache_enabled` | During inference, the cached sink K/V is computed from the student's OWN frame-0 prediction (not the anchor-overwritten reference image). Display output still shows the anchored frame 0. |
| **F3** | `skip_clean_cache_pass` | Skip the separate `t=0` cache-prefill pass; last denoise step writes K/V directly. Saves ~20% of inference forwards. |

Plus a refinement to the anchor-mode system:
- New `teacher_anchor_disabled` config — hard-disables teacher's frame-0 anchor (teacher is always in eval mode, so the existing `_anchor_eval_only` would paradoxically leave it anchoring).
- Combined with `student_anchor_eval_only=True` and `fake_score_anchor_eval_only=True`, gives a "train-time anchor-free" mode where no model anchors during training but student anchors at eval/inference.

All features landed through a 10-task TDD plan. 46 unit/integration tests passing, all mock-based (no 14B model load required). Full end-to-end runtime verification via the production inference CLI on a single real sample.

---

## Two bugs caught + fixed

### Bug 1: SF wandb callback tensor layout (5-15 KB garbage videos)

**Symptom:** SF validation videos logged to wandb were 5-15 KB and visually garbled; DF videos (same code path otherwise) were 150-800 KB and correct.

**Root cause:** `InfiniteTalkSFWandbCallback._tensor_to_wandb_video` applied a stray `t.permute(0, 2, 3, 1)` before calling `wandb.Video`, producing `[T, H, W, C]` shape. `wandb.Video` expects `[T, C, H, W]` (it reads `shape[-3]` as channels) — with the wrong layout it silently misinterprets dimensions and produces tiny corrupted MP4s.

**Fix (commit `b0e0669`):** removed the permute. Also ported the remaining DF-callback features (`_mux_video_audio` for H.264+AAC, GT upload at `on_dataloader_init_end`, GT reconstruction fallback).

### Bug 2: Lookahead trigger fired on chunk 0 (would have corrupted training)

**Symptom:** caught in code review before any training ran. Would have been very hard to diagnose post-facto (symptom: training just doesn't converge).

**Root cause:** the trigger condition `sink_tokens > 0 AND sink_tokens < k_win.shape[1]` used `sink_tokens` which is a **constant** (`self.sink_size * frame_seqlen`), not reflecting whether `k_win` actually contains a cached sink slab. On chunk 0 with `chunk_size=3, sink_size=1`, the trigger fired because `sink_tokens=4 < k_win.shape[1]=12` — but k_win on chunk 0 is just the current chunk's K, with no cached sink. The code then treated the first frame of the current chunk (the sink being generated) as a stale sink to shift, rotating it at position `F_window-1+L` while frames 1-2 of the same chunk got positions `[1, 2]`. Nonsensical relative distances → garbage gradients across every training step.

**Fix (commit `c53a8d6`):** added `query_offset_in_win > 0` as a fourth conjunct. `query_offset_in_win = new_local_start - k_win_start` is 0 iff no cached content in k_win (chunk 0), non-zero otherwise. Regression test `test_lookahead_does_not_fire_on_chunk_zero` locks this in — fails before fix (3 rope calls), passes after (2 rope calls).

---

## Design decisions with rationale

### Why sink at `F_window - 1 + lookahead_distance` (not `rest_frames - 1 + distance`)

First-pass formula was `sink_pos = rest_frames - 1 + lookahead_distance` with Q shifted to `(query_offset_in_win - sink_tokens) // frame_seqlen`. That shifted BOTH sink and Q when the flag flipped, so pretrained-DF weights would have to adapt to new Q positions, not just a new sink position.

Revised formula: `sink_pos = F_window - 1 + lookahead_distance`, rest at `[sink_frames_n, ..., F_window-1]`, Q at natural position `query_offset_in_win // frame_seqlen`. Only the sink's absolute position changes when the flag flips — rolling/Q positions and all non-sink relative distances are identical to no-lookahead case. Minimizes drift from pretrained-DF weights.

### Why F2 overrides F3 for chunk 0

F2 wants the sink K/V cache to contain the student's OWN frame-0 prediction (not reference). The K/V cached by any forward call is computed from that forward's **input**, never its output. The model's prediction only exists as OUTPUT after a forward returns; to cache it, another forward call must consume it as input. That's the separate cache pass F3 tries to skip.

Resolution: F2 forces the cache pass alive for chunk 0. For chunks > 0 (where F2 is a no-op anyway), F3 applies normally. Net cost of F2+F3 vs F3 alone: 1 extra forward per sample (chunk 0's cache pass).

### Why stochastic lookahead distance is training-only

Mirrors `_sample_attn_config` for the existing DF stochastic-attention path. Training randomness helps generalization; inference needs reproducibility. Gated on `self.training`.

### Why teacher uses `_enable_first_frame_anchor = False` not `_anchor_eval_only = True`

Teacher is always in eval mode (it's frozen, `.eval()` is called at init). The `_anchor_eval_only` flag says "anchor only when `self.training == False`" — but for the teacher, `self.training` is already False, so `_anchor_eval_only = True` would leave it anchoring. To get teacher to NEVER anchor, we use the hard-disable path `_enable_first_frame_anchor = False`.

### Why all SF variants share the `infinitetalk_sf` wandb group

Early drafts had each variant in its own group (e.g., `infinitetalk_sf_w9s1_lookahead_noanchor_f2_f3`). Fragmented the dashboard and made cross-variant comparisons hard. Consolidated to a single canonical group; run names encode the variant.

---

## Files changed

### Core implementation
- `fastgen/configs/methods/config_infinitetalk_sf.py` — added 6 new fields (4 feature flags + 2 stochastic-distance range fields + `teacher_anchor_disabled`)
- `fastgen/networks/InfiniteTalk/network_causal.py`:
  - Added `_apply_window_rope` module-level helper with lookahead logic
  - Added `_maybe_apply_first_frame_anchor` module-level helper
  - Added `apply_anchor: bool = True` kwarg to `CausalInfiniteTalkWan.forward`
  - Added lookahead/stochastic-distance params to `CausalInfiniteTalkWan.__init__` with validation
  - Added stochastic distance sampling at the top of `forward` (training-only)
  - Added default `lookahead_sink_enabled/distance` attrs on `CausalSelfAttention.__init__`
- `fastgen/methods/infinitetalk_self_forcing.py::_apply_anchor_config` — extended to stamp the 4 new F1/F2/F3 fields onto `self.net` + propagate lookahead to each `block.self_attn`. Handles `teacher_anchor_disabled` (hard-disables teacher's anchor permanently).
- `fastgen/methods/distribution_matching/causvid.py::_student_sample_loop` — rewritten to honor F2+F3 matrix with `apply_anchor` plumbing, manual display anchor when F2 active on chunk 0, debug trace gated by `LOOKAHEAD_DEBUG_TRACE=1`.
- `fastgen/callbacks/infinitetalk_sf_wandb.py` — tensor layout fix + audio muxing + GT upload port from DF callback.
- `scripts/inference/inference_causal.py` — 5 new CLI flags + F2+F3 matrix in `run_inference`'s inline AR loop + ephemeral `[sample_loop]` debug trace.

### Experiment configs (variant explosion)
- `config_sf_w9s1.py` — base w9s1 attention (local_attn_size=10, sink_size=1)
- `config_sf_w9s1_valtest.py` — validation-only variant
- `config_sf_w9s1_noanchor.py` — softboth + teacher_anchor_disabled
- `config_sf_w9s1_noanchor_valtest.py` — valtest with noanchor
- `config_sf_w9s1_lookahead.py` — w9s1 + F1 (+ opt F2/F3/F1b via env)
- `config_sf_w9s1_lookahead_valtest.py` — valtest (also disables lazy caching + points train at 2-sample list for fast init)
- `config_sf_w9s1_lookahead_noanchor.py` — all features combined (current canonical training config)

### Run scripts
- `run_sf_w9s1.sh`, `run_sf_w9s1_valtest.sh`
- `run_sf_w9s1_noanchor.sh`, `run_sf_w9s1_noanchor_valtest.sh`
- `run_sf_w9s1_lookahead.sh`, `run_sf_w9s1_lookahead_valtest.sh`
- **`run_sf_w9s1_lookahead_noanchor.sh`** ← canonical training run for the current experiment

### Tests (46 total, all mock-based, <30s)
- `test_config_fields.py` (2) — attrs field presence
- `test_apply_anchor_kwarg.py` (4) — anchor helper + kwarg plumbing
- `test_lookahead_sink_attention.py` (4) — constructor param propagation + validation
- `test_apply_window_rope.py` (8) — RoPE math, sink position formula, chunk-0 exclusion
- `test_sample_loop_toggles.py` (7) — F2+F3 matrix via MockNet
- `test_apply_attention_config.py` (4) — `_apply_anchor_config` stamping
- `test_lookahead_pipeline_integration.py` (3) — end-to-end log parsing
- `test_inference_causal_toggles.py` (4) — inference CLI F2+F3
- `test_stochastic_lookahead_distance.py` (10) — sampling behavior + validation

### Data
- `data/precomputed_talkvid/val_quarter_2.txt` — 2-sample val list for fast valtests

### Docs
- `docs/superpowers/plans/2026-04-13-lookahead-sink-and-cache-toggles.md` — full TDD plan (10 tasks)
- `CLAUDE.md` — repo-level codebase notes (patterns, footguns, quick diagnostics)
- `docs/2026-04-13-lookahead-anchor-free-session.md` — THIS DOCUMENT

### Verification artifacts
- `scripts/verify_lookahead_logic.py` — standalone runnable, exits 0/1, prints full debug trace across all F2/F3 combinations (mock-based, runs in seconds)

---

## Running the canonical training experiment

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
bash scripts/run_sf_w9s1_lookahead_noanchor.sh
```

**Baked-in defaults:**
- w9s1 attention (`local_attn_size=10, sink_size=1`)
- Lookahead sink ENABLED, fixed `lookahead_distance=4` for eval
- Stochastic distance sampling during training: `[1, 5]` inclusive per forward
- F2 (model-generated sink cache): ON
- F3 (skip clean cache pass): ON
- Anchor (training): OFF on student, teacher, fake_score
- Anchor (eval/inference): ON for student only
- Student init: stochastic DF checkpoint `quarter_stoch_r128_bs4_accum1_8gpu_0409_2248/checkpoints/0009500.pth`
- Dynamic RoPE: ON (required by lookahead)

**Expected startup logs** (the critical verification that everything's wired):
```
[anchor] Student: eval-only (no anchor during training rollout)
[anchor] Fake score: eval-only (tracks student distribution)
[anchor] Teacher: DISABLED (anchor-free target distribution)
[attn] Lookahead sink ENABLED, stochastic distance in [1, 5] (eval uses fixed=4)
[attn] Model-generated sink cache: ENABLED (F2)
[attn] Skip clean cache pass: ENABLED (F3)
```

**Wandb:** group `infinitetalk_sf`, run name `sf_w9s1_la1-5_noanchor_f2_f3_freq5_lr1e5_<timestamp>`.

**Overrides** (all env-var based):
- `LOOKAHEAD_DISTANCE=6` — change eval-time fixed distance
- `LOOKAHEAD_DISTANCE_MIN=0 LOOKAHEAD_DISTANCE_MAX=0` — disable stochastic sampling (deterministic training at fixed distance)
- `MODEL_SINK_CACHE=` — disable F2
- `SKIP_CLEAN_CACHE=` — disable F3
- `INFINITETALK_DF_CKPT=/path` — override the auto-detected student init
- `INFINITETALK_TRAIN_LIST=/path` — override the train list

---

## Known caveats / unresolved

### Valtest hangs at GT upload (WIP)

`scripts/run_sf_w9s1_lookahead_valtest.sh` reliably initializes the 3× 14B FSDP stack and reaches `on_dataloader_init_end`'s `Uploading GT validation videos to wandb...` log line — then GPUs sit at 100% for 15+ min with no further log output. Exact cause not diagnosed; possible culprits:
- Rank-0 VAE decode + wandb upload of 2 videos shouldn't take this long
- Possible buffering issue in loguru-through-torchrun stdout
- Possible deadlock in the SF callback's `synchronize()` after rank-0 work

**Workaround for now:** skip valtest for end-to-end verification. The production inference CLI (`inference_causal.py`) is faster, uses a single model (no FSDP teacher+fake_score), and was successfully used to verify the runtime forward path under real data:
- Output: `/tmp/inference_lookahead_smoke.mp4` (1.2 MB, 7 chunks, 21 frames)
- 1200 `[lookahead]` log lines parsed — RoPE math verified against formula at F_window ∈ {6, 9, 10}:
  - F=6 → sink_pos=9 = 5+4 ✓
  - F=9 → sink_pos=12 = 8+4 ✓
  - F=10 → sink_pos=13 = 9+4 ✓ (steady-state when window saturates at L=10)
- 35 `[sample_loop]` log lines match the expected F2+F3 pattern across all 7 chunks

### Dataloader init NFS scan

`InfiniteTalkDataLoader.__init__` runs `torch.load(vae_latents.pt)` for every sample as a shape check. With the full train list (147k samples) on NFS, this is 20+ min. Two known mitigations:
1. Warm NFS cache before training (run the script once, kill after init, rerun — hot cache makes the second init ~5 min)
2. For valtest only: point train list at `val_quarter_2.txt` AND set `raw_data_root=None` on the train dataloader (disables lazy caching encoder loads — VAE+CLIP+wav2vec on every rank)

### Ephemeral debug traces to remove

`[sample_loop]` trace lines in `_student_sample_loop`, `inference_causal.py::run_inference` are gated by `LOOKAHEAD_DEBUG_TRACE=1` and were added specifically to verify the F2+F3 matrix at runtime. Task 10 of the plan is to remove them once training verifies the feature in practice. The `[lookahead]` trace in `_apply_window_rope` is permanent (useful long-term debug tool).

### What hasn't been directly tested

- End-to-end FSDP SF training with the full pipeline → valtest hanging is a blocker. Inference CLI is a strong proxy since it shares all the helpers.
- Long-video inference via `generator_fn_extrapolation` — intentionally skipped (not used by InfiniteTalk; only by `video_model_inference.py`).
- Multi-GPU `inference_causal.py` with FSDP — the CLI loads a single-GPU model, so FSDP interactions during attention forward aren't exercised here. SF training does exercise FSDP and reached the validation phase successfully (just hung on GT upload, not on the attention forward itself).

---

## Timeline of the session

1. Diagnosed the SF wandb callback bug (DF produces 150-800KB videos, SF produces 5-15KB videos) — traced to wrong tensor layout at `wandb.Video` boundary.
2. Added `teacher_anchor_disabled` flag for the "train-time anchor-free on all three models" mode.
3. Wrote the lookahead plan (`docs/superpowers/plans/2026-04-13-lookahead-sink-and-cache-toggles.md`), 10 tasks, full TDD.
4. Executed tasks 1-7 via subagent-driven development. Two-stage review per task (spec compliance + code quality). Opus reviewer caught the critical chunk-0 bug in Task 4.
5. Implemented Task 6b (`inference_causal.py` CLI plumbing) out of order after it became clear the valtest was not a viable end-to-end verification path.
6. Ran inference smoke test — full pass, verified RoPE formula with real tensors across F_window ∈ {6, 9, 10}.
7. Added F1b (stochastic lookahead distance sampling per forward).
8. Consolidated wandb group to `infinitetalk_sf` so all SF experiments share a dashboard.
9. Pointed SF training to the stochastic DF checkpoint (which has 20% exposure to `local_attn_size=10, sink_size=1` — ideal init for w9s1 SF).
10. Committed + pushed all work.

Total commits on this branch for this session: ~18. See `git log --oneline feat/infinitetalk-adaptation` for the full series.
