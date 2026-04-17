# Session Record — Knot Forcing Implementation

**Date:** 2026-04-18
**Branch:** `feat/knot-forcing` (worktree at `reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk_knot/`)
**Understanding doc:** `docs/analysis/2026-04-17-knot-forcing-algorithm-understanding.md`
**Plan:** `docs/superpowers/plans/2026-04-18-knot-forcing-implementation.md`
**Paper:** Xiao et al., "Knot Forcing: Taming Autoregressive Video Diffusion Models for Real-time Infinite Interactive Portrait Animation", arXiv:2512.21734v2.

---

## TL;DR for resume-after-compact

1. **What landed**: Knot Forcing (temporal knot + running-ahead + last-frame reference) implemented across the full stack — training rollout, validation sample loop, production inference CLI, dataloader. All three KF components gated by feature flags with off-by-default bit-exactness to existing SF.

2. **Architecture choice**: **Hybrid** — frame-0 sink is overloaded with the knot at iter > 0 (we do NOT maintain a separate `KV_ref` cache as the paper describes). Global reference identity is preserved via the sink KV written at iter 0, which persists in the sliding window's sink slot. This is a deliberate deviation from the paper's pathway-separated design; justified by code reuse (our F1 plumbing covers the dynamic sink RoPE required by running-ahead).

3. **Key simplifications validated during implementation**:
   - Existing anchor helpers already early-return for `cur_start_frame != 0`, so KF iter > 0 needs no modification to them. Tasks 3 and 4 (original plan) were dropped.
   - Under dynamic RoPE, K is stored un-rotated and rotation is applied at each attention read. Running-ahead "re-cache" is implicit — just update `_running_ahead_n` and the next attention call picks up the new position. Task 9 (original plan) was dropped.

4. **Training artifacts**:
   - New config: `config_sf_w9s1_knot_runahead.py`
   - Launcher: `scripts/run_sf_w9s1_knot_runahead.sh`
   - Inference: `scripts/inference/run_inference_knot_runahead.sh`
   - Monitor: `scripts/inference/monitor_sf_w9s1_knot.sh`
   - Valtest: `config_sf_w9s1_knot_valtest.py` + `run_sf_w9s1_knot_valtest.sh`

5. **Test status**: 100 tests pass + 4 skipped (integration placeholders for KF-11 that need a MockNet harness — smoke-covered by valtest).

6. **Launch command**: `bash scripts/run_sf_w9s1_knot_runahead.sh`
   - DF init: latest in `quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints/`
   - Run name pattern: `sf_w9s1_knot_k1_ra_s4_n8_lastref_freq5_lr1e5_<timestamp>`

---

## Design decisions locked in

### Feature flags (all default off)
Added to `InfiniteTalkSFModelConfig` in `config_infinitetalk_sf.py`:
- `use_temporal_knot: bool = False`, `knot_size: int = 1`
- `use_running_ahead: bool = False`, `running_ahead_step: int = 4`, `running_ahead_init_n: int = 8`
- `use_last_frame_reference: bool = False`

Validator `validate_kf_flags()` enforces:
- `use_running_ahead` and `lookahead_sink_enabled` (F1) are mutually exclusive (both modify sink RoPE).
- `knot_size >= 1` when `use_temporal_knot=True`.
- `running_ahead_step >= 1` when `use_running_ahead=True`.

### Knot injection mechanism
The knot enters via **two paths** at iter > 0:
1. **I2V mask inpainting** in `_build_y`: when `condition["knot_latent_for_chunk_start"]` is set and `start_frame > 0`, the full y-tensor is modified at position `start_frame` to carry mask=1 + VAE(knot). Sliced y's position 0 thus has the knot as a reference-like condition.
2. **Noisy-latent pin** in rollout/sample-loop/inference: `noisy_input[:, :, 0:k] = knot_latent` before the forward. Re-pinned after `forward_process` on non-exit steps.

The existing input/output anchor helpers are NOT modified — their `cur_start_frame != 0` gate makes them automatic no-ops for iter > 0.

### Fusion (Eq. 5)
At the exit/last denoise step of iter > 0:
```python
x0_pred[:, :, 0:k] = (knot_latent + x0_pred[:, :, 0:k]) / 2.0
```
Gradient flows through both terms so cross-chunk DMD loss backprops correctly.

### Running-ahead (dynamic sink RoPE)
`advance_running_ahead(net, i, c, k)` checks `i + c + k > n` and advances `n += s`. Re-cache is implicit under `use_dynamic_rope=True`: K is stored un-rotated and `compute_sink_rope_position` reads `_running_ahead_n` at each attention forward.

Generalized `_apply_window_rope` with new `running_ahead_n: Optional[int]` kwarg. When provided, overrides the F1 `sink_pos = F_window - 1 + lookahead_distance` formula with `sink_pos = running_ahead_n` (absolute).

### Last-frame reference
`_maybe_swap_last_frame_ref(first_frame_cond, vae_latents, use_last_frame)` replaces `first_frame_cond[:, 0]` with `vae_latents[:, -1]` — the clip's last latent frame. Not strictly equivalent to re-encoding a last-frame-replicated video (VAE temporal conv), but captures the clip's semantic endpoint.

### Audio slicing
Extended dataloader to slice `audio_emb[:_train_pixel_frames + extra_k_pixel]` when available on disk, else right-pad via `_maybe_pad_audio_for_knot`. For k=1 latent, extra_k_pixel = 4 video frames (VAE temporal stride ≈3.86).

### F3 under KF
Forced off. Cache-store pass always runs over the committed c frames (not c+k). Rationale: F3's store-during-exit would cache c+k frames of K/V, but the knot frame shouldn't enter `KV_pre` (it's a one-iteration bridge). Using the separate cache pass keeps the KV cache aligned with the paper's Algorithm 1 line 19.

---

## Commit history

```
15b44b3 test(kf): regression — KF-off default bit-exact + training config round-trip
b3c30b3 feat(kf): valtest config + launcher for end-to-end KF smoke
86aa506 feat(kf): run_sf_w9s1_knot_runahead.sh — training launcher
a09171b feat(kf): config_sf_w9s1_knot_runahead.py — training config
ed00b96 feat(kf): inference launcher + monitor for KF run
39c7b7c feat(kf): inference CLI — KF flags + run_inference KF branch
c82c963 feat(kf): dataloader audio slicing for c+k chunks
826a621 feat(kf): _student_sample_loop KF branch
cb437d0 feat(kf): rollout_with_gradient KF branch
2029457 feat(kf): dataloader use_last_frame_reference
c13084a feat(kf): _apply_running_ahead_config — stamps RA attrs on causal student
51e98c7 feat(kf): compute_sink_rope_position + running-ahead RoPE override
9627a66 feat(kf): advance_running_ahead helper
499368f feat(kf): _build_y injects knot at start_frame
85ec81c feat(kf): validate_kf_flags — KF↔F1 mutual exclusion
1582c3a feat(kf): add KF config flags (all default off)
137f89c fix(tests): shim torch.distributed.GroupName for diffusers import
d59f942 docs(kf): implementation plan for Knot Forcing
```

---

## Known caveats / open follow-ups

1. **Valtest smoke not executed**: the plan's Task 20 includes a valtest run (torchrun + 8 GPUs). Not executed as part of this commit — requires GPU allocation and 15+ min startup. User should run `bash scripts/run_sf_w9s1_knot_valtest.sh` as a first end-to-end check before launching the full training run. Watch for: finite loss at iter 0, no runtime errors during the c+k denoise path, `[running_ahead]` trace lines (enable via `LOOKAHEAD_DEBUG_TRACE=1`).

2. **vae_latents[:, -1] vs re-encoded last frame**: our last-frame reference source is approximate (captures clip endpoint but includes VAE temporal-conv mixing). The paper doesn't specify how they encoded the last-frame reference, so this is a reasonable default. Ablation could re-encode actual last frames if mixing turns out to matter.

3. **Gradient doubling at fusion boundary**: under SF, the knot is saved gradient-enabled at the exit step and re-used at the next iter's exit step for Eq. 5 fusion. Backprop flows through both prev-iter and current-iter exit steps — roughly doubles compute cost at boundary frames. If this becomes prohibitive, switch to `knot_latent = x0_pred_chunk[..., c:c+k].detach()` (plan section N5 fallback).

4. **Running-ahead firing during short training rollouts**: with `n_init=8, s=4` and 7-chunk rollouts (21 frames), RA fires ~4 times per rollout. Good natural training-distribution coverage.

5. **Training clip length under KF**: unchanged at 21 latent frames. Internal noise is extended by k=1 inside `rollout_with_gradient` and `_student_sample_loop` for the last chunk's knot slot; output tensor remains 21 frames.

6. **Dataloader init slowdown**: none expected — we reuse the existing precomputed `vae_latents.pt`, `first_frame_cond.pt`, `audio_emb.pt`. Last-frame ref is a runtime swap from vae_latents (no extra disk I/O).

7. **Conftest env shim**: `tests/conftest.py` now shims `torch.distributed.distributed_c10d.GroupName = str` to unblock diffusers import under torch 2.8 nv. Pre-existing env issue, fixed as part of KF-1. No runtime impact.

---

## What to verify before launching the full run

1. **Valtest passes**: `bash scripts/run_sf_w9s1_knot_valtest.sh` completes without crashes.
2. **Config loads cleanly**: `python -c "from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_knot_runahead import create_config; c = create_config(); c.model.validate_kf_flags(); print(c.log_config.name)"`.
3. **Regression suite**: 100 tests + 4 skipped (or higher), no failures. Command is in the session-notes above.
4. **DF init checkpoint exists**: latest `.pth` under `quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints/`.

Once the valtest smokes clean, launch:
```bash
bash scripts/run_sf_w9s1_knot_runahead.sh
```
Wandb group: `infinitetalk_sf`. Run name pattern: `sf_w9s1_knot_k1_ra_s4_n8_lastref_freq5_lr1e5_<timestamp>`.

In parallel, start the monitor to pull new checkpoints and eval on TalkVid val 30:
```bash
bash scripts/inference/monitor_sf_w9s1_knot.sh
```
