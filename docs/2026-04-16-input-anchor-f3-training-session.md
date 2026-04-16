# Session Record — Input Anchor Fix + F3-in-Training Plumbing

**Date:** 2026-04-15 through 2026-04-16
**Branch:** `feat/infinitetalk-adaptation`
**Active run at end of session:** `FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/sf_w9s1_la4_f3_anchor_freq5_lr1e5_0415_2326/`
**Launch script:** `scripts/run_sf_w9s1_lookahead_f3.sh`

---

## TL;DR for resume-after-compact

1. **Root problem:** SF training in the anchor-free regime (`config_sf_w9s1_lookahead_noanchor.py`) was producing garbage at frame 0 of validation videos. Verified empirically that the InfiniteTalk teacher itself produces garbage at frame 0 without the inference-time overwrite (via `batch_inference.py --disable_i2v_overwrite`). Inferred: InfiniteTalk was trained with clean `x[:, :, 0]` at every timestep (Option A training regime). The anchor-free SF training was asking the student to match a teacher that can't produce meaningful frame-0 targets without the overwrite.

2. **Fix:** added a new input-side anchor (`_maybe_apply_input_anchor` / `_maybe_apply_input_anchor_bidir`) that pins `x_t[:, :, 0:1] = first_frame_cond[:, :, 0:1]` at the top of every `forward()` call on both the causal student and the bidirectional teacher/fake. Complements (doesn't replace) the existing output-side anchor. Default `apply_input_anchor=True` on every forward; respects the same `_enable_first_frame_anchor` / `_anchor_eval_only` / `teacher_anchor_disabled` attribute semantics as the output anchor.

3. **F3 follow-up:** plumbed the existing `_skip_clean_cache_pass` (F3) feature into training (`rollout_with_gradient`) — previously it only applied to validation/inference. When on, the exit step sets `store_kv=True` and the separate clean-cache pass is skipped. Default off → behavior unchanged. **Autograd verified to survive the `store_kv=True` + gradient-enabled forward combination.**

4. **F2 deliberately NOT added to training.** Reasoning documented below. Short version: with anchors-on defaults, F2's benefit is near-zero, and propagating its complexity into `rollout_with_gradient` wasn't worth it right now.

5. **Current run:** `run_sf_w9s1_lookahead_f3.sh` → `config_sf_w9s1_lookahead_f3.py`. Bakes F1 (lookahead, dist=4), F3 on, F2 off, all anchors on. Launched 2026-04-15 23:26.

---

## Background — what went wrong before this session

User was running `config_sf_w9s1_lookahead_noanchor.py` with:

- `student_anchor_eval_only = True` (no output anchor on student during training)
- `fake_score_anchor_eval_only = True`
- `teacher_anchor_disabled = True` (teacher never anchors)
- F1 (lookahead) on, F2 on, F3 on

Validation videos had garbage at frame 0 that propagated to adjacent frames.

The debugging path earlier in the session:

1. Read through `InfiniteTalk/wan/multitalk.py` end-to-end. Established that InfiniteTalk's native inference does two overwrites per denoising step: **L711 (pre-model)** and **L773 (post-step)** — both forcing `latent[:, :cur_motion_frames_latent_num] = latent_motion_frames`. These are *input-side* hard overwrites, not output-side pins.

2. Initially hypothesized that training used noisy `x[:, :, 0]` and the inference overwrite was cosmetic. User pushed back. Ran `batch_inference.py --disable_i2v_overwrite` on the InfiniteTalk teacher — **produced utter garbage at frame 0, recovering over subsequent frames**. This is the diagnostic signature of "model has no trained signal for frame-0 velocity from noise."

3. Concluded: InfiniteTalk was trained with Option A (clean `x[:, :, 0]` at all timesteps). The teacher literally cannot denoise frame 0 from noise. Any training setup that removes the frame-0 pin is training the student against a broken VSD target at frame 0.

4. FastGen's `_maybe_apply_first_frame_anchor` (introduced in commit `66defb9`, refined in `60b62ca`) had been doing **output-side** pinning (`out[:, :, 0:1] = ffc[:, :, 0:1]`). This makes the student's output at frame 0 correct, but leaves the teacher/fake's *input* at frame 0 as noisy/drifted, putting them out-of-distribution relative to how they were trained (clean x at frame 0 always). Output-side alone is a partial substitute — the internal attention pattern still routes through an OOD frame 0.

The plan: add an input-side anchor that mirrors the output-side one, so both teacher input distribution and student I/O end up matching training.

---

## The main plan — 7 tasks

Plan file: `docs/superpowers/plans/2026-04-15-input-anchor-and-full-inference-convention.md`
Executed via superpowers:subagent-driven-development with two-stage review per task (spec compliance + code quality).

| Task | What | Commit |
|------|------|--------|
| 1 | `_maybe_apply_input_anchor` helper in `network_causal.py` + 6 unit tests | `da67ee0` |
| 2 | `apply_input_anchor: bool = True` kwarg on `CausalInfiniteTalkWan.forward`. First attempt introduced a scope-violation by changing `self._forward_ar` → `CausalInfiniteTalkWan._forward_ar(self, ...)` to make a `monkeypatch.setattr` test work. Spec reviewer caught it. | `31077d3` (initial), `2492276` (fix) |
| 3 | `_maybe_apply_input_anchor_bidir` helper + `apply_input_anchor` kwarg on bidirectional `InfiniteTalkWan.forward` (used by teacher and fake_score) | `b44f9a3` |
| 4 | In `CausVidModel._student_sample_loop`: propagate `apply_input_anchor=apply_anchor_here` on denoise forwards (F2-aware — moves in lockstep with output anchor); same on cache-store forward; add explicit post-`forward_process` pin on chunk 0 to match InfiniteTalk's `multitalk.py:773` convention | `df4ba6d` |
| 5 | Same three edits applied to `scripts/inference/inference_causal.py::run_inference` (production AR loop). This is the CLI path that `inference_causal.py` uses. | `83624b5` |
| 6 | Gradient-sanity tests for the new helpers. Original plan test targeted `_student_sample_loop` with backprop — failed because that loop uses an in-place `x[:, :, start:end] = x_next` pattern that's pre-existing and gradient-unsafe (it's only used under `torch.no_grad()` in production). Replaced with 6 tests targeting the helpers directly to verify the clone+slice-assign pattern routes gradients correctly. | `073952a` |
| 7 | `CLAUDE.md` documentation section | `e683117` |

Also committed during the session:
- `c2a4317` — bump `fake_score` LR to 4e-6 in the `_noanchor` config (pre-existing uncommitted work, separate of the plan)
- `7ba53de` — `--anchor_modes` CLI flag + FSDP-sidecar guard in `inference_causal.py` (pre-existing uncommitted work)

### Key design decisions in the main plan

- **Keep both anchors (input + output) — don't remove output.** Output-side anchor already existed, was used by F2's `apply_anchor=False` paths. Removing it would require rewriting F2. Adding input-side alongside is minimally invasive. They're cheap, redundant-but-safe when both True. F2 disables both in lockstep (see below).
- **`apply_input_anchor` default True, same attribute semantics as output anchor.** `_enable_first_frame_anchor=False` hard-disables (used when `teacher_anchor_disabled=True`). `_anchor_eval_only=True` disables during training. Causal helper additionally gates on `cur_start_frame == 0`; bidirectional is unconditional.
- **Post-step pin only on validation/inference paths, not training rollout.** `SelfForcingModel.rollout_with_gradient` doesn't need post-step pin because the next step's input anchor covers the same work, and rollout uses a gradient-safe list-and-cat pattern (vs sample_loop's in-place writeback).

### The subtle ripple effect in the cache pass

Before this plan, the non-F2 cache pass with `context_noise > 0` produced K/V at position 0 from a noised input. With input anchor now pinning `x_t[:, :, 0:1]` at the top of the cache-store forward, position 0's K/V now comes from **clean** `first_frame_cond` regardless of `context_noise`. This is a subtle but correct behavior change — aligns with training distribution where frame 0 was always clean.

---

## Task 8 (off-plan) — F3 in training

**Commit:** `c436033`
**File:** `fastgen/methods/distribution_matching/self_forcing.py::rollout_with_gradient`
**Test:** `tests/test_rollout_f3_toggle.py` (5 tests)

User wanted F3 (skip clean cache pass) to also apply during training, not just validation/inference. The pre-existing F3 feature (`60b62ca`, `d5ff7ac`) was deliberately scoped to inference/validation — `rollout_with_gradient` had no F3 plumbing.

### What F3 does

At inference under F3: the separate cache-store forward at the end of a chunk is skipped; instead, the last denoise step has `store_kv=True` and writes KV during its forward. Saves one forward per chunk (~25% chunk-wise compute). The cached KV is at the last denoise step's input noise level (`t_list[-2] = 0.700` for the SF 4-step schedule).

### The stochastic-exit wrinkle in training

`rollout_with_gradient` has a **stochastic exit step**: `_sample_denoising_end_steps` samples an exit index in `[0, num_denoise_steps)`, and the loop `break`s there. The `x0_pred_chunk` appended to `denoised_blocks` is the exit step's output, not a fully denoised output. The separate cache pass then runs at `t=0` (or `context_noise`) over `x0_pred_chunk`, producing a consistent cache distribution regardless of exit step.

With F3 in training, "move store_kv to the last step" is ambiguous: last step of what, the loop (fixed = step N-1, requires completing denoising) or the exit step (stochastic)?

### The three options evaluated

| Option | Where KV is stored under F3 | Training cache noise level | Train/infer cache match |
|--------|------------------------------|---------------------------|--------------------------|
| **A** | Exit step (stochastic) | `t = t_list[exit_step]` (varies among 0.999, 0.955, 0.875, 0.700) | Partial — content matched, timestep varies |
| B | Last step of loop after no_grad completion through remaining steps | `t = t_list[-2] = 0.700` (fixed) | Exact distribution match, but loses F3's speed benefit |
| C | Keep separate cache pass, bump `context_noise` to 0.7 | `t = 0.7` (fixed, via context_noise) | Exact distribution match, zero code change, not actually F3 |

**User chose Option A.** Justification: simplest code change; the cache *content* at position 0 is the clean first-frame latent in both train and infer (because input anchor pins it on every forward); only the timestep embedding baked into K/V varies; residual mismatch is inherent to SF's stochastic-exit design.

### What landed

```python
# self_forcing.py::rollout_with_gradient — additions only
skip_clean_cache = getattr(self.net, "_skip_clean_cache_pass", False)
# ...
else:  # exit step branch
    with torch.set_grad_enabled(enable_grad):
        x0_pred_chunk = self.net(
            noisy_input, t_chunk_cur, ...,
            store_kv=skip_clean_cache,   # ← F3 stores here when on
            ...
        )
    break

# Separate cache pass — gated by F3
if not skip_clean_cache:
    with torch.no_grad():
        # ...unchanged...
```

Default (`SKIP_CLEAN_CACHE` unset): `skip_clean_cache=False` → behavior bit-identical to pre-commit.

### F2 deliberately NOT added to training

Multiple reasons:

1. **F2's design premise (drifted frame-0 input → model's raw prediction cached as sink) doesn't hold under anchors-on.** With input anchor on, the model always sees clean frame 0 regardless of F2. Its "raw prediction" with clean input is approximately `clean_ref` anyway (model trained to output velocity ≈ 0 at frame 0 when input is clean). F2 in anchors-on mode is near-no-op.

2. **F2 was originally designed against the anchors-off regime**, where drifted inputs produce meaningfully different "raw predictions." That regime is now considered broken (produced the garbage frame 0), so F2's original use case is gone.

3. **Sink K/V under F2-off + anchors-on is exactly what we want**: position 0 = `projection(clean first_frame_cond)`. The lookahead mechanism (F1) rotates this clean-ref K to a "future" RoPE position, giving the model a stable identity anchor in its attention's future-slot. This is a strictly better target than F2's "model's raw prediction" in the anchors-on regime.

4. **Adding F2 to training would require**: per-block `f2_active = model_sink_cache and cur_start_frame == 0`, per-exit-step `apply_anchor_here = not f2_active` + `apply_input_anchor_here = apply_anchor_here`, per-cache-pass `apply_anchor_cache = not f2_active` + mirror. Non-trivial change for a feature that's near-no-op in the anchors-on regime we're actually running.

**If F2 is ever wanted in training later**: the path is clear — mirror the sample_loop's F2-aware gating into `rollout_with_gradient`. Pattern is already in `causvid.py::_student_sample_loop` and `inference_causal.py::run_inference`.

---

## Interaction matrix — who affects what

### Sink K/V (position 0 in the KV cache subsequent chunks attend to)

| Config | Sink K/V content | Sink K/V timestep embedding |
|--------|------------------|------------------------------|
| Anchors on, F2 off, F3 off (default training pre-Task 8) | `proj(clean_ref)` via input anchor | `t=0` (separate cache pass with context_noise=0) |
| Anchors on, F2 off, F3 on (current run) | `proj(clean_ref)` via input anchor | `t=t_list[exit_step]` (stochastic in training) / `t=t_list[-2]=0.7` (inference) |
| Anchors off, F2 on, F3 on (previous experiment, produced garbage) | `proj(model's drifted frame-0 prediction)` (F2) | `t=t_list[-2]` |

### Teacher/fake input at frame 0 (in DMD2's `_student_update_step`)

Regardless of F2/F3: teacher/fake forward has `apply_input_anchor=True` default from Task 3 plumbing. At the top of their forward, `x_t[:, :, 0:1]` is pinned to `first_frame_cond[:, :, 0:1]`. The `forward_process` that noises `gen_data` into `perturbed_data` noises frame 0 too, but the teacher/fake input anchor overrides that immediately inside their forward. Frame 0 of teacher input = clean ref, always.

### `context_noise` interaction

`context_noise` noises the cache-pass input before the forward. With input anchor on, the anchor immediately re-pins frame 0 to clean, so `context_noise` only affects frames 1+. Under F3 on (no separate cache pass), `context_noise` does nothing — the cache is populated on the exit step which reads `noisy_input` at `t=t_list[exit_step]` directly. Not a bug — `context_noise` and F3 are parallel mechanisms for injecting KV noise; use one or the other.

### Gradient flow under F3 on

Verified empirically in `test_rollout_f3_toggle.py::test_rollout_f3_on_gradient_flows`: combining `store_kv=True` with a gradient-enabled forward does NOT break autograd. The KV cache buffers are treated as detached side-effect state, not part of the autograd graph. Gradient flows normally to the learnable parameters via the returned `x0_pred_chunk`.

This was untested territory in the codebase (all prior `store_kv=True` calls were under `torch.no_grad()`). Confirmed safe before shipping.

---

## Current state

### Commit history since session start

```
bc7aaba feat: w9s1 + lookahead + F3 baked config/scripts for training and inference
c436033 feat: F3 (skip_clean_cache_pass) plumbed into rollout_with_gradient
7ba53de feat: --anchor_modes CLI + FSDP-sidecar guard in inference_causal.py      (pre-existing)
c2a4317 feat: bump fake_score lr to 4e-6 in lookahead-noanchor config               (pre-existing)
e683117 docs: CLAUDE.md — document input-anchor + F2 interaction
073952a test: gradient-sanity tests for input-anchor helpers
83624b5 feat: full InfiniteTalk inference convention in inference_causal.py
df4ba6d feat: full InfiniteTalk inference convention in _student_sample_loop
b44f9a3 feat: add apply_input_anchor kwarg to InfiniteTalkWan.forward (bidirectional)
2492276 fix: revert scope-creep _forward_ar dispatch; instance-level test mock
31077d3 feat: add apply_input_anchor kwarg to CausalInfiniteTalkWan.forward
da67ee0 feat: add _maybe_apply_input_anchor helper for input-side frame-0 pinning
```

(`38384a9` was the pre-session HEAD.)

### Test suite

44 tests pass across all plan-relevant suites: `test_input_anchor_plumbing.py` (9), `test_post_step_pin_inference.py` (4), `test_rollout_gradient_sanity.py` (6), `test_rollout_f3_toggle.py` (5), `test_sample_loop_toggles.py` (7), `test_inference_causal_toggles.py` (5), `test_apply_anchor_kwarg.py` (4), `test_apply_attention_config.py` (4). Zero regressions from pre-session state. Full suite (`pytest tests/`) has 4 pre-existing failures unrelated to this work: 2 gated HF repos, 1 test-code bug, 1 flaky under concurrent pytest.

### The running job

**Launch:** `bash scripts/run_sf_w9s1_lookahead_f3.sh` at 2026-04-15 23:26.

**Config:** `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_f3.py`
- w9s1 attention (local_attn_size=10, sink_size=1)
- F1 lookahead on, distance=4, dynamic RoPE
- F2 baked OFF
- F3 baked ON (applies to training rollout, validation sample loop, and inference CLI)
- Anchors ON for all three models (student + fake_score + teacher)

**Output dir:** `FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/sf_w9s1_la4_f3_anchor_freq5_lr1e5_0415_2326/`

**Checkpoints land at:** `.../checkpoints/<iter>.pth` + `.../checkpoints/<iter>.net_model/*.distcp`, every 100 iters per `save_ckpt_iter` default. Use `scripts/inference/consolidate_and_infer.py` to turn FSDP shards into a single `<iter>_net_consolidated.pth` before passing to `inference_causal.py` via `INFINITETALK_CKPT_PATH`.

---

## Known gaps / caveats / follow-ups

1. **Train/infer cache-noise-level mismatch under F3.** Training's stochastic exit means cache is at `t_list[exit_step]` (varies across 0.999, 0.955, 0.875, 0.700). Inference F3 is always at `t_list[-2] = 0.700`. Content matches (clean_ref at position 0, x0-ish at positions 1+); only timestep embedding differs. To close this fully, set `config.model.last_step_only = True` which forces exit at step N-1 always — but that changes SF's stochastic-exit distillation regime and is a separate research choice.

2. **`motion_frame` vs `first_frame_cond[:, :, 0:1]` subtle VAE-context difference.** Native InfiniteTalk uses `latent_motion_frames = vae.encode(cond_image)` (just 1 frame). Our anchor uses `first_frame_cond[:, :, 0:1]` which is the first slice of `vae.encode([cond_image, zeros × 92])`. These differ slightly due to VAE temporal convolutions. Dataloader already precomputes `motion_frame.pt` — swapping the anchor source is a small follow-up, not done yet.

3. **F2 in training:** deliberately not implemented. If needed later, mirror the sample_loop gating pattern into `rollout_with_gradient`.

4. **`_student_sample_loop` is gradient-unsafe.** Its `x[:, :, start:end] = x_next` pre-existing in-place pattern triggers autograd errors if run with gradients. Fine under `torch.no_grad()` (its only production use site). If anyone ever wants to backprop through it, that pattern needs replacing with a list-and-cat pattern like `rollout_with_gradient` uses.

5. **Validation videos lag real progress.** `validation_iter = 100` in the SF config; first validation video lands at iter 100. Plus dataloader init (~20 min per CLAUDE.md). Expect the first validation video ~25-30 min after launch.

---

## Files added/modified in this session

### New files
- `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_f3.py` — training config
- `scripts/run_sf_w9s1_lookahead_f3.sh` — training launcher
- `scripts/inference/run_inference_w9s1_lookahead_f3.sh` — inference launcher (mirrors training flags)
- `tests/test_input_anchor_plumbing.py` — 9 tests for helpers + kwarg plumbing
- `tests/test_post_step_pin_inference.py` — 4 tests for sample_loop F2-aware propagation + post-step pin
- `tests/test_rollout_gradient_sanity.py` — 6 gradient-flow tests for both helpers
- `tests/test_rollout_f3_toggle.py` — 5 tests for F3 in training rollout
- `docs/superpowers/plans/2026-04-15-input-anchor-and-full-inference-convention.md` — executed plan
- `docs/2026-04-16-input-anchor-f3-training-session.md` — this file

### Modified files
- `fastgen/networks/InfiniteTalk/network_causal.py` — `_maybe_apply_input_anchor` helper + `apply_input_anchor` kwarg on forward
- `fastgen/networks/InfiniteTalk/network.py` — `_maybe_apply_input_anchor_bidir` + `apply_input_anchor` kwarg on bidirectional forward
- `fastgen/methods/distribution_matching/causvid.py` — F2-aware input anchor propagation + post-step pin in `_student_sample_loop`
- `fastgen/methods/distribution_matching/self_forcing.py` — F3 (skip_clean_cache) plumbing in `rollout_with_gradient`
- `scripts/inference/inference_causal.py` — F2-aware input anchor propagation + post-step pin in `run_inference`, plus earlier CLI flags + FSDP-sidecar guard
- `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_noanchor.py` — `fake_score_optimizer.lr` bumped to 4e-6
- `CLAUDE.md` — new "Anchor plumbing (input-side + output-side)" section

---

## What to check next

1. First validation video at iter 100 (~25-30 min after launch). Look for clean frame 0.
2. Loss trend — should be finite and decreasing.
3. wandb group `infinitetalk_sf`, run name `sf_w9s1_la4_f3_anchor_freq5_lr1e5_0415_2326`.
4. If frame 0 is clean at iter 100: anchor fix validated end-to-end.
5. If frame 0 is still bad: something is wrong outside the anchor plumbing (config, checkpoint, data). Cross-check with the 44 passing unit tests to rule out plumbing bugs.

To correlate with upstream inference behavior (optional): generate a matching sample via `run_inference_w9s1_lookahead_f3.sh` with the same reference image/audio once a good checkpoint exists. The training rollout's KV distribution at `t_list[exit_step]` will differ from inference's `t_list[-2]`, but clean frame 0 should hold in both.
