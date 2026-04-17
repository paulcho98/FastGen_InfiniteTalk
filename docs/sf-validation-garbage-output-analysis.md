# SF Validation Garbage Output — Root Cause Analysis

**Date:** 2026-04-13
**Scope:** Why InfiniteTalk Self-Forcing (SF) validation generates garbage while Diffusion Forcing (DF) validation works correctly.

## Summary

SF validation produces garbage video because its `validation_step` uses the wrong network reference (`self.net` instead of `self.net_inference`), never sets the network to eval mode, and duplicates generation work that the wandb callback also performs.

---

## Two Validation Code Paths

The trainer (`fastgen/trainer.py:361-396`) dispatches validation differently depending on whether the model implements `validation_step`:

```python
# trainer.py:386-391
if hasattr(model, "validation_step"):
    loss_map, outputs = model.validation_step(data, iteration)    # SF path
else:
    loss_map, outputs = model_ddp.single_train_step(data, iteration)  # DF path
```

The trainer **never calls `model.eval()`** before entering the validation loop.

### DF Path (works correctly)

DF has no `validation_step` override. The trainer falls through to `single_train_step`, which just computes training loss on the validation data — no video generation happens here.

Video generation is handled entirely by the **wandb callback** in `on_validation_end` (`fastgen/callbacks/infinitetalk_wandb.py:181`):

```python
# infinitetalk_wandb.py:216-223
latent = model.generator_fn(
    net=model.net_inference,    # <-- correct: respects EMA
    noise=noise,
    condition=cond_gpu,
    student_sample_steps=model.config.student_sample_steps,
    t_list=model.config.sample_t_cfg.t_list,
    precision_amp=model.precision_amp_infer,
)
```

This works because:
- Uses `model.net_inference` which returns the EMA network when available (`fastgen/methods/model.py:676-685`)
- Runs under `torch.no_grad()`
- Network mode (train/eval) doesn't affect DF's training forward pass, so the callback-generated videos are fine

### SF Path (garbage output)

SF overrides `validation_step` (`fastgen/methods/infinitetalk_self_forcing.py:448`):

```python
# infinitetalk_self_forcing.py:463-473
with torch.no_grad():
    gen_data = CausVidModel.generator_fn(
        net=self.net,               # <-- BUG: should be self.net_inference
        noise=noise,
        condition=condition,
        student_sample_steps=self.config.student_sample_steps,
        student_sample_type=self.config.student_sample_type,
        t_list=self.config.sample_t_cfg.t_list,
        context_noise=context_noise,
        precision_amp=self.precision_amp_infer,
    )
```

---

## Bug #1: Wrong Network Reference (`self.net` vs `self.net_inference`)

| Location | Network used | Effect |
|----------|-------------|--------|
| SF `validation_step` (line 464) | `self.net` | Raw training network — no EMA |
| Wandb callback `on_validation_end` (line 216) | `model.net_inference` | EMA network when available |

`net_inference` is a property (`fastgen/methods/model.py:676-685`) that returns the EMA network when EMA is enabled and FSDP is not active. If EMA is configured, `self.net` has noisy training weights while `self.net_inference` has the smoothed EMA weights.

**Fix:** Change `net=self.net` to `net=self.net_inference` in `validation_step`.

## Bug #2: Network Never Set to Eval Mode

The trainer calls `validate()` without calling `model.eval()` first. The network remains in training mode from the previous training step.

For DF this is irrelevant (no generation happens in the train step path). For SF, the `validation_step` generates video with the network still in training mode, meaning:
- Dropout layers remain active (randomly zeroing activations)
- Any training-mode-specific behavior (e.g., stochastic attention mask rebuilding) is triggered

This also affects the first-frame anchor logic:

```python
# network_causal.py:1940-1943
anchor_active = getattr(self, "_enable_first_frame_anchor", True)
if anchor_active and getattr(self, "_anchor_eval_only", False):
    anchor_active = not self.training  # <-- checks self.training, which is still True
```

When `_anchor_eval_only=True`, the anchor is disabled during validation because `self.training` is `True`.

**Fix:** Add `self.net.eval()` at the start of `validation_step`, restore with `self.net.train()` after.

## Bug #3: Double Generation

SF's `validation_step` generates video AND VAE-decodes it (lines 486-489). Then the wandb callback `on_validation_end` **also** generates videos from stored conditions (line 216).

This means:
1. `validation_step` generates with `self.net` (wrong) — this output goes into `outputs` dict
2. The wandb callback generates again with `model.net_inference` (correct) — this is what gets logged to wandb

However, the first generation wastes compute and, depending on what `outputs` is used for downstream, could propagate garbage data.

---

## The AR Sample Loop (shared by both paths)

Both paths ultimately call `CausVidModel._student_sample_loop` (`fastgen/methods/distribution_matching/causvid.py:88-185`):

```
For each chunk (0 .. num_chunks-1):
    For each denoising step (0 .. len(t_list)-2):
        x_next = net(x_cur, t_cur, is_ar=True, cache_tag="pos",
                     cur_start_frame=start, store_kv=False)
        if t_next > 0:
            x_next = forward_process(x_next, noise, t_next)   # re-noise for next step

    # Cache: run final clean output through net with store_kv=True
    net(x_cache, t_cache, is_ar=True, store_kv=True, ...)
```

Key parameters:
- `t_list = [0.999, 0.955, 0.875, 0.700, 0.0]` → 4 denoising steps
- `student_sample_type = "sde"` → fresh noise at each re-noising step
- `context_noise > 0` → adds noise to cached context frames

This loop is correct in structure. The garbage comes from the network state (bugs #1 and #2), not the loop logic.

---

## Proposed Fix

```python
# infinitetalk_self_forcing.py:validation_step

def validation_step(self, data, iteration):
    real_data, condition, neg_condition = self._prepare_training_data(data)
    B, C, T, H, W = real_data.shape

    noise = torch.randn_like(real_data)
    context_noise = getattr(self.config, "context_noise", 0)

    was_training = self.net.training
    self.net.eval()                          # Fix #2: eval mode
    try:
        with torch.no_grad():
            gen_data = CausVidModel.generator_fn(
                net=self.net_inference,      # Fix #1: use EMA net
                noise=noise,
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                context_noise=context_noise,
                precision_amp=self.precision_amp_infer,
            )
    finally:
        if was_training:
            self.net.train()                 # Restore training mode

    # ... rest of validation_step ...
```

## File References

| File | Lines | What |
|------|-------|------|
| `fastgen/trainer.py` | 361-396 | Trainer `validate()` — no `model.eval()` call |
| `fastgen/trainer.py` | 386-391 | Dispatch: `validation_step` vs `single_train_step` |
| `fastgen/methods/infinitetalk_self_forcing.py` | 448-489 | SF `validation_step` — uses `self.net`, no eval mode |
| `fastgen/methods/model.py` | 675-685 | `net_inference` property — returns EMA when available |
| `fastgen/callbacks/infinitetalk_wandb.py` | 204-228 | Callback generation — uses `model.net_inference` (correct) |
| `fastgen/methods/distribution_matching/causvid.py` | 88-185 | `_student_sample_loop` — AR chunk-by-chunk denoising |
| `fastgen/networks/InfiniteTalk/network_causal.py` | 1940-1943 | First-frame anchor `self.training` check |
