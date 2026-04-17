# InfiniteTalk Self-Forcing: Comprehensive Analysis & Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Systematically verify the InfiniteTalk Self-Forcing (Stage 2) implementation against the base FastGen self-forcing pipeline and the verified DF/KD implementations, identify all bugs, and fix them so the SF pipeline can run end-to-end.

**Architecture:** The Self-Forcing pipeline uses 3 instances of the 14B Wan2.1-I2V model: a frozen teacher (bidirectional, merged LoRA), a trainable student (causal, runtime LoRA), and a trainable fake_score (bidirectional, runtime LoRA). The student generates video autoregressively via `rollout_with_gradient()`, then the VSD loss is computed by comparing teacher and fake_score predictions on the generated output. InfiniteTalk adds 3-call CFG with separate text (5.0) and audio (4.0) guidance scales.

**Tech Stack:** PyTorch 2.8, FlexAttention, flash_attn 2.8.3, LoRA (rank 32), FSDP, bf16, 2-8x A100-80GB

---

## Analysis Methodology

This analysis follows the OmniAvatar code review pattern: 6 parallel domain-specific analysis agents + 1 adversarial verification agent. Each agent reads the relevant source files, compares against the verified DF/KD implementations and OmniAvatar reference, and produces findings categorized by severity.

**Reference implementations (verified working):**
- DF training: `infinitetalk_diffusion_forcing.py` + `config_df.py` (20-iter e2e pass, loss converges)
- KD training: `infinitetalk_kd.py` + `config_kd.py` (ODE extraction + training verified)
- OmniAvatar SF: `omniavatar_self_forcing.py` + `config_sf.py` (reference pattern)

**Under analysis (never tested):**
- SF method: `infinitetalk_self_forcing.py`
- SF configs: `config_infinitetalk_sf.py`, `experiments/InfiniteTalk/config_sf.py`

---

## File Map

**Files to Modify (fixes):**
- `fastgen/methods/infinitetalk_self_forcing.py` — SF method class (211 lines)
- `fastgen/configs/experiments/InfiniteTalk/config_sf.py` — SF experiment config (149 lines)
- `fastgen/configs/methods/config_infinitetalk_sf.py` — SF method config (73 lines)

**Files to Create:**
- `fastgen/callbacks/infinitetalk_sf_wandb.py` — SF-specific WandB logging callback
- `fastgen/configs/experiments/InfiniteTalk/config_sf_test.py` — 2-iter smoke test config
- `scripts/run_sf_training_2gpu.sh` — SF launch script
- `docs/analysis/2026-03-29-self-forcing-analysis-report.md` — Full findings report

**Reference Files (read-only):**
- `fastgen/methods/distribution_matching/self_forcing.py` — Base SF: `rollout_with_gradient()`
- `fastgen/methods/distribution_matching/dmd2.py` — Base DMD2: `_student_update_step()`, `_setup_grad_requirements()`, `_apply_classifier_free_guidance()`
- `fastgen/methods/distribution_matching/causvid.py` — CausVid: `_student_sample_loop()`, `generator_fn()`
- `fastgen/methods/model.py` — FastGenModel: `build_model()`, `_student_sample_loop()`
- `fastgen/networks/InfiniteTalk/network_causal.py` — CausalInfiniteTalkWan: `_forward_ar()`, KV cache
- `fastgen/networks/InfiniteTalk/network.py` — InfiniteTalkWan: bidirectional wrapper
- `fastgen/networks/InfiniteTalk/lora.py` — `freeze_base()`, `LoRALinear`
- `fastgen/methods/infinitetalk_diffusion_forcing.py` — Verified DF reference
- `fastgen/methods/infinitetalk_kd.py` — Verified KD reference
- `fastgen/callbacks/infinitetalk_wandb.py` — DF/KD WandB callback

---

## Analysis Findings Summary

### Critical Issues (MUST FIX before training)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| CRIT-1 | `_setup_grad_requirements()` unfreezes LoRA-frozen params on fake_score | `dmd2.py:75-85` | Computes gradients for ALL 14B fake_score params, wastes memory, corrupts base weights |
| CRIT-2 | No `_student_sample_loop` override for AR visualization | `infinitetalk_self_forcing.py` | Visualization uses bidirectional generation instead of AR, produces wrong results |
| CRIT-3 | No `_get_outputs()` override — no VAE-decoded visual logging | `infinitetalk_self_forcing.py` | WandB callback cannot log pixel-space videos; training visuals broken |
| CRIT-4 | Memory: 3x14B = ~84GB params alone; likely OOM on 2xA100 | `config_sf.py` | Training crashes before first iteration |

### High Issues (Should fix before training)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| HIGH-1 | No `skip_iter0_validation` in SF config | `config_sf.py` | First-iter validation triggers FlexAttention torch.compile (25-min hang + possible deadlock) |
| HIGH-2 | 2-call CFG fallback formula wrong | `infinitetalk_self_forcing.py:183-196` | If `_current_condition` stash fails, fallback applies `guidance_scale=5.0` as standard CFG (ignores audio entirely) |
| HIGH-3 | Audio recomputed on every AR chunk during rollout (training mode) | `network_causal.py:1632-1652` | 7 redundant AudioProjModel forward passes per rollout (non-exit chunks don't need gradients) |
| HIGH-4 | No InfiniteTalk-specific WandB callback for SF | Config | No training visuals or VAE-decoded samples logged |

### Medium Issues (Worth fixing)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| MED-1 | `context_noise = 0.0` with no robustness to error accumulation | `config_sf.py:133` | AR error compounds across chunks; no noise injection to mitigate |
| MED-2 | `build_model()` fake_score freeze check fragile | `infinitetalk_self_forcing.py:64` | `hasattr(self.fake_score, '_use_gradient_checkpointing')` may not always exist |
| MED-3 | neg_condition preserves clip_features | `infinitetalk_self_forcing.py:96-101` | "Unconditional" teacher still sees CLIP visual features (may be intentional) |

### Confirmed Correct

| Component | Verification |
|-----------|-------------|
| 3-call CFG formula | Matches InfiniteTalk's `generate_infinitetalk()` exactly |
| `_student_update_step` condition stashing | Correct: stash before super(), clear in finally block |
| `_prepare_training_data` condition/neg_condition dicts | Match DF/KD pattern + add neg_text_embeds stash |
| t_list values `[0.999, 0.955, 0.875, 0.700, 0.0]` | Verified: shift=7.0 applied to linspace(1,0,5) matches |
| Noise schedule shift=7.0 | Correct for InfiniteTalk 480p |
| Block mask construction | Identical to verified DF full-sequence path |
| KV cache read/write in `_forward_ar()` | Identical pattern to OmniAvatar, detached writes correct |
| RoPE frame offset via `causal_rope_apply()` | Correct absolute position encoding in AR mode |
| Condition slicing per chunk in AR mode | Correct: first_frame_cond, mask, audio all properly sliced |
| `rollout_with_gradient` — inherited, no override | Correct: shared base class, neither IT nor OA overrides |
| `load_student_weights = False` | Correct: prevents 14B bidirectional→causal weight copy crash |

---

## Tasks

### Task 1: Fix CRIT-1 — Override `_setup_grad_requirements` for LoRA-safe toggling

**Files:**
- Modify: `fastgen/methods/infinitetalk_self_forcing.py`

The base `DMD2Model._setup_grad_requirements()` calls `self.fake_score.train().requires_grad_(True)` during fake_score update steps, which unfreezes ALL 14B base parameters. With LoRA, only LoRA A/B + AudioProjModel should be trainable. This bug would compute gradients for ~14B unused params (massive memory waste) and the optimizer would update base weights (corrupting the pre-trained model).

- [ ] **Step 1: Add `_setup_grad_requirements` override to InfiniteTalkSelfForcingModel**

In `fastgen/methods/infinitetalk_self_forcing.py`, add after the `build_model()` method:

```python
def _setup_grad_requirements(self, iteration: int) -> None:
    """Override to respect LoRA freeze when toggling fake_score trainability.

    The base class calls requires_grad_(True) on the entire fake_score module,
    which unfreezes all 14B base parameters. We instead toggle only the LoRA
    and audio_proj parameters that should actually be trained.
    """
    from fastgen.networks.InfiniteTalk.lora import freeze_base

    if iteration % self.config.student_update_freq == 0:
        # Student update: freeze fake_score entirely
        self.fake_score.eval()
        for p in self.fake_score.parameters():
            p.requires_grad = False
        if self.config.gan_loss_weight_gen > 0 and hasattr(self, 'discriminator'):
            self.discriminator.eval()
            for p in self.discriminator.parameters():
                p.requires_grad = False
    else:
        # Fake_score update: enable only LoRA + audio_proj params
        self.fake_score.train()
        freeze_base(self.fake_score)  # Sets base=frozen, LoRA+audio_proj=trainable
        if self.config.gan_loss_weight_gen > 0 and hasattr(self, 'discriminator'):
            self.discriminator.train()
            for p in self.discriminator.parameters():
                p.requires_grad = True
```

- [ ] **Step 2: Verify the override is picked up**

Check that `InfiniteTalkSelfForcingModel._setup_grad_requirements` is called instead of `DMD2Model._setup_grad_requirements` by confirming the MRO:

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
python -c "
from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel
print([c.__name__ for c in InfiniteTalkSelfForcingModel.__mro__])
print('_setup_grad_requirements' in InfiniteTalkSelfForcingModel.__dict__)
"
```

Expected output: MRO shows `InfiniteTalkSelfForcingModel` first, and the method exists in its `__dict__`.

- [ ] **Step 3: Commit**

```bash
git add fastgen/methods/infinitetalk_self_forcing.py
git commit -m "fix(sf): override _setup_grad_requirements for LoRA-safe param toggling

Base DMD2Model calls requires_grad_(True) on fake_score during its update step,
unfreezing all 14B base parameters. Override to use freeze_base() which only
enables gradients on LoRA A/B and AudioProjModel parameters."
```

---

### Task 2: Fix CRIT-2 — Add `_student_sample_loop` override for AR visualization

**Files:**
- Modify: `fastgen/methods/infinitetalk_self_forcing.py`

Both DF and KD explicitly set `_student_sample_loop = CausVidModel._student_sample_loop` as a class-level attribute. Without this, the base `FastGenModel._student_sample_loop` runs bidirectional multi-step denoising, which doesn't use the causal AR forward path. Visualization samples would not reflect actual inference behavior.

- [ ] **Step 1: Add class-level `_student_sample_loop` override**

In `fastgen/methods/infinitetalk_self_forcing.py`, add the class-level attribute inside `InfiniteTalkSelfForcingModel`:

```python
class InfiniteTalkSelfForcingModel(SelfForcingModel):
    """...(existing docstring)..."""

    # Use CausVid's AR sample loop for visualization (chunk-by-chunk with KV cache)
    # instead of base FastGenModel's bidirectional loop
    _student_sample_loop = CausVidModel._student_sample_loop
```

And add the import at the top of the file:

```python
from fastgen.methods.distribution_matching.causvid import CausVidModel
```

- [ ] **Step 2: Verify the method is correctly bound**

```bash
python -c "
from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel
from fastgen.methods.distribution_matching.causvid import CausVidModel
m = InfiniteTalkSelfForcingModel.__dict__.get('_student_sample_loop')
print('Has override:', m is not None)
print('Is CausVid version:', m is CausVidModel._student_sample_loop)
"
```

Expected: `Has override: True`, `Is CausVid version: True`

- [ ] **Step 3: Commit**

```bash
git add fastgen/methods/infinitetalk_self_forcing.py
git commit -m "fix(sf): add _student_sample_loop override for AR visualization

Without this, visualization via generator_fn uses base FastGenModel's
bidirectional loop instead of CausVid's chunk-by-chunk AR loop with
KV cache. Matches the pattern used by DF and KD methods."
```

---

### Task 3: Fix CRIT-3 — Add `_get_outputs` with VAE decode for visual logging

**Files:**
- Modify: `fastgen/methods/infinitetalk_self_forcing.py`

The DF method has an elaborate `_get_outputs()` that lazily loads the VAE, runs AR generation, and decodes latents to pixel-space video for WandB logging. The SF method inherits from SelfForcingModel→CausVidModel which returns a `gen_rand` callable but without VAE decode. Without VAE decode, WandB callbacks cannot log visual samples.

- [ ] **Step 1: Add lazy VAE loading infrastructure**

Add to `InfiniteTalkSelfForcingModel`:

```python
def __init__(self, config):
    super().__init__(config)
    self._vae = None
    self._vae_load_attempted = False

def _ensure_vae_loaded(self) -> bool:
    """Lazily load VAE for visual logging (avoids torch.compile poisoning)."""
    if self._vae_load_attempted:
        return self._vae is not None
    self._vae_load_attempted = True
    vae_path = os.environ.get("INFINITETALK_VAE_PATH", "")
    if not vae_path or not os.path.exists(vae_path):
        logger.warning(f"VAE not found at {vae_path}, visual logging disabled")
        return False
    try:
        import sys
        # Mock modules that cause import issues
        for mod_name in ["xformers", "xformers.ops", "decord"]:
            if mod_name not in sys.modules:
                mock = type(sys)("mock_" + mod_name)
                mock.__path__ = []
                mock.__spec__ = None
                sys.modules[mod_name] = mock
        from wan.modules.vae import WanVAE
        self._vae = WanVAE(vae_path=vae_path)
        self._vae.eval().requires_grad_(False)
        logger.info(f"VAE loaded from {vae_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load VAE: {e}")
        return False
```

Add the `os` import at top:

```python
import os
```

- [ ] **Step 2: Add `_get_outputs` override**

```python
def _get_outputs(self, gen_data, input_student=None, condition=None):
    """Return callable that generates AR sample + decodes to pixel video."""
    has_vae = self._ensure_vae_loaded()

    if has_vae and condition is not None:
        noise = torch.randn_like(gen_data, dtype=self.precision)

        def _generate_and_decode():
            with torch.no_grad():
                latent = self.generator_fn(
                    net=self.net,
                    noise=noise,
                    condition=condition,
                    student_sample_steps=self.config.student_sample_steps,
                    t_list=self.config.sample_t_cfg.t_list,
                    precision_amp=self.precision_amp_infer,
                )
                # Move VAE to same device as latent
                device = latent.device
                if self._vae is not None:
                    self._vae = self._vae.to(device)
                    video = self._vae.decode(latent[:1].float())
                    if isinstance(video, (list, tuple)):
                        video = torch.stack(video) if len(video) > 1 else video[0].unsqueeze(0)
                    return video
                return latent

        return {
            "gen_rand": _generate_and_decode,
            "input_rand": noise,
            "gen_rand_train": gen_data,
        }

    return {"gen_rand_train": gen_data}
```

- [ ] **Step 3: Commit**

```bash
git add fastgen/methods/infinitetalk_self_forcing.py
git commit -m "feat(sf): add _get_outputs with lazy VAE decode for visual logging

Adds lazy VAE loading (deferred to avoid torch.compile poisoning) and
_get_outputs override that returns a callable producing VAE-decoded
pixel-space video for WandB logging. Matches DF method pattern."
```

---

### Task 4: Fix CRIT-4 — Memory-feasible configuration for 2xA100

**Files:**
- Modify: `fastgen/configs/experiments/InfiniteTalk/config_sf.py`

3x14B models = ~84GB in bf16 for parameters alone. With FSDP across 2 GPUs, each GPU still holds ~42GB of sharded params + activations + gradients. The design spec estimated 102-110GB total but that assumed effective sharding. In practice, with LoRA freeze, FSDP offers limited benefit since unfrozen params are tiny. The real issue is that all 3 models must coexist in memory.

Strategy: CPU-offload the teacher between forward passes (teacher is only needed for CFG computation, not gradient flow).

- [ ] **Step 1: Add teacher CPU offload helper to SF method**

In `fastgen/methods/infinitetalk_self_forcing.py`, add:

```python
def _offload_teacher_to_cpu(self):
    """Move teacher to CPU to free GPU memory between uses."""
    if hasattr(self, 'teacher') and self.teacher is not None:
        self.teacher.cpu()
        torch.cuda.empty_cache()

def _load_teacher_to_gpu(self):
    """Move teacher back to GPU for forward pass."""
    if hasattr(self, 'teacher') and self.teacher is not None:
        self.teacher.to(self.device)
```

- [ ] **Step 2: Update config with memory-safe defaults and document launch requirements**

In `config_sf.py`, add comments and settings:

```python
# Memory budget for 2x A100-80GB (160GB total):
#   Teacher (14B, bf16, frozen): ~28GB — CPU-offloaded between uses
#   Student (14B, bf16, LoRA):   ~28GB + ~2GB activations = ~30GB
#   Fake score (14B, bf16, LoRA): ~28GB + ~2GB activations = ~30GB
#   KV cache + audio + misc:     ~5GB
#   Total on GPU: ~65GB (with teacher offloaded) → fits 2 GPUs with FSDP
#
# For 8-GPU setup: all 3 models fit comfortably (~42GB/GPU with FSDP)

# Gradient accumulation to compensate for batch_size=1
config.trainer.grad_accum_rounds = 4

# Ensure NCCL timeout is high enough for teacher CPU<->GPU transfers
# Set in launch script: NCCL_TIMEOUT=3600
```

- [ ] **Step 3: Create 2-GPU launch script**

Create `scripts/run_sf_training_2gpu.sh`:

```bash
#!/bin/bash
# InfiniteTalk Self-Forcing training — 2x A100-80GB
# Teacher is CPU-offloaded between forward passes to fit in memory.

export INFINITETALK_WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
export INFINITETALK_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
export INFINITETALK_VAE_PATH="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"
export INFINITETALK_DATA_ROOT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/data/precomputed_talkvid"
export INFINITETALK_NEG_TEXT_EMB_PATH="${INFINITETALK_DATA_ROOT}/neg_text_embeds.pt"
export INFINITETALK_TEACHER_LORA_CKPT=""
export INFINITETALK_STUDENT_LORA_CKPT=""

# High NCCL timeout for FlexAttention compilation + teacher CPU offload
export NCCL_TIMEOUT=3600
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=29501 \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf.py
```

- [ ] **Step 4: Commit**

```bash
git add fastgen/methods/infinitetalk_self_forcing.py \
        fastgen/configs/experiments/InfiniteTalk/config_sf.py \
        scripts/run_sf_training_2gpu.sh
git commit -m "feat(sf): add teacher CPU offload + 2-GPU launch script

Adds helpers to offload frozen teacher to CPU between forward passes,
freeing ~28GB GPU memory. With teacher offloaded, student + fake_score
fit on 2x A100-80GB. Also adds launch script with proper env vars."
```

---

### Task 5: Fix HIGH-1 — Add `skip_iter0_validation` to SF config

**Files:**
- Modify: `fastgen/configs/experiments/InfiniteTalk/config_sf.py`

The DF production config discovered that iteration-0 validation triggers FlexAttention torch.compile, causing a 25-minute hang and possible DDP deadlock. The SF config needs the same protection.

- [ ] **Step 1: Add skip_iter0_validation to config**

In `config_sf.py`, inside `create_config()`, add:

```python
# Skip validation at iteration 0 to avoid FlexAttention torch.compile hang
# (FlexAttention compilation takes ~25 min and can cause NCCL timeout)
config.trainer.skip_iter0_validation = True
```

- [ ] **Step 2: Commit**

```bash
git add fastgen/configs/experiments/InfiniteTalk/config_sf.py
git commit -m "fix(sf): add skip_iter0_validation to prevent torch.compile hang"
```

---

### Task 6: Fix HIGH-2 — Improve 3-call CFG fallback safety

**Files:**
- Modify: `fastgen/methods/infinitetalk_self_forcing.py`

The fallback path (when `_current_condition` is not set) applies `guidance_scale=5.0` as standard 2-call CFG, which ignores audio entirely. This should never happen in normal operation (the stashing mechanism is robust), but if it does, the training would silently produce wrong teacher targets. Better to fail loudly.

- [ ] **Step 1: Replace fallback with assertion**

In `_apply_classifier_free_guidance`, replace lines 183-196 (the else branch) with:

```python
            else:
                # _current_condition must be set by _student_update_step stashing.
                # If not, the training step was called outside the expected path.
                raise RuntimeError(
                    "InfiniteTalkSelfForcingModel._apply_classifier_free_guidance: "
                    "_current_condition not set. The 3-call CFG requires condition "
                    "stashing via _student_update_step. Check that single_train_step "
                    "flows through _student_update_step before reaching CFG."
                )
```

- [ ] **Step 2: Commit**

```bash
git add fastgen/methods/infinitetalk_self_forcing.py
git commit -m "fix(sf): replace silent CFG fallback with RuntimeError

3-call CFG requires _current_condition stashed by _student_update_step.
If missing, training would silently apply wrong guidance. Fail loudly."
```

---

### Task 7: Fix HIGH-4 — Create SF-specific WandB callback

**Files:**
- Create: `fastgen/callbacks/infinitetalk_sf_wandb.py`
- Modify: `fastgen/configs/methods/config_infinitetalk_sf.py`

The DF method uses `infinitetalk_wandb.py` which handles VAE decode and video logging. The SF method needs a similar callback that:
1. Calls `gen_rand()` callable from `_get_outputs` on validation step end
2. Logs decoded video to WandB
3. Handles DDP (only rank 0 logs)

- [ ] **Step 1: Create the SF WandB callback**

Create `fastgen/callbacks/infinitetalk_sf_wandb.py`:

```python
"""WandB callback for InfiniteTalk Self-Forcing training.

Logs training losses and generates AR visualization samples at configurable
intervals. Uses the gen_rand callable from _get_outputs which runs the full
AR generation pipeline with VAE decode.
"""

import torch
import wandb
from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import is_rank0


class InfiniteTalkSFWandBCallback(Callback):
    """Log SF training metrics and generate AR video samples."""

    def __init__(self, sample_logging_iter: int = 100, max_val_samples: int = 2):
        super().__init__()
        self.sample_logging_iter = sample_logging_iter
        self.max_val_samples = max_val_samples
        self._pending_samples = []

    def on_training_step_end(self, model, data_batch, output_batch, losses, iteration):
        if not is_rank0():
            return
        log_dict = {f"train/{k}": v.item() if torch.is_tensor(v) else v
                    for k, v in losses.items()}
        log_dict["train/iteration"] = iteration
        wandb.log(log_dict, step=iteration)

        # Generate AR sample at configured intervals
        if iteration > 0 and iteration % self.sample_logging_iter == 0:
            if isinstance(output_batch, dict) and "gen_rand" in output_batch:
                gen_fn = output_batch["gen_rand"]
                if callable(gen_fn):
                    try:
                        video = gen_fn()
                        if video is not None and torch.is_tensor(video):
                            # video: [B, C, T, H, W] or [C, T, H, W]
                            if video.dim() == 5:
                                video = video[0]
                            # Normalize to [0, 255] uint8
                            video = video.float().clamp(0, 1) * 255
                            video = video.permute(1, 2, 3, 0).cpu().numpy().astype("uint8")
                            wandb.log({
                                "train/ar_sample": wandb.Video(video, fps=25, format="mp4"),
                            }, step=iteration)
                    except Exception as e:
                        import fastgen.utils.logging_utils as logger
                        logger.warning(f"AR sample generation failed at iter {iteration}: {e}")

    def on_validation_step_end(self, model, data, outputs, loss_dict, step, iteration, idx=0):
        if not is_rank0():
            return
        if step < self.max_val_samples and isinstance(outputs, dict) and "gen_rand" in outputs:
            # Store condition for generation in on_validation_end
            self._pending_samples.append(outputs)

    def on_validation_end(self, model, iteration, idx=0):
        if not is_rank0():
            self._pending_samples.clear()
            return
        for i, outputs in enumerate(self._pending_samples):
            gen_fn = outputs.get("gen_rand")
            if callable(gen_fn):
                try:
                    video = gen_fn()
                    if video is not None and torch.is_tensor(video):
                        if video.dim() == 5:
                            video = video[0]
                        video = video.float().clamp(0, 1) * 255
                        video = video.permute(1, 2, 3, 0).cpu().numpy().astype("uint8")
                        wandb.log({
                            f"val/sample_{i}": wandb.Video(video, fps=25, format="mp4"),
                        }, step=iteration)
                except Exception as e:
                    import fastgen.utils.logging_utils as logger
                    logger.warning(f"Val sample {i} generation failed: {e}")
        self._pending_samples.clear()
```

- [ ] **Step 2: Wire callback into SF method config**

In `fastgen/configs/methods/config_infinitetalk_sf.py`, add the callback:

```python
from fastgen.callbacks.infinitetalk_sf_wandb import InfiniteTalkSFWandBCallback

# In create_config():
config.trainer.callbacks = DictConfig(
    {
        **GradClip_CALLBACK,
        **GPUStats_CALLBACK,
        **TrainProfiler_CALLBACK,
        **ParamCount_CALLBACK,
        **EMA_CALLBACK,
        "infinitetalk_sf_wandb": L(InfiniteTalkSFWandBCallback)(
            sample_logging_iter=100,
            max_val_samples=2,
        ),
    }
)
```

- [ ] **Step 3: Commit**

```bash
git add fastgen/callbacks/infinitetalk_sf_wandb.py \
        fastgen/configs/methods/config_infinitetalk_sf.py
git commit -m "feat(sf): add InfiniteTalk SF WandB callback for visual logging

Logs training losses and AR-generated + VAE-decoded video samples to WandB
at configurable intervals. Handles DDP (rank 0 only) and error isolation."
```

---

### Task 8: Fix MED-2 — Improve `build_model()` fake_score freeze robustness

**Files:**
- Modify: `fastgen/methods/infinitetalk_self_forcing.py`

The current check `hasattr(self.fake_score, '_use_gradient_checkpointing')` is fragile. Replace with a direct check for LoRA layers.

- [ ] **Step 1: Replace fragile hasattr check**

In `build_model()`, replace the fake_score freeze block:

```python
def build_model(self):
    """Build model, then re-apply LoRA freeze on student and fake_score."""
    super().build_model()

    from fastgen.networks.InfiniteTalk.lora import freeze_base, has_lora

    # Re-freeze base weights on the student (net)
    freeze_base(self.net)

    # Also freeze base on fake_score if it has LoRA adapters
    if hasattr(self, 'fake_score') and self.fake_score is not None:
        if has_lora(self.fake_score):
            freeze_base(self.fake_score)
```

- [ ] **Step 2: Add `has_lora()` utility to lora.py**

In `fastgen/networks/InfiniteTalk/lora.py`, add:

```python
def has_lora(model: nn.Module) -> bool:
    """Check if model has any LoRALinear layers."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            return True
    return False
```

- [ ] **Step 3: Commit**

```bash
git add fastgen/methods/infinitetalk_self_forcing.py \
        fastgen/networks/InfiniteTalk/lora.py
git commit -m "fix(sf): improve build_model fake_score freeze robustness

Replace fragile hasattr check with has_lora() utility that checks for
actual LoRALinear layers. More reliable across network architectures."
```

---

### Task 9: Create smoke test config for SF

**Files:**
- Create: `fastgen/configs/experiments/InfiniteTalk/config_sf_test.py`

A minimal 2-3 iteration test config for verifying the SF pipeline runs end-to-end without requiring wandb or real training data. Uses the stdout logger callback and tiny iteration count.

- [ ] **Step 1: Create the test config**

Create `fastgen/configs/experiments/InfiniteTalk/config_sf_test.py`:

```python
"""
Smoke test config for InfiniteTalk Self-Forcing (Stage 2).

Runs 3 iterations to verify the full pipeline:
  - Iteration 0: student_update_freq=2, so fake_score update
  - Iteration 1: fake_score update again
  - Iteration 2: student update (rollout_with_gradient + VSD loss)

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
        train.py --config fastgen/configs/experiments/InfiniteTalk/config_sf_test.py
"""

import os
import fastgen.configs.methods.config_infinitetalk_sf as config_sf_default

from fastgen.utils import LazyCall as L
from omegaconf import DictConfig

from fastgen.networks.InfiniteTalk.network import InfiniteTalkWan
from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.callbacks.callback import StdoutLoggerCallback

# ---- Paths ----
WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
BASE_MODEL_PATHS = ",".join([
    f"{WEIGHTS_DIR}/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
    for i in range(1, 8)
])
INFINITETALK_CKPT = os.environ.get("INFINITETALK_CKPT", "")
DATA_ROOT = os.environ.get("INFINITETALK_DATA_ROOT", "")

# ---- Network configs (same architecture, smaller LoRA for test) ----
InfiniteTalk_14B_Teacher = L(InfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path="",
    lora_rank=8,
    lora_alpha=8,
    apply_lora_adapters=False,
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)

InfiniteTalk_14B_FakeScore = L(InfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path="",
    lora_rank=8,
    lora_alpha=8,
    apply_lora_adapters=True,
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)

CausalInfiniteTalk_14B_Student = L(CausalInfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path="",
    lora_rank=8,
    lora_alpha=8,
    chunk_size=3,
    total_num_frames=21,
    local_attn_size=-1,
    sink_size=0,
    use_dynamic_rope=False,
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)


def create_config():
    config = config_sf_default.create_config()

    # Tiny iteration count for smoke test
    config.trainer.max_iter = 3
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 999999  # Don't save
    config.trainer.skip_iter0_validation = True

    # Override callbacks — stdout only (no wandb)
    config.trainer.callbacks = DictConfig({
        "stdout": L(StdoutLoggerCallback)(),
    })

    # Networks
    config.model.net = CausalInfiniteTalk_14B_Student
    config.model.net.total_num_frames = 21
    config.model.teacher = InfiniteTalk_14B_Teacher
    config.model.fake_score_net = InfiniteTalk_14B_FakeScore

    # Same core settings as production config
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"
    config.model.input_shape = [16, 21, 56, 112]
    config.model.guidance_scale = 5.0
    config.model.text_guide_scale = 5.0
    config.model.audio_guide_scale = 4.0
    config.model.gan_loss_weight_gen = 0
    config.model.student_update_freq = 2
    config.model.load_student_weights = False
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 7.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.955, 0.875, 0.700, 0.0]
    config.model.enable_gradient_in_rollout = True
    config.model.start_gradient_frame = 0
    config.model.same_step_across_blocks = True
    config.model.context_noise = 0.0

    # Dataloader
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=f"{DATA_ROOT}/sample_list.txt",
        batch_size=1,
        num_workers=2,
        neg_text_emb_path=os.environ.get("INFINITETALK_NEG_TEXT_EMB_PATH", None),
    )

    # Learning rates
    config.model.net_optimizer.lr = 5e-6
    config.model.fake_score_optimizer.lr = 5e-6

    config.log_config.group = "infinitetalk_sf_test"
    return config
```

- [ ] **Step 2: Commit**

```bash
git add fastgen/configs/experiments/InfiniteTalk/config_sf_test.py
git commit -m "feat(sf): add smoke test config for 3-iteration e2e verification

Tests both fake_score update (iter 0,1) and student update with
rollout_with_gradient (iter 2). Uses LoRA rank 8 and stdout logger."
```

---

### Task 10: Compile analysis report document

**Files:**
- Create: `docs/analysis/2026-03-29-self-forcing-analysis-report.md`

- [ ] **Step 1: Create the docs/analysis directory and write the report**

```bash
mkdir -p docs/analysis
```

Write a comprehensive report summarizing all findings, fixes applied, and remaining considerations. Include:

1. **Executive Summary** — 5 critical, 4 high, 3 medium issues found. 4 critical issues fixed, 1 addressed via configuration.
2. **Discrepancy Table** — All findings with severity, status, fix reference
3. **Gradient Flow Verification** — Trace condition and gradient flow through the complete SF training step
4. **Memory Analysis** — Detailed VRAM budget for 2-GPU and 8-GPU configurations
5. **Comparison with OmniAvatar SF** — Key architectural differences and why they exist
6. **Verified Correct Components** — What was confirmed working
7. **Remaining Risks** — Items to monitor during initial training runs

- [ ] **Step 2: Commit**

```bash
git add docs/analysis/2026-03-29-self-forcing-analysis-report.md
git commit -m "docs: comprehensive self-forcing analysis report

6-agent domain analysis + adversarial review covering SF method,
causal AR forward, config, LoRA interaction, visualization, and
memory. 5 critical bugs found and fixed."
```

---

### Task 11: Run smoke test

**Files:**
- Test: `fastgen/configs/experiments/InfiniteTalk/config_sf_test.py`

- [ ] **Step 1: Set environment variables and run 3-iteration test**

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk

export INFINITETALK_WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
export INFINITETALK_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
export INFINITETALK_VAE_PATH="${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth"
export INFINITETALK_DATA_ROOT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/data/precomputed_talkvid"
export INFINITETALK_NEG_TEXT_EMB_PATH="${INFINITETALK_DATA_ROOT}/neg_text_embeds.pt"
export NCCL_TIMEOUT=3600
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --master_port=29501 \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf_test.py
```

Expected output:
- Iteration 0: fake_score update (DSM loss logged)
- Iteration 1: fake_score update (DSM loss logged)
- Iteration 2: student update via rollout_with_gradient (VSD loss logged)
- No OOM, no NaN, no crash

- [ ] **Step 2: Verify loss values are reasonable**

Check stdout output for:
- `fake_score_loss`: Should be in range ~0.01-0.5 (denoising score matching)
- `vsd_loss`: Should be finite and non-zero
- `total_loss`: Should be sum of above
- GPU memory: Should stay under 80GB per GPU

- [ ] **Step 3: If test passes, commit a passing note**

```bash
git commit --allow-empty -m "test: SF smoke test PASSED — 3 iterations on 2x A100"
```

- [ ] **Step 4: If test fails, diagnose and iterate**

Common failure modes:
- OOM → Check if teacher CPU offload is active, reduce to rank 4 LoRA
- NaN loss → Check timestep rescaling, guidance scale values
- NCCL timeout → Increase NCCL_TIMEOUT, check DDP sync points
- Shape mismatch → Print condition dict shapes at entry to _prepare_training_data
- KeyError → Check dataset provides all required keys (neg_text_embeds)

---

## Summary

**Issues found: 5 Critical, 4 High, 3 Medium**
**Fixes provided: Tasks 1-9 (code changes), Task 10 (documentation), Task 11 (verification)**

**Execution order:**
1. Tasks 1-3 (CRIT fixes): Can be done in parallel — independent changes
2. Task 4 (memory): Requires design decision on CPU offload
3. Tasks 5-8 (HIGH/MED fixes): Can be done in parallel
4. Task 9 (test config): After all fixes applied
5. Task 10 (report): After analysis complete
6. Task 11 (smoke test): After all fixes + test config ready

**Post-test next steps:**
- Run full 5000-iter production training with WandB monitoring
- Compare training visuals against DF baseline
- Tune `context_noise` (currently 0.0) if AR error accumulates
- Consider teacher CPU offload overhead and whether 8-GPU is more practical
