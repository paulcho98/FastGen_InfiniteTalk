# InfiniteTalk Self-Forcing Adaptation — Design Spec

## Goal

Adapt FastGen's Self-Forcing distillation framework for InfiniteTalk, a 14B Wan2.1-I2V-based audio-driven talking-face model. The 14B model serves as both teacher and student, with LoRA-based parameter-efficient training to fit within 2×A100-80GB.

## Architecture

InfiniteTalk is an I2V (image-to-video) model that generates lip-synced talking-face video from a reference image + audio. Unlike OmniAvatar (V2V, 65ch), InfiniteTalk uses standard Wan I2V conditioning (36ch = 16 noise + 4 mask + 16 VAE ref) plus audio cross-attention injected per transformer block.

The adaptation follows the OmniAvatar pattern (7 components from the adaptation guide) with two key differences:
1. **Same-size teacher and student (14B)** — LoRA adapters make the student trainable without duplicating the full model
2. **3-call CFG** — separate text and audio guidance matching InfiniteTalk's default (text_scale=5.0, audio_scale=4.0)

## Tech Stack

- PyTorch (bf16), FSDP for multi-GPU
- FastGen training infrastructure (trainer, callbacks, checkpointing)
- Wan2.1-I2V-14B base model (7 safetensor shards from HuggingFace)
- InfiniteTalk audio modules (`infinitetalk.safetensors`)
- wav2vec2 (chinese-wav2vec2-base) for audio encoding
- CLIP ViT-H/14 for image encoding
- T5 UMT5-XXL for text encoding
- LoRA (rank=32-64) for parameter-efficient training

---

## 1. Network Roles and Memory

All three network roles use the same 14B architecture. FastGen instantiates them as separate `nn.Module` instances — there is **no automatic weight sharing** between teacher, student, and fake_score. FSDP shards each model's parameters across GPUs.

| Role | Class | Forward Mode | Trainable Params | When Used |
|------|-------|-------------|-----------------|-----------|
| Teacher | `InfiniteTalkWan` | Bidirectional (full sequence) | None (frozen base + merged LoRA if any) | SF Stage 2: 3-call CFG + VSD target |
| Student | `CausalInfiniteTalkWan` | Causal (AR chunks + KV cache) or full-sequence (KD/DF) | LoRA adapters + audio modules | All stages |
| Fake Score | `InfiniteTalkWan` | Bidirectional (full sequence) | LoRA adapters + audio modules | SF Stage 2: VSD loss |

**Memory estimates:**

*Stage 1 (DF or KD) — student only:*
- 1× 14B in bf16: ~28GB
- LoRA adapters + audio modules: ~0.5-1GB
- Activations with gradient checkpointing: ~15-20GB
- Optimizer states (AdamW on LoRA only): ~2-4GB
- Total: ~46-53GB across 2 GPUs (~23-27GB/GPU with FSDP). Comfortable.

*Stage 2 (Self-Forcing DMD) — all three models:*
- 3× 14B in bf16: ~84GB (no weight sharing — separate instantiations)
- LoRA adapters (student + fake_score): ~1-2GB
- Activations with gradient checkpointing: ~15-20GB
- Optimizer states (AdamW on LoRA only): ~2-4GB
- Total: ~102-110GB across 2 GPUs (~51-55GB/GPU with FSDP). **Tight but feasible.**

*OOM mitigations if needed:*
- CPU offload teacher between forward passes (teacher is frozen, only needed for CFG)
- Reduce batch size to 1
- Activation offloading (checkpoint + offload to CPU)
- Disable GAN entirely (already default)

---

## 2. Training Pipeline

```
Stage 0: Verification
    Prove ported DiT produces identical outputs to InfiniteTalk's original WanModel.
    Three levels: weight loading, single forward pass, full 40-step denoising.

Stage 1 (choose one):

    Option A — Diffusion Forcing (no teacher needed):
        Student (14B + LoRA, causal full-sequence mode)
        Add Gaussian noise to real data at per-chunk inhomogeneous timesteps
        Loss: L2(student_output, real_data)

    Option B — ODE KD (requires teacher ODE generation):
        Pre-generate ODE trajectories with 3-call CFG teacher
        Student (14B + LoRA, causal full-sequence mode)
        Loss: L2(student_output, clean_data)

Stage 2 — Self-Forcing DMD:
    Teacher (14B, frozen): 3-call CFG for VSD target
    Student (14B + LoRA, causal AR mode): generates via rollout
    Fake Score (14B + LoRA, bidirectional): VSD loss
    Optional: Discriminator (MLP on teacher features)
    Loss: VSD + optional GAN
```

---

## 3. File Structure

```
fastgen/networks/InfiniteTalk/
    __init__.py                        # Package exports
    wan_model.py                       # Component 1: Standalone DiT
    audio_modules.py                   # AudioProjModel + SingleStreamAttention
    network.py                         # Component 2: Bidirectional wrapper
    network_causal.py                  # Component 3: Causal wrapper
    lora.py                            # LoRA utilities (LoRALinear, apply/merge helpers)
fastgen/datasets/
    infinitetalk_dataloader.py         # Component 4: Dataset adapter
fastgen/methods/
    infinitetalk_self_forcing.py       # Component 5a: SF method subclass
    infinitetalk_kd.py                 # Component 5b: KD method subclass
    infinitetalk_diffusion_forcing.py  # Component 5c: DF method subclass
fastgen/configs/methods/
    config_infinitetalk_sf.py          # Component 6a: SF method config
    config_infinitetalk_kd.py          # Component 6b: KD method config
    config_infinitetalk_df.py          # Component 6c: DF method config
fastgen/configs/experiments/InfiniteTalk/
    __init__.py
    config_sf.py                       # Component 6d: SF experiment config
    config_kd.py                       # Component 6e: KD experiment config
    config_df.py                       # Component 6f: DF experiment config
scripts/
    generate_infinitetalk_ode_pairs.py # Component 7: ODE trajectory generation
    verify_infinitetalk_equivalence.py # Stage 0: Verification script
    precompute_infinitetalk_data.py    # Data preprocessing utility
```

---

## 4. Component Details

### 4.1 Standalone DiT (`wan_model.py` + `audio_modules.py`)

**Source:** InfiniteTalk's `wan/modules/multitalk_model.py`

**Port (keep):**
- `WanModel` class with all 40 transformer blocks
- `WanAttentionBlock`: self-attention + text/CLIP cross-attention + audio cross-attention + FFN
- `WanSelfAttention`, `WanI2VCrossAttention`
- `AudioProjModel`: wav2vec2 → windowed projection → 32 context tokens/frame
- `SingleStreamAttention`: audio cross-attention (single-speaker path)
- `Head`, `MLPProj` (CLIP projection)
- RoPE computation (`rope_params`, `rope_apply`)
- Sinusoidal time embedding
- Patch embedding (Conv3d, in_dim=36 → dim=5120) and unpatchify

**Strip (remove):**
- TeaCache acceleration logic
- VRAM management / model offloading
- Quantization support (optimum.quanto)
- xfuser / sequence parallel patches
- sageattn conditional import
- Multi-speaker `SingleStreamMutiAttention` (keep only `SingleStreamAttention` for single-speaker)
- `ref_target_masks` / human mask logic (single-speaker, not needed)

**Add for FastGen:**
- `feature_indices` parameter in `forward()` — collect intermediate block outputs for GAN discriminator
- `return_features_early` parameter — exit after collecting features
- `_unpatchify_features()` method — convert patched features to spatial tensors
- Configurable gradient checkpointing (`use_gradient_checkpointing` param)

**`audio_modules.py` contains:**
- `AudioProjModel`: projects windowed wav2vec2 features to context tokens
  - `proj1`: Linear(5×12×768=46080 → 512) for first frame
  - `proj1_vf`: Linear(8×12×768=73728 → 512) for latter frames (variable window within VAE temporal groups)
  - `proj2`: Linear(512 → 512)
  - `proj3`: Linear(512 → 32×768) → 32 context tokens per frame
- `SingleStreamAttention`: cross-attention from visual tokens to audio tokens
  - Per-frame: reshape sequence to `(B*N_t, S, C)`, cross-attend to audio, reshape back
  - Q from visual, KV from audio embeddings
  - Original uses `xformers.ops.memory_efficient_attention`; port replaces with
    `torch.nn.functional.scaled_dot_product_attention` (PyTorch native SDPA) or flash_attn
    for broader compatibility

### 4.2 Bidirectional Wrapper (`network.py`)

**Class:** `InfiniteTalkWan(FastGenNetwork)`

**Constructor params:**
- `base_model_paths: str` — comma-separated safetensor shard paths for base Wan I2V 14B
- `infinitetalk_ckpt_path: str` — path to `infinitetalk.safetensors`
- `lora_ckpt_path: Optional[str]` — path to LoRA checkpoint (for merging into teacher)
- `lora_rank: int` — LoRA rank (default 32)
- `net_pred_type: str = "flow"`
- `schedule_type: str = "rf"`
- `shift: float = 7.0`
- `disable_grad_ckpt: bool = False`

**Weight loading (`_load_weights()`):**
1. Load base Wan I2V safetensors (7 shards) → merge into single state dict
2. Load `infinitetalk.safetensors` → merge (audio modules + any overridden base weights)
3. Handle key name mapping between InfiniteTalk's naming and our DiT's naming
4. If `lora_ckpt_path` provided: merge LoRA into base weights (`W = W + alpha/rank * B @ A`, on GPU for speed)
5. Load into `self.model`

**`forward(x_t, t, condition, fwd_pred_type, feature_indices, return_features_early, ...)`:**
1. Extract from condition dict: `text_embeds`, `first_frame_cond`, `clip_features`, `audio_emb`
2. Construct 4ch temporal mask deterministically in the wrapper:
   ```python
   # First frame is conditioned (mask=1), rest are noisy (mask=0)
   # Follows InfiniteTalk's mask construction in multitalk.py lines 552-559
   msk = torch.ones(B, 1, frame_num, H_lat, W_lat)
   msk[:, :, 1:] = 0
   # Expand first frame to match VAE temporal stride, reshape to [B, 4, T_lat, H_lat, W_lat]
   msk = _construct_i2v_mask(frame_num, H_lat, W_lat, device, dtype)
   ```
3. Build `y = cat([msk, first_frame_cond], dim=1)` — 20ch condition tensor
4. Rescale timestep: `timestep = noise_scheduler.rescale_t(t)`
5. Forward through DiT: the model's own `forward()` concatenates `x` and `y` channel-wise
   (`x = [cat(u, v) for u, v in zip(x, y)]`) producing the 36ch input, then passes through
   patch_embedding, transformer blocks (with text/CLIP cross-attn + audio cross-attn), and head
6. Convert model output: `noise_scheduler.convert_model_output(x_t, model_output, t)`
7. Return output (and features if requested)

**Note on tensor convention:** InfiniteTalk's `WanModel.forward()` accepts `x` and `y` as **lists** of
unbatched tensors (one element per sample). Our standalone DiT port will adopt the same list convention
to maintain compatibility for verification. The wrapper handles the batched→list conversion.

**Properties:**
- `noise_scheduler`: `RFNoiseSchedule(shift=7.0)`
- `net_pred_type`: `"flow"`

### 4.3 Causal Wrapper (`network_causal.py`)

**Class:** `CausalInfiniteTalkWan(CausalFastGenNetwork)`

**Same base DiT as bidirectional** but with causal modifications:

**Causal attention components (following OmniAvatar pattern):**
- `CausalSelfAttention`: replaces `WanSelfAttention` with KV cache read/write
- `CausalDiTBlock`: wraps `WanAttentionBlock` with per-frame modulation
- `CausalHead`: per-frame head modulation
- `causal_rope_apply`: RoPE with frame offset (`start_frame` parameter)
- `_build_block_mask`: FlexAttention chunk-wise causal mask for full-sequence mode

**Two forward modes:**

`_forward_full_sequence(is_ar=False)` — Used by KD and DF:
- Input: full noisy latent `[B, 16, 21, H, W]` with per-frame timesteps `[B, 21]`
- FlexAttention chunk-wise causal block mask (chunk_size=3, 7 chunks)
- Per-frame timestep: flatten → embed → reshape to `[B, num_frames, dim]`
- Audio: process full sequence, aligned per-frame

`_forward_ar(is_ar=True)` — Used by Self-Forcing:
- Input: single chunk `[B, 16, chunk_frames, H, W]` with scalar timestep `[B]`
- KV cache: pre-allocated for `total_num_frames` (21), reads cached past, writes current
- Dynamic RoPE: `start_frame` offset applied so positions are absolute
- Audio: processed once and cached (`self._cached_audio`), sliced per chunk
- Condition `y`: sliced to current chunk's frames

**Cache management:**
- `_init_caches(batch_size, total_tokens, frame_seqlen, device, dtype)` — pre-allocate K/V per block
- `clear_caches()` — reset all caches + `_cached_audio` between samples

**LoRA training:**
- `_init_lora(rank, alpha)` — wraps all eligible `nn.Linear` layers with `LoRALinear`
- `freeze_base()` — sets `requires_grad=False` on all base params, `True` on LoRA + audio modules
- Checkpoint save/load for LoRA params only

**Auto-detect routing (same as OmniAvatar):**
```python
def forward(self, x_t, t, condition, is_ar=True, ...):
    if t.dim() == 2:  # per-frame timesteps from sample_t_inhom
        return self._forward_full_sequence(...)
    elif is_ar:
        return self._forward_ar(...)
    else:
        return self._forward_full_sequence(...)
```

### 4.4 LoRA Utilities (`lora.py`)

```python
class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adapters."""
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float):
        self.base = base_linear  # frozen
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_out = F.linear(x, self.base.weight, self.base.bias)  # no grad
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

def apply_lora(model, rank, alpha, target_modules=None):
    """Replace eligible nn.Linear layers with LoRALinear."""

def merge_lora(model):
    """Merge LoRA weights into base: W = W + scaling * B @ A. For inference/teacher."""

def extract_lora_state_dict(model):
    """Extract only LoRA A/B params for checkpoint saving."""

def load_lora_state_dict(model, state_dict):
    """Load LoRA params into existing LoRALinear layers."""
```

**Target modules:** All `nn.Linear` in transformer blocks (Q, K, V, O, FFN). Excludes: `patch_embedding` (Conv3d), `time_embedding`, `text_embedding`, `head`, `img_emb`, all audio modules.

### 4.5 Dataset Adapter (`infinitetalk_dataloader.py`)

**Pre-computed tensors per sample directory:**

| File | Tensor Shape | Description |
|------|-------------|-------------|
| `vae_latents.pt` | `[16, 21, H_lat, W_lat]` | VAE-encoded 81-frame video |
| `first_frame_cond.pt` | `[16, 21, H_lat, W_lat]` | VAE-encoded first frame + zero padding |
| `clip_features.pt` | `[1, 257, 1280]` | CLIP ViT-H/14 on reference frame |
| `audio_emb.pt` | `[num_video_frames, 12, 768]` | wav2vec2 hidden states (all 12 layers), interpolated to 25fps |
| `text_embeds.pt` | `[1, 512, 4096]` | T5 UMT5-XXL text embeddings |
| `ode_path.pt` | `[num_steps, 16, 21, H_lat, W_lat]` | ODE trajectory (KD only) |

**Shared tensors (loaded once):**
- `neg_text_embeds.pt` — negative prompt T5 embeddings (same for all samples)

**Dataset output dict:**

| Key | Shape | Used By |
|-----|-------|---------|
| `real` | `[16, 21, H, W]` | All stages |
| `first_frame_cond` | `[16, 21, H, W]` | All stages |
| `clip_features` | `[1, 257, 1280]` | All stages |
| `audio_emb` | `[81, 12, 768]` | All stages |
| `text_embeds` | `[1, 512, 4096]` | All stages |
| `neg_text_embeds` | `[1, 512, 4096]` | SF (neg condition) |
| `path` | `[num_steps, 16, 21, H, W]` | KD only |

**Audio windowing** is applied in the model forward (inside `WanModel.forward()`), not the dataloader.
The dataloader outputs raw per-frame embeddings `[num_video_frames, 12, 768]`. The pipeline's
denoising loop (or the wrapper's `forward()`) applies the 5-frame sliding window to produce
`[1, clip_length, 5, 12, 768]` before passing to the DiT. The `AudioProjModel` inside the DiT then
processes this windowed input. This matches InfiniteTalk's original behavior where windowing happens
in `generate_infinitetalk()` (multitalk.py lines 523-533), not inside `WanModel.forward()`.

**Mask construction:** The 4ch temporal mask is **not** stored in the dataset. It is constructed
deterministically by the network wrapper at forward time (first frame = conditioned, rest = noisy).
This follows InfiniteTalk's original pattern (multitalk.py lines 552-559). The mask depends only on
`frame_num` and spatial dimensions, which are known at forward time.

**`InfiniteTalkDataLoader` class:** Infinite iterator with `DistributedSampler` for multi-GPU, same pattern as OmniAvatar's `OmniAvatarDataLoader`.

### 4.6 Method Subclasses

**5a: `InfiniteTalkSelfForcingModel(SelfForcingModel)`**

Overrides:
- `_prepare_training_data(data)` — maps dataset output to standard 3-tuple (real_data, condition, neg_condition)
- `_compute_teacher_x0(...)` or equivalent — overrides CFG computation to perform 3 forward passes

**Important:** The base `DMD2Model.single_train_step()` destructures exactly 3 return values:
`real_data, condition, neg_condition = self._prepare_training_data(data)`. We **cannot** return 4
values. Instead, we embed the `neg_text_embeds` inside the `condition` dict and override the
teacher CFG computation.

```python
def _prepare_training_data(self, data):
    real_data = data["real"]
    condition = {
        "text_embeds": data["text_embeds"],
        "first_frame_cond": data["first_frame_cond"],
        "clip_features": data["clip_features"],
        "audio_emb": data["audio_emb"],
        # Embed neg_text for 3-call CFG (used by our CFG override, not by base class)
        "neg_text_embeds": data["neg_text_embeds"],
    }
    # neg_condition: drop BOTH text and audio (standard uncond)
    neg_condition = {
        **condition,
        "text_embeds": data["neg_text_embeds"],
        "audio_emb": torch.zeros_like(data["audio_emb"]),
    }
    return real_data, condition, neg_condition  # standard 3-tuple
```

The 3-call CFG is integrated by overriding the method that computes the teacher's guided output
(either `_apply_classifier_free_guidance` or the relevant section of `_student_update_step`).
The override constructs the intermediate `neg_text_condition` on the fly from `condition`:

```python
def _compute_teacher_guided_output(self, x_t, t, condition, neg_condition):
    # Full condition (text + audio)
    teacher_cond = self.teacher(x_t, t, condition)
    # Drop text only (neg text + audio)
    neg_text_condition = {**condition, "text_embeds": condition["neg_text_embeds"]}
    teacher_drop_text = self.teacher(x_t, t, neg_text_condition)
    # Drop both (neg text + zero audio) — this is neg_condition
    teacher_uncond = self.teacher(x_t, t, neg_condition)

    text_scale = self.config.text_guide_scale  # 5.0
    audio_scale = self.config.audio_guide_scale  # 4.0
    teacher_x0 = teacher_uncond + text_scale * (teacher_cond - teacher_drop_text) \
                                + audio_scale * (teacher_drop_text - teacher_uncond)
    return teacher_x0
```

**Note on linearity:** The CFG formula is applied to x0 predictions (after `convert_model_output`).
Since the flow→x0 conversion is linear (`x0 = x_t - v*t`), applying CFG before or after conversion
yields identical results. This matches InfiniteTalk's original which applies CFG to raw flow
predictions before the Euler step.

**5b: `InfiniteTalkKDModel(CausalKDModel)`**

Overrides:
- `_build_condition(data)` — same as SF's positive condition dict
- `single_train_step(data, iteration)` — gathers from ODE path, computes L2 loss

**5c: `InfiniteTalkDiffusionForcingModel(KDModel)`**

Overrides:
- `_build_condition(data)` — same condition dict
- `single_train_step(data, iteration)` — adds noise to real data at inhomogeneous timesteps, L2 loss

### 4.7 Experiment Configs

**SF config (`config_sf.py`):**
```python
Teacher_Config = L(InfiniteTalkWan)(
    base_model_paths="...Wan2.1-I2V-14B shards...",
    infinitetalk_ckpt_path="...infinitetalk.safetensors",
    shift=7.0,
)
Student_Config = L(CausalInfiniteTalkWan)(
    base_model_paths="...same shards...",
    infinitetalk_ckpt_path="...infinitetalk.safetensors",
    lora_rank=32,
    chunk_size=3,
    total_num_frames=21,
    shift=7.0,
)
FakeScore_Config = L(InfiniteTalkWan)(
    base_model_paths="...same shards...",
    infinitetalk_ckpt_path="...infinitetalk.safetensors",
    lora_rank=32,
    shift=7.0,
)

config.model.net = Student_Config
config.model.teacher = Teacher_Config
config.model.fake_score_net = FakeScore_Config
config.model.load_student_weights = False
config.model.gan_loss_weight_gen = 0  # GAN disabled to save VRAM
config.model.sample_t_cfg.shift = 7.0
config.model.text_guide_scale = 5.0
config.model.audio_guide_scale = 4.0
```

**Config plumbing for guide scales:** The base `BaseModelConfig` has `guidance_scale` (single float).
We add `text_guide_scale` and `audio_guide_scale` to the SF method config class
(`InfiniteTalkSFModelConfig`) as new `attrs` fields. These flow to the method class via
`self.config.text_guide_scale` / `self.config.audio_guide_scale` in the 3-call CFG override.

**KD config (`config_kd.py`):** Student only, `load_ode_path=True` in dataloader.

**DF config (`config_df.py`):** Student only, no ODE paths needed.

### 4.8 ODE Trajectory Generation (`generate_infinitetalk_ode_pairs.py`)

**Pipeline per sample:**
1. Load pre-computed data (VAE latents, audio, text, CLIP, mask)
2. Build condition, neg_text_condition, neg_condition dicts
3. Sample noise
4. Run deterministic multi-step ODE solve with 3-call CFG:
   ```
   for step in range(num_steps):
       x0_cond = teacher(x_t, t, condition)
       x0_drop_text = teacher(x_t, t, neg_text_condition)
       x0_uncond = teacher(x_t, t, neg_condition)
       x0 = x0_uncond + text_scale * (x0_cond - x0_drop_text)
                       + audio_scale * (x0_drop_text - x0_uncond)
       eps = noise_scheduler.x0_to_eps(x_t, x0, t)
       x_t_next = noise_scheduler.forward_process(x0, eps, t_next)
       trajectory.append(x_t)
   ```
5. Subsample trajectory at target `t_list` indices
6. Save `ode_path.pt` per sample

---

## 5. Verification Plan (Stage 0)

This is a **hard gate** — no training work proceeds until verification passes.

### Step 1: Weight Loading Verification

- Load base Wan I2V safetensors + `infinitetalk.safetensors` into our standalone DiT
- Compare parameter names and shapes against InfiniteTalk's original `WanModel`
- Verify total parameter count matches
- Verify audio module weights are correctly loaded (spot-check a few tensors for value equality)

### Step 2: Single Forward Pass Equivalence

- Load identical weights into both our DiT and InfiniteTalk's original `WanModel`
- Construct identical inputs:
  - Noise latent: random `[16, 21, H_lat, W_lat]`
  - Timestep: scalar `[t]`
  - Text context: random `[1, 512, 4096]`
  - CLIP features: random `[1, 257, 1280]`
  - Audio embeddings: random `[1, 81, 5, 12, 768]` (pre-windowed, matching InfiniteTalk's convention at the DiT input boundary)
  - Condition y (mask + VAE ref): `[20, 21, H_lat, W_lat]` (mask constructed deterministically, VAE ref from first frame)
- Run both models in eval mode with `torch.no_grad()`
- Assert `torch.allclose(output_ours, output_original, atol=1e-3)` (bf16 tolerance)
- Test with at least 3 different random seeds

### Step 3: Full Denoising Equivalence

- Run a complete 40-step Euler ODE sample with:
  - Our pipeline: `InfiniteTalkWan.sample()` with shift=7.0, text_scale=5.0, audio_scale=4.0
  - InfiniteTalk's pipeline: `InfiniteTalkPipeline.generate_infinitetalk()` with same params
- Use identical noise seed, same conditioning inputs
- Compare final denoised latents: `torch.allclose(atol=1e-2)` (accumulated error over 40 steps)
- Optionally: VAE decode both and visually compare

### Verification Script

`scripts/verify_infinitetalk_equivalence.py`:
- Loads both models
- Runs all 3 verification steps
- Prints PASS/FAIL with max absolute difference for each step
- Exits with error code if any step fails

---

## 6. Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Model size | 14B (dim=5120, 40 layers, 40 heads) | InfiniteTalk config |
| `in_dim` | 36 (16 noise + 4 mask + 16 VAE ref) | Wan I2V architecture |
| `out_dim` | 16 | Wan I2V architecture |
| VAE stride | (4, 8, 8) | Wan VAE |
| Patch size | (1, 2, 2) | InfiniteTalk config |
| Noise schedule | Rectified flow, shift=7.0 (480p) | InfiniteTalk default |
| `text_guide_scale` | 5.0 | InfiniteTalk default |
| `audio_guide_scale` | 4.0 | InfiniteTalk default |
| Sampling steps | 40 (teacher) | InfiniteTalk default |
| Frame count | 81 pixel → 21 latent | InfiniteTalk default |
| `chunk_size` | 3 (7 chunks of 3 latent frames) | FastGen default |
| Audio encoder | chinese-wav2vec2-base (12 layers × 768 dim) | InfiniteTalk |
| Audio window | 5 frames centered | InfiniteTalk |
| Audio context tokens | 32 per frame | InfiniteTalk AudioProjModel |
| LoRA rank | 32 (configurable) | Design decision |
| LoRA target | All nn.Linear in transformer blocks | Design decision |
| Resolution | 640×640 (480p default bucket) | InfiniteTalk default |
| FPS | 25 | InfiniteTalk default |
| Audio sample rate | 16kHz | InfiniteTalk default |

---

## 7. Reference Documents

- **InfiniteTalk pipeline analysis:** `reference_FastGen_OmniAvatar/FastGen/docs/infinitetalk-analysis.md`
- **FastGen adaptation guide:** `reference_FastGen_OmniAvatar/FastGen/docs/fastgen-adaptation-guide.md`
- **OmniAvatar adaptation (working reference):** `reference_FastGen_OmniAvatar/FastGen/`
- **Original FastGen (starting point):** `original_FastGen/FastGen/`
- **InfiniteTalk source (read-only):** `reference_FastGen_InfiniteTalk/InfiniteTalk/`

## 8. Out of Scope (Deferred)

- Multi-speaker support (spatial attention routing, `SingleStreamMutiAttention`)
- 720p resolution (shift=11.0, different aspect ratio buckets)
- Streaming/infinite-length generation
- TeaCache or other inference acceleration
- Quantization (int8/fp8)
- APG (Adaptive Projected Guidance)
- Color correction
