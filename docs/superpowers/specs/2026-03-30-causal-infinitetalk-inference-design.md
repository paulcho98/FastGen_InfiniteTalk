# Causal InfiniteTalk Inference — Design Spec

**Date:** 2026-03-30
**Script location:** `FastGen_InfiniteTalk/scripts/inference/inference_causal.py`
**Reference implementation:** `reference_FastGen_OmniAvatar/FastGen/scripts/inference/inference_causal.py`
**Porting guide:** `reference_FastGen_OmniAvatar/FastGen/docs/causal-inference-porting-guide.md`

---

## 1. Overview

Block-wise autoregressive inference script for `CausalInfiniteTalkWan` — the 14B causal DiT student trained via Diffusion Forcing (Stage 1) or Self-Forcing (Stage 2). Takes a reference image (or video first frame) + audio and generates a lip-synced talking-head video.

**No CFG.** The Self-Forcing pipeline distills the teacher's 3-call CFG outputs into the student, so the student generates with a single conditioned forward pass per denoising step.

**Audio determines length.** Generation length is derived from audio duration, rounded down to the nearest multiple of `chunk_size` latent frames.

**Batch-ready architecture.** The script is structured so that model loading happens once, then a per-sample loop processes each input. This makes it straightforward to extend to batch processing (directory of inputs, metadata file, etc.) in a follow-up without restructuring.

---

## 2. Pipeline Stages

```
=== ONE-TIME SETUP ===
1. Parse args
2. Load T5 (if --prompt) → encode text → unload T5
3. Load VAE (kept for entire session — used for both encode and decode)
4. Load CLIP, wav2vec2 encoders
5. [For each sample] Encode inputs:
   a. Reference image → VAE encode → first_frame_cond [16, T_lat, H_lat, W_lat]
   b. Reference image → CLIP encode → clip_features [1, 257, 1280]
   c. Audio → wav2vec2 encode → audio_emb [num_video_frames, 12, 768]
   d. Compute generation length from audio duration
6. Unload CLIP + wav2vec2, free VRAM
7. Load CausalInfiniteTalkWan (14B) + checkpoint

=== PER-SAMPLE LOOP ===
8. Build condition dict (from encoded tensors or pre-computed .pt files)
9. Set model.total_num_frames, clear caches
10. Block-wise AR loop (no CFG, no gradients)
11. VAE decode (float32)
12. Save video + mux audio
13. Clear caches, free intermediate tensors
```

Sequential encoder loading is critical: the 14B model (~28GB bf16) + VAE (~1GB) + KV caches leave limited headroom on an 80GB A100. T5 (20GB), CLIP (~2GB), and wav2vec2 (~360MB) are loaded, used, and unloaded before the DiT.

Steps 1-7 (setup + encode + load model) happen once. Steps 8-13 repeat per sample in batch mode. The VAE is kept loaded throughout (needed for both encode and decode, ~1GB). For batch mode, CLIP + wav2vec2 encode all samples first (step 5), then unload before loading the DiT.

---

## 3. Input Modes

### 3.1 Raw inputs (primary)

```
--image /path/to/reference.png        # Reference face image
--audio /path/to/audio.wav            # Driving audio
# OR
--video /path/to/video.mp4            # Extract first frame + audio
```

When `--video` is provided, extract:
- First frame as reference image (PIL/cv2)
- Audio track via ffmpeg → temp wav file

When `--image` + `--audio` are provided, use directly.

Text conditioning:
```
--prompt "A person speaking"           # Free-text (load T5, encode, unload)
# OR
--text_embeds_path /path/to/text_embeds.pt  # Pre-computed [1, 512, 4096]
```

If neither is provided, use zeros `[1, 512, 4096]` (unconditional text, acceptable since CFG is not used).

### 3.2 Pre-computed inputs (debugging)

```
--precomputed_dir /path/to/sample_dir/
```

Loads directly from pre-computed `.pt` files matching the training dataloader format:
- `vae_latents.pt` → `[16, T_lat, H, W]` (only used to derive spatial dims if needed)
- `first_frame_cond.pt` → `[16, T_lat, H, W]` (sliced to `num_latent_frames`)
- `clip_features.pt` → `[1, 257, 1280]`
- `audio_emb.pt` → `[T_video, 12, 768]` (sliced to `num_video_frames`)
- `text_embeds.pt` → `[1, 512, 4096]`

Also accepts the `_81frames` variants (`audio_emb_81frames.pt`, `first_frame_cond_81frames.pt`) if present — these are pre-sliced to 81 video frames / 21 latent frames.

When using `--precomputed_dir`, no encoders need to be loaded. Only the DiT + VAE (for decode) are loaded.

### 3.3 Neg text embedding (for potential future CFG)

```
--neg_text_embeds_path /path/to/neg_text_embeds.pt  # Optional, not used without CFG
```

Stored but not used in the current no-CFG design. If CFG is added later, this provides the negative text embedding for the unconditional call.

---

## 4. Generation Length Computation

Audio duration determines generation length:

```python
duration = librosa.get_duration(path=audio_path)
fps = 25  # InfiniteTalk uses 25 FPS (generate_infinitetalk.py line 325, save_video_ffmpeg default)
num_video_raw = int(duration * fps)  # floor

# VAE temporal compression: first frame is 1:1, then groups of 4
num_latent_raw = 1 + (num_video_raw - 1) // 4

# Round DOWN to nearest multiple of chunk_size
chunk_size = 3
num_latent = (num_latent_raw // chunk_size) * chunk_size
num_latent = max(num_latent, chunk_size)  # at least one chunk

# Back to video frames
num_video = 1 + (num_latent - 1) * 4

# Allow override
if args.num_latent_frames is not None:
    num_latent = args.num_latent_frames
    assert num_latent % chunk_size == 0
    num_video = 1 + (num_latent - 1) * 4
```

For pre-computed data, use the audio tensor's first dimension directly:
```python
num_video = audio_emb.shape[0]  # already 81 or 93
num_latent = 1 + (num_video - 1) // 4
num_latent = (num_latent // chunk_size) * chunk_size
num_video = 1 + (num_latent - 1) * 4
audio_emb = audio_emb[:num_video]
```

---

## 5. Encoding Pipeline

### 5.1 Reference image → VAE latent

```python
# Input: PIL Image or numpy array [H, W, 3] (uint8)
# Resize to target resolution (448, 896) — center crop to maintain aspect ratio
# Normalize to [-1, 1]
# Stack as single-frame video: [1, 3, 1, H, W]
# VAE encode (float32): [1, 16, 1, H_lat, W_lat]
# Expand to T_lat frames: first_frame_cond [1, 16, T_lat, H_lat, W_lat]
#   Frame 0 = encoded ref, frames 1..T_lat-1 = zeros
# (Matches first_frame_cond.pt format from precompute script)
```

The `first_frame_cond` tensor has the reference frame in position 0 and zeros elsewhere. The model's `_build_y()` constructs the I2V mask (4 channels: frame 0 = 1, rest = 0) and concatenates with `first_frame_cond` (16 channels) to form the 20-channel conditioning input.

### 5.2 Reference image → CLIP features

```python
# Load CLIP ViT-H/14 (open_clip xlm-roberta-large-vit-huge-14)
# Preprocess image (resize, normalize per CLIP spec)
# Encode: clip_features [1, 257, 1280] (CLS + 256 patch tokens)
```

Reuse `_load_clip()` and `_encode_clip()` from the lazy dataloader (`infinitetalk_dataloader.py` lines 147-175).

### 5.3 Audio → wav2vec2 embeddings

```python
# Load wav2vec2 (Chinese base model from InfiniteTalk weights)
# Load audio at 16kHz via librosa
# Feature extractor → input_values
# wav2vec2 forward with output_hidden_states=True
# Stack hidden states from layers 1-12: [seq_len, 12, 768]
# Slice to num_video_frames
```

Reuse `_encode_audio()` from the lazy dataloader (`infinitetalk_dataloader.py` lines 367-409). Note: InfiniteTalk uses layers 1-12 (12 layers, 768 dim each) NOT all 14 layers concatenated like OmniAvatar (which gives 10752 dim).

**Wav2Vec2 SDPA gotcha:** Load with `attn_implementation="eager"` to avoid SDPA incompatibility in newer transformers versions.

### 5.4 Text → T5 embeddings

```python
# If --prompt provided:
#   Load T5 UMT5-XXL encoder (bf16)
#   Tokenize prompt (max 512 tokens)
#   Encode: text_embeds [1, 512, 4096]
#   Unload T5 (del model, torch.cuda.empty_cache())
# Elif --text_embeds_path provided:
#   Load .pt file directly
# Else:
#   Use zeros [1, 512, 4096]
```

The T5 encoder path is from InfiniteTalk weights dir: `models_t5_umt5-xxl-enc-bf16.pth`. The tokenizer is `google/umt5-xxl`.

---

## 6. Model Loading

### 6.1 Constructor

```python
model = CausalInfiniteTalkWan(
    base_model_paths=args.base_model_paths,       # 7 safetensor shards
    infinitetalk_ckpt_path=args.infinitetalk_ckpt, # infinitetalk.safetensors
    lora_ckpt_path="",                             # No pre-merge LoRA (SF ckpt replaces)
    lora_rank=32,
    lora_alpha=32,
    chunk_size=3,
    total_num_frames=num_latent,                   # Set from audio duration
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
    local_attn_size=args.local_attn_size,          # -1 = global (default)
    sink_size=0,
    use_dynamic_rope=False,
)
```

The constructor handles:
1. Load base Wan I2V-14B shards (with patch_embedding expansion 16→36)
2. Load InfiniteTalk checkpoint (audio modules + weight overrides)
3. Apply runtime LoRA adapters + freeze base

### 6.2 Checkpoint overlay (DF or SF)

```python
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)

# Handle FSDP checkpoint nesting
if isinstance(ckpt, dict):
    if "model" in ckpt and isinstance(ckpt["model"], dict) and "net" in ckpt["model"]:
        state_dict = ckpt["model"]["net"]
    elif "net" in ckpt:
        state_dict = ckpt["net"]
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# Load (InfiniteTalk has no _core prefix, so no prefix handling needed)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Checkpoint: {len(state_dict)} params, {len(missing)} missing, {len(unexpected)} unexpected")

# Verify: LoRA keys should match, base keys should be unexpected (already loaded by constructor)
# Expected: missing=0 for LoRA+audio keys, unexpected=many (base weights already in model)
```

### 6.3 Move to device

```python
model = model.to(device=device, dtype=torch.bfloat16)
model.eval()
```

---

## 7. Condition Dict

```python
condition = {
    "text_embeds": text_embeds,           # [1, 512, 4096]
    "first_frame_cond": first_frame_cond, # [1, 16, T_lat, H_lat, W_lat]
    "clip_features": clip_features,       # [1, 257, 1280]
    "audio_emb": audio_emb,              # [1, num_video_frames, 12, 768]
}
```

All tensors moved to device and cast to model dtype (bf16) before the AR loop.

Note: `audio_emb` is 4D `[B, T, 12, 768]`. The model's `forward()` applies the 5-frame sliding window internally (lines 1777-1784 of `network_causal.py`), converting to `[B, T, 5, 12, 768]` before passing to `_process_audio()`.

---

## 8. AR Inference Loop

```python
@torch.no_grad()
def run_inference(model, condition, num_latent_frames, t_list,
                  chunk_size, context_noise, seed, device, dtype):
    B = 1
    C = 16  # latent channels
    # Derive spatial dims from first_frame_cond
    H_lat = condition["first_frame_cond"].shape[3]
    W_lat = condition["first_frame_cond"].shape[4]

    model.total_num_frames = num_latent_frames
    model.clear_caches()

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(B, C, num_latent_frames, H_lat, W_lat,
                        generator=generator, device=device, dtype=dtype)
    output = torch.zeros_like(noise)

    t_list_t = torch.tensor(t_list, device=device, dtype=torch.float64)
    num_blocks = num_latent_frames // chunk_size

    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * chunk_size
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        # Multi-step denoising (4 steps)
        for step_idx in range(len(t_list_t) - 1):
            t_cur = t_list_t[step_idx]
            t_next = t_list_t[step_idx + 1]

            x0_pred = model(
                noisy_input,
                t_cur.float().expand(B),
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=False,
                is_ar=True,
                fwd_pred_type="x0",
                use_gradient_checkpointing=False,
            )

            if t_next > 0:
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.float().expand(B),
                )
            else:
                noisy_input = x0_pred

        output[:, :, cur_start_frame:cur_start_frame + chunk_size] = x0_pred

        # Cache update: store denoised output for next block's context
        cache_input = x0_pred
        t_cache = torch.full((B,), context_noise, device=device, dtype=dtype)
        if context_noise > 0:
            cache_eps = torch.randn_like(x0_pred)
            cache_input = model.noise_scheduler.forward_process(
                x0_pred, cache_eps,
                torch.tensor(context_noise, device=device, dtype=torch.float64).expand(B),
            )

        model(
            cache_input,
            t_cache,
            condition=condition,
            cur_start_frame=cur_start_frame,
            store_kv=True,
            is_ar=True,
            fwd_pred_type="x0",
            use_gradient_checkpointing=False,
        )

    model.clear_caches()
    return output
```

### 8.1 Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `chunk_size` | 3 | Latent frames per AR block |
| `t_list` | `[0.999, 0.955, 0.875, 0.700, 0.0]` | 4 denoising steps, shift=7.0 |
| `context_noise` | 0.0 | Clean context (default from config_sf.py) |
| `store_kv=False` | During denoising | Don't pollute cache with noisy intermediates |
| `store_kv=True` | After block completes | Cache clean output for next block |
| `is_ar=True` | Always | Triggers `_forward_ar` with KV cache |
| `fwd_pred_type="x0"` | Always | Predict clean output directly |

### 8.2 Forward call count

Per block: 4 denoising steps + 1 cache update = 5 forward passes.
For 21 latent frames / 3 chunk_size = 7 blocks → **35 total forward passes**.

---

## 9. Post-Processing

### 9.1 VAE decode

```python
# VAE must stay loaded (used for both encode and decode)
# Always float32 for VAE
latent_for_vae = output[0].to(torch.float32)  # [16, T_lat, H_lat, W_lat]
video_tensor = vae.decode(latent_for_vae)
# Returns [3, T_video, H, W] in [-1, 1] range
```

### 9.2 Normalize and save

```python
video = (video_tensor.clamp(-1, 1) + 1) / 2 * 255
video = video.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # [T, H, W, 3]

# Save silent video via imageio
import imageio.v3 as iio
iio.imwrite(silent_path, video, fps=fps, codec="libx264")

# Mux audio
ffmpeg = _get_ffmpeg()
subprocess.run([
    ffmpeg, "-y",
    "-i", silent_path,
    "-i", audio_path,
    "-c:v", "copy", "-c:a", "aac",
    "-t", str(num_video / fps),  # clip audio to video duration
    output_path
], check=True)
```

---

## 10. CLI Arguments

```
Required (raw mode):
  --image PATH              Reference face image
  --audio PATH              Driving audio file
  OR
  --video PATH              Video file (extract first frame + audio)

Required (precomputed mode):
  --precomputed_dir PATH    Directory with .pt files

Required (always):
  --output_path PATH        Output video path
  --ckpt_path PATH          DF or SF trained checkpoint
  --base_model_paths STR    Comma-separated safetensor shard paths
  --infinitetalk_ckpt PATH  infinitetalk.safetensors
  --vae_path PATH           Wan2.1_VAE.pth
  --wav2vec_path PATH       wav2vec2 model directory

Optional:
  --prompt STR              Text prompt (loads T5 for encoding)
  --text_embeds_path PATH   Pre-computed text_embeds.pt
  --neg_text_embeds_path PATH  Negative text embeddings (unused without CFG)
  --t5_path PATH            T5 UMT5-XXL encoder path (required if --prompt)
  --clip_path PATH          CLIP ViT-H/14 path
  --num_latent_frames INT   Override generation length (must be divisible by 3)
  --chunk_size INT          AR chunk size (default: 3)
  --seed INT                Random seed (default: 42)
  --context_noise FLOAT     Context noise for cache update (default: 0.0)
  --local_attn_size INT     Rolling attention window in frames (-1=global, default: -1)
  --dtype STR               Model dtype (default: bfloat16)
  --fps INT                 Output video FPS (default: 25)
```

### 10.2 Batch mode (future extension point)

The script architecture separates:
- **One-time setup**: model loading, encoder loading
- **Per-sample processing**: encode inputs → build condition → AR loop → decode → save

This makes it straightforward to add batch args like:
```
--input_dir PATH           Directory of video/audio files
--input_list PATH          Metadata file (CSV/txt) with per-sample args
--output_dir PATH          Output directory (one video per input)
```

The per-sample loop would iterate over inputs, calling the same `encode_inputs()` → `run_inference()` → `decode_and_save()` pipeline. Model loading happens once. The `model.total_num_frames` and `model.clear_caches()` are called per sample to handle variable-length generation.

---

## 11. Differences from OmniAvatar Reference

| Aspect | OmniAvatar | InfiniteTalk |
|--------|------------|--------------|
| Model size | 1.3B | 14B (LoRA) |
| Conditioning | V2V 65ch (ref+mask+masked_video+ref_seq) | I2V 20ch (4-ch temporal mask + 16-ch first_frame_cond) |
| `in_dim` | 65 | 36 |
| Audio format | `[B, T_video, 10752]` (14 layers concat) | `[B, T_video, 12, 768]` (layers 1-12 separate) |
| Audio processing | `AudioPack` additive injection | `AudioProjModel` + `SingleStreamAttention` cross-attn |
| CLIP | None | Yes — `MLPProj(1280→5120)` + I2V cross-attention |
| Cross-attention | Text only (`WanCrossAttention`) | I2V: `CausalI2VCrossAttention` (text + CLIP image) |
| CFG | No (single conditioned pass) | No (SF distills CFG; same single-pass approach) |
| Spatial mask | LatentSync mouth mask | None (I2V temporal mask only) |
| Resolution | 512×512 | 448×896 |
| Shift | 3.0 | 7.0 |
| t_list | `[0.999, 0.900, 0.750, 0.500, 0.0]` | `[0.999, 0.955, 0.875, 0.700, 0.0]` |
| FPS | 25 | 25 |
| Length | Fixed 21 latent / 81 video | Variable, audio-determined |
| `_core` wrapper | Yes | No |
| Weight loading | Constructor (base + OA ckpt + merge LoRA) | Constructor (base shards + IT ckpt + runtime LoRA) |
| Checkpoint overlay | SF student weights | DF or SF LoRA + audio weights |

---

## 12. Gotchas (InfiniteTalk-specific)

1. **No `_core` prefix** — Unlike OmniAvatar, InfiniteTalk's `CausalInfiniteTalkWan` does not wrap submodules in `self._core`. State dict keys match directly. FSDP checkpoint nesting still applies.

2. **Audio is `[T, 12, 768]` not `[T, 10752]`** — InfiniteTalk uses wav2vec2 layers 1-12 kept separate. The model's `forward()` applies 5-frame sliding window internally when `audio_emb.dim() == 4`.

3. **VAE requires float32** — Same gotcha as OmniAvatar. Always `.to(torch.float32)` before VAE encode/decode.

4. **FPS is 25** — InfiniteTalk uses 25 FPS (confirmed in `generate_infinitetalk.py` line 325 and `save_video_ffmpeg` default). Audio frame count = `int(duration * 25)`. The `sample_fps=16` in `shared_config.py` is the base Wan default and does NOT apply to InfiniteTalk.

5. **`first_frame_cond` construction** — The precomputed format is `[16, T_lat, H, W]` where frame 0 is the VAE-encoded reference and frames 1+ are zeros. When encoding from raw image, must construct this full tensor.

6. **LoRA checkpoint keys** — The SF/DF checkpoint contains LoRA adapter weights (keys like `blocks.0.self_attn.q.lora_A.weight`). These should match the runtime LoRA adapters applied by the constructor. If the checkpoint also contains base weights, they'll be unexpected (already loaded).

7. **Patch embedding expansion** — Base Wan I2V has `in_dim=16`, InfiniteTalk has `in_dim=36` (16 noise + 20 I2V cond). The constructor handles this expansion automatically (zero-pad extra channels).

8. **Wav2Vec2 SDPA** — Load with `attn_implementation="eager"` to avoid SDPA incompatibility.

9. **T5 sequential loading** — T5 (20GB) + DiT (28GB) = 48GB. On 80GB A100 with KV caches and VAE, this is tight. Load T5 first, encode, unload before loading DiT.

---

## 13. Memory Estimate

| Component | VRAM |
|-----------|------|
| CausalInfiniteTalkWan (14B, bf16) | ~28 GB |
| KV caches (40 layers × 21 frames × 3136 tokens × 2 × 128d × 40heads × bf16) | ~13 GB |
| VAE (float32, shared encode/decode) | ~1 GB |
| Activations (inference, no grad) | ~5 GB |
| **Total** | **~47 GB** |

Fits comfortably on a single A100-80GB for inference. T5/CLIP/wav2vec2 are loaded and unloaded before the DiT, so their memory is not additive.

---

## 14. File Structure

```
FastGen_InfiniteTalk/
├── scripts/
│   └── inference/
│       ├── inference_causal.py          # This script
│       └── run_inference_causal.sh      # Shell wrapper with default paths
```

The script imports from:
- `fastgen.networks.InfiniteTalk.network_causal` — `CausalInfiniteTalkWan`
- `fastgen.datasets.infinitetalk_dataloader` — `_load_vae`, `_load_clip`, `_encode_audio` (reuse lazy encoding helpers)
- `wan.modules.vae` — `WanVAE` (for decode)

---

## 15. Verification Checklist

- [ ] **Precomputed round-trip**: Load precomputed `.pt` files, run inference, verify output is not garbage (shapes correct, values in reasonable range)
- [ ] **KV cache indices**: After each `store_kv=True` call, `global_end_index` advances by `chunk_size * frame_seqlen`
- [ ] **Audio slicing**: `audio_emb` sliced correctly per chunk (model handles internally)
- [ ] **VAE encode-decode**: Encode reference frame, decode, verify visual similarity
- [ ] **Variable length**: Test with different audio durations (e.g., 3s = 48 frames = 12 latent → 12/3 = 4 blocks)
- [ ] **Checkpoint loading**: Print missing/unexpected key counts, verify 0 missing for LoRA+audio keys
