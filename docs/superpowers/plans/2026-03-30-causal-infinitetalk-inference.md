# Causal InfiniteTalk Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a block-wise autoregressive inference script for the CausalInfiniteTalkWan 14B student model, supporting both raw inputs (image+audio) and pre-computed tensors.

**Architecture:** Single-file script (`inference_causal.py`) organized as composable functions: `encode_*()` for input encoding, `load_*()` for model loading, `run_inference()` for the AR loop, `decode_and_save()` for output. VAE stays loaded throughout; T5/CLIP/wav2vec2 are loaded sequentially then freed before the DiT. No CFG — single conditioned pass per denoising step.

**Tech Stack:** PyTorch (bf16), CausalInfiniteTalkWan (14B DiT + LoRA), WanVAE (float32), CLIP ViT-H/14, wav2vec2, T5 UMT5-XXL, librosa, imageio, ffmpeg

**Spec:** `docs/superpowers/specs/2026-03-30-causal-infinitetalk-inference-design.md`
**Reference:** `reference_FastGen_OmniAvatar/FastGen/scripts/inference/inference_causal.py`
**Porting guide:** `reference_FastGen_OmniAvatar/FastGen/docs/causal-inference-porting-guide.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/inference/inference_causal.py` | Create | Main inference script — all functions + CLI |
| `scripts/inference/run_inference_causal.sh` | Create | Shell wrapper with default env-var paths |

All logic goes in the single `inference_causal.py` file (following the OmniAvatar pattern). Reuses existing code via imports:
- `fastgen.networks.InfiniteTalk.network_causal.CausalInfiniteTalkWan`
- `fastgen.datasets.infinitetalk_dataloader._load_vae`, `_load_clip`, `_encode_audio`, `_add_infinitetalk_to_path`
- `wan.modules.vae.WanVAE` (via the InfiniteTalk source tree)

No tests directory — this is a standalone inference script. Verification is done via shape assertions and visual inspection of outputs (spec Section 15).

---

## Task 0: Scaffold and CLI

**Files:**
- Create: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Create the file with imports, path setup, and `parse_args()`**

```python
#!/usr/bin/env python3
"""Causal InfiniteTalk inference — block-wise AR generation with audio conditioning.

Generates lip-synced video from a reference image and driving audio using the
14B CausalInfiniteTalkWan student model trained via Diffusion Forcing or Self-Forcing.

Usage (raw inputs):
    python scripts/inference/inference_causal.py \
        --image /path/to/reference.png \
        --audio /path/to/audio.wav \
        --output_path /path/to/output.mp4 \
        --ckpt_path /path/to/trained_student.pth \
        --base_model_paths "shard1.safetensors,shard2.safetensors,..." \
        --infinitetalk_ckpt /path/to/infinitetalk.safetensors \
        --vae_path /path/to/Wan2.1_VAE.pth \
        --wav2vec_path /path/to/wav2vec2-dir \
        --clip_path /path/to/clip-ckpt.pth

Usage (pre-computed):
    python scripts/inference/inference_causal.py \
        --precomputed_dir /path/to/sample_dir/ \
        --output_path /path/to/output.mp4 \
        --ckpt_path /path/to/trained_student.pth \
        --base_model_paths "shard1.safetensors,..." \
        --infinitetalk_ckpt /path/to/infinitetalk.safetensors \
        --vae_path /path/to/Wan2.1_VAE.pth
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — add FastGen and InfiniteTalk source roots to sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, FASTGEN_ROOT)

# Add InfiniteTalk source tree (for wan.modules.vae, etc.)
from fastgen.datasets.infinitetalk_dataloader import _add_infinitetalk_to_path
_add_infinitetalk_to_path()


def _get_ffmpeg():
    """Return path to ffmpeg binary (system or imageio_ffmpeg fallback)."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg or pip install imageio-ffmpeg.")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Causal InfiniteTalk inference (block-wise AR with audio)"
    )

    # --- Input: raw mode ---
    p.add_argument("--image", type=str, default=None,
                   help="Reference face image path")
    p.add_argument("--audio", type=str, default=None,
                   help="Driving audio file path")
    p.add_argument("--video", type=str, default=None,
                   help="Video file (extract first frame + audio)")

    # --- Input: precomputed mode ---
    p.add_argument("--precomputed_dir", type=str, default=None,
                   help="Directory with pre-computed .pt files")

    # --- Output ---
    p.add_argument("--output_path", type=str, required=True,
                   help="Output video path")

    # --- Model paths ---
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="DF or SF trained checkpoint (.pth)")
    p.add_argument("--base_model_paths", type=str, required=True,
                   help="Comma-separated Wan I2V-14B safetensor shard paths")
    p.add_argument("--infinitetalk_ckpt", type=str, required=True,
                   help="Path to infinitetalk.safetensors")
    p.add_argument("--vae_path", type=str, required=True,
                   help="Path to Wan2.1_VAE.pth")

    # --- Encoder paths (required for raw mode) ---
    p.add_argument("--wav2vec_path", type=str, default=None,
                   help="Path to wav2vec2 model directory")
    p.add_argument("--clip_path", type=str, default=None,
                   help="Path to CLIP ViT-H/14 checkpoint")

    # --- Text conditioning ---
    p.add_argument("--prompt", type=str, default=None,
                   help="Text prompt (loads T5 for encoding)")
    p.add_argument("--text_embeds_path", type=str, default=None,
                   help="Pre-computed text_embeds.pt path")
    p.add_argument("--t5_path", type=str, default=None,
                   help="T5 UMT5-XXL encoder path (required if --prompt)")
    p.add_argument("--t5_tokenizer", type=str, default="google/umt5-xxl",
                   help="T5 tokenizer name or path")

    # --- Generation parameters ---
    p.add_argument("--num_latent_frames", type=int, default=None,
                   help="Override generation length (must be divisible by chunk_size)")
    p.add_argument("--chunk_size", type=int, default=3,
                   help="AR chunk size in latent frames (default: 3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--context_noise", type=float, default=0.0,
                   help="Context noise for cache update (default: 0.0)")
    p.add_argument("--local_attn_size", type=int, default=-1,
                   help="Rolling attention window in frames (-1=global)")
    p.add_argument("--lora_rank", type=int, default=32,
                   help="LoRA rank (must match trained checkpoint)")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (must match trained checkpoint)")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model dtype")
    p.add_argument("--fps", type=int, default=25,
                   help="Output video FPS (default: 25)")

    args = p.parse_args()

    # Validation
    has_raw = args.image is not None or args.video is not None
    has_precomputed = args.precomputed_dir is not None
    if not has_raw and not has_precomputed:
        p.error("Must provide --image/--audio, --video, or --precomputed_dir")
    if has_raw and has_precomputed:
        p.error("Cannot use both raw inputs and --precomputed_dir")
    if args.image is not None and args.audio is None:
        p.error("--audio is required when using --image")
    if has_raw and args.wav2vec_path is None:
        p.error("--wav2vec_path is required for raw input mode")
    if has_raw and args.clip_path is None:
        p.error("--clip_path is required for raw input mode")
    if args.prompt is not None and args.t5_path is None:
        p.error("--t5_path is required when using --prompt")
    if args.num_latent_frames is not None and args.num_latent_frames % args.chunk_size != 0:
        p.error(f"--num_latent_frames ({args.num_latent_frames}) must be divisible by chunk_size ({args.chunk_size})")

    # Resolve dtype
    args.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"Args parsed. Mode: {'precomputed' if args.precomputed_dir else 'raw'}")
```

- [ ] **Step 2: Verify the script runs and parses args**

Run:
```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
python scripts/inference/inference_causal.py --help
```

Expected: Help text printed with all arguments listed, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): scaffold causal inference script with CLI"
```

---

## Task 1: Pre-computed Input Loading

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Add `load_precomputed()` function**

Add after the `parse_args()` function:

```python
# ===========================================================================
# Pre-computed input loading
# ===========================================================================

def load_precomputed(precomputed_dir, chunk_size):
    """Load pre-computed .pt files from a sample directory.

    Matches the training dataloader format (infinitetalk_dataloader.py).
    Returns dict with all conditioning tensors + generation length info.
    """
    def _load(name):
        path = os.path.join(precomputed_dir, name)
        if not os.path.exists(path):
            return None
        t = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = next(v for v in t.values() if isinstance(v, torch.Tensor))
        return t

    # Prefer _81frames variants if they exist (pre-sliced to 81 video / 21 latent)
    first_frame_cond = _load("first_frame_cond_81frames.pt")
    if first_frame_cond is None:
        first_frame_cond = _load("first_frame_cond.pt")
        if first_frame_cond is None:
            raise FileNotFoundError(f"first_frame_cond.pt not found in {precomputed_dir}")

    audio_emb = _load("audio_emb_81frames.pt")
    if audio_emb is None:
        audio_emb = _load("audio_emb.pt")
        if audio_emb is None:
            raise FileNotFoundError(f"audio_emb.pt not found in {precomputed_dir}")

    clip_features = _load("clip_features.pt")
    if clip_features is None:
        raise FileNotFoundError(f"clip_features.pt not found in {precomputed_dir}")

    text_embeds = _load("text_embeds.pt")
    if text_embeds is None:
        raise FileNotFoundError(f"text_embeds.pt not found in {precomputed_dir}")

    # Compute generation length from audio
    num_video = audio_emb.shape[0]
    num_latent_raw = 1 + (num_video - 1) // 4
    num_latent = (num_latent_raw // chunk_size) * chunk_size
    num_latent = max(num_latent, chunk_size)
    num_video = 1 + (num_latent - 1) * 4

    # Slice tensors to computed length
    audio_emb = audio_emb[:num_video]
    first_frame_cond = first_frame_cond[:, :num_latent]

    # Ensure correct shapes: add batch dim if missing
    if first_frame_cond.dim() == 4:
        first_frame_cond = first_frame_cond.unsqueeze(0)  # [16, T, H, W] -> [1, 16, T, H, W]
    if audio_emb.dim() == 3:
        audio_emb = audio_emb.unsqueeze(0)  # [T, 12, 768] -> [1, T, 12, 768]
    if clip_features.dim() == 3:
        clip_features = clip_features.unsqueeze(0)  # [1, 257, 1280] -> already has batch
    if text_embeds.dim() == 3:
        text_embeds = text_embeds.unsqueeze(0)  # [1, 512, 4096] -> already has batch

    print(f"  Precomputed: {num_latent} latent frames, {num_video} video frames")
    print(f"  first_frame_cond: {list(first_frame_cond.shape)}")
    print(f"  audio_emb: {list(audio_emb.shape)}")
    print(f"  clip_features: {list(clip_features.shape)}")
    print(f"  text_embeds: {list(text_embeds.shape)}")

    return {
        "first_frame_cond": first_frame_cond.to(torch.bfloat16),
        "clip_features": clip_features.to(torch.bfloat16),
        "audio_emb": audio_emb.to(torch.bfloat16),
        "text_embeds": text_embeds.to(torch.bfloat16),
        "num_latent": num_latent,
        "num_video": num_video,
        "audio_path": None,  # No source audio for precomputed
    }
```

- [ ] **Step 2: Add a quick smoke test in `__main__` for precomputed loading**

Replace the `if __name__` block:

```python
if __name__ == "__main__":
    args = parse_args()
    print(f"Args parsed. Mode: {'precomputed' if args.precomputed_dir else 'raw'}")

    if args.precomputed_dir:
        data = load_precomputed(args.precomputed_dir, args.chunk_size)
        print(f"Loaded {data['num_latent']} latent / {data['num_video']} video frames")
```

- [ ] **Step 3: Test with real precomputed data**

Run:
```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
python scripts/inference/inference_causal.py \
    --precomputed_dir data/precomputed_talkvid/data_-0F1owya2oo_-0F1owya2oo_106956_111998 \
    --output_path /tmp/test_inference.mp4 \
    --ckpt_path dummy --base_model_paths dummy --infinitetalk_ckpt dummy --vae_path dummy
```

Expected: Prints tensor shapes, `21 latent / 81 video frames`, exit 0 (will fail on dummy paths only if we try to load models, which we don't yet).

- [ ] **Step 4: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): add precomputed .pt file loading"
```

---

## Task 2: Raw Input Encoding — Video/Image Extraction + Audio

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Add helper functions for video/image extraction and generation length**

Add after `load_precomputed()`:

```python
# ===========================================================================
# Raw input helpers
# ===========================================================================

def extract_first_frame_and_audio(video_path):
    """Extract first frame (PIL Image) and audio (temp wav) from a video file."""
    from PIL import Image
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame from: {video_path}")

    # BGR -> RGB -> PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Extract audio to temp wav
    ffmpeg = _get_ffmpeg()
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_tmp.close()
    subprocess.run([
        ffmpeg, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_tmp.name,
    ], check=True, capture_output=True)

    return pil_image, audio_tmp.name


def compute_generation_length(audio_path, override_frames, chunk_size, fps=25):
    """Compute generation length from audio duration.

    Returns (num_latent, num_video).
    """
    if override_frames is not None:
        num_latent = override_frames
        num_video = 1 + (num_latent - 1) * 4
        return num_latent, num_video

    duration = librosa.get_duration(path=audio_path)
    num_video_raw = int(duration * fps)  # floor
    num_latent_raw = 1 + (num_video_raw - 1) // 4
    num_latent = (num_latent_raw // chunk_size) * chunk_size
    num_latent = max(num_latent, chunk_size)
    num_video = 1 + (num_latent - 1) * 4

    print(f"  Audio duration: {duration:.2f}s -> {num_video_raw} raw frames -> "
          f"{num_latent} latent ({num_latent // chunk_size} blocks) / {num_video} video frames")
    return num_latent, num_video


def resize_and_center_crop(image, target_h, target_w):
    """Resize PIL image to target size with center crop (no padding).

    Matches multitalk.py::resize_and_centercrop logic.
    """
    from PIL import Image

    orig_w, orig_h = image.size
    scale = max(target_h / orig_h, target_w / orig_w)
    new_h = math.ceil(scale * orig_h)
    new_w = math.ceil(scale * orig_w)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))
    return image
```

- [ ] **Step 2: Add `encode_reference_image()` for VAE + first_frame_cond construction**

```python
# ===========================================================================
# Encoding functions
# ===========================================================================

@torch.no_grad()
def encode_reference_image(vae, pil_image, num_latent, target_h=448, target_w=896, device="cuda"):
    """Encode reference image → first_frame_cond [1, 16, T_lat, H_lat, W_lat].

    Builds the padded first-frame tensor: VAE-encode(ref_frame + zeros),
    matching the precompute pipeline (infinitetalk_dataloader.py::_encode_vae).
    """
    import torchvision.transforms as T

    # Resize and crop to target resolution
    image = resize_and_center_crop(pil_image, target_h, target_w)

    # To tensor [-1, 1]
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img_tensor = transform(image)  # [3, H, W]

    # Build padded video: [1, 3, T_video, H, W] with frame 0 = ref, rest = 0
    num_video = 1 + (num_latent - 1) * 4
    video_padded = torch.zeros(1, 3, num_video, target_h, target_w, dtype=torch.float32)
    video_padded[0, :, 0] = img_tensor

    # VAE encode (float32)
    vae_dtype = getattr(vae, "dtype", None) or next(vae.model.parameters()).dtype
    video_for_vae = video_padded[0].to(device=device, dtype=vae_dtype)  # [3, T, H, W]

    latent = vae.encode([video_for_vae], device=device)
    if isinstance(latent, (list, tuple)):
        latent = torch.stack(latent)
    first_frame_cond = latent.to(torch.bfloat16).cpu()  # [1, 16, T_lat, H_lat, W_lat]

    if first_frame_cond.dim() == 4:
        first_frame_cond = first_frame_cond.unsqueeze(0)

    # Slice to num_latent (in case VAE produces extra frames)
    first_frame_cond = first_frame_cond[:, :, :num_latent]

    print(f"  first_frame_cond: {list(first_frame_cond.shape)}")
    return first_frame_cond
```

- [ ] **Step 3: Add `encode_clip()` and `encode_audio()` wrappers**

```python
@torch.no_grad()
def encode_clip(clip_model, pil_image, device="cuda"):
    """Encode reference image → clip_features [1, 1, 257, 1280].

    Reuses _encode_clip logic from infinitetalk_dataloader.py.
    """
    import torchvision.transforms as T

    # CLIP preprocessing: resize to 224x224, normalize
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    img_tensor = transform(pil_image).unsqueeze(0).to(device=device)  # [1, 3, 224, 224]

    # CLIP encode
    clip_dtype = next(clip_model.parameters()).dtype
    img_tensor = img_tensor.to(dtype=clip_dtype)
    clip_out = clip_model.encode_image(img_tensor)

    # clip_out should be [1, 257, 1280] (CLS + 256 patches)
    if clip_out.dim() == 2:
        clip_out = clip_out.unsqueeze(1)
    clip_features = clip_out.unsqueeze(0).to(torch.bfloat16).cpu()  # [1, 1, 257, 1280]

    print(f"  clip_features: {list(clip_features.shape)}")
    return clip_features


@torch.no_grad()
def encode_audio_from_file(wav2vec_fe, wav2vec_model, audio_path, num_video_frames, device="cuda"):
    """Encode audio file → audio_emb [1, num_video_frames, 12, 768].

    Reuses _encode_audio from infinitetalk_dataloader.py.
    """
    from fastgen.datasets.infinitetalk_dataloader import _encode_audio

    # Load audio at 16kHz
    audio_array, _ = librosa.load(audio_path, sr=16000)

    audio_emb = _encode_audio(
        wav2vec_fe, wav2vec_model, audio_array,
        num_video_frames=num_video_frames, device=device,
    )
    audio_emb = audio_emb.unsqueeze(0).to(torch.bfloat16)  # [1, T, 12, 768]

    print(f"  audio_emb: {list(audio_emb.shape)}")
    return audio_emb
```

- [ ] **Step 4: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): add raw input encoding (VAE, CLIP, wav2vec2, image helpers)"
```

---

## Task 3: Text Encoding (T5 or Pre-computed)

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Add `encode_text()` function**

Add after the other encoding functions:

```python
@torch.no_grad()
def encode_text(prompt, t5_path, t5_tokenizer_name, text_embeds_path, device="cuda"):
    """Encode text prompt → text_embeds [1, 1, 512, 4096].

    Three modes:
    1. --prompt: Load T5, encode, unload (sequential to save VRAM)
    2. --text_embeds_path: Load pre-computed .pt
    3. Neither: Return zeros (unconditional text)
    """
    if text_embeds_path is not None:
        text_embeds = torch.load(text_embeds_path, map_location="cpu", weights_only=False)
        if isinstance(text_embeds, dict):
            text_embeds = next(v for v in text_embeds.values() if isinstance(v, torch.Tensor))
        text_embeds = text_embeds.to(torch.bfloat16)
        if text_embeds.dim() == 3:
            text_embeds = text_embeds.unsqueeze(0)
        print(f"  text_embeds (pre-computed): {list(text_embeds.shape)}")
        return text_embeds

    if prompt is not None:
        print(f"  Loading T5 encoder from {t5_path}...")
        from wan.modules.t5 import T5EncoderModel
        t5_model = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device(device),
            checkpoint_path=t5_path,
            tokenizer_path=t5_tokenizer_name,
        )

        text_embeds = t5_model([prompt], device=torch.device(device))
        # text_embeds: [1, 512, 4096]
        text_embeds = text_embeds.unsqueeze(0).to(torch.bfloat16).cpu()  # [1, 1, 512, 4096]

        # Unload T5 to free VRAM
        del t5_model
        torch.cuda.empty_cache()
        print(f"  text_embeds (T5 encoded): {list(text_embeds.shape)}, T5 unloaded")
        return text_embeds

    # No text conditioning — use zeros
    text_embeds = torch.zeros(1, 1, 512, 4096, dtype=torch.bfloat16)
    print(f"  text_embeds (zeros): {list(text_embeds.shape)}")
    return text_embeds
```

- [ ] **Step 2: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): add text encoding (T5 sequential load or pre-computed)"
```

---

## Task 4: Encoder Loading/Unloading Functions

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Add `load_vae()`, `load_clip()`, `load_wav2vec()` functions**

Add after the encoding functions:

```python
# ===========================================================================
# Encoder loading
# ===========================================================================

def load_vae(vae_path, device="cpu"):
    """Load WanVAE. Kept on CPU, moved to device per encode/decode call."""
    from wan.modules.vae import WanVAE
    print(f"Loading VAE from {vae_path}...")
    vae = WanVAE(vae_pth=vae_path, device=device)
    return vae


def load_clip(clip_path, device="cuda"):
    """Load CLIP ViT-H/14 visual encoder."""
    from fastgen.datasets.infinitetalk_dataloader import _load_clip
    print(f"Loading CLIP from {clip_path}...")
    clip_model = _load_clip(clip_path, device=device)
    return clip_model


def load_wav2vec(wav2vec_path, device="cuda"):
    """Load wav2vec2 feature extractor and model."""
    from transformers import Wav2Vec2FeatureExtractor
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model

    print(f"Loading wav2vec2 from {wav2vec_path}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_path, local_files_only=True
    )
    audio_encoder = Wav2Vec2Model.from_pretrained(
        wav2vec_path, local_files_only=True
    )
    audio_encoder = audio_encoder.to(device=device)
    audio_encoder.eval()

    return feature_extractor, audio_encoder
```

- [ ] **Step 2: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): add encoder loading functions (VAE, CLIP, wav2vec2)"
```

---

## Task 5: DiT Model Loading + Checkpoint Overlay

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Add `load_diffusion_model()` function**

```python
# ===========================================================================
# Diffusion model loading
# ===========================================================================

def load_diffusion_model(args, num_latent, device, dtype):
    """Load CausalInfiniteTalkWan and apply DF/SF checkpoint."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    print(f"Loading CausalInfiniteTalkWan (14B)...")
    print(f"  base_model_paths: {args.base_model_paths.split(',')[0]}... ({len(args.base_model_paths.split(','))} shards)")
    print(f"  infinitetalk_ckpt: {args.infinitetalk_ckpt}")
    print(f"  lora_rank: {args.lora_rank}, lora_alpha: {args.lora_alpha}")
    print(f"  chunk_size: {args.chunk_size}, total_num_frames: {num_latent}")

    model = CausalInfiniteTalkWan(
        base_model_paths=args.base_model_paths,
        infinitetalk_ckpt_path=args.infinitetalk_ckpt,
        lora_ckpt_path="",  # SF/DF ckpt loaded separately below
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        chunk_size=args.chunk_size,
        total_num_frames=num_latent,
        net_pred_type="flow",
        schedule_type="rf",
        shift=7.0,
        local_attn_size=args.local_attn_size,
        sink_size=0,
        use_dynamic_rope=False,
    )

    # --- Load DF/SF checkpoint overlay ---
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        print(f"  Loading checkpoint: {args.ckpt_path}")
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

        # Print first few keys for debugging
        keys_sample = list(state_dict.keys())[:5]
        print(f"  Checkpoint keys sample: {keys_sample}")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Checkpoint loaded: {len(state_dict)} params, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")

        if missing:
            lora_missing = [k for k in missing if "lora" in k.lower()]
            other_missing = [k for k in missing if "lora" not in k.lower()]
            if lora_missing:
                print(f"  WARNING: {len(lora_missing)} LoRA keys missing — "
                      f"rank/alpha mismatch? Sample: {lora_missing[:3]}")
            if other_missing:
                print(f"  Other missing: {other_missing[:5]}")

        del ckpt, state_dict
    else:
        print(f"  WARNING: No checkpoint file at {args.ckpt_path}, using base weights only")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"  Model on {device}, dtype={dtype}")
    return model
```

- [ ] **Step 2: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): add diffusion model loading with FSDP checkpoint overlay"
```

---

## Task 6: AR Inference Loop

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Add `run_inference()` function**

```python
# ===========================================================================
# AR inference loop
# ===========================================================================

@torch.no_grad()
def run_inference(model, condition, num_latent_frames, chunk_size,
                  context_noise, seed, device, dtype):
    """Block-wise autoregressive inference — no CFG, no gradients.

    Mirrors rollout_with_gradient / _student_sample_loop from training,
    but without exit steps, gradient tracking, or VSD loss.
    """
    B = 1
    C = 16

    # Derive spatial dims from conditioning
    H_lat = condition["first_frame_cond"].shape[3]
    W_lat = condition["first_frame_cond"].shape[4]

    # t_list for shift=7.0: derived from new_t = 7*t / (1 + 6*t) applied to linspace(1,0,5)
    t_list = [0.999, 0.955, 0.875, 0.700, 0.0]

    model.total_num_frames = num_latent_frames
    model.clear_caches()

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        B, C, num_latent_frames, H_lat, W_lat,
        generator=generator, device=device, dtype=dtype,
    )
    output = torch.zeros_like(noise)

    t_list_t = torch.tensor(t_list, device=device, dtype=torch.float64)
    num_blocks = num_latent_frames // chunk_size
    num_denoise_steps = len(t_list_t) - 1

    print(f"  AR loop: {num_blocks} blocks x {num_denoise_steps} denoising steps "
          f"= {num_blocks * (num_denoise_steps + 1)} forward passes")

    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * chunk_size
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        # Multi-step denoising
        for step_idx in range(num_denoise_steps):
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

        # Store denoised chunk
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

        print(f"    Block {block_idx + 1}/{num_blocks} done "
              f"(frames {cur_start_frame}-{cur_start_frame + chunk_size - 1})")

    model.clear_caches()
    return output
```

- [ ] **Step 2: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): add block-wise AR inference loop"
```

---

## Task 7: VAE Decode + Video Saving

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Add `decode_and_save()` function**

```python
# ===========================================================================
# Post-processing
# ===========================================================================

@torch.no_grad()
def decode_and_save(vae, output_latents, audio_path, output_path, fps, device):
    """VAE decode latents → normalize → save video → mux audio."""
    import imageio

    print("Decoding latents with VAE (float32)...")

    # VAE decode — expects [C, T_lat, H_lat, W_lat] in float32
    latent = output_latents[0].to(device=device, dtype=torch.float32)
    video_tensor = vae.decode([latent], device=device)

    # Handle list/tuple output
    if isinstance(video_tensor, (list, tuple)):
        video_tensor = video_tensor[0] if len(video_tensor) == 1 else torch.stack(video_tensor)
    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]  # Remove batch dim: [3, T, H, W]

    # Normalize [-1, 1] → [0, 255] uint8
    video = (video_tensor.clamp(-1, 1) + 1) / 2 * 255
    video = video.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)  # [T, H, W, 3]
    print(f"  Decoded video: {video.shape} (T, H, W, C)")

    # Determine output paths
    if audio_path is not None:
        # Save silent video first, then mux
        base, ext = os.path.splitext(output_path)
        silent_path = base + "_silent" + ext
    else:
        silent_path = output_path

    # Save video via imageio
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(silent_path, fps=fps, codec="libx264", quality=8)
    for frame in video:
        writer.append_data(frame)
    writer.close()
    print(f"  Saved silent video: {silent_path}")

    # Mux audio if available
    if audio_path is not None:
        duration_s = video.shape[0] / fps
        ffmpeg = _get_ffmpeg()
        subprocess.run([
            ffmpeg, "-y",
            "-i", silent_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-t", f"{duration_s:.3f}",
            output_path,
        ], check=True, capture_output=True)
        # Clean up silent file
        if os.path.exists(output_path) and silent_path != output_path:
            os.remove(silent_path)
        print(f"  Muxed with audio: {output_path}")
    else:
        print(f"  No audio to mux. Output: {output_path}")

    return output_path
```

- [ ] **Step 2: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): add VAE decode and video saving with audio muxing"
```

---

## Task 8: Wire Up `main()` — Full Pipeline

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Replace the `__main__` block with the full `main()` function**

```python
# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype

    print("=" * 60)
    print("Causal InfiniteTalk Inference")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Encode text (T5 loaded/unloaded first to save VRAM)
    # ------------------------------------------------------------------
    print("\n[1/5] Text encoding...")
    text_embeds = encode_text(
        args.prompt, args.t5_path, args.t5_tokenizer,
        args.text_embeds_path, device=device,
    )

    # ------------------------------------------------------------------
    # Step 2: Load VAE (kept for entire session)
    # ------------------------------------------------------------------
    print("\n[2/5] Loading VAE...")
    vae = load_vae(args.vae_path, device="cpu")

    # ------------------------------------------------------------------
    # Step 3: Encode inputs (raw or precomputed)
    # ------------------------------------------------------------------
    print("\n[3/5] Encoding inputs...")
    if args.precomputed_dir:
        data = load_precomputed(args.precomputed_dir, args.chunk_size)
        first_frame_cond = data["first_frame_cond"]
        clip_features = data["clip_features"]
        audio_emb = data["audio_emb"]
        text_embeds = data["text_embeds"]  # Override with precomputed
        num_latent = data["num_latent"]
        num_video = data["num_video"]
        audio_path = data["audio_path"]
    else:
        # Extract reference image and audio
        if args.video is not None:
            pil_image, audio_path = extract_first_frame_and_audio(args.video)
            print(f"  Extracted first frame and audio from {args.video}")
        else:
            from PIL import Image
            pil_image = Image.open(args.image).convert("RGB")
            audio_path = args.audio

        # Compute generation length from audio
        num_latent, num_video = compute_generation_length(
            audio_path, args.num_latent_frames, args.chunk_size, args.fps
        )

        # Load encoders
        clip_model = load_clip(args.clip_path, device=device)
        wav2vec_fe, wav2vec_model = load_wav2vec(args.wav2vec_path, device=device)

        # Encode
        first_frame_cond = encode_reference_image(
            vae, pil_image, num_latent, device=device
        )
        clip_features = encode_clip(clip_model, pil_image, device=device)
        audio_emb = encode_audio_from_file(
            wav2vec_fe, wav2vec_model, audio_path, num_video, device=device
        )

        # Unload encoders to free VRAM for DiT
        del clip_model, wav2vec_fe, wav2vec_model
        torch.cuda.empty_cache()
        print("  Encoders unloaded, VRAM freed")

    # Apply override if specified
    if args.num_latent_frames is not None:
        num_latent = args.num_latent_frames
        num_video = 1 + (num_latent - 1) * 4
        first_frame_cond = first_frame_cond[:, :, :num_latent]
        audio_emb = audio_emb[:, :num_video]

    # ------------------------------------------------------------------
    # Step 4: Load diffusion model
    # ------------------------------------------------------------------
    print("\n[4/5] Loading diffusion model...")
    model = load_diffusion_model(args, num_latent, device, dtype)

    # Build condition dict — move to device
    condition = {
        "text_embeds": text_embeds.to(device=device, dtype=dtype),
        "first_frame_cond": first_frame_cond.to(device=device, dtype=dtype),
        "clip_features": clip_features.to(device=device, dtype=dtype),
        "audio_emb": audio_emb.to(device=device, dtype=dtype),
    }

    # ------------------------------------------------------------------
    # Step 5: Run inference + decode + save
    # ------------------------------------------------------------------
    print(f"\n[5/5] Running AR inference ({num_latent} latent / {num_video} video frames)...")
    output_latents = run_inference(
        model, condition, num_latent, args.chunk_size,
        args.context_noise, args.seed, device, dtype,
    )

    # Free model VRAM before decode
    del model
    torch.cuda.empty_cache()

    # Decode and save
    print("\nDecoding and saving...")
    output_file = decode_and_save(
        vae, output_latents, audio_path, args.output_path, args.fps, device,
    )

    print(f"\nDone! Output: {output_file}")

    # Clean up temp audio if extracted from video
    if args.video is not None and audio_path and os.path.exists(audio_path):
        if audio_path.startswith(tempfile.gettempdir()):
            os.remove(audio_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the full script imports and `--help` still works**

Run:
```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
python scripts/inference/inference_causal.py --help
```

Expected: Full help text, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(inference): wire up main() — full end-to-end pipeline"
```

---

## Task 9: Shell Wrapper Script

**Files:**
- Create: `scripts/inference/run_inference_causal.sh`

- [ ] **Step 1: Create the shell wrapper with default paths**

```bash
#!/usr/bin/env bash
# Run causal InfiniteTalk inference with default weight paths.
#
# Usage:
#   ./scripts/inference/run_inference_causal.sh \
#       --image /path/to/reference.png \
#       --audio /path/to/audio.wav \
#       --output_path /path/to/output.mp4
#
# Or with pre-computed:
#   ./scripts/inference/run_inference_causal.sh \
#       --precomputed_dir data/precomputed_talkvid/data_xxx \
#       --output_path /path/to/output.mp4
#
# Override any default by passing it explicitly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTGEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- Default paths (adjust for your setup) ----------
# InfiniteTalk weights directory
IT_WEIGHTS="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights}"

# Base Wan I2V-14B safetensor shards (comma-separated)
BASE_SHARDS=""
for i in $(seq 1 7); do
    shard="${IT_WEIGHTS}/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000${i}-of-00007.safetensors"
    if [ -n "$BASE_SHARDS" ]; then BASE_SHARDS="${BASE_SHARDS},"; fi
    BASE_SHARDS="${BASE_SHARDS}${shard}"
done

# Other weight paths
IT_CKPT="${IT_WEIGHTS}/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH="${IT_WEIGHTS}/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"
WAV2VEC_PATH="${IT_WEIGHTS}/InfiniteTalk/single/wav2vec2-base-960h-zh"
CLIP_PATH="${IT_WEIGHTS}/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
T5_PATH="${IT_WEIGHTS}/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"

# DF/SF checkpoint (override with INFINITETALK_CKPT_PATH env var)
CKPT_PATH="${INFINITETALK_CKPT_PATH:-}"

# ---------- Run ----------
cd "$FASTGEN_ROOT"
python scripts/inference/inference_causal.py \
    --base_model_paths "$BASE_SHARDS" \
    --infinitetalk_ckpt "$IT_CKPT" \
    --vae_path "$VAE_PATH" \
    --wav2vec_path "$WAV2VEC_PATH" \
    --clip_path "$CLIP_PATH" \
    --t5_path "$T5_PATH" \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"} \
    "$@"
```

- [ ] **Step 2: Make executable and test help**

```bash
chmod +x scripts/inference/run_inference_causal.sh
./scripts/inference/run_inference_causal.sh --help
```

Expected: Prints help from `inference_causal.py`, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/inference/run_inference_causal.sh
git commit -m "feat(inference): add shell wrapper with default weight paths"
```

---

## Task 10: End-to-End Smoke Test with Pre-computed Data

**Files:**
- No new files — test using existing precomputed data

- [ ] **Step 1: Run with pre-computed data (requires GPU + weights)**

This is the first real test. It loads the full 14B model and runs inference on a pre-computed sample.

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk

# Pick a sample from validation set
SAMPLE_DIR="data/precomputed_talkvid/data_yrHiEeOu5Pw_yrHiEeOu5Pw_273606_278647"

# Run with base weights only (no SF/DF checkpoint yet — verifies the pipeline runs)
python scripts/inference/inference_causal.py \
    --precomputed_dir "$SAMPLE_DIR" \
    --output_path /tmp/inference_smoke_test.mp4 \
    --ckpt_path "none" \
    --base_model_paths "$(echo /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000{1..7}-of-00007.safetensors | tr ' ' ',')" \
    --infinitetalk_ckpt /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors \
    --vae_path /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
    --seed 42
```

Expected output:
- Prints progress through all 5 stages
- AR loop prints 7 block completions
- Saves `/tmp/inference_smoke_test.mp4`
- No crashes, no CUDA OOM

- [ ] **Step 2: Verify output video**

```bash
# Check video file exists and has reasonable size
ls -lh /tmp/inference_smoke_test.mp4

# Check video properties
python3 -c "
import imageio
r = imageio.get_reader('/tmp/inference_smoke_test.mp4')
meta = r.get_meta_data()
print(f'FPS: {meta.get(\"fps\", \"?\")}')
print(f'Duration: {meta.get(\"duration\", \"?\")}s')
print(f'Size: {meta.get(\"size\", \"?\")}')
n = 0
for _ in r:
    n += 1
print(f'Frames: {n}')
r.close()
"
```

Expected: 81 frames, ~3.24s at 25fps, 448x896 resolution.

- [ ] **Step 3: Inspect output quality**

The output will look noisy/incoherent with base weights only (no DF/SF training). What matters is:
- Correct spatial resolution (448x896)
- Correct frame count (81)
- No NaN/inf values
- Video is watchable (not a solid color or noise pattern indicating catastrophic failure)

- [ ] **Step 4: Commit (if any bug fixes were needed)**

```bash
git add scripts/inference/inference_causal.py
git commit -m "fix(inference): bug fixes from smoke test"
```

---

## Task 11: Debug and Fix Issues from Smoke Test

**Files:**
- Modify: `scripts/inference/inference_causal.py`

This task is a placeholder for the inevitable issues discovered during the smoke test. Common problems from the porting guide:

- [ ] **Step 1: Check for VAE float32 issues**

If you see `RuntimeError: Input type (c10::BFloat16) and bias type (float)`:
- Ensure all VAE encode/decode calls cast input to float32
- The `encode_reference_image()` and `decode_and_save()` functions should handle this

- [ ] **Step 2: Check for audio shape issues**

If you see audio dimension errors:
- Verify `audio_emb` has batch dim: `[1, T, 12, 768]`
- The model's `forward()` handles 4D→5D windowing (line 1777-1784 of network_causal.py)
- Print `audio_emb.shape` before and after entering the model

- [ ] **Step 3: Check for CLIP feature shape issues**

If clip_features shape is wrong:
- The lazy dataloader's `_load_clip` / `_encode_clip` may return different shapes depending on the CLIP model variant
- Print shapes and adjust `encode_clip()` accordingly

- [ ] **Step 4: Check for checkpoint key mismatches**

If `missing` count is high after checkpoint load:
- Print first 10 missing and first 10 checkpoint keys
- Check if keys need a prefix transformation
- The constructor applies LoRA adapters, so LoRA keys (containing `lora_A`/`lora_B`) should be present in the model's state dict

- [ ] **Step 5: Commit any fixes**

```bash
git add scripts/inference/inference_causal.py
git commit -m "fix(inference): address issues from end-to-end smoke test"
```

---

## Summary

| Task | Description | ~Size |
|------|-------------|-------|
| 0 | Scaffold + CLI | Small |
| 1 | Pre-computed input loading | Small |
| 2 | Raw input encoding (VAE, CLIP, wav2vec2) | Medium |
| 3 | Text encoding (T5 or pre-computed) | Small |
| 4 | Encoder loading functions | Small |
| 5 | DiT model loading + checkpoint overlay | Medium |
| 6 | AR inference loop | Medium |
| 7 | VAE decode + video saving | Small |
| 8 | Wire up `main()` | Medium |
| 9 | Shell wrapper | Small |
| 10 | Smoke test with precomputed data | Test |
| 11 | Debug fixes from smoke test | Variable |

Total: ~600-700 lines of Python (matching OmniAvatar's ~1000 lines minus the V2V-specific conditioning, spatial mask, and video ping-pong logic that InfiniteTalk doesn't need).
