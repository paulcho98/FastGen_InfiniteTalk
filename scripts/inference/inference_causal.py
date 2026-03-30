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
        first_frame_cond = first_frame_cond.unsqueeze(0)
    if audio_emb.dim() == 3:
        audio_emb = audio_emb.unsqueeze(0)
    if clip_features.dim() == 3:
        clip_features = clip_features.unsqueeze(0)
    if text_embeds.dim() == 3:
        text_embeds = text_embeds.unsqueeze(0)

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
        "audio_path": None,
    }


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

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

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
    """Compute generation length from audio duration. Returns (num_latent, num_video)."""
    if override_frames is not None:
        num_latent = override_frames
        num_video = 1 + (num_latent - 1) * 4
        return num_latent, num_video

    duration = librosa.get_duration(path=audio_path)
    num_video_raw = int(duration * fps)
    num_latent_raw = 1 + (num_video_raw - 1) // 4
    num_latent = (num_latent_raw // chunk_size) * chunk_size
    num_latent = max(num_latent, chunk_size)
    num_video = 1 + (num_latent - 1) * 4

    print(f"  Audio duration: {duration:.2f}s -> {num_video_raw} raw frames -> "
          f"{num_latent} latent ({num_latent // chunk_size} blocks) / {num_video} video frames")
    return num_latent, num_video


def resize_and_center_crop(image, target_h, target_w):
    """Resize PIL image to target size with center crop."""
    from PIL import Image as PILImage

    orig_w, orig_h = image.size
    scale = max(target_h / orig_h, target_w / orig_w)
    new_h = math.ceil(scale * orig_h)
    new_w = math.ceil(scale * orig_w)
    image = image.resize((new_w, new_h), PILImage.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))
    return image


# ===========================================================================
# Encoding functions
# ===========================================================================

@torch.no_grad()
def encode_reference_image(vae, pil_image, num_latent, target_h=448, target_w=896, device="cuda"):
    """Encode reference image -> first_frame_cond [1, 16, T_lat, H_lat, W_lat].

    Builds the padded first-frame tensor: VAE-encode(ref_frame + zeros),
    matching the precompute pipeline (infinitetalk_dataloader.py::_encode_vae).
    """
    import torchvision.transforms as T

    image = resize_and_center_crop(pil_image, target_h, target_w)

    transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img_tensor = transform(image)  # [3, H, W]

    # Build padded video: first frame = ref, rest = zeros
    num_video = 1 + (num_latent - 1) * 4
    video_padded = torch.zeros(1, 3, num_video, target_h, target_w, dtype=torch.float32)
    video_padded[0, :, 0] = img_tensor

    vae_dtype = getattr(vae, "dtype", None) or next(vae.model.parameters()).dtype
    video_for_vae = video_padded[0].to(device=device, dtype=vae_dtype)

    latent = vae.encode([video_for_vae], device=device)
    if isinstance(latent, (list, tuple)):
        latent = torch.stack(latent)
    first_frame_cond = latent.to(torch.bfloat16).cpu()

    if first_frame_cond.dim() == 4:
        first_frame_cond = first_frame_cond.unsqueeze(0)
    first_frame_cond = first_frame_cond[:, :, :num_latent]

    print(f"  first_frame_cond: {list(first_frame_cond.shape)}")
    return first_frame_cond


@torch.no_grad()
def encode_clip(clip_model, pil_image, device="cuda"):
    """Encode reference image -> clip_features [1, 1, 257, 1280]."""
    from fastgen.datasets.infinitetalk_dataloader import _encode_clip
    clip_features = _encode_clip(clip_model, device=device, cond_image=pil_image)
    if isinstance(clip_features, torch.Tensor):
        clip_features = clip_features.to(torch.bfloat16).cpu()
    if clip_features.dim() == 3:
        clip_features = clip_features.unsqueeze(0)
    print(f"  clip_features: {list(clip_features.shape)}")
    return clip_features


@torch.no_grad()
def encode_audio_from_file(wav2vec_fe, wav2vec_model, audio_path, num_video_frames, device="cuda"):
    """Encode audio file -> audio_emb [1, num_video_frames, 12, 768]."""
    from fastgen.datasets.infinitetalk_dataloader import _encode_audio

    audio_array, _ = librosa.load(audio_path, sr=16000)
    audio_emb = _encode_audio(
        wav2vec_fe, wav2vec_model, audio_array,
        num_video_frames=num_video_frames, device=device,
    )
    audio_emb = audio_emb.unsqueeze(0).to(torch.bfloat16)
    print(f"  audio_emb: {list(audio_emb.shape)}")
    return audio_emb


@torch.no_grad()
def encode_text(prompt, t5_path, t5_tokenizer_name, text_embeds_path, device="cuda"):
    """Encode text prompt -> text_embeds [1, 1, 512, 4096].

    Three modes: --prompt (T5), --text_embeds_path (pre-computed), or zeros.
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
        text_embeds = text_embeds.unsqueeze(0).to(torch.bfloat16).cpu()
        del t5_model
        torch.cuda.empty_cache()
        print(f"  text_embeds (T5 encoded): {list(text_embeds.shape)}, T5 unloaded")
        return text_embeds

    text_embeds = torch.zeros(1, 1, 512, 4096, dtype=torch.bfloat16)
    print(f"  text_embeds (zeros): {list(text_embeds.shape)}")
    return text_embeds


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
        lora_ckpt_path="",
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

        if isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], dict) and "net" in ckpt["model"]:
                state_dict = ckpt["model"]["net"]
            elif "net" in ckpt:
                state_dict = ckpt["net"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

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


# ===========================================================================
# AR inference loop
# ===========================================================================

@torch.no_grad()
def run_inference(model, condition, num_latent_frames, chunk_size,
                  context_noise, seed, device, dtype):
    """Block-wise autoregressive inference — no CFG, no gradients."""
    B = 1
    C = 16
    H_lat = condition["first_frame_cond"].shape[3]
    W_lat = condition["first_frame_cond"].shape[4]

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

        output[:, :, cur_start_frame:cur_start_frame + chunk_size] = x0_pred

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


# ===========================================================================
# Post-processing
# ===========================================================================

@torch.no_grad()
def decode_and_save(vae, output_latents, audio_path, output_path, fps, device):
    """VAE decode latents -> normalize -> save video -> mux audio."""
    import imageio

    print("Decoding latents with VAE (float32)...")

    latent = output_latents[0].to(device=device, dtype=torch.float32)
    video_tensor = vae.decode([latent], device=device)

    if isinstance(video_tensor, (list, tuple)):
        video_tensor = video_tensor[0] if len(video_tensor) == 1 else torch.stack(video_tensor)
    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]

    video = (video_tensor.clamp(-1, 1) + 1) / 2 * 255
    video = video.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    print(f"  Decoded video: {video.shape} (T, H, W, C)")

    if audio_path is not None:
        base, ext = os.path.splitext(output_path)
        silent_path = base + "_silent" + ext
    else:
        silent_path = output_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = imageio.get_writer(silent_path, fps=fps, codec="libx264", quality=8)
    for frame in video:
        writer.append_data(frame)
    writer.close()
    print(f"  Saved silent video: {silent_path}")

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
        if os.path.exists(output_path) and silent_path != output_path:
            os.remove(silent_path)
        print(f"  Muxed with audio: {output_path}")
    else:
        print(f"  No audio to mux. Output: {output_path}")

    return output_path


if __name__ == "__main__":
    args = parse_args()
    print(f"Args parsed. Mode: {'precomputed' if args.precomputed_dir else 'raw'}")

    if args.precomputed_dir:
        data = load_precomputed(args.precomputed_dir, args.chunk_size)
        print(f"Loaded {data['num_latent']} latent / {data['num_video']} video frames")
