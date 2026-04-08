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
    p.add_argument("--video_dir", type=str, default=None,
                   help="Directory of video files for batch inference (requires --output_dir)")

    # --- Input: precomputed mode ---
    p.add_argument("--precomputed_dir", type=str, default=None,
                   help="Directory with pre-computed .pt files")
    p.add_argument("--precomputed_list", type=str, default=None,
                   help="Text file listing precomputed dirs (one per line, for batch mode)")

    # --- Audio for precomputed mode ---
    p.add_argument("--audio_data_root", type=str, default=None,
                   help="Root dir for source audio (TalkVid/data/). "
                        "Resolves audio from precomputed sample name.")
    p.add_argument("--source_audio", type=str, default=None,
                   help="Explicit source audio path for muxing (any mode)")

    # --- Output ---
    p.add_argument("--output_path", type=str, default=None,
                   help="Output video path (single mode)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (batch mode — one video per sample)")

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
    p.add_argument("--quarter_res", action="store_true",
                   help="Use quarter-resolution precomputed files (*_quarter.pt)")
    p.add_argument("--target_h", type=int, default=448,
                   help="Target height for raw/video encoding (default: 448)")
    p.add_argument("--target_w", type=int, default=896,
                   help="Target width for raw/video encoding (default: 896)")
    p.add_argument("--no_anchor_first_frame", action="store_true",
                   help="Disable hard-overwrite of first latent frame with clean "
                        "reference at every denoising step (on by default to match "
                        "original InfiniteTalk inference)")
    p.add_argument("--preprocess_mode", type=str, default="crop",
                   choices=["crop", "pad"],
                   help="Preprocessing for raw/video inputs: 'crop' = resize+center-crop "
                        "(default), 'pad' = resize to short side + zero-pad to target")

    args = p.parse_args()

    # Validation
    has_raw = args.image is not None or args.video is not None
    has_video_dir = args.video_dir is not None
    has_precomputed = args.precomputed_dir is not None
    has_batch = args.precomputed_list is not None
    if not has_raw and not has_precomputed and not has_batch and not has_video_dir:
        p.error("Must provide --image/--audio, --video, --video_dir, --precomputed_dir, or --precomputed_list")
    if sum([has_raw, has_precomputed, has_batch, has_video_dir]) > 1:
        p.error("Cannot combine raw inputs, --precomputed_dir, --precomputed_list, and --video_dir")
    if (has_batch or has_video_dir) and args.output_dir is None:
        p.error("--output_dir is required for batch mode (--precomputed_list or --video_dir)")
    if not has_batch and not has_video_dir and args.output_path is None:
        p.error("--output_path is required for single-sample mode")
    if args.image is not None and args.audio is None:
        p.error("--audio is required when using --image")
    if (has_raw or has_video_dir) and args.wav2vec_path is None:
        p.error("--wav2vec_path is required for raw/video_dir input mode")
    if (has_raw or has_video_dir) and args.clip_path is None:
        p.error("--clip_path is required for raw/video_dir input mode")
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

def resolve_audio_for_precomputed(precomputed_dir, audio_data_root):
    """Resolve source audio path from precomputed sample dir name.

    Sample dir: data_VIDEOID_VIDEOID_START_END
    Audio at:   audio_data_root/VIDEOID/VIDEOID_START_END.wav
    """
    if not audio_data_root:
        return None
    basename = os.path.basename(precomputed_dir)
    if basename.startswith("data_"):
        basename = basename[5:]
    parts = basename.split("_")
    if len(parts) >= 4 and parts[0] == parts[1]:
        video_id = parts[0]
        clip_name = f"{video_id}_{'_'.join(parts[2:])}"
        wav_path = os.path.join(audio_data_root, video_id, f"{clip_name}.wav")
        if os.path.exists(wav_path):
            return wav_path
    return None


def load_precomputed(precomputed_dir, chunk_size, quarter_res=False):
    """Load pre-computed .pt files from a sample directory.

    Matches the training dataloader format (infinitetalk_dataloader.py).
    Returns dict with all conditioning tensors + generation length info.

    Args:
        quarter_res: If True, prefer *_quarter.pt files (224x448 → 28x56 latent).
    """
    def _load(name):
        path = os.path.join(precomputed_dir, name)
        if not os.path.exists(path):
            return None
        t = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = next(v for v in t.values() if isinstance(v, torch.Tensor))
        return t

    # Quarter-res: prefer *_quarter.pt files
    if quarter_res:
        first_frame_cond = _load("first_frame_cond_quarter.pt")
    else:
        first_frame_cond = None

    if first_frame_cond is None:
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
    # audio_emb may be [T, 12, 768] (unbatched) or [1, T, 12, 768] (batched)
    num_video = audio_emb.shape[-3]  # T dimension (12 and 768 are always last two)
    num_latent_raw = 1 + (num_video - 1) // 4
    num_latent = (num_latent_raw // chunk_size) * chunk_size
    num_latent = max(num_latent, chunk_size)
    num_video = 1 + (num_latent - 1) * 4

    # Slice tensors to computed length (temporal dim is -3 for both)
    audio_emb = audio_emb[..., :num_video, :, :]
    first_frame_cond = first_frame_cond[..., :num_latent, :, :]

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

    # Read audio path if saved by precompute script
    audio_path = None
    audio_path_file = os.path.join(precomputed_dir, "audio_path.txt")
    if os.path.isfile(audio_path_file):
        with open(audio_path_file) as f:
            ap = f.read().strip()
            if ap and os.path.isfile(ap):
                audio_path = ap

    return {
        "first_frame_cond": first_frame_cond.to(torch.bfloat16),
        "clip_features": clip_features.to(torch.bfloat16),
        "audio_emb": audio_emb.to(torch.bfloat16),
        "text_embeds": text_embeds.to(torch.bfloat16),
        "num_latent": num_latent,
        "num_video": num_video,
        "audio_path": audio_path,
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


def resize_and_pad(image, target_h, target_w):
    """Resize PIL image to fit target height, zero-pad width to target."""
    from PIL import Image as PILImage

    orig_w, orig_h = image.size
    scale = target_h / orig_h
    new_h = target_h
    new_w = round(orig_w * scale)
    image = image.resize((new_w, new_h), PILImage.LANCZOS)

    if new_w < target_w:
        padded = PILImage.new("RGB", (target_w, target_h), (0, 0, 0))
        padded.paste(image, ((target_w - new_w) // 2, 0))
        return padded
    elif new_w > target_w:
        left = (new_w - target_w) // 2
        return image.crop((left, 0, left + target_w, target_h))
    return image


def preprocess_image(image, target_h, target_w, mode="crop"):
    """Dispatch to crop or pad preprocessing."""
    if mode == "pad":
        return resize_and_pad(image, target_h, target_w)
    return resize_and_center_crop(image, target_h, target_w)


# ===========================================================================
# Encoding functions
# ===========================================================================

@torch.no_grad()
def encode_reference_image(vae, pil_image, num_latent, target_h=448, target_w=896, device="cuda", preprocess_mode="crop"):
    """Encode reference image -> first_frame_cond [1, 16, T_lat, H_lat, W_lat].

    Builds the padded first-frame tensor: VAE-encode(ref_frame + zeros),
    matching the precompute pipeline (infinitetalk_dataloader.py::_encode_vae).
    """
    import torchvision.transforms as T

    image = preprocess_image(pil_image, target_h, target_w, mode=preprocess_mode)

    transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img_tensor = transform(image)  # [3, H, W]

    # Build padded video: first frame = ref, rest = zeros
    num_video = 1 + (num_latent - 1) * 4
    video_padded = torch.zeros(1, 3, num_video, target_h, target_w, dtype=torch.float32)
    video_padded[0, :, 0] = img_tensor

    # Move VAE to GPU for encoding
    vae.model = vae.model.to(device)
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]

    vae_dtype = getattr(vae, "dtype", None) or next(vae.model.parameters()).dtype
    video_for_vae = video_padded[0].to(device=device, dtype=vae_dtype)

    latent = vae.encode([video_for_vae])
    if isinstance(latent, (list, tuple)):
        latent = torch.stack(latent)
    first_frame_cond = latent.to(torch.bfloat16).cpu()

    if first_frame_cond.dim() == 4:
        first_frame_cond = first_frame_cond.unsqueeze(0)
    first_frame_cond = first_frame_cond[:, :, :num_latent]

    print(f"  first_frame_cond: {list(first_frame_cond.shape)}")
    return first_frame_cond


@torch.no_grad()
def encode_clip(clip_model, pil_image, device="cuda", target_h=448, target_w=896, preprocess_mode="crop"):
    """Encode reference image -> clip_features [1, 1, 257, 1280]."""
    import torchvision.transforms as T
    from fastgen.datasets.infinitetalk_dataloader import _encode_clip

    # _encode_clip expects cond_image as [1, C, 1, H, W] float32 in [-1, 1]
    image = preprocess_image(pil_image, target_h, target_w, mode=preprocess_mode)
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img_tensor = transform(image)  # [3, H, W]
    cond_image = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]

    clip_features = _encode_clip(clip_model, device=device, cond_image=cond_image)
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
    # _load_clip expects a directory; if given a file path, use its parent
    if os.path.isfile(clip_path):
        clip_path = os.path.dirname(clip_path)
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
        wav2vec_path, local_files_only=True, attn_implementation="eager"
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
                  context_noise, seed, device, dtype, anchor_first_frame=True):
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

    # Extract clean first-frame latent for optional I2V overwrite (matches original
    # InfiniteTalk inference: multitalk.py line 711 overwrites frame 0 at every step)
    first_frame_latent = condition["first_frame_cond"][:, :, 0:1]  # [B, 16, 1, H, W]

    print(f"  AR loop: {num_blocks} blocks x {num_denoise_steps} denoising steps "
          f"= {num_blocks * (num_denoise_steps + 1)} forward passes"
          f"{' (anchor_first_frame=ON)' if anchor_first_frame else ''}")

    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * chunk_size
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        for step_idx in range(num_denoise_steps):
            t_cur = t_list_t[step_idx]
            t_next = t_list_t[step_idx + 1]

            # No pre-forward input anchor: the model expects noisy input at
            # timestep t_cur. Injecting clean latent without timestep masking
            # creates a distribution mismatch. The network's forward() handles
            # frame 0 via output-level replacement (matching SF training).

            x0_pred = model(
                noisy_input,
                t_cur.expand(B),  # keep float64 to match _student_sample_loop
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=False,
                is_ar=True,
                fwd_pred_type="x0",
                use_gradient_checkpointing=False,
            )

            # Network forward() already anchors frame 0 in output.
            # This is redundant but kept as explicit safety.
            if anchor_first_frame and cur_start_frame == 0:
                x0_pred = x0_pred.clone()
                x0_pred[:, :, 0:1] = first_frame_latent

            if t_next > 0:
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.expand(B),
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

    # Move VAE (model + scale tensors) to GPU for decoding
    vae.model = vae.model.to(device)
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    latent = output_latents[0].to(device=device, dtype=torch.float32)
    video_tensor = vae.decode([latent])

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
    # Step 3: Build sample list
    # ------------------------------------------------------------------
    if args.precomputed_list:
        # Batch mode: read list of precomputed dirs
        list_dir = os.path.dirname(os.path.abspath(args.precomputed_list))
        with open(args.precomputed_list) as f:
            sample_dirs = [line.strip() for line in f if line.strip()]
        # Resolve relative paths against the list file's location or CWD
        resolved = []
        for d in sample_dirs:
            if os.path.isabs(d):
                resolved.append(d)
            elif os.path.isdir(d):
                resolved.append(os.path.abspath(d))
            elif os.path.isdir(os.path.join(list_dir, d)):
                resolved.append(os.path.join(list_dir, d))
            else:
                print(f"  WARNING: skipping non-existent dir: {d}")
        sample_dirs = resolved
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n[3/5] Batch mode: {len(sample_dirs)} samples")
    elif args.precomputed_dir:
        sample_dirs = [args.precomputed_dir]
    elif args.video_dir:
        sample_dirs = None  # handled below as video_dir batch
    else:
        sample_dirs = None  # raw input mode

    # ------------------------------------------------------------------
    # Step 3b: Encode raw inputs (if not precomputed)
    # ------------------------------------------------------------------
    raw_encoded = None
    raw_encoded_list = None  # for --video_dir batch mode

    if args.video_dir:
        # Batch video directory mode: encode all videos up front, then load model once
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_files = sorted([
            os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir)
            if os.path.splitext(f)[1].lower() in video_exts
        ])
        if not video_files:
            raise RuntimeError(f"No video files found in {args.video_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n[3/5] Video dir batch mode: {len(video_files)} videos in {args.video_dir}")

        clip_model = load_clip(args.clip_path, device=device)
        wav2vec_fe, wav2vec_model = load_wav2vec(args.wav2vec_path, device=device)

        raw_encoded_list = []
        for vi, vpath in enumerate(video_files):
            vname = os.path.splitext(os.path.basename(vpath))[0]
            print(f"  Encoding {vi+1}/{len(video_files)}: {vname}")
            pil_image, audio_path = extract_first_frame_and_audio(vpath)
            num_latent_v, num_video_v = compute_generation_length(
                audio_path, args.num_latent_frames, args.chunk_size, args.fps
            )
            ffc = encode_reference_image(vae, pil_image, num_latent_v,
                                         target_h=args.target_h, target_w=args.target_w, device=device,
                                         preprocess_mode=args.preprocess_mode)
            cf = encode_clip(clip_model, pil_image, device=device,
                            target_h=args.target_h, target_w=args.target_w,
                            preprocess_mode=args.preprocess_mode)
            ae = encode_audio_from_file(wav2vec_fe, wav2vec_model, audio_path, num_video_v, device=device)
            raw_encoded_list.append({
                "first_frame_cond": ffc.cpu(), "clip_features": cf.cpu(),
                "audio_emb": ae.cpu(), "text_embeds": text_embeds.cpu(),
                "num_latent": num_latent_v, "num_video": num_video_v,
                "audio_path": audio_path, "name": vname,
            })

        del clip_model, wav2vec_fe, wav2vec_model
        torch.cuda.empty_cache()
        print(f"  All {len(video_files)} videos encoded, encoders unloaded")

        # Set sample_dirs sentinel for the loop
        sample_dirs = [None] * len(raw_encoded_list)

    elif sample_dirs is None:
        print("\n[3/5] Encoding raw inputs...")
        if args.video is not None:
            pil_image, audio_path = extract_first_frame_and_audio(args.video)
            print(f"  Extracted first frame and audio from {args.video}")
        else:
            from PIL import Image
            pil_image = Image.open(args.image).convert("RGB")
            audio_path = args.audio

        num_latent, num_video = compute_generation_length(
            audio_path, args.num_latent_frames, args.chunk_size, args.fps
        )

        clip_model = load_clip(args.clip_path, device=device)
        wav2vec_fe, wav2vec_model = load_wav2vec(args.wav2vec_path, device=device)

        first_frame_cond = encode_reference_image(vae, pil_image, num_latent,
                                                     target_h=args.target_h, target_w=args.target_w, device=device,
                                                     preprocess_mode=args.preprocess_mode)
        clip_features = encode_clip(clip_model, pil_image, device=device,
                                    target_h=args.target_h, target_w=args.target_w,
                                    preprocess_mode=args.preprocess_mode)
        audio_emb = encode_audio_from_file(wav2vec_fe, wav2vec_model, audio_path, num_video, device=device)

        del clip_model, wav2vec_fe, wav2vec_model
        torch.cuda.empty_cache()
        print("  Encoders unloaded, VRAM freed")

        if args.source_audio:
            audio_path = args.source_audio

        if args.num_latent_frames is not None:
            num_latent = args.num_latent_frames
            num_video = 1 + (num_latent - 1) * 4
            first_frame_cond = first_frame_cond[:, :, :num_latent]
            audio_emb = audio_emb[:, :num_video]

        raw_encoded = {
            "first_frame_cond": first_frame_cond, "clip_features": clip_features,
            "audio_emb": audio_emb, "text_embeds": text_embeds,
            "num_latent": num_latent, "num_video": num_video,
            "audio_path": audio_path,
        }
        sample_dirs = [None]  # single-item sentinel for the loop

    # ------------------------------------------------------------------
    # Step 4: Load diffusion model (once)
    # ------------------------------------------------------------------
    # Use first sample to get num_latent for initial cache allocation
    if raw_encoded:
        init_num_latent = raw_encoded["num_latent"]
    elif raw_encoded_list:
        init_num_latent = max(d["num_latent"] for d in raw_encoded_list)
    else:
        first_data = load_precomputed(sample_dirs[0], args.chunk_size, quarter_res=args.quarter_res)
        init_num_latent = first_data["num_latent"]
        del first_data

    print("\n[4/5] Loading diffusion model...")
    model = load_diffusion_model(args, init_num_latent, device, dtype)

    # ------------------------------------------------------------------
    # Step 5: Per-sample inference loop
    # ------------------------------------------------------------------
    total = len(sample_dirs)
    for sample_idx, sample_dir in enumerate(sample_dirs):
        print(f"\n{'='*60}")
        sample_name = None
        if raw_encoded_list:
            sample_name = raw_encoded_list[sample_idx]["name"]
        elif sample_dir:
            sample_name = os.path.basename(sample_dir)
        print(f"[5/5] Sample {sample_idx + 1}/{total}" +
              (f": {sample_name}" if sample_name else ""))
        print("=" * 60)

        # Load/resolve condition tensors
        if raw_encoded_list:
            cond_data = raw_encoded_list[sample_idx]
        elif raw_encoded:
            cond_data = raw_encoded
        else:
            cond_data = load_precomputed(sample_dir, args.chunk_size, quarter_res=args.quarter_res)
            # Resolve audio: CLI override > precomputed audio_path.txt > audio_data_root lookup
            if args.source_audio:
                cond_data["audio_path"] = args.source_audio
            elif not cond_data.get("audio_path") and args.audio_data_root and sample_dir:
                audio_path = resolve_audio_for_precomputed(sample_dir, args.audio_data_root)
                if audio_path:
                    print(f"  Resolved source audio: {audio_path}")
                cond_data["audio_path"] = audio_path

        num_latent = cond_data["num_latent"]
        num_video = cond_data["num_video"]
        audio_path = cond_data.get("audio_path")

        # Override length if specified
        if args.num_latent_frames is not None:
            num_latent = args.num_latent_frames
            num_video = 1 + (num_latent - 1) * 4

        # Build condition dict
        te = cond_data.get("text_embeds", text_embeds)
        condition = {
            "text_embeds": te.to(device=device, dtype=dtype),
            "first_frame_cond": cond_data["first_frame_cond"][:, :, :num_latent].to(device=device, dtype=dtype),
            "clip_features": cond_data["clip_features"].to(device=device, dtype=dtype),
            "audio_emb": cond_data["audio_emb"][:, :num_video].to(device=device, dtype=dtype),
        }

        # Update model for variable length
        model.total_num_frames = num_latent

        # Run AR inference
        print(f"  Running AR inference ({num_latent} latent / {num_video} video frames)...")
        anchor = not args.no_anchor_first_frame
        output_latents = run_inference(
            model, condition, num_latent, args.chunk_size,
            args.context_noise, args.seed + sample_idx, device, dtype,
            anchor_first_frame=anchor,
        )

        # Determine output path
        if args.output_dir:
            name = sample_name or f"sample_{sample_idx}"
            out_path = os.path.join(args.output_dir, f"{name}.mp4")
        else:
            out_path = args.output_path

        # Decode and save
        print("  Decoding and saving...")
        decode_and_save(vae, output_latents, audio_path, out_path, args.fps, device)
        print(f"  Output: {out_path}")

        # Free per-sample tensors
        del output_latents, condition
        torch.cuda.empty_cache()

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print(f"\nDone! Processed {total} sample(s).")


if __name__ == "__main__":
    main()
