#!/usr/bin/env python3
"""Precompute VAE latents, CLIP features, T5 text embeddings, and wav2vec2
audio embeddings from raw video+audio data using InfiniteTalk's own encoders.

Outputs per sample (saved as .pt files):
    vae_latents.pt       [16, T_lat, H_lat, W_lat]  VAE-encoded video
    first_frame_cond.pt  [16, T_lat, H_lat, W_lat]  VAE-encoded first-frame + zero pad
    clip_features.pt     [1, 257, 1280]              CLIP ViT-H/14 on reference frame
    audio_emb.pt         [num_video_frames, 12, 768] wav2vec2 hidden states
    text_embeds.pt       [1, 512, 4096]              T5 UMT5-XXL text embeddings

One shared file:
    neg_text_embeds.pt   [1, 512, 4096]              Negative-prompt T5 embeddings
"""

import argparse
import csv
import logging
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults matching InfiniteTalk-14B config
# ---------------------------------------------------------------------------
DEFAULT_NEG_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)
T5_TEXT_LEN = 512  # max token length for T5 encoder
VAE_STRIDE = (4, 8, 8)  # temporal, height, width stride of WanVAE


# ===================================================================
# Encoder loading helpers
# ===================================================================

def _add_infinitetalk_to_path():
    """Add InfiniteTalk source root to sys.path and install mock modules
    for heavy dependencies (xfuser, xformers) that are not needed for
    pure encoding workloads."""
    it_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     "../../InfiniteTalk"))
    if it_root not in sys.path:
        sys.path.insert(0, it_root)

    # wan.modules.attention imports xfuser and xformers at module level.
    # We only need the CLIP visual encoder (which uses flash_attention) and
    # the VAE / T5 (which do not).  Install lightweight mocks so the import
    # graph resolves without installing those large packages.
    import types

    class _MockModule(types.ModuleType):
        """A mock module that returns dummy callables for unknown attributes
        while preserving standard module attributes."""
        def __getattr__(self, name):
            # Let standard module attrs raise normally so inspect/importlib
            # do not get confused.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return lambda *a, **k: None

    def _ensure_mock(module_path):
        """Create a mock module at *module_path* (e.g. 'a.b.c') if absent."""
        import importlib.machinery
        parts = module_path.split(".")
        for i in range(len(parts)):
            partial = ".".join(parts[: i + 1])
            if partial not in sys.modules:
                mod = _MockModule(partial)
                # Mark intermediate mocks as packages so sub-imports work
                if i < len(parts) - 1:
                    mod.__path__ = []
                # Provide a proper __spec__ so importlib.util.find_spec()
                # does not choke when diffusers checks for xformers.
                mod.__spec__ = importlib.machinery.ModuleSpec(partial, None)
                sys.modules[partial] = mod

    for mock_mod in [
        "xfuser",
        "xfuser.core",
        "xfuser.core.distributed",
        "xformers",
        "xformers.ops",
        "optimum",
        "optimum.quanto",
        "optimum.quanto.nn",
        "optimum.quanto.nn.qlinear",
    ]:
        _ensure_mock(mock_mod)

    # Provide dummy symbols that t5.py imports from optimum.quanto
    oq = sys.modules["optimum.quanto"]
    oq.quantize = lambda *a, **k: None
    oq.freeze = lambda *a, **k: None
    oq.qint8 = None
    oq.requantize = lambda *a, **k: None

    # Provide specific dummy functions that attention.py imports by name
    xfuser_dist = sys.modules["xfuser.core.distributed"]
    xfuser_dist.get_sequence_parallel_rank = lambda: 0
    xfuser_dist.get_sequence_parallel_world_size = lambda: 1
    xfuser_dist.get_sp_group = lambda: None

    # wan/__init__.py eagerly imports InfiniteTalkPipeline and other heavy
    # pipeline classes (they require accelerate.init_empty_weights, FSDP,
    # etc.).  We only need the encoder sub-modules.  Install a thin "wan"
    # package stub so that ``from wan.modules.vae import WanVAE`` resolves
    # without executing the real wan/__init__.py.
    import importlib

    # Create wan package stub (empty, just marks it as a package)
    wan_pkg = types.ModuleType("wan")
    wan_pkg.__path__ = [os.path.join(it_root, "wan")]
    wan_pkg.__package__ = "wan"
    wan_pkg.__file__ = os.path.join(it_root, "wan", "__init__.py")
    sys.modules["wan"] = wan_pkg

    # Pre-import the sub-packages we need so they attach to the stub
    for sub in ("wan.modules", "wan.utils", "wan.configs"):
        importlib.import_module(sub)

    return it_root


def load_vae(weights_dir: str, device: str):
    """Load WanVAE encoder (float32 on device)."""
    from wan.modules.vae import WanVAE

    vae_path = os.path.join(weights_dir, "Wan2.1_VAE.pth")
    logger.info("Loading VAE from %s", vae_path)
    vae = WanVAE(vae_pth=vae_path, device=device)
    return vae


def load_clip(weights_dir: str, device: str):
    """Load CLIP ViT-H/14 visual encoder.

    The CLIPModel class relies on ``flash_attention`` from
    ``wan.modules.attention``, which uses flash_attn (available in this env).
    No monkey-patching needed — the original flash_attention function is used
    directly for exact numerical match with the InfiniteTalk pipeline.
    """
    from wan.modules.clip import CLIPModel

    clip_ckpt = os.path.join(
        weights_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    clip_tok = os.path.join(weights_dir, "xlm-roberta-large")
    logger.info("Loading CLIP from %s", clip_ckpt)
    clip_model = CLIPModel(
        dtype=torch.float16,
        device=device,
        checkpoint_path=clip_ckpt,
        tokenizer_path=clip_tok,
    )
    return clip_model


def load_t5(weights_dir: str, device: str):
    """Load T5 UMT5-XXL encoder (bf16)."""
    from wan.modules.t5 import T5EncoderModel

    t5_ckpt = os.path.join(weights_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    t5_tok = os.path.join(weights_dir, "google/umt5-xxl")
    logger.info("Loading T5 from %s", t5_ckpt)
    t5_encoder = T5EncoderModel(
        text_len=T5_TEXT_LEN,
        dtype=torch.bfloat16,
        device=torch.device(device),
        checkpoint_path=t5_ckpt,
        tokenizer_path=t5_tok,
    )
    return t5_encoder


def load_wav2vec(wav2vec_dir: str, device: str):
    """Load wav2vec2 model and feature extractor.

    Uses InfiniteTalk's custom Wav2Vec2Model that adds a
    ``linear_interpolation`` step inside ``forward()``.
    """
    from transformers import Wav2Vec2FeatureExtractor
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model

    logger.info("Loading wav2vec2 from %s", wav2vec_dir)
    audio_encoder = Wav2Vec2Model.from_pretrained(
        wav2vec_dir, local_files_only=True, attn_implementation="eager"
    ).to(device).eval()
    audio_encoder.feature_extractor._freeze_parameters()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_dir, local_files_only=True
    )
    return feature_extractor, audio_encoder


# ===================================================================
# Video / audio loading helpers
# ===================================================================

def load_video_frames(video_path: str, frame_count: int = 81):
    """Load ``frame_count`` frames from *video_path* using decord or imageio.

    Returns:
        frames: np.ndarray  [T, H, W, 3] uint8
        fps: float
    """
    # Try decord first (fast, C++ backend)
    try:
        from decord import VideoReader, cpu as decord_cpu
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        fps = float(vr.get_avg_fps())
        total = len(vr)
        n = min(total, frame_count)
        indices = list(range(n))
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3]
        return frames, fps
    except Exception:
        pass

    # Fallback: imageio
    try:
        import imageio.v3 as iio
        frames_list = []
        meta = iio.improps(video_path, plugin="pyav")
        fps = 25.0  # default
        reader = iio.imiter(video_path, plugin="pyav")
        for i, frame in enumerate(reader):
            if i >= frame_count:
                break
            frames_list.append(frame)
        frames = np.stack(frames_list, axis=0)
        return frames, fps
    except Exception:
        pass

    # Fallback: av
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 25.0
    frames_list = []
    for frame in container.decode(video=0):
        if len(frames_list) >= frame_count:
            break
        frames_list.append(frame.to_ndarray(format="rgb24"))
    container.close()
    frames = np.stack(frames_list, axis=0)
    return frames, fps


def resize_and_center_crop(frames: np.ndarray, target_h: int, target_w: int):
    """Resize + center-crop a batch of frames to (target_h, target_w).

    Uses torch.nn.functional.interpolate (bilinear) for batch processing.
    Note: For the FIRST frame used as conditioning (CLIP + VAE first_frame_cond),
    use resize_and_centercrop_pil() instead — it matches the original pipeline
    exactly (PIL BILINEAR resize → numpy → torch → torchvision center_crop).

    Args:
        frames: [T, H, W, 3] uint8
    Returns:
        tensor: [3, T, target_h, target_w] float32 in [-1, 1]
    """
    t_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
    _, _, h, w = t_tensor.shape

    # Scale so the smaller dimension matches target, then center-crop
    scale_h = target_h / h
    scale_w = target_w / w
    scale = max(scale_h, scale_w)
    new_h = math.ceil(scale * h)
    new_w = math.ceil(scale * w)

    t_tensor = F.interpolate(
        t_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )  # [T, 3, new_h, new_w]

    # Center crop
    crop_top = (new_h - target_h) // 2
    crop_left = (new_w - target_w) // 2
    t_tensor = t_tensor[:, :, crop_top : crop_top + target_h,
                        crop_left : crop_left + target_w]

    # Normalize to [-1, 1]
    t_tensor = (t_tensor / 255.0 - 0.5) * 2.0

    # Rearrange to [C, T, H, W]
    t_tensor = t_tensor.permute(1, 0, 2, 3)
    return t_tensor


def resize_and_centercrop_pil(pil_image, target_h: int, target_w: int):
    """Resize + center-crop a PIL Image to (target_h, target_w).

    This matches the EXACT preprocessing from InfiniteTalk's multitalk.py
    resize_and_centercrop() function (PIL path): PIL BILINEAR resize,
    numpy → torch, torchvision center_crop, resulting in [1, C, 1, H, W].

    The result is normalized to [-1, 1].

    Args:
        pil_image: PIL Image (RGB)
        target_h, target_w: target spatial dimensions

    Returns:
        tensor: [1, C, 1, target_h, target_w] float32 in [-1, 1]
    """
    from PIL import Image
    import torchvision.transforms as transforms

    orig_h, orig_w = pil_image.height, pil_image.width
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    scale = max(scale_h, scale_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)

    resized_image = pil_image.resize((final_w, final_h), resample=Image.BILINEAR)
    resized_image = np.array(resized_image)
    resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
    cropped_tensor = transforms.functional.center_crop(resized_tensor, (target_h, target_w))
    cropped_tensor = cropped_tensor[:, :, None, :, :]  # [1, C, 1, H, W]

    # Normalize to [-1, 1]
    cropped_tensor = cropped_tensor.float() / 255.0
    cropped_tensor = (cropped_tensor - 0.5) * 2.0

    return cropped_tensor


def load_audio_array(audio_path: str, sample_rate: int = 16000):
    """Load audio waveform as 1-D numpy float array at *sample_rate*."""
    import librosa
    import pyloudnorm as pyln

    speech, sr = librosa.load(audio_path, sr=sample_rate)
    # Loudness normalisation (same as InfiniteTalk)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(speech)
    if abs(loudness) <= 100:
        speech = pyln.normalize.loudness(speech, loudness, -23.0)
    return speech


# ===================================================================
# Per-encoder processing
# ===================================================================

@torch.no_grad()
def encode_vae(vae, video_tensor, device, cond_image=None):
    """Encode a full video and the first-frame condition through the VAE.

    Args:
        video_tensor: [C, T, H, W] float32, values in [-1, 1]
        cond_image:   [1, C, 1, H, W] float32, values in [-1, 1]
                      PIL-processed first frame (from resize_and_centercrop_pil).
                      If None, falls back to using video_tensor[:, :1].

    Returns:
        latents:            [C_lat, T_lat, H_lat, W_lat]
        first_frame_cond:   [C_lat, T_lat, H_lat, W_lat]  (first frame encoded + zero-padded)
    """
    video = video_tensor.to(device)  # [C, T, H, W]

    # Full video latents
    latents = vae.encode([video])  # list of [C_lat, T_lat, H_lat, W_lat]
    latents = latents[0].cpu()

    # First-frame condition: first frame + zero padding (same as original pipeline)
    # Original: torch.concat([cond_image, video_frames], dim=2) where cond_image
    # is [1, C, 1, H, W] from PIL-based resize_and_centercrop
    c, t, h, w = video.shape
    if cond_image is not None:
        # Use the PIL-processed first frame (matches original exactly)
        cond_image_dev = cond_image.to(device)  # [1, C, 1, H, W]
        # t is the number of pixel frames (e.g. 81)
        # Original: zeros(1, C, frame_num - 1, H, W) where frame_num = t = 81
        zero_padding = torch.zeros(
            1, cond_image_dev.shape[1], t - 1,
            h, w, device=device
        )
        first_frame_padded = torch.cat(
            [cond_image_dev, zero_padding], dim=2
        )  # [1, C, t, H, W]
        first_frame_cond = vae.encode(first_frame_padded)
        # Original casts to param_dtype=bfloat16: y = torch.stack(y).to(self.param_dtype)
        first_frame_cond = torch.stack(first_frame_cond).to(torch.bfloat16)[0].cpu()
    else:
        # Fallback: use video_tensor's first frame
        first_frame = video[:, :1, :, :]  # [C, 1, H, W]
        padding = torch.zeros(c, t - 1, h, w, device=device)
        first_frame_padded = torch.cat([first_frame, padding], dim=1)  # [C, T, H, W]
        first_frame_cond = vae.encode([first_frame_padded])
        first_frame_cond = first_frame_cond[0].cpu()

    # Also encode ONLY the reference image (no temporal context) for motion anchoring.
    # Original: latent_motion_frames = self.vae.encode(cond_image)[0]
    # cond_image is [1, C, 1, H, W] — a single frame, not padded.
    # This produces a different latent than first_frame_cond[:, :1] because the
    # VAE's temporal convolutions don't see the zero padding.
    if cond_image is not None:
        motion_frame = vae.encode(cond_image_dev)
        motion_frame = torch.stack(motion_frame).to(torch.bfloat16)[0].cpu()
        # motion_frame: [16, 1, H, W] — single latent frame, no temporal context
    else:
        single_frame = video[:, :1, :, :].unsqueeze(0).to(device)  # [1, C, 1, H, W]
        motion_frame = vae.encode(single_frame)
        motion_frame = torch.stack(motion_frame)[0].cpu()

    return latents, first_frame_cond, motion_frame


@torch.no_grad()
def encode_clip(clip_model, device, cond_image=None, video_tensor=None):
    """Extract CLIP visual features from the first frame.

    Args:
        cond_image:   [1, C, 1, H, W] float32 in [-1, 1]
                      PIL-processed first frame (from resize_and_centercrop_pil).
                      If None, falls back to using video_tensor.
        video_tensor: [C, T, H, W] float32 in [-1, 1] (fallback)

    Returns:
        clip_features: [1, 257, 1280]  (CLS + 256 patches from ViT-H/14)
    """
    # Original pipeline: clip.visual(cond_image[:, :, -1:, :, :])
    # cond_image is [1, C, 1, H, W] from PIL resize_and_centercrop
    if cond_image is not None:
        first_frame = cond_image[:, :, -1:, :, :].to(device)  # [1, C, 1, H, W]
    else:
        first_frame = video_tensor[:, :1, :, :].unsqueeze(0).to(device)  # [1, C, 1, H, W]
    clip_features = clip_model.visual(first_frame)  # [1, 257, 1280]
    # Original pipeline casts to param_dtype=bfloat16 after CLIP
    clip_features = clip_features.to(torch.bfloat16)
    return clip_features.cpu()


@torch.no_grad()
def encode_t5(t5_encoder, text: str, device):
    """Encode text through T5 UMT5-XXL.

    Returns the FULL padded embedding (not trimmed to actual token length)
    so it can be loaded directly during training.

    Returns:
        text_embeds: [1, 512, 4096]
    """
    # T5EncoderModel.__call__ returns a list of trimmed tensors.
    # We call the underlying model directly to get the padded version.
    ids, mask = t5_encoder.tokenizer(
        [text], return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    context = t5_encoder.model(ids, mask)  # [1, 512, 4096]
    return context.float().cpu()


@torch.no_grad()
def encode_audio(feature_extractor, audio_encoder, audio_array,
                 num_video_frames: int, device, sr: int = 16000):
    """Extract wav2vec2 embeddings from audio.

    Follows the exact same logic as ``get_embedding()`` in
    ``generate_infinitetalk.py``.

    Returns:
        audio_emb: [num_video_frames, 12, 768]
    """
    audio_duration = len(audio_array) / sr
    video_length = audio_duration * 25  # assume 25 fps

    # Feature extraction
    audio_feature = np.squeeze(
        feature_extractor(audio_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device)
    audio_feature = audio_feature.unsqueeze(0)

    # Forward through wav2vec2 with linear interpolation to video_length
    embeddings = audio_encoder(
        audio_feature,
        seq_len=int(video_length),
        output_hidden_states=True,
    )

    # Stack hidden states from layers 1-12 (skip layer 0 which is the feature projection)
    # embeddings.hidden_states: tuple of 13 tensors, each [1, seq_len, 768]
    # After stacking layers 1-12: [1, 12, seq_len, 768]
    # After rearrange "b s d -> s b d" on the stacked: [seq_len, 12, 768]
    from einops import rearrange
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    # audio_emb: [12, seq_len, 768]
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    # audio_emb: [seq_len, 12, 768]

    # Trim or pad to num_video_frames
    seq_len = audio_emb.shape[0]
    if seq_len > num_video_frames:
        audio_emb = audio_emb[:num_video_frames]
    elif seq_len < num_video_frames:
        pad = torch.zeros(
            num_video_frames - seq_len, 12, 768,
            dtype=audio_emb.dtype, device=audio_emb.device)
        audio_emb = torch.cat([audio_emb, pad], dim=0)

    return audio_emb.cpu()


# ===================================================================
# Main loop
# ===================================================================

def output_complete(sample_dir: str):
    """Return True if all expected output files exist for a sample."""
    required = [
        "vae_latents.pt",
        "first_frame_cond.pt",
        "clip_features.pt",
        "audio_emb.pt",
        "text_embeds.pt",
    ]
    return all(os.path.isfile(os.path.join(sample_dir, f)) for f in required)


def process_sample(
    row: dict,
    data_root: str,
    output_dir: str,
    vae,
    clip_model,
    t5_encoder,
    wav2vec_fe,
    audio_encoder,
    device: str,
    resolution: int,
    frame_count: int,
):
    """Process a single sample: encode all modalities and save."""
    video_rel = row["video_path"]
    audio_rel = row["audio_path"]
    text = row["text"]
    sample_name = os.path.splitext(video_rel.replace("/", "_"))[0]

    sample_dir = os.path.join(output_dir, sample_name)
    if output_complete(sample_dir):
        logger.info("  Skipping (already complete): %s", sample_name)
        return True

    os.makedirs(sample_dir, exist_ok=True)
    video_path = os.path.join(data_root, video_rel)
    audio_path = os.path.join(data_root, audio_rel)

    if not os.path.isfile(video_path):
        logger.warning("  Video not found: %s", video_path)
        return False
    if not os.path.isfile(audio_path):
        logger.warning("  Audio not found: %s", audio_path)
        return False

    target_h = resolution
    target_w = resolution

    # Per-file skip: only compute what's missing
    def _exists(name):
        return os.path.isfile(os.path.join(sample_dir, name))

    need_vae = not _exists("vae_latents.pt")
    need_ffc = not _exists("first_frame_cond.pt")
    need_clip = not _exists("clip_features.pt")
    need_t5 = not _exists("text_embeds.pt")
    need_audio = not _exists("audio_emb.pt")

    # Video loading + PIL first frame needed for VAE, first_frame_cond, or CLIP
    need_video = need_vae or need_ffc or need_clip
    video_tensor = None
    cond_image = None

    if need_video:
        frames, fps = load_video_frames(video_path, frame_count)
        actual_frames = frames.shape[0]
        if actual_frames < frame_count:
            logger.warning(
                "  Video has %d frames (need %d), zero-padding", actual_frames, frame_count)
            pad = np.zeros(
                (frame_count - actual_frames, frames.shape[1], frames.shape[2], 3),
                dtype=frames.dtype)
            frames = np.concatenate([frames, pad], axis=0)

        video_tensor = resize_and_center_crop(frames, target_h, target_w)
        # video_tensor: [3, T, H, W] float32 in [-1, 1]

        # Extract first frame as PIL and apply original preprocessing
        from PIL import Image
        first_frame_pil = Image.fromarray(frames[0])  # [H, W, 3] uint8 → PIL
        cond_image = resize_and_centercrop_pil(first_frame_pil, target_h, target_w)
        # cond_image: [1, C, 1, H, W] float32 in [-1, 1]

    # ---- VAE encode ----
    need_motion = not _exists("motion_frame.pt")
    if need_vae or need_ffc or need_motion:
        logger.info("  Encoding VAE...")
        latents, first_frame_cond, motion_frame = encode_vae(
            vae, video_tensor, device, cond_image=cond_image)
        if need_vae:
            torch.save(latents, os.path.join(sample_dir, "vae_latents.pt"))
        if need_ffc:
            torch.save(first_frame_cond, os.path.join(sample_dir, "first_frame_cond.pt"))
        if need_motion:
            torch.save(motion_frame, os.path.join(sample_dir, "motion_frame.pt"))
        del latents, first_frame_cond, motion_frame
        torch.cuda.empty_cache()
    else:
        logger.info("  Skipping VAE (exists)")

    # ---- CLIP encode ----
    if need_clip:
        logger.info("  Encoding CLIP...")
        clip_features = encode_clip(clip_model, device, cond_image=cond_image)
        torch.save(clip_features, os.path.join(sample_dir, "clip_features.pt"))
        del clip_features
        torch.cuda.empty_cache()
    else:
        logger.info("  Skipping CLIP (exists)")

    # ---- T5 encode ----
    if need_t5:
        logger.info("  Encoding T5...")
        text_embeds = encode_t5(t5_encoder, text, device)
        torch.save(text_embeds, os.path.join(sample_dir, "text_embeds.pt"))
        del text_embeds
        torch.cuda.empty_cache()
    else:
        logger.info("  Skipping T5 (exists)")

    # ---- Audio encode ----
    if need_audio:
        logger.info("  Encoding audio (wav2vec2)...")
        audio_array = load_audio_array(audio_path)
        audio_emb = encode_audio(
            wav2vec_fe, audio_encoder, audio_array,
            num_video_frames=frame_count, device=device)
        torch.save(audio_emb, os.path.join(sample_dir, "audio_emb.pt"))
        del audio_emb, audio_array
        torch.cuda.empty_cache()
    else:
        logger.info("  Skipping audio (exists)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Precompute InfiniteTalk training data (VAE, CLIP, T5, wav2vec2)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to video_list.csv")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset (paths in CSV are relative to this)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write precomputed .pt files")
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="InfiniteTalk weights dir (Wan2.1-I2V-14B-480P/)")
    parser.add_argument("--wav2vec_dir", type=str, required=True,
                        help="Path to chinese-wav2vec2-base/ directory")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Process only first N samples (0 = all)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index in CSV (for multi-GPU sharding)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End index in CSV, exclusive (-1 = all)")
    parser.add_argument("--resolution", type=int, default=640,
                        help="Target square resolution (default 640 for 480p bucket)")
    parser.add_argument("--frame_count", type=int, default=81,
                        help="Number of frames to extract per video (must be 4n+1)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device for encoding")
    parser.add_argument("--neg_prompt", type=str, default=DEFAULT_NEG_PROMPT,
                        help="Negative prompt for T5 encoding")
    args = parser.parse_args()

    assert (args.frame_count - 1) % 4 == 0, \
        f"frame_count must be 4n+1, got {args.frame_count}"

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Add InfiniteTalk to path ----
    it_root = _add_infinitetalk_to_path()
    logger.info("InfiniteTalk source root: %s", it_root)

    # ---- Load CSV ----
    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    total = len(rows)
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    # Apply start_idx / end_idx sharding (for multi-GPU parallel runs)
    if args.start_idx > 0 or args.end_idx >= 0:
        end = args.end_idx if args.end_idx >= 0 else len(rows)
        rows = rows[args.start_idx : end]
    logger.info("Loaded %d / %d samples from CSV (range [%d:%d])",
                len(rows), total, args.start_idx,
                args.end_idx if args.end_idx >= 0 else total)

    # ---- Load encoders ----
    logger.info("Loading encoders to %s ...", args.device)
    t0 = time.time()

    vae = load_vae(args.weights_dir, args.device)
    logger.info("  VAE loaded (%.1fs)", time.time() - t0)

    t1 = time.time()
    clip_model = load_clip(args.weights_dir, args.device)
    logger.info("  CLIP loaded (%.1fs)", time.time() - t1)

    t2 = time.time()
    t5_encoder = load_t5(args.weights_dir, args.device)
    logger.info("  T5 loaded (%.1fs)", time.time() - t2)

    t3 = time.time()
    wav2vec_fe, audio_encoder = load_wav2vec(args.wav2vec_dir, args.device)
    logger.info("  wav2vec2 loaded (%.1fs)", time.time() - t3)

    logger.info("All encoders loaded in %.1fs", time.time() - t0)

    # ---- Precompute negative-prompt T5 embedding (shared) ----
    neg_path = os.path.join(args.output_dir, "neg_text_embeds.pt")
    if not os.path.isfile(neg_path):
        logger.info("Computing negative-prompt T5 embedding...")
        neg_embeds = encode_t5(t5_encoder, args.neg_prompt, args.device)
        torch.save(neg_embeds, neg_path)
        logger.info("  Saved neg_text_embeds.pt  %s", list(neg_embeds.shape))
        del neg_embeds
        torch.cuda.empty_cache()
    else:
        logger.info("neg_text_embeds.pt already exists, skipping")

    # ---- Process samples ----
    success = 0
    failed = 0
    skipped = 0
    for idx, row in enumerate(rows):
        sample_name = os.path.splitext(row["video_path"].replace("/", "_"))[0]
        sample_dir = os.path.join(args.output_dir, sample_name)
        if output_complete(sample_dir):
            logger.info("[%d/%d] Skipping (complete): %s", idx + 1, len(rows), sample_name)
            skipped += 1
            continue

        logger.info("[%d/%d] Processing: %s", idx + 1, len(rows), sample_name)
        t_sample = time.time()

        try:
            ok = process_sample(
                row=row,
                data_root=args.data_root,
                output_dir=args.output_dir,
                vae=vae,
                clip_model=clip_model,
                t5_encoder=t5_encoder,
                wav2vec_fe=wav2vec_fe,
                audio_encoder=audio_encoder,
                device=args.device,
                resolution=args.resolution,
                frame_count=args.frame_count,
            )
            elapsed = time.time() - t_sample
            if ok:
                success += 1
                logger.info("  Done in %.1fs", elapsed)
            else:
                failed += 1
                logger.warning("  Failed (missing input) in %.1fs", elapsed)
        except Exception:
            failed += 1
            elapsed = time.time() - t_sample
            logger.error(
                "  ERROR processing %s (%.1fs):\n%s",
                sample_name, elapsed, traceback.format_exc())

        torch.cuda.empty_cache()

    logger.info(
        "Finished. success=%d  skipped=%d  failed=%d  total=%d",
        success, skipped, failed, len(rows))


if __name__ == "__main__":
    main()
