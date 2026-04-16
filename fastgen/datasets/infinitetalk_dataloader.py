# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch Dataset for InfiniteTalk precomputed training data.

Each sample directory contains precomputed .pt files:
    - vae_latents.pt: [16, T_lat, H, W] — VAE-encoded video (T_lat=24 for 93 frames, or 21 for 81)
    - first_frame_cond.pt: [16, T_lat, H, W] — VAE-encoded reference frame + zero padding
    - clip_features.pt: [1, 257, 1280] — CLIP ViT-H/14 on reference frame
    - audio_emb.pt: [T_pixel, 12, 768] — wav2vec2 hidden states (T_pixel=93 or 81)
    - text_embeds.pt: [1, 512, 4096] — T5 UMT5-XXL text embeddings
    - neg_text_embeds.pt: [1, 512, 4096] — negative text embedding (shared across samples)
    - ode_path.pt: [num_steps, 16, T_lat, H, W] — ODE trajectory (KD only, optional)

The dataloader slices to num_latent_frames (default 21) at load time, so precomputed
data can have more frames than needed for training.

Audio windowing (5-frame sliding window) is NOT applied in the dataloader.
It is handled inside WanModel.forward() at inference time, matching InfiniteTalk's
original behavior (multitalk.py lines 523-533).

Lazy caching mode (optional):
    When raw_data_root, csv_path, weights_dir, and wav2vec_dir are provided,
    missing .pt files (except text_embeds.pt) are encoded on-the-fly using
    VAE, CLIP, and wav2vec2 encoders loaded once in __init__. Results are
    cached to disk atomically for subsequent epochs. T5 text_embeds.pt must
    be pre-computed (the 20GB T5 encoder is too large to co-reside with the
    14B training model on an 80GB GPU). Requires num_workers=0 since GPU
    encoders cannot cross process boundaries.
"""

import csv
import importlib
import importlib.machinery
import logging
import math
import os
import sys
import types
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

# Aspect-ratio buckets for 480p (627-class) — matches InfiniteTalk's original pipeline
ASPECT_RATIO_627 = {
    '0.26': [320, 1216], '0.38': [384, 1024], '0.50': [448, 896],
    '0.67': [512, 768],  '0.82': [576, 704],  '1.00': [640, 640],
    '1.22': [704, 576],  '1.50': [768, 512],  '1.86': [832, 448],
    '2.00': [896, 448],  '2.50': [960, 384],  '2.83': [1088, 384],
    '3.60': [1152, 320], '3.80': [1216, 320],  '4.00': [1280, 320],
}


# ===================================================================
# Lazy-mode helpers (module setup, encoder loading, encoding)
# ===================================================================

def _add_infinitetalk_to_path():
    """Add InfiniteTalk source root to sys.path and install mock modules
    for heavy dependencies (xfuser, xformers) that are not needed for
    pure encoding workloads.

    Mirrors scripts/precompute_infinitetalk_data.py::_add_infinitetalk_to_path().
    """
    it_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     "../../../InfiniteTalk"))
    if it_root not in sys.path:
        sys.path.insert(0, it_root)

    class _MockModule(types.ModuleType):
        """A mock module that returns dummy callables for unknown attributes."""
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return lambda *a, **k: None

    def _ensure_mock(module_path):
        parts = module_path.split(".")
        for i in range(len(parts)):
            partial = ".".join(parts[: i + 1])
            if partial not in sys.modules:
                mod = _MockModule(partial)
                if i < len(parts) - 1:
                    mod.__path__ = []
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

    # wan/__init__.py eagerly imports heavy pipeline classes.
    # Install a thin stub so encoder sub-modules resolve without it.
    if "wan" not in sys.modules:
        wan_pkg = types.ModuleType("wan")
        wan_pkg.__path__ = [os.path.join(it_root, "wan")]
        wan_pkg.__package__ = "wan"
        wan_pkg.__file__ = os.path.join(it_root, "wan", "__init__.py")
        sys.modules["wan"] = wan_pkg

        for sub in ("wan.modules", "wan.utils", "wan.configs"):
            importlib.import_module(sub)

    return it_root


def _load_vae(weights_dir: str, device: str):
    """Load WanVAE encoder (float32 on device)."""
    from wan.modules.vae import WanVAE

    vae_path = os.path.join(weights_dir, "Wan2.1_VAE.pth")
    logger.info("Lazy cache: loading VAE from %s", vae_path)
    vae = WanVAE(vae_pth=vae_path, device=device)
    return vae


def _load_clip(weights_dir: str, device: str):
    """Load CLIP ViT-H/14 visual encoder.

    On CPU, uses float32 (float16 ops unsupported on CPU).
    On GPU, uses float16 (native precision, uses flash_attn).
    """
    from wan.modules.clip import CLIPModel

    clip_ckpt = os.path.join(
        weights_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    clip_tok = os.path.join(weights_dir, "xlm-roberta-large")
    # CPU doesn't support float16 ops — use float32
    clip_dtype = torch.float32 if device == "cpu" else torch.float16
    logger.info("Lazy cache: loading CLIP from %s (dtype=%s)", clip_ckpt, clip_dtype)
    clip_model = CLIPModel(
        dtype=clip_dtype,
        device=device,
        checkpoint_path=clip_ckpt,
        tokenizer_path=clip_tok,
    )
    return clip_model


def _load_wav2vec(wav2vec_dir: str, device: str):
    """Load wav2vec2 model and feature extractor.

    Uses attn_implementation='eager' to avoid SDPA/output_attentions conflict.
    """
    from transformers import Wav2Vec2FeatureExtractor
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model

    logger.info("Lazy cache: loading wav2vec2 from %s", wav2vec_dir)
    audio_encoder = Wav2Vec2Model.from_pretrained(
        wav2vec_dir, local_files_only=True, attn_implementation="eager"
    ).to(device).eval()
    audio_encoder.feature_extractor._freeze_parameters()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_dir, local_files_only=True
    )
    return feature_extractor, audio_encoder


def _load_video_frames(video_path: str, frame_count: int = 93, target_fps: float = 25.0):
    """Load frames from video, resampled to target_fps.

    If the source FPS differs from target_fps, selects frames at timestamps
    corresponding to target_fps (nearest-neighbor in time). This matches the
    precompute script's behavior and the wav2vec2 assumption of 25fps.

    Returns:
        frames: np.ndarray [T, H, W, 3] uint8 (T <= frame_count)
        fps: float (always target_fps)
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    native_fps = float(stream.average_rate) if stream.average_rate else target_fps
    total_frames = stream.frames or 10000

    # Compute which source frames to grab for target_fps output
    if abs(native_fps - target_fps) < 0.5:
        # Close enough — no resampling needed
        indices = list(range(min(frame_count, total_frames)))
    else:
        indices = []
        for i in range(frame_count):
            t = i / target_fps  # target timestamp in seconds
            src_idx = round(t * native_fps)
            if src_idx >= total_frames:
                break
            indices.append(src_idx)

    # Decode needed frames
    index_set = set(indices)
    max_idx = max(indices) if indices else 0
    all_frames = {}
    for i, frame in enumerate(container.decode(video=0)):
        if i in index_set:
            all_frames[i] = frame.to_ndarray(format="rgb24")
        if i >= max_idx:
            break
    container.close()

    frames = np.stack([all_frames[i] for i in indices if i in all_frames], axis=0)
    return frames, target_fps


def _resize_and_center_crop(frames: np.ndarray, target_h: int, target_w: int):
    """Resize + center-crop frames with torch bilinear interpolation.

    Args:
        frames: [T, H, W, 3] uint8
    Returns:
        tensor: [3, T, target_h, target_w] float32 in [-1, 1]
    """
    t_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
    _, _, h, w = t_tensor.shape

    scale = max(target_h / h, target_w / w)
    new_h = math.ceil(scale * h)
    new_w = math.ceil(scale * w)

    t_tensor = F.interpolate(
        t_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )  # [T, 3, new_h, new_w]

    crop_top = (new_h - target_h) // 2
    crop_left = (new_w - target_w) // 2
    t_tensor = t_tensor[:, :, crop_top: crop_top + target_h,
                        crop_left: crop_left + target_w]

    # Normalize to [-1, 1]
    t_tensor = (t_tensor / 255.0 - 0.5) * 2.0

    # Rearrange to [C, T, H, W]
    t_tensor = t_tensor.permute(1, 0, 2, 3)
    return t_tensor


def _resize_and_centercrop_pil(pil_image, target_h: int, target_w: int):
    """PIL BILINEAR resize + center-crop for reference frame.

    Matches InfiniteTalk's multitalk.py resize_and_centercrop() exactly.

    Returns:
        tensor: [1, C, 1, target_h, target_w] float32 in [-1, 1]
    """
    from PIL import Image
    import torchvision.transforms as transforms

    scale = max(target_h / pil_image.height, target_w / pil_image.width)
    fh = math.ceil(scale * pil_image.height)
    fw = math.ceil(scale * pil_image.width)

    resized = np.array(pil_image.resize((fw, fh), resample=Image.BILINEAR))
    t = torch.from_numpy(resized)[None, ...].permute(0, 3, 1, 2).contiguous()
    cond_image = transforms.functional.center_crop(t, (target_h, target_w))
    cond_image = cond_image[:, :, None, :, :]  # [1, C, 1, H, W]
    cond_image = (cond_image.float() / 255.0 - 0.5) * 2.0
    return cond_image


def _load_audio_array(audio_path: str, sample_rate: int = 16000):
    """Load audio waveform as 1-D numpy float array with loudness normalization."""
    import librosa
    import pyloudnorm as pyln

    speech, sr = librosa.load(audio_path, sr=sample_rate)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(speech)
    if abs(loudness) <= 100:
        speech = pyln.normalize.loudness(speech, loudness, -23.0)
    return speech


@torch.no_grad()
def _encode_vae(vae, video_tensor, device, cond_image):
    """Encode video through VAE — produces latents, first_frame_cond, motion_frame.

    Matches scripts/precompute_infinitetalk_data.py::encode_vae() exactly.

    Args:
        video_tensor: [C, T, H, W] float32 in [-1, 1]
        cond_image: [1, C, 1, H, W] float32 in [-1, 1]
    Returns:
        latents: [C_lat, T_lat, H_lat, W_lat]
        first_frame_cond: [C_lat, T_lat, H_lat, W_lat]
        motion_frame: [C_lat, 1, H_lat, W_lat]
    """
    # Cast to match VAE weight dtype (WanVAE exposes .dtype; fallback to model params)
    vae_dtype = getattr(vae, "dtype", None) or next(vae.model.parameters()).dtype
    video = video_tensor.to(device=device, dtype=vae_dtype)

    # Full video latents (biggest activation spike — 93 frames)
    latents = vae.encode([video])
    latents = latents[0].cpu()
    # Free the big input immediately
    del video
    if device != "cpu":
        torch.cuda.empty_cache()

    # First-frame condition: first frame + zero padding
    c, t, h, w = video_tensor.shape
    cond_image_dev = cond_image.to(device=device, dtype=vae_dtype)  # [1, C, 1, H, W]
    zero_padding = torch.zeros(
        1, cond_image_dev.shape[1], t - 1, h, w, device=device, dtype=vae_dtype
    )
    first_frame_padded = torch.cat([cond_image_dev, zero_padding], dim=2)  # [1, C, t, H, W]
    del zero_padding
    first_frame_cond = vae.encode(first_frame_padded)
    first_frame_cond = torch.stack(first_frame_cond).to(torch.bfloat16)[0].cpu()
    del first_frame_padded

    # Motion frame: single reference frame (no temporal context)
    motion_frame = vae.encode(cond_image_dev)
    motion_frame = torch.stack(motion_frame).to(torch.bfloat16)[0].cpu()
    del cond_image_dev

    return latents, first_frame_cond, motion_frame


@torch.no_grad()
def _encode_clip(clip_model, device, cond_image):
    """Extract CLIP visual features from the reference frame.

    Matches scripts/precompute_infinitetalk_data.py::encode_clip() exactly.

    Args:
        cond_image: [1, C, 1, H, W] float32 in [-1, 1]
    Returns:
        clip_features: [1, 257, 1280] bf16
    """
    clip_dtype = getattr(clip_model, "dtype", None) or next(clip_model.model.parameters()).dtype
    first_frame = cond_image[:, :, -1:, :, :].to(device=device, dtype=clip_dtype)  # [1, C, 1, H, W]
    clip_features = clip_model.visual(first_frame)  # [1, 257, 1280]
    clip_features = clip_features.to(torch.bfloat16)
    return clip_features.cpu()


@torch.no_grad()
def _encode_audio(feature_extractor, audio_encoder, audio_array,
                  num_video_frames: int, device, sr: int = 16000):
    """Extract wav2vec2 embeddings from audio.

    Matches scripts/precompute_infinitetalk_data.py::encode_audio() exactly.

    Returns:
        audio_emb: [num_video_frames, 12, 768]
    """
    from einops import rearrange

    video_length = (len(audio_array) / sr) * 25  # assume 25 fps

    audio_feature = np.squeeze(
        feature_extractor(audio_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device)
    audio_feature = audio_feature.unsqueeze(0)

    embeddings = audio_encoder(
        audio_feature,
        seq_len=int(video_length),
        output_hidden_states=True,
    )

    # Stack hidden states from layers 1-12
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


def _save_atomic(tensor, path):
    """Save tensor atomically: write to .tmp then rename."""
    tmp_path = path + ".tmp"
    torch.save(tensor, tmp_path)
    os.rename(tmp_path, path)  # atomic on same filesystem


def _move_encoder_to(encoder, device):
    """Move a WanVAE / CLIPModel / nn.Module to device.

    WanVAE and CLIPModel aren't nn.Module — they wrap .model (nn.Module)
    and track .device as a string attribute. WanVAE also has .mean, .std,
    .scale tensors used in encode/decode that must be moved too.
    """
    if hasattr(encoder, "model") and hasattr(encoder.model, "to"):
        encoder.model.to(device)
    elif hasattr(encoder, "to"):
        encoder.to(device)
    # Move any tensor attributes (WanVAE: .mean, .std, .scale)
    for attr in ("mean", "std"):
        if hasattr(encoder, attr):
            val = getattr(encoder, attr)
            if isinstance(val, torch.Tensor):
                setattr(encoder, attr, val.to(device))
    if hasattr(encoder, "scale") and isinstance(getattr(encoder, "scale"), list):
        encoder.scale = [s.to(device) if isinstance(s, torch.Tensor) else s
                         for s in encoder.scale]
    # Update .device attr if it's a plain attribute (not a property)
    if hasattr(encoder, "device") and not isinstance(
        getattr(type(encoder), "device", None), property
    ):
        encoder.device = str(device)


class InfiniteTalkDataset(Dataset):
    """
    Dataset for InfiniteTalk training data with precomputed tensors.

    Returns dict with:
        real: [16, 21, H, W] -- clean video latents (bf16)
        first_frame_cond: [16, 21, H, W] -- reference frame conditioning (bf16)
        clip_features: [1, 257, 1280] -- CLIP features (bf16)
        audio_emb: [81, 12, 768] -- wav2vec2 audio embeddings (bf16)
        text_embeds: [1, 512, 4096] -- T5 text embedding (bf16)
        neg_text_embeds: [1, 512, 4096] -- negative text embedding for CFG (bf16)
        path: [num_steps, 16, 21, H, W] -- ODE trajectory (bf16, optional, for KD)
    """

    # Files that can be lazily encoded (VAE + CLIP + audio)
    _LAZY_ENCODABLE = {
        "vae_latents.pt",
        "first_frame_cond.pt",
        "motion_frame.pt",
        "clip_features.pt",
        "audio_emb.pt",
    }

    def __init__(
        self,
        data_list_path: str,
        neg_text_emb_path: str = None,
        load_ode_path: bool = False,
        num_video_frames: int = 93,
        num_latent_frames: int = 21,
        expected_latent_shape: tuple = None,
        quarter_res: bool = False,
        audio_data_root: str = None,
        # --- Lazy caching params ---
        raw_data_root: str = None,
        csv_path: str = None,
        weights_dir: str = None,
        wav2vec_dir: str = None,
        encode_device: str = "cuda:0",
    ):
        """
        Args:
            data_list_path: Path to text file listing sample directories (one per line).
            neg_text_emb_path: Path to shared negative text embedding file.
                If None, uses zeros [1, 512, 4096].
            load_ode_path: If True, load ode_path.pt for KD training.
            num_video_frames: Number of pixel frames stored on disk (for audio truncation).
                Set to 93 (24 latent frames) for extra temporal context.
            num_latent_frames: Number of latent frames to return for training.
                Precomputed data may have more (e.g. 24); this slices to the
                training length (e.g. 21). Pixel-frame equivalent = (N-1)*4+1.
            expected_latent_shape: If set, e.g. (16, 21, 56, 112), filter out samples
                whose vae_latents.pt has a different shape. This handles datasets with
                mixed aspect ratios — only samples matching the training resolution
                are kept.

                NOTE: For future multi-resolution training, replace this filter with
                aspect-ratio bucketed sampling (group by resolution so each batch
                has uniform spatial dims). The model's transformer is resolution-
                agnostic (RoPE adapts), but noise/loss shapes must match within a batch.
            raw_data_root: Root directory of raw TalkVid dataset. When set, enables
                lazy caching mode — missing .pt files are encoded on-the-fly.
            csv_path: Path to video_list.csv mapping sample names to video/audio paths.
                Required when raw_data_root is set.
            weights_dir: Path to Wan2.1-I2V-14B-480P weights (for VAE + CLIP).
                Required when raw_data_root is set.
            wav2vec_dir: Path to chinese-wav2vec2-base directory.
                Required when raw_data_root is set.
            encode_device: Device for on-the-fly encoding (default: 'cuda:0').
        """
        self.load_ode_path = load_ode_path
        self.num_video_frames = num_video_frames
        self.num_latent_frames = num_latent_frames
        self._train_pixel_frames = (num_latent_frames - 1) * 4 + 1  # e.g. 81 for 21 latent frames
        self.expected_latent_shape = tuple(expected_latent_shape) if expected_latent_shape else None
        self.quarter_res = quarter_res
        self._vae_suffix = "_quarter" if quarter_res else ""
        self.audio_data_root = audio_data_root
        self._lazy_enabled = False

        # --- Lazy caching setup ---
        if raw_data_root is not None:
            assert csv_path is not None, "csv_path required for lazy caching"
            assert weights_dir is not None, "weights_dir required for lazy caching"
            assert wav2vec_dir is not None, "wav2vec_dir required for lazy caching"

            self._lazy_enabled = True
            self._raw_data_root = raw_data_root
            self._device = encode_device

            # Set up InfiniteTalk module path and mock modules
            _add_infinitetalk_to_path()

            # Load encoders on CPU — they'll be moved to GPU on-demand during
            # encoding, then moved back. This avoids OOM since the training model
            # uses ~77GB of 80GB, but the VAE only needs ~7.3GB peak temporarily
            # (freed after each encode via empty_cache).
            logger.info("Lazy caching enabled — loading encoders to CPU (will offload to %s for encoding)", encode_device)
            self._vae = _load_vae(weights_dir, "cpu")
            self._clip = _load_clip(weights_dir, "cpu")
            self._wav2vec_fe, self._audio_encoder = _load_wav2vec(wav2vec_dir, "cpu")
            logger.info("All lazy-cache encoders loaded on CPU")

            # Build mapping: sample directory basename -> CSV row
            # The precompute script names sample dirs as:
            #   os.path.splitext(video_rel.replace("/", "_"))[0]
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self._csv_map = {}
            for row in rows:
                sample_name = os.path.splitext(
                    row["video_path"].replace("/", "_")
                )[0]
                self._csv_map[sample_name] = row

        # Read sample directories from text file
        with open(data_list_path) as f:
            all_dirs = [line.strip() for line in f if line.strip()]

        # Filter out samples missing required files
        # In lazy mode, only text_embeds.pt is strictly required (T5 cannot be lazily encoded).
        # Other missing files will be encoded on-the-fly.
        valid_dirs = []
        required_files = [
            "vae_latents.pt",
            "first_frame_cond.pt",
            "clip_features.pt",
            "audio_emb.pt",
            "text_embeds.pt",
        ]
        missing_count = 0
        lazy_eligible_count = 0
        for d in all_dirs:
            missing = [fn for fn in required_files if not os.path.exists(os.path.join(d, fn))]
            if missing:
                if self._lazy_enabled:
                    # In lazy mode: only skip if text_embeds.pt is missing
                    if "text_embeds.pt" in missing:
                        warnings.warn(f"Skipping {d}: text_embeds.pt missing (T5 must be pre-computed)")
                        missing_count += 1
                    else:
                        # Check that we have a CSV row for this sample
                        sample_name = os.path.basename(d)
                        if sample_name in self._csv_map:
                            valid_dirs.append(d)
                            lazy_eligible_count += 1
                        else:
                            warnings.warn(f"Skipping {d}: no CSV row for sample '{sample_name}'")
                            missing_count += 1
                else:
                    warnings.warn(f"Skipping {d}: missing {missing}")
                    missing_count += 1
            else:
                valid_dirs.append(d)

        if lazy_eligible_count > 0:
            logger.info(
                "[InfiniteTalkDataset] %d samples will be lazily encoded on first access",
                lazy_eligible_count,
            )

        # Filter by resolution if expected_latent_shape is set
        shape_mismatch_count = 0
        if self.expected_latent_shape is not None:
            self.dirs = []
            for d in valid_dirs:
                vae_path = os.path.join(d, f"vae_latents{self._vae_suffix}.pt")
                if not os.path.exists(vae_path):
                    # Lazy mode: can't check shape before encoding — include it
                    # and let __getitem__ handle shape mismatch after encoding
                    self.dirs.append(d)
                    continue
                # Quick shape check: load only metadata (mmap), don't load full tensor
                try:
                    lat = torch.load(vae_path, map_location="cpu", weights_only=False)
                    if isinstance(lat, dict):
                        lat = next(v for v in lat.values() if isinstance(v, torch.Tensor))
                    # Compare spatial dims only (C, H, W), ignoring temporal
                    # since we slice temporal to num_latent_frames at load time.
                    # expected: (C, T_train, H, W), stored: (C, T_stored, H, W)
                    stored_spatial = (lat.shape[0],) + tuple(lat.shape[2:])  # (C, H, W)
                    expected_spatial = (self.expected_latent_shape[0],) + tuple(self.expected_latent_shape[2:])
                    if stored_spatial == expected_spatial and lat.shape[1] >= self.num_latent_frames:
                        self.dirs.append(d)
                    else:
                        shape_mismatch_count += 1
                    del lat
                except Exception:
                    shape_mismatch_count += 1
        else:
            self.dirs = valid_dirs

        skipped = missing_count + shape_mismatch_count
        if skipped > 0:
            print(
                f"[InfiniteTalkDataset] Kept {len(self.dirs)}/{len(all_dirs)} samples "
                f"(skipped: {missing_count} missing files, {shape_mismatch_count} resolution mismatch)"
            )

        # Load negative text embedding (shared across all samples, for CFG)
        if neg_text_emb_path is not None and os.path.exists(neg_text_emb_path):
            neg_emb = torch.load(neg_text_emb_path, map_location="cpu", weights_only=False)
            if isinstance(neg_emb, dict):
                # Handle dict format — grab the first tensor value
                neg_emb = next(v for v in neg_emb.values() if isinstance(v, torch.Tensor))
            self.neg_text_embeds = neg_emb.to(torch.bfloat16)
        else:
            self.neg_text_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)

        # Ensure correct shape [1, 512, 4096]
        if self.neg_text_embeds.dim() == 2:
            self.neg_text_embeds = self.neg_text_embeds.unsqueeze(0)

    def _encode_and_cache(self, sample_dir, csv_row):
        """Encode missing modalities on-the-fly and cache to disk.

        This must match EXACTLY what scripts/precompute_infinitetalk_data.py does.
        T5 (text_embeds.pt) is NOT handled here — it must be pre-computed.
        """
        from PIL import Image

        video_rel = csv_row["video_path"]
        audio_rel = csv_row["audio_path"]
        src_h = int(csv_row.get("height", 640))
        src_w = int(csv_row.get("width", 640))

        video_path = os.path.join(self._raw_data_root, video_rel)
        audio_path = os.path.join(self._raw_data_root, audio_rel)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        os.makedirs(sample_dir, exist_ok=True)

        # Select aspect-ratio bucket
        ratio = src_h / src_w
        closest_bucket = sorted(
            ASPECT_RATIO_627.keys(), key=lambda x: abs(float(x) - ratio)
        )[0]
        target_h, target_w = ASPECT_RATIO_627[closest_bucket]

        if self.quarter_res:
            target_h = target_h // 2
            target_w = target_w // 2

        # Per-file skip: only compute what's missing
        def _exists(name):
            return os.path.isfile(os.path.join(sample_dir, name))

        need_vae = not _exists(f"vae_latents{self._vae_suffix}.pt")
        need_ffc = not _exists(f"first_frame_cond{self._vae_suffix}.pt")
        need_motion = not _exists(f"motion_frame{self._vae_suffix}.pt")
        need_clip = not _exists("clip_features.pt")
        need_audio = not _exists("audio_emb.pt")

        # In quarter_res mode, CLIP and audio are resolution-independent —
        # only skip re-encoding if they already exist on disk.
        # (Don't unconditionally skip: samples that were never precomputed still need them.)

        need_video = need_vae or need_ffc or need_motion or need_clip
        video_tensor = None
        cond_image = None

        if need_video:
            frames, fps = _load_video_frames(video_path, self.num_video_frames)
            actual_frames = frames.shape[0]
            if actual_frames < self.num_video_frames:
                logger.warning(
                    "Lazy cache: video %s has %d frames (need %d), zero-padding",
                    video_path, actual_frames, self.num_video_frames,
                )
                pad = np.zeros(
                    (self.num_video_frames - actual_frames,
                     frames.shape[1], frames.shape[2], 3),
                    dtype=frames.dtype,
                )
                frames = np.concatenate([frames, pad], axis=0)

            video_tensor = _resize_and_center_crop(frames, target_h, target_w)
            # video_tensor: [3, T, H, W] float32 in [-1, 1]

            first_frame_pil = Image.fromarray(frames[0])  # [H, W, 3] uint8 -> PIL
            cond_image = _resize_and_centercrop_pil(first_frame_pil, target_h, target_w)
            # cond_image: [1, C, 1, H, W] float32 in [-1, 1]

        # ---- Encode with GPU offloading ----
        # Encoders live on CPU. For each encode step:
        #   1. empty_cache() to free training activation memory
        #   2. Move encoder to GPU
        #   3. Encode (outputs moved to CPU immediately)
        #   4. Move encoder back to CPU
        #   5. empty_cache() to reclaim GPU memory for training
        # This allows encoding even when training uses ~77GB of 80GB,
        # since we only need temporary GPU headroom for one encoder at a time.
        device = self._device

        if need_vae or need_ffc or need_motion:
            logger.info("Lazy cache: encoding VAE for %s", os.path.basename(sample_dir))
            torch.cuda.empty_cache()
            _move_encoder_to(self._vae, device)
            latents, first_frame_cond, motion_frame = _encode_vae(
                self._vae, video_tensor, device, cond_image
            )
            _move_encoder_to(self._vae, "cpu")
            torch.cuda.empty_cache()
            if need_vae:
                _save_atomic(latents, os.path.join(sample_dir, f"vae_latents{self._vae_suffix}.pt"))
            if need_ffc:
                _save_atomic(first_frame_cond, os.path.join(sample_dir, f"first_frame_cond{self._vae_suffix}.pt"))
            if need_motion:
                _save_atomic(motion_frame, os.path.join(sample_dir, f"motion_frame{self._vae_suffix}.pt"))
            del latents, first_frame_cond, motion_frame

        if need_clip:
            logger.info("Lazy cache: encoding CLIP for %s", os.path.basename(sample_dir))
            torch.cuda.empty_cache()
            _move_encoder_to(self._clip, device)
            clip_features = _encode_clip(self._clip, device, cond_image)
            _move_encoder_to(self._clip, "cpu")
            torch.cuda.empty_cache()
            _save_atomic(clip_features, os.path.join(sample_dir, "clip_features.pt"))
            del clip_features

        if need_audio:
            logger.info("Lazy cache: encoding audio for %s", os.path.basename(sample_dir))
            torch.cuda.empty_cache()
            _move_encoder_to(self._audio_encoder, device)
            audio_array = _load_audio_array(audio_path)
            audio_emb = _encode_audio(
                self._wav2vec_fe, self._audio_encoder, audio_array,
                num_video_frames=self.num_video_frames, device=device,
            )
            _move_encoder_to(self._audio_encoder, "cpu")
            torch.cuda.empty_cache()
            _save_atomic(audio_emb, os.path.join(sample_dir, "audio_emb.pt"))
            del audio_emb, audio_array

        # ---- Future anchor latents (extra frames past clip boundary) ----
        need_future_anchor = not _exists(f"future_anchor_latents{self._vae_suffix}.pt")
        if need_future_anchor:
            # Future anchors need frames BEYOND the training clip.
            # The clip-level video file only has [0, num_video_frames) frames.
            # We need to find the full source video and read frames past the clip end.
            #
            # Strategy: parse sample_dir name to get (video_id, start, end),
            # resolve the full video, read [end-5, end+20) = 25 frames
            # (5 overlap for VAE temporal context, 20 future = 5 latent frames).
            future_anchor = self._encode_future_anchor(sample_dir, target_h, target_w)
            if future_anchor is not None:
                _save_atomic(
                    future_anchor,
                    os.path.join(sample_dir, f"future_anchor_latents{self._vae_suffix}.pt"),
                )
            del future_anchor

        # Clean up video data
        del video_tensor, cond_image

    # ------------------------------------------------------------------
    # Future anchor encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sample_dir_name(basename):
        """Parse ``data_VIDEOID_VIDEOID_START_END`` → ``(video_id, start, end)``.

        The sample directory naming convention (set in the precompute script) is::

            os.path.splitext(video_rel.replace("/", "_"))[0]

        where ``video_rel`` looks like ``data/VID_VID/VID_VID_START_END.mp4``.
        This produces ``data_VID_VID_VID_VID_START_END``.

        Returns (video_id, start, end) or None if unparseable.
        ``video_id`` is the repeated portion (e.g. ``VID_VID``).
        """
        if not basename.startswith("data_"):
            return None
        rest = basename[5:]  # strip "data_"
        # The last two underscore-separated tokens are START and END (ints).
        parts = rest.rsplit("_", 2)
        if len(parts) != 3:
            return None
        prefix, start_str, end_str = parts
        try:
            start = int(start_str)
            end = int(end_str)
        except ValueError:
            return None
        # prefix should be VID_VID (video_id repeated twice separated by _).
        # Find the split point where prefix[:i] == prefix[i+1:].
        for i in range(1, len(prefix)):
            if prefix[i] == '_' and prefix[:i] == prefix[i + 1:]:
                return (prefix[:i], start, end)
        return None

    def _resolve_full_video(self, video_id):
        """Find the full source video for *video_id* under ``raw_data_root``.

        Tries the directory layout used by the TalkVid dataset::

            raw_data_root/data/<video_id>/<video_id>.mp4

        Returns the path string, or ``None`` if not found.
        """
        if not self._raw_data_root:
            return None
        for pattern in [
            os.path.join(self._raw_data_root, "data", video_id, f"{video_id}.mp4"),
            os.path.join(self._raw_data_root, "videos", f"{video_id}.mp4"),
        ]:
            if os.path.isfile(pattern):
                return pattern
        return None

    @staticmethod
    def _read_video_range(video_path, start_frame, num_frames, target_fps=25.0):
        """Read a range of frames from a video, resampled to target_fps.

        ``start_frame`` and ``num_frames`` are in target_fps space.  If the
        source video's native FPS differs, frame indices are mapped accordingly.

        Returns ``[T, H, W, 3]`` uint8 ndarray, or ``None`` if the video is
        too short or cannot be opened.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Map target-fps frame indices to native-fps source indices
        if abs(native_fps - target_fps) < 0.5:
            src_indices = list(range(start_frame, start_frame + num_frames))
        else:
            src_indices = [
                round(i / target_fps * native_fps)
                for i in range(start_frame, start_frame + num_frames)
            ]

        # Check bounds
        if any(idx < 0 or idx >= total for idx in src_indices):
            cap.release()
            return None

        frames = []
        for idx in src_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames)

    def _encode_future_anchor(self, sample_dir, target_h, target_w):
        """Encode 5 future latent frames past the clip boundary.

        Returns ``[16, 5, H_lat, W_lat]`` bf16 tensor, or ``None`` if the
        source video doesn't have enough frames.
        """
        basename = os.path.basename(sample_dir)
        parsed = self._parse_sample_dir_name(basename)
        if parsed is None:
            logger.warning(
                "Lazy cache: cannot parse sample dir name for future anchor: %s",
                basename,
            )
            return None

        video_id, _clip_start, clip_end = parsed

        # Need the full video (not clip file) to get frames past clip_end
        full_video_path = self._resolve_full_video(video_id)
        if full_video_path is None:
            logger.warning(
                "Lazy cache: full video not found for future anchor: video_id=%s",
                video_id,
            )
            return None

        # Read 25 pixel frames: 5 overlap (VAE temporal context) + 20 future
        # Starting at clip_end - 5 in the full video.
        # 25 pixel frames → 7 latent frames via VAE.
        # Discard first 2 (overlap context), keep 5 future latent frames.
        read_start = clip_end - 5
        needed = 25

        frames = self._read_video_range(full_video_path, read_start, needed)
        if frames is None or frames.shape[0] < needed:
            logger.warning(
                "Lazy cache: not enough frames for future anchor in %s "
                "(need %d from frame %d)",
                full_video_path, needed, read_start,
            )
            return None

        # Resize + center crop (same transform as training frames)
        video_tensor = _resize_and_center_crop(frames, target_h, target_w)
        # video_tensor: [3, 25, target_h, target_w] float32

        # VAE encode — same offload pattern as main encoding block
        device = self._device
        logger.info("Lazy cache: encoding future anchor VAE for %s", basename)
        torch.cuda.empty_cache()
        _move_encoder_to(self._vae, device)

        vae_dtype = getattr(self._vae, "dtype", None) or next(self._vae.model.parameters()).dtype
        with torch.no_grad():
            latents = self._vae.encode([video_tensor.to(device=device, dtype=vae_dtype)])
            if isinstance(latents, (list, tuple)):
                latents = latents[0]
            latents = latents.to(torch.bfloat16).cpu()

        _move_encoder_to(self._vae, "cpu")
        torch.cuda.empty_cache()

        # 25 pixel frames → 7 latent frames. Discard first 2 (overlap context), keep 5.
        if latents.shape[1] < 7:
            logger.warning(
                "Lazy cache: VAE produced only %d latent frames (need 7) for %s",
                latents.shape[1], basename,
            )
            return None
        future_anchor = latents[:, 2:7]  # [16, 5, H_lat, W_lat]
        return future_anchor

    def __len__(self):
        return len(self.dirs)

    def _check_sample_shape(self, sample: dict, sample_dir: str):
        """Validate spatial dims match expected_latent_shape. Returns sample or None."""
        if self.expected_latent_shape is None or sample is None:
            return sample
        expected_spatial = (self.expected_latent_shape[0],) + tuple(self.expected_latent_shape[2:])
        real = sample["real"]
        actual_spatial = (real.shape[0],) + tuple(real.shape[2:])
        if actual_spatial != expected_spatial:
            warnings.warn(
                f"Shape mismatch after load: {os.path.basename(sample_dir)} "
                f"has {list(real.shape)}, expected spatial {list(expected_spatial)}"
            )
            return None
        return sample

    def __getitem__(self, idx) -> dict:
        sample_dir = self.dirs[idx]

        try:
            sample = self._load_sample(sample_dir)
            return self._check_sample_shape(sample, sample_dir)
        except Exception:
            pass

        # First load failed — attempt lazy encoding if enabled
        if self._lazy_enabled:
            # Check if text_embeds.pt exists (T5 must be pre-computed)
            if not os.path.exists(os.path.join(sample_dir, "text_embeds.pt")):
                warnings.warn(
                    f"Skipping {sample_dir}: text_embeds.pt missing "
                    f"(T5 must be pre-computed, cannot lazy-encode)"
                )
                return None

            # Look up CSV row for this sample
            sample_name = os.path.basename(sample_dir)
            csv_row = self._csv_map.get(sample_name)
            if csv_row is None:
                warnings.warn(
                    f"Skipping {sample_dir}: no CSV row for '{sample_name}'"
                )
                return None

            try:
                self._encode_and_cache(sample_dir, csv_row)
                sample = self._load_sample(sample_dir)
                return self._check_sample_shape(sample, sample_dir)
            except Exception as e:
                warnings.warn(f"Lazy encoding failed for {sample_dir}: {e}")
                return None
        else:
            return None

    def _load_sample(self, sample_dir: str) -> dict:
        """Load all .pt files from a sample directory.

        Raises on any missing file or load error (caller handles fallback).
        """
        # --- VAE latents (clean video) ---
        real = torch.load(
            os.path.join(sample_dir, f"vae_latents{self._vae_suffix}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(real, dict):
            real = next(v for v in real.values() if isinstance(v, torch.Tensor))
        real = real.to(torch.bfloat16)
        # Slice to training length (precomputed data may have more latent frames)
        real = real[:, :self.num_latent_frames]  # [16, num_latent_frames, H, W]

        # --- First frame conditioning ---
        first_frame_cond = torch.load(
            os.path.join(sample_dir, f"first_frame_cond{self._vae_suffix}.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(first_frame_cond, dict):
            first_frame_cond = next(
                v for v in first_frame_cond.values() if isinstance(v, torch.Tensor)
            )
        first_frame_cond = first_frame_cond.to(torch.bfloat16)
        assert first_frame_cond.shape[0] == 16, (
            f"Expected first_frame_cond with 16 channels, got {first_frame_cond.shape[0]}. "
            f"Sample: {sample_dir}"
        )
        # Slice to training length (must match real's temporal dim)
        first_frame_cond = first_frame_cond[:, :self.num_latent_frames]

        # --- CLIP features ---
        clip_features = torch.load(
            os.path.join(sample_dir, "clip_features.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(clip_features, dict):
            clip_features = next(
                v for v in clip_features.values() if isinstance(v, torch.Tensor)
            )
        clip_features = clip_features.to(torch.bfloat16)  # [1, 257, 1280]
        assert clip_features.shape[-2:] == (257, 1280), (
            f"Expected clip_features [..., 257, 1280], got {list(clip_features.shape)}. "
            f"Sample: {sample_dir}"
        )

        # --- Audio embeddings ---
        audio_emb = torch.load(
            os.path.join(sample_dir, "audio_emb.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(audio_emb, dict):
            audio_emb = next(
                v for v in audio_emb.values() if isinstance(v, torch.Tensor)
            )
        # Validate and slice to training pixel frames
        assert audio_emb.dim() == 3 and audio_emb.shape[1:] == (12, 768), (
            f"Expected audio_emb shape [T, 12, 768], got {list(audio_emb.shape)}. "
            f"Sample: {sample_dir}"
        )
        assert audio_emb.shape[0] >= self._train_pixel_frames, (
            f"audio_emb has {audio_emb.shape[0]} frames, need >= {self._train_pixel_frames}. "
            f"Sample: {sample_dir}"
        )
        audio_emb = audio_emb[:self._train_pixel_frames]  # [train_pixel_frames, 12, 768]
        audio_emb = audio_emb.to(torch.bfloat16)

        # --- Text embeddings ---
        text_embeds = torch.load(
            os.path.join(sample_dir, "text_embeds.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(text_embeds, dict):
            text_embeds = next(
                v for v in text_embeds.values() if isinstance(v, torch.Tensor)
            )
        text_embeds = text_embeds.to(torch.bfloat16)
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)  # [1, 512, 4096]

        # --- Future anchor latents (optional, for DF lookahead anchoring) ---
        future_anchor_path = os.path.join(
            sample_dir, f"future_anchor_latents{self._vae_suffix}.pt"
        )
        has_future_anchor = os.path.exists(future_anchor_path)
        if has_future_anchor:
            future_anchor_latents = torch.load(
                future_anchor_path, map_location="cpu", weights_only=False,
            )
            if isinstance(future_anchor_latents, dict):
                future_anchor_latents = next(
                    v for v in future_anchor_latents.values()
                    if isinstance(v, torch.Tensor)
                )
            future_anchor_latents = future_anchor_latents.to(torch.bfloat16)
        else:
            # Zero fallback so default_collate works across mixed batches
            H_lat = real.shape[2]
            W_lat = real.shape[3]
            future_anchor_latents = torch.zeros(
                16, 5, H_lat, W_lat, dtype=torch.bfloat16
            )

        result = {
            "real": real,
            "first_frame_cond": first_frame_cond,
            "clip_features": clip_features,
            "audio_emb": audio_emb,
            "text_embeds": text_embeds,
            "neg_text_embeds": self.neg_text_embeds.clone(),
            "future_anchor_latents": future_anchor_latents,
            "has_future_anchor": has_future_anchor,
        }

        # --- Source audio path (optional, for wandb video muxing) ---
        # Always include the key (empty string if not found) so that
        # default_collate doesn't crash on inconsistent keys at BS>1.
        audio_wav_path = ""
        if self.audio_data_root:
            basename = os.path.basename(sample_dir)
            if basename.startswith("data_"):
                basename = basename[5:]
            parts = basename.split("_")
            # Format: VIDEOID_VIDEOID_START_END → audio at audio_root/VIDEOID/VIDEOID_START_END.wav
            if len(parts) >= 4 and parts[0] == parts[1]:
                video_id = parts[0]
                clip_name = f"{video_id}_{'_'.join(parts[2:])}"
                wav_path = os.path.join(self.audio_data_root, video_id, f"{clip_name}.wav")
                if os.path.exists(wav_path):
                    audio_wav_path = wav_path
        result["audio_path"] = audio_wav_path

        # --- ODE trajectory (optional, for KD training) ---
        if self.load_ode_path:
            ode_path_file = os.path.join(sample_dir, "ode_path.pt")
            if os.path.exists(ode_path_file):
                result["path"] = torch.load(
                    ode_path_file, map_location="cpu", weights_only=False
                ).to(torch.bfloat16)
            else:
                # Also check path.pth (alternative filename)
                alt_path_file = os.path.join(sample_dir, "path.pth")
                if os.path.exists(alt_path_file):
                    result["path"] = torch.load(
                        alt_path_file, map_location="cpu", weights_only=False
                    ).to(torch.bfloat16)

        return result


class InfiniteTalkDataLoader:
    """Infinite-iterator DataLoader wrapper with DistributedSampler support.

    FastGen's trainer expects an infinite iterator. This class wraps a standard
    DataLoader and yields batches indefinitely, cycling through the dataset.
    """

    def __init__(
        self,
        data_list_path: str = None,
        datatags: list = None,
        batch_size: int = 1,
        num_workers: int = 4,
        load_ode_path: bool = False,
        quarter_res: bool = False,
        **kwargs,
    ):
        # Support both data_list_path and datatags (list of paths)
        if data_list_path is None and datatags is not None:
            data_list_path = datatags[0] if isinstance(datatags, list) else datatags
        assert data_list_path is not None, "Must provide data_list_path or datatags"

        # Enforce num_workers=0 when lazy caching is enabled
        if kwargs.get("raw_data_root"):
            if num_workers > 0:
                warnings.warn(
                    "Lazy caching requires num_workers=0 (GPU encoders "
                    "cannot cross process boundaries). Overriding."
                )
                num_workers = 0

        self.dataset = InfiniteTalkDataset(
            data_list_path=data_list_path,
            load_ode_path=load_ode_path,
            quarter_res=quarter_res,
            **kwargs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Use DistributedSampler for multi-GPU training
        if dist.is_initialized():
            self._sampler = DistributedSampler(self.dataset, shuffle=True)
            shuffle = False
        else:
            self._sampler = None
            shuffle = True

        def collate_fn(batch):
            """Filter out None samples from failed loads."""
            valid = [s for s in batch if s is not None]
            if not valid:
                return {}
            return torch.utils.data.default_collate(valid)

        self._dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self._sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def __iter__(self):
        """Infinite iterator -- cycles through the dataset, skipping empty batches."""
        epoch = 0
        while True:
            if self._sampler is not None:
                self._sampler.set_epoch(epoch)
            for batch in self._dataloader:
                if batch and "real" in batch:
                    yield batch
                else:
                    logger.warning("Skipping empty batch (all samples in batch failed to load)")
            epoch += 1

    def __len__(self):
        return len(self.dataset)


# Keep the simple function for backward compatibility
def create_infinitetalk_dataloader(
    data_list_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    load_ode_path: bool = False,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for InfiniteTalk training data (non-infinite, no DDP support).

    For training, prefer InfiniteTalkDataLoader which provides infinite iteration
    and DistributedSampler support.
    """
    dataset = InfiniteTalkDataset(
        data_list_path=data_list_path,
        load_ode_path=load_ode_path,
        **kwargs,
    )

    def collate_fn(batch):
        """Filter out None samples from failed loads."""
        valid = [s for s in batch if s is not None]
        if not valid:
            return {}
        return torch.utils.data.default_collate(valid)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
