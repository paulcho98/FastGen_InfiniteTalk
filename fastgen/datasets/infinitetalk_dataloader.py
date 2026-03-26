# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch Dataset for InfiniteTalk precomputed training data.

Each sample directory contains precomputed .pt files:
    - vae_latents.pt: [16, 21, H, W] — VAE-encoded 81-frame video
    - first_frame_cond.pt: [16, 21, H, W] — VAE-encoded reference frame + zero padding
    - clip_features.pt: [1, 257, 1280] — CLIP ViT-H/14 on reference frame
    - audio_emb.pt: [81, 12, 768] — wav2vec2 hidden states (all 12 layers)
    - text_embeds.pt: [1, 512, 4096] — T5 UMT5-XXL text embeddings
    - neg_text_embeds.pt: [1, 512, 4096] — negative text embedding (shared across samples)
    - ode_path.pt: [num_steps, 16, 21, H, W] — ODE trajectory (KD only, optional)

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
    """Load CLIP ViT-H/14 visual encoder (uses flash_attn natively)."""
    from wan.modules.clip import CLIPModel

    clip_ckpt = os.path.join(
        weights_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    clip_tok = os.path.join(weights_dir, "xlm-roberta-large")
    logger.info("Lazy cache: loading CLIP from %s", clip_ckpt)
    clip_model = CLIPModel(
        dtype=torch.float16,
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


def _load_video_frames(video_path: str, frame_count: int = 81):
    """Load frames from video using av (PyAV).

    Returns:
        frames: np.ndarray [T, H, W, 3] uint8
        fps: float
    """
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
    video = video_tensor.to(device)

    # Full video latents
    latents = vae.encode([video])
    latents = latents[0].cpu()

    # First-frame condition: first frame + zero padding
    c, t, h, w = video.shape
    cond_image_dev = cond_image.to(device)  # [1, C, 1, H, W]
    zero_padding = torch.zeros(
        1, cond_image_dev.shape[1], t - 1, h, w, device=device
    )
    first_frame_padded = torch.cat([cond_image_dev, zero_padding], dim=2)  # [1, C, t, H, W]
    first_frame_cond = vae.encode(first_frame_padded)
    first_frame_cond = torch.stack(first_frame_cond).to(torch.bfloat16)[0].cpu()

    # Motion frame: single reference frame (no temporal context)
    motion_frame = vae.encode(cond_image_dev)
    motion_frame = torch.stack(motion_frame).to(torch.bfloat16)[0].cpu()
    # motion_frame: [16, 1, H_lat, W_lat]

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
    first_frame = cond_image[:, :, -1:, :, :].to(device)  # [1, C, 1, H, W]
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
        num_video_frames: int = 81,
        expected_latent_shape: tuple = None,
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
            num_video_frames: Number of video frames (for audio truncation).
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
        self.expected_latent_shape = tuple(expected_latent_shape) if expected_latent_shape else None
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

            # Load encoders (once, kept for the lifetime of the dataset)
            logger.info("Lazy caching enabled — loading encoders to %s", encode_device)
            self._vae = _load_vae(weights_dir, encode_device)
            self._clip = _load_clip(weights_dir, encode_device)
            self._wav2vec_fe, self._audio_encoder = _load_wav2vec(wav2vec_dir, encode_device)
            logger.info("All lazy-cache encoders loaded")

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
                vae_path = os.path.join(d, "vae_latents.pt")
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
                    if tuple(lat.shape) == self.expected_latent_shape:
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

        # Per-file skip: only compute what's missing
        def _exists(name):
            return os.path.isfile(os.path.join(sample_dir, name))

        need_vae = not _exists("vae_latents.pt")
        need_ffc = not _exists("first_frame_cond.pt")
        need_motion = not _exists("motion_frame.pt")
        need_clip = not _exists("clip_features.pt")
        need_audio = not _exists("audio_emb.pt")

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

        # ---- VAE encode ----
        if need_vae or need_ffc or need_motion:
            logger.info("Lazy cache: encoding VAE for %s", os.path.basename(sample_dir))
            latents, first_frame_cond, motion_frame = _encode_vae(
                self._vae, video_tensor, self._device, cond_image
            )
            if need_vae:
                _save_atomic(latents, os.path.join(sample_dir, "vae_latents.pt"))
            if need_ffc:
                _save_atomic(first_frame_cond, os.path.join(sample_dir, "first_frame_cond.pt"))
            if need_motion:
                _save_atomic(motion_frame, os.path.join(sample_dir, "motion_frame.pt"))
            del latents, first_frame_cond, motion_frame
            torch.cuda.empty_cache()

        # ---- CLIP encode ----
        if need_clip:
            logger.info("Lazy cache: encoding CLIP for %s", os.path.basename(sample_dir))
            clip_features = _encode_clip(self._clip, self._device, cond_image)
            _save_atomic(clip_features, os.path.join(sample_dir, "clip_features.pt"))
            del clip_features
            torch.cuda.empty_cache()

        # ---- Audio encode ----
        if need_audio:
            logger.info("Lazy cache: encoding audio for %s", os.path.basename(sample_dir))
            audio_array = _load_audio_array(audio_path)
            audio_emb = _encode_audio(
                self._wav2vec_fe, self._audio_encoder, audio_array,
                num_video_frames=self.num_video_frames, device=self._device,
            )
            _save_atomic(audio_emb, os.path.join(sample_dir, "audio_emb.pt"))
            del audio_emb, audio_array
            torch.cuda.empty_cache()

        # Clean up video data
        del video_tensor, cond_image

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx) -> dict:
        sample_dir = self.dirs[idx]

        try:
            return self._load_sample(sample_dir)
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
                return self._load_sample(sample_dir)
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
            os.path.join(sample_dir, "vae_latents.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(real, dict):
            real = next(v for v in real.values() if isinstance(v, torch.Tensor))
        real = real.to(torch.bfloat16)  # [16, 21, H, W]

        # --- First frame conditioning ---
        first_frame_cond = torch.load(
            os.path.join(sample_dir, "first_frame_cond.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(first_frame_cond, dict):
            first_frame_cond = next(
                v for v in first_frame_cond.values() if isinstance(v, torch.Tensor)
            )
        first_frame_cond = first_frame_cond.to(torch.bfloat16)  # [16, 21, H, W]

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
        # Truncate to num_video_frames if longer
        audio_emb = audio_emb[: self.num_video_frames]  # [81, 12, 768]
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

        result = {
            "real": real,
            "first_frame_cond": first_frame_cond,
            "clip_features": clip_features,
            "audio_emb": audio_emb,
            "text_embeds": text_embeds,
            "neg_text_embeds": self.neg_text_embeds.clone(),
        }

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
        """Infinite iterator -- cycles through the dataset."""
        epoch = 0
        while True:
            if self._sampler is not None:
                self._sampler.set_epoch(epoch)
            yield from self._dataloader
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
