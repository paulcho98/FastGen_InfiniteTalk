# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate ODE trajectory pairs from InfiniteTalk teacher for KD pre-training (Stage 1).

Runs a multi-step deterministic ODE solve with 3-call classifier-free guidance
(separate text and audio guidance) using the InfiniteTalk bidirectional teacher
(14B Wan 2.1 I2V), then saves subsampled noisy states as ``ode_path.pt`` in each
sample directory.

Input:  InfiniteTalk precomputed .pt files
        (vae_latents, first_frame_cond, clip_features, audio_emb, text_embeds)
Output: Per-sample ``ode_path.pt`` with shape [4, 16, 21, H, W] (4 noisy states, bf16)

The clean target is already available as ``vae_latents.pt`` in the sample directory.

3-call CFG:
    InfiniteTalk uses separate text and audio guidance scales.
    x0 = uncond + text_scale * (cond - drop_text) + audio_scale * (drop_text - uncond)
    where:
      - cond:      full conditioning (text + audio)
      - drop_text: negative text + audio (drops text guidance)
      - uncond:    negative text + zero audio (drops both)

Audio windowing:
    Raw precomputed audio is [81, 12, 768] (per-frame wav2vec2 hidden states).
    Before passing to the teacher, we apply a 5-frame sliding window to produce
    [81, 5, 12, 768] per sample, matching InfiniteTalk's multitalk.py lines 500-531.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_infinitetalk_ode_pairs.py \\
        --data_list_path /path/to/sample_list.txt \\
        --neg_text_emb_path /path/to/neg_text_embeds.pt \\
        --weights_dir /path/to/Wan2.1-I2V-14B-480P/ \\
        --infinitetalk_ckpt /path/to/infinitetalk.safetensors \\
        --num_steps 40 --text_guide_scale 5.0 --audio_guide_scale 4.0

Usage (distributed, 8 GPUs):
    torchrun --nproc_per_node=8 scripts/generate_infinitetalk_ode_pairs.py \\
        --data_list_path /path/to/sample_list.txt \\
        --neg_text_emb_path /path/to/neg_text_embeds.pt \\
        --weights_dir /path/to/Wan2.1-I2V-14B-480P/ \\
        --infinitetalk_ckpt /path/to/infinitetalk.safetensors
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastgen.networks.InfiniteTalk.network import InfiniteTalkWan
import fastgen.utils.logging_utils as logger


# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ODE trajectory pairs from InfiniteTalk teacher for KD"
    )

    # Model weights
    parser.add_argument(
        "--weights_dir", type=str, required=True,
        help="Directory containing Wan2.1-I2V-14B-480P safetensor shards "
             "(diffusion_pytorch_model-*.safetensors)",
    )
    parser.add_argument(
        "--infinitetalk_ckpt", type=str, default="",
        help="Path to infinitetalk.safetensors (audio modules + weight overrides)",
    )
    parser.add_argument(
        "--lora_ckpt", type=str, default="",
        help="Path to external LoRA checkpoint to merge into base weights",
    )

    # Data
    parser.add_argument(
        "--data_list_path", type=str, required=True,
        help="Path to text file with one sample directory per line",
    )
    parser.add_argument(
        "--neg_text_emb_path", type=str, default=None,
        help="Path to precomputed negative text embedding .pt "
             "(shared across samples, or per-sample fallback)",
    )
    parser.add_argument(
        "--num_video_frames", type=int, default=81,
        help="Number of video frames (for audio truncation)",
    )

    # ODE parameters
    parser.add_argument(
        "--num_steps", type=int, default=40,
        help="Number of ODE solver steps",
    )
    parser.add_argument(
        "--shift", type=float, default=7.0,
        help="Timestep shift for RF noise schedule "
             "(InfiniteTalk uses 7.0 for 480p)",
    )
    parser.add_argument(
        "--text_guide_scale", type=float, default=5.0,
        help="Text classifier-free guidance scale",
    )
    parser.add_argument(
        "--audio_guide_scale", type=float, default=4.0,
        help="Audio classifier-free guidance scale",
    )
    parser.add_argument(
        "--t_list", type=float, nargs="+",
        default=[0.999, 0.937, 0.833, 0.624, 0.0],
        help="Target noise levels for trajectory subsampling "
             "(last value should be 0.0 for clean state)",
    )

    # Output
    parser.add_argument(
        "--output_key", type=str, default="ode_path.pt",
        help="Filename for saved ODE path tensors in each sample dir",
    )

    # Processing
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size per GPU (1 recommended for 80GB, 2 may work)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of samples (for testing)",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index for partial processing",
    )
    parser.add_argument(
        "--end_idx", type=int, default=None,
        help="End index for partial processing",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=False,
        help="Skip samples that already have ode_path.pt",
    )

    # Seed
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for reproducibility "
             "(per-sample seed = base_seed + sample_index)",
    )

    # Audio
    parser.add_argument(
        "--audio_window", type=int, default=5,
        help="Audio context window size for sliding window",
    )

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load_tensor(path: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Load a .pt file, handling dict-wrapped tensors."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, torch.Tensor))
    return data.to(dtype)


def build_audio_windows(
    audio_emb: torch.Tensor,
    audio_window: int = 5,
) -> torch.Tensor:
    """Apply sliding window to raw per-frame audio embeddings.

    Reproduces InfiniteTalk's multitalk.py lines 500-531:
        indices = (torch.arange(2 * 2 + 1) - 2) * 1  # [-2, -1, 0, 1, 2]
        center_indices = frame_indices.unsqueeze(1) + indices.unsqueeze(0)
        center_indices = clamp(center_indices, 0, num_frames-1)
        audio_windowed = audio_emb[center_indices]

    Args:
        audio_emb: [num_frames, 12, 768] raw wav2vec2 hidden states.
        audio_window: Window size (default 5).

    Returns:
        [num_frames, audio_window, 12, 768] windowed audio.
    """
    num_frames = audio_emb.shape[0]
    half_win = audio_window // 2

    # Build offset indices: [-2, -1, 0, 1, 2] for window=5
    offsets = torch.arange(audio_window) - half_win  # [audio_window]

    # Center indices for each frame
    frame_indices = torch.arange(num_frames).unsqueeze(1)  # [num_frames, 1]
    window_indices = frame_indices + offsets.unsqueeze(0)   # [num_frames, audio_window]

    # Clamp to valid range (boundary padding by clamping)
    window_indices = window_indices.clamp(0, num_frames - 1)

    # Gather: [num_frames, audio_window, 12, 768]
    return audio_emb[window_indices]


def load_sample(
    sample_dir: str,
    neg_text_embeds: torch.Tensor,
    num_video_frames: int = 81,
    audio_window: int = 5,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Optional[Dict[str, Any]]:
    """Load a single sample's precomputed tensors and build condition dicts.

    Returns:
        Dict with keys:
            vae_latents: [16, 21, H, W] clean latents
            condition: full condition dict for CFG call 1
            neg_text_condition: drop-text condition for CFG call 2
            neg_condition: fully unconditional for CFG call 3
        or None if loading fails.
    """
    try:
        # --- VAE latents (clean video) ---
        vae_latents = _load_tensor(
            os.path.join(sample_dir, "vae_latents.pt"), dtype
        )  # [16, 21, H, W]

        # --- First frame conditioning ---
        first_frame_cond = _load_tensor(
            os.path.join(sample_dir, "first_frame_cond.pt"), dtype
        )  # [16, 21, H, W]

        # --- Motion frame (single-frame VAE encode for anchoring) ---
        motion_frame_path = os.path.join(sample_dir, "motion_frame.pt")
        motion_frame = None
        if os.path.isfile(motion_frame_path):
            motion_frame = _load_tensor(motion_frame_path, dtype)  # [16, 1, H, W]

        # --- CLIP features ---
        clip_features = _load_tensor(
            os.path.join(sample_dir, "clip_features.pt"), dtype
        )  # [1, 257, 1280]

        # --- Audio embeddings ---
        audio_emb = _load_tensor(
            os.path.join(sample_dir, "audio_emb.pt"), dtype
        )
        # Truncate/pad to num_video_frames
        if audio_emb.shape[0] > num_video_frames:
            audio_emb = audio_emb[:num_video_frames]
        elif audio_emb.shape[0] < num_video_frames:
            pad = torch.zeros(
                num_video_frames - audio_emb.shape[0], *audio_emb.shape[1:],
                dtype=dtype,
            )
            audio_emb = torch.cat([audio_emb, pad], dim=0)
        # audio_emb: [81, 12, 768]

        # Apply sliding window: [81, 12, 768] -> [81, 5, 12, 768]
        audio_emb = build_audio_windows(audio_emb, audio_window)

        # --- Text embeddings ---
        text_embeds = _load_tensor(
            os.path.join(sample_dir, "text_embeds.pt"), dtype
        )
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0)  # [1, 512, 4096]

        # --- Build condition dicts (add batch dim) ---
        # Full condition (CFG call 1: text + audio)
        condition = {
            "text_embeds": text_embeds.unsqueeze(0).to(device),       # [1, 1, 512, 4096]
            "first_frame_cond": first_frame_cond.unsqueeze(0).to(device),  # [1, 16, 21, H, W]
            "clip_features": clip_features.to(device),                # [1, 257, 1280]
            "audio_emb": audio_emb.unsqueeze(0).to(device),           # [1, 81, 5, 12, 768]
        }
        if motion_frame is not None:
            condition["motion_frame"] = motion_frame.unsqueeze(0).to(device)  # [1, 16, 1, H, W]

        # Negative text embedding for CFG
        neg_text = neg_text_embeds.to(device=device, dtype=dtype)
        if neg_text.dim() == 2:
            neg_text = neg_text.unsqueeze(0)  # [1, 512, 4096]

        # Drop-text condition (CFG call 2: neg text + audio)
        neg_text_condition = {
            "text_embeds": neg_text.unsqueeze(0),                     # [1, 1, 512, 4096]
            "first_frame_cond": condition["first_frame_cond"],        # same ref frame
            "clip_features": condition["clip_features"],              # same CLIP
            "audio_emb": condition["audio_emb"],                      # same audio
        }

        # Fully unconditional (CFG call 3: neg text + zero audio)
        neg_condition = {
            "text_embeds": neg_text.unsqueeze(0),                     # [1, 1, 512, 4096]
            "first_frame_cond": condition["first_frame_cond"],        # same ref frame
            "clip_features": condition["clip_features"],              # same CLIP
            "audio_emb": torch.zeros_like(condition["audio_emb"]),    # zero audio
        }

        return {
            "vae_latents": vae_latents.to(device),  # [16, 21, H, W]
            "condition": condition,
            "neg_text_condition": neg_text_condition,
            "neg_condition": neg_condition,
        }

    except Exception as e:
        logger.warning(f"Failed to load sample {sample_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Timestep Schedule
# ─────────────────────────────────────────────────────────────────────────────

def get_shifted_timesteps(
    num_steps: int,
    shift: float = 7.0,
    num_timesteps: int = 1000,
    device: torch.device = None,
) -> torch.Tensor:
    """Compute the shifted timestep schedule matching InfiniteTalk's flow matching.

    Produces ``num_steps + 1`` timesteps from T_max down to 0, with shift transform
    applied.  The result is in the [0, num_timesteps] range; divide by
    ``num_timesteps`` to get FastGen's [0, 1) range.

    Args:
        num_steps: Number of ODE solver steps.
        shift: Timestep shift parameter (InfiniteTalk uses 7.0 for 480p).
        num_timesteps: Total number of discrete timesteps (1000).
        device: Target device.

    Returns:
        [num_steps + 1] tensor of timesteps in [0, num_timesteps] range (descending).
    """
    timesteps = torch.linspace(
        num_timesteps, 0, num_steps + 1, device=device, dtype=torch.float64,
    )
    # Apply shift transform: t_shifted = shift * t / (1 + (shift - 1) * t / T)
    timesteps = shift * timesteps / (1 + (shift - 1) * timesteps / num_timesteps)
    return timesteps


# ─────────────────────────────────────────────────────────────────────────────
# ODE Trajectory Extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_subsample_indices(
    ode_t_list: torch.Tensor,
    target_t_values: List[float],
) -> List[int]:
    """Map target t-values to the closest ODE trajectory step indices.

    Args:
        ode_t_list: [num_steps+1] noise levels in [0, 1) range (descending).
        target_t_values: Target noise levels (e.g., [0.999, 0.937, 0.833, 0.624, 0.0]).

    Returns:
        List of trajectory indices.  For t=0.0, returns -1 (final clean state).
    """
    target_non_zero = [t for t in target_t_values if t > 0]

    indices = []
    for t_target in target_non_zero:
        diffs = (ode_t_list - t_target).abs()
        best_idx = diffs.argmin().item()
        indices.append(best_idx)

    # t=0.0 maps to the last trajectory state (clean/near-clean)
    if any(t == 0.0 for t in target_t_values):
        indices.append(-1)

    return indices


@torch.no_grad()
def extract_ode_trajectory(
    teacher: InfiniteTalkWan,
    noise_scheduler,
    latent_shape: Tuple[int, ...],
    condition: Dict[str, torch.Tensor],
    neg_text_condition: Dict[str, torch.Tensor],
    neg_condition: Dict[str, torch.Tensor],
    num_steps: int = 40,
    shift: float = 7.0,
    text_guide_scale: float = 5.0,
    audio_guide_scale: float = 4.0,
    target_t_list: List[float] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run teacher ODE solve with 3-call CFG and return subsampled trajectory.

    Performs a deterministic Euler ODE solve using the InfiniteTalk bidirectional
    teacher, collecting latent states at each step.  The trajectory is then
    subsampled at the noise levels specified by ``target_t_list``.

    The 3-call CFG formula:
        x0 = uncond + text_scale * (cond - drop_text)
                     + audio_scale * (drop_text - uncond)

    Args:
        teacher: InfiniteTalkWan bidirectional teacher (frozen 14B).
        noise_scheduler: RF noise scheduler from teacher.
        latent_shape: [C, T, H, W] shape of the latent (e.g. [16, 21, 80, 80]).
        condition: Full condition dict (text + audio).
        neg_text_condition: Drop-text condition dict (neg text + audio).
        neg_condition: Fully unconditional dict (neg text + zero audio).
        num_steps: Number of ODE steps.
        shift: Timestep shift for the schedule.
        text_guide_scale: Text CFG scale.
        audio_guide_scale: Audio CFG scale.
        target_t_list: Target noise levels for subsampling.
        device: CUDA device.
        dtype: Compute dtype.

    Returns:
        Tensor [num_subsample, C, T, H, W] -- trajectory states from noisiest
        to clean.  The clean state (t=0) is the last element.
    """
    if target_t_list is None:
        target_t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    num_timesteps = 1000

    # Compute shifted timestep schedule: [num_steps+1] in [0, 1000] range
    timesteps = get_shifted_timesteps(
        num_steps, shift=shift, num_timesteps=num_timesteps, device=device,
    )

    # Convert to [0, 1) range for FastGen convention
    t_schedule = timesteps / num_timesteps  # [num_steps+1] in [0, 1) descending

    # Compute subsample indices from the schedule
    subsample_indices = compute_subsample_indices(t_schedule, target_t_list)

    # Early stopping: only run ODE steps up to the last needed subsample index.
    # The clean state (t=0.0, index -1) uses GT data["real"], not the ODE solve,
    # so we don't need to run the full solve.
    max_needed_step = max(
        idx for idx in subsample_indices if idx >= 0
    )
    actual_num_steps = min(num_steps, max_needed_step + 1)

    # Initialize from pure noise (matching InfiniteTalk's original initialization)
    # CRITICAL: Original uses float32 for latent (line 548 of multitalk.py)
    # to avoid bf16 precision loss over 40 accumulated Euler steps.
    noise = torch.randn(1, *latent_shape, device=device, dtype=torch.float32)
    x_t = noise

    # Motion frame anchoring (matching multitalk.py lines 574-577, 711, 773):
    # The original forces the first latent frame to stay equal to the
    # VAE-encoded reference image at EVERY step. This is essential for I2V —
    # without it, the reference frame drifts and the output becomes blurry.
    #
    # CRITICAL: The motion anchor must be from vae.encode(single_ref_image),
    # NOT from first_frame_cond[:, :, :1]. The VAE's temporal convolutions
    # produce DIFFERENT values when encoding a single frame vs the first frame
    # of a 81-frame sequence (even if the rest is zero-padded).
    #
    # motion_frame.pt is precomputed separately as vae.encode(cond_image)[0].
    # Fallback: use first_frame_cond[:, :, :1] (less accurate but functional).
    if "motion_frame" in condition:
        motion_frame = condition["motion_frame"].to(torch.float32)  # [B, 16, 1, H, W]
    else:
        # Fallback: slice from first_frame_cond (not ideal — temporal conv bleed)
        first_frame_cond = condition["first_frame_cond"]
        motion_frame = first_frame_cond[:, :, :1, :, :].to(torch.float32)

    # Collect trajectory states (including initial noisy state)
    trajectory = [x_t.clone()]

    for step_idx in range(actual_num_steps):
        t_cur = t_schedule[step_idx]
        t_next = t_schedule[step_idx + 1]

        # Anchor first frame BEFORE model forward (multitalk.py line 711)
        x_t[:, :, :1, :, :] = motion_frame

        # Broadcast timestep to batch dim [B=1]
        t_tensor = torch.full((1,), t_cur.item(), device=device, dtype=dtype)

        # Cast to model dtype for forward, keep latent in float32
        x_t_model = x_t.to(dtype)

        # --- 3-call CFG on raw flow velocity ---
        v_cond = teacher(x_t_model, t_tensor, condition=condition)
        v_drop_text = teacher(x_t_model, t_tensor, condition=neg_text_condition)
        v_uncond = teacher(x_t_model, t_tensor, condition=neg_condition)

        # Combine with 3-call CFG formula (on velocity, same as original)
        v_guided = (
            v_uncond
            + text_guide_scale * (v_cond - v_drop_text)
            + audio_guide_scale * (v_drop_text - v_uncond)
        )

        # --- Euler step (matching InfiniteTalk multitalk.py lines 758-763) ---
        v_guided = -v_guided.float()  # negate + cast to float32 for accumulation

        dt = (t_cur - t_next)  # in [0, 1) range
        x_t = x_t + v_guided * dt

        # Anchor first frame AFTER Euler step (multitalk.py line 773)
        x_t[:, :, :1, :, :] = motion_frame

        trajectory.append(x_t.clone())

    # Stack trajectory: [1, actual_num_steps+1, C, T, H, W]
    trajectory = torch.stack(trajectory, dim=1)

    # Subsample: only keep non-negative indices (skip t=0.0 which uses GT)
    valid_indices = [idx for idx in subsample_indices if idx >= 0]
    subsampled = trajectory[:, valid_indices]

    return subsampled.squeeze(0)  # [num_subsample, C, T, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Model Construction
# ─────────────────────────────────────────────────────────────────────────────

def resolve_base_model_paths(weights_dir: str) -> str:
    """Find Wan 2.1 I2V safetensor shards in a directory.

    Looks for ``diffusion_pytorch_model*.safetensors`` files and returns them
    as a comma-separated string (the format expected by InfiniteTalkWan).

    Args:
        weights_dir: Directory containing the safetensor shard files.

    Returns:
        Comma-separated string of shard paths, sorted alphabetically.

    Raises:
        FileNotFoundError: If no safetensor files are found.
    """
    patterns = [
        os.path.join(weights_dir, "diffusion_pytorch_model*.safetensors"),
        os.path.join(weights_dir, "*.safetensors"),
    ]

    shard_paths = []
    for pattern in patterns:
        shard_paths = sorted(glob.glob(pattern))
        if shard_paths:
            break

    if not shard_paths:
        raise FileNotFoundError(
            f"No .safetensors files found in {weights_dir}. "
            f"Expected Wan2.1-I2V-14B-480P shards."
        )

    return ",".join(shard_paths)


def build_teacher(
    weights_dir: str,
    infinitetalk_ckpt: str = "",
    lora_ckpt: str = "",
    shift: float = 7.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> InfiniteTalkWan:
    """Build and load the frozen InfiniteTalk 14B teacher model.

    Args:
        weights_dir: Directory with Wan 2.1 I2V-14B safetensor shards.
        infinitetalk_ckpt: Path to InfiniteTalk checkpoint.
        lora_ckpt: Path to external LoRA checkpoint to merge.
        shift: Timestep shift for the RF noise schedule.
        device: Target CUDA device.
        dtype: Model dtype (bf16).

    Returns:
        Frozen InfiniteTalkWan teacher model on the specified device.
    """
    base_model_paths = resolve_base_model_paths(weights_dir)

    logger.info(
        f"Building InfiniteTalk teacher: "
        f"{len(base_model_paths.split(','))} base shards, "
        f"infinitetalk_ckpt={'set' if infinitetalk_ckpt else 'none'}, "
        f"lora_ckpt={'set' if lora_ckpt else 'none'}, "
        f"shift={shift}"
    )

    teacher = InfiniteTalkWan(
        base_model_paths=base_model_paths,
        infinitetalk_ckpt_path=infinitetalk_ckpt,
        lora_ckpt_path=lora_ckpt,
        apply_lora_adapters=False,  # Teacher is frozen, no runtime LoRA
        net_pred_type="flow",
        schedule_type="rf",
        shift=shift,
        disable_grad_ckpt=True,  # No need for gradient checkpointing in inference
    ).to(device, dtype=dtype).eval()

    teacher.requires_grad_(False)
    return teacher


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Distributed setup ──
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = 0
        world_size = 1

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Load teacher model ──
    if global_rank == 0:
        logger.info("=" * 70)
        logger.info("InfiniteTalk ODE Trajectory Generation")
        logger.info("=" * 70)

    t_model_start = time.time()
    teacher = build_teacher(
        weights_dir=args.weights_dir,
        infinitetalk_ckpt=args.infinitetalk_ckpt,
        lora_ckpt=args.lora_ckpt,
        shift=args.shift,
        device=device,
        dtype=dtype,
    )
    noise_scheduler = teacher.noise_scheduler

    if global_rank == 0:
        t_model_elapsed = time.time() - t_model_start
        logger.info(
            f"Teacher loaded in {t_model_elapsed:.1f}s. "
            f"Noise schedule: RF, max_t={noise_scheduler.max_t}, shift={args.shift}"
        )

    # ── Log trajectory subsampling plan ──
    if global_rank == 0:
        timesteps = get_shifted_timesteps(
            args.num_steps, shift=args.shift, device="cpu",
        )
        t_schedule = timesteps / 1000.0
        subsample_idx = compute_subsample_indices(t_schedule, args.t_list)

        logger.info(f"ODE steps: {args.num_steps}, target t_list: {args.t_list}")
        logger.info(f"Computed subsample indices: {subsample_idx}")
        total_states = args.num_steps + 1
        for i, idx in enumerate(subsample_idx):
            resolved = idx if idx >= 0 else total_states + idx
            actual_t = t_schedule[resolved].item()
            target_t = args.t_list[i] if i < len(args.t_list) else 0.0
            logger.info(
                f"  Step {i}: target t={target_t:.4f} -> "
                f"index {idx} (resolved={resolved}), actual t={actual_t:.4f}"
            )
        logger.info(
            f"CFG: text_guide_scale={args.text_guide_scale}, "
            f"audio_guide_scale={args.audio_guide_scale}"
        )

    # ── Load negative text embedding ──
    if args.neg_text_emb_path is not None and os.path.exists(args.neg_text_emb_path):
        neg_text_embeds = _load_tensor(args.neg_text_emb_path, dtype)
    else:
        neg_text_embeds = torch.zeros(1, 512, 4096, dtype=dtype)

    if neg_text_embeds.dim() == 2:
        neg_text_embeds = neg_text_embeds.unsqueeze(0)

    if global_rank == 0:
        logger.info(f"Negative text embedding shape: {neg_text_embeds.shape}")

    # ── Gather sample directories ──
    with open(args.data_list_path) as f:
        all_dirs = [line.strip() for line in f if line.strip()]

    # Apply index range
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else len(all_dirs)
    all_dirs = all_dirs[start:end]

    # Apply max_samples
    if args.max_samples is not None:
        all_dirs = all_dirs[: args.max_samples]

    # Filter: check required files exist
    required_files = [
        "vae_latents.pt",
        "first_frame_cond.pt",
        "clip_features.pt",
        "audio_emb.pt",
        "text_embeds.pt",
    ]
    valid_dirs = []
    for d in all_dirs:
        missing = [fn for fn in required_files if not os.path.exists(os.path.join(d, fn))]
        if missing:
            if global_rank == 0:
                logger.warning(f"Skipping {d}: missing {missing}")
        else:
            valid_dirs.append(d)

    if global_rank == 0:
        logger.info(
            f"Processing {len(valid_dirs)} samples "
            f"(range [{start}:{end}], {len(all_dirs) - len(valid_dirs)} skipped)"
        )

    # Skip existing if requested
    if args.skip_existing:
        before = len(valid_dirs)
        valid_dirs = [
            d for d in valid_dirs
            if not os.path.exists(os.path.join(d, args.output_key))
        ]
        if global_rank == 0:
            logger.info(
                f"Skipped {before - len(valid_dirs)} samples "
                f"with existing {args.output_key}"
            )

    # ── Distribute across ranks ──
    rank_dirs = valid_dirs[global_rank::world_size]

    if global_rank == 0:
        logger.info(
            f"Rank {global_rank}/{world_size}: processing {len(rank_dirs)} samples"
        )

    # ── Process samples ──
    total_time = 0.0
    success_count = 0
    fail_count = 0

    pbar = tqdm(
        rank_dirs,
        disable=global_rank != 0,
        desc="Generating ODE trajectories",
    )

    for sample_idx, sample_dir in enumerate(pbar):
        t_start = time.time()

        # Per-sample deterministic seed: base_seed + global sample index
        # Find global index of this sample in valid_dirs for reproducibility
        global_sample_idx = valid_dirs.index(sample_dir)
        sample_seed = args.seed + global_sample_idx
        torch.manual_seed(sample_seed)

        # Load sample
        sample = load_sample(
            sample_dir=sample_dir,
            neg_text_embeds=neg_text_embeds,
            num_video_frames=args.num_video_frames,
            audio_window=args.audio_window,
            device=device,
            dtype=dtype,
        )

        if sample is None:
            fail_count += 1
            continue

        vae_latents = sample["vae_latents"]             # [16, 21, H, W]
        condition = sample["condition"]
        neg_text_condition = sample["neg_text_condition"]
        neg_condition = sample["neg_condition"]

        latent_shape = tuple(vae_latents.shape)  # (16, 21, H, W)

        try:
            # Extract ODE trajectory
            trajectory = extract_ode_trajectory(
                teacher=teacher,
                noise_scheduler=noise_scheduler,
                latent_shape=latent_shape,
                condition=condition,
                neg_text_condition=neg_text_condition,
                neg_condition=neg_condition,
                num_steps=args.num_steps,
                shift=args.shift,
                text_guide_scale=args.text_guide_scale,
                audio_guide_scale=args.audio_guide_scale,
                target_t_list=args.t_list,
                device=device,
                dtype=dtype,
            )
            # trajectory: [num_subsample, C, T, H, W]
            # extract_ode_trajectory already filters out t=0.0 (clean state)
            # since the clean target is vae_latents.pt, not the ODE solve.
            ode_path = trajectory  # [num_subsample, 16, 21, H, W]

            # Save to sample directory
            save_path = os.path.join(sample_dir, args.output_key)
            torch.save(ode_path.to(torch.bfloat16).cpu(), save_path)

            t_elapsed = time.time() - t_start
            total_time += t_elapsed
            success_count += 1

            pbar.set_postfix({
                "done": success_count,
                "fail": fail_count,
                "shape": list(ode_path.shape),
                "time": f"{t_elapsed:.1f}s",
            })

        except Exception as e:
            logger.warning(f"Failed ODE extraction for {sample_dir}: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            continue

        # Free memory
        del sample, trajectory, ode_path, vae_latents
        del condition, neg_text_condition, neg_condition
        torch.cuda.empty_cache()

    # ── Summary ──
    if world_size > 1:
        dist.barrier()

    avg_time = total_time / max(success_count, 1)
    if global_rank == 0:
        logger.info("=" * 70)
        logger.info(
            f"ODE trajectory generation complete. "
            f"Success: {success_count}, Failed: {fail_count}, "
            f"Avg time: {avg_time:.1f}s/sample"
        )

        if success_count > 0:
            # Verify a saved file
            test_dir = rank_dirs[0] if rank_dirs else valid_dirs[0]
            test_path = os.path.join(test_dir, args.output_key)
            if os.path.exists(test_path):
                saved = torch.load(test_path, map_location="cpu", weights_only=True)
                logger.info(
                    f"Verification: {test_path}\n"
                    f"  shape={list(saved.shape)}, dtype={saved.dtype}, "
                    f"min={saved.float().min():.4f}, max={saved.float().max():.4f}, "
                    f"size={os.path.getsize(test_path) / 1024 / 1024:.1f}MB"
                )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
