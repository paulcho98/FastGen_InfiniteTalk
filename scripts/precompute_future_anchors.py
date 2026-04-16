#!/usr/bin/env python3
"""Precompute future anchor latents for Diffusion Forcing training.

For each precomputed sample directory, this script:
  1. Resolves the source video from the sample dir name
  2. Reads 25 video frames: 5 overlap from the clip tail + 20 future frames
  3. VAE-encodes all 25 frames → 7 latent frames
  4. Discards the first 2 latent frames (temporal context), keeps 5 future anchors
  5. Saves as future_anchor_latents[_quarter].pt with shape [16, 5, H_lat, W_lat]

Sample directory naming convention:
    data_{VIDEOID}_{VIDEOID}_{START}_{END}
    e.g. data_abc123_abc123_0_81

The clip covers video frames [START, END).  We read frames [END-5, END+20)
from the source video, giving 5 overlap frames for VAE temporal context and
20 future frames that compress into 5 latent anchor frames.

VAE temporal compression:
    1st pixel frame → 1 latent frame (no compression)
    Subsequent: 4 pixel frames → 1 latent frame
    25 pixel frames → 1 + 24/4 = 7 latent frames
    First 2 latent frames = temporal overlap context (discarded)
    Last 5 latent frames = future anchors (saved)

Source video resolution:
    {raw_data_root}/data/{VIDEOID}/{VIDEOID}_{START}_{END}.mp4  (clip-level)
    {raw_data_root}/videos/{VIDEOID}.mp4                        (full video)
"""

import argparse
import logging
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

VAE_STRIDE = (4, 8, 8)  # temporal, height, width stride of WanVAE

# Aspect-ratio buckets for 480p — matches InfiniteTalk's original pipeline
ASPECT_RATIO_627 = {
    '0.26': [320, 1216], '0.38': [384, 1024], '0.50': [448, 896],
    '0.67': [512, 768],  '0.82': [576, 704],  '1.00': [640, 640],
    '1.22': [704, 576],  '1.50': [768, 512],  '1.86': [832, 448],
    '2.00': [896, 448],  '2.50': [960, 384],  '2.83': [1088, 384],
    '3.60': [1152, 320], '3.80': [1216, 320],  '4.00': [1280, 320],
}
ASPECT_RATIO_627_QUARTER = {
    k: [h // 2, w // 2] for k, [h, w] in ASPECT_RATIO_627.items()
}


# ===================================================================
# InfiniteTalk path / module setup
# ===================================================================

def _add_infinitetalk_to_path():
    """Add InfiniteTalk source root to sys.path and install mock modules
    for heavy dependencies (xfuser, xformers) that are not needed for
    pure encoding workloads.

    Mirrors scripts/precompute_infinitetalk_data.py::_add_infinitetalk_to_path().
    """
    import types
    import importlib
    import importlib.machinery

    it_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     "../../InfiniteTalk"))
    if it_root not in sys.path:
        sys.path.insert(0, it_root)

    class _MockModule(types.ModuleType):
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

    oq = sys.modules["optimum.quanto"]
    oq.quantize = lambda *a, **k: None
    oq.freeze = lambda *a, **k: None
    oq.qint8 = None
    oq.requantize = lambda *a, **k: None

    xfuser_dist = sys.modules["xfuser.core.distributed"]
    xfuser_dist.get_sequence_parallel_rank = lambda: 0
    xfuser_dist.get_sequence_parallel_world_size = lambda: 1
    xfuser_dist.get_sp_group = lambda: None

    wan_pkg = types.ModuleType("wan")
    wan_pkg.__path__ = [os.path.join(it_root, "wan")]
    wan_pkg.__package__ = "wan"
    wan_pkg.__file__ = os.path.join(it_root, "wan", "__init__.py")
    sys.modules["wan"] = wan_pkg

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


# ===================================================================
# Video loading helpers
# ===================================================================

def _resample_indices(native_fps, target_fps, frame_count, total_frames):
    """Compute frame indices to resample from native_fps to target_fps."""
    if abs(native_fps - target_fps) < 0.5:
        return list(range(min(frame_count, total_frames)))
    indices = []
    for i in range(frame_count):
        t = i / target_fps
        src_idx = round(t * native_fps)
        if src_idx >= total_frames:
            break
        indices.append(src_idx)
    return indices


def load_video_frames_range(
    video_path: str,
    start_frame: int,
    frame_count: int,
    target_fps: float = 25.0,
):
    """Load a specific range of frames from a video, resampled to target_fps.

    Reads frames [start_frame, start_frame + frame_count) from the video,
    where start_frame and frame_count are in target_fps space. If the source
    video FPS differs from target_fps, appropriate resampling is applied.

    Returns:
        frames: np.ndarray [T, H, W, 3] uint8 (T <= frame_count)
        fps: float (always target_fps)
    """
    # Try decord first (fast, C++ backend)
    try:
        from decord import VideoReader, cpu as decord_cpu
        vr = VideoReader(video_path, ctx=decord_cpu(0))
        native_fps = float(vr.get_avg_fps())
        total = len(vr)

        # Map target-fps frame indices to source-fps indices
        if abs(native_fps - target_fps) < 0.5:
            src_indices = list(range(start_frame, start_frame + frame_count))
        else:
            src_indices = []
            for i in range(start_frame, start_frame + frame_count):
                t_sec = i / target_fps
                src_idx = round(t_sec * native_fps)
                src_indices.append(src_idx)

        # Filter out-of-range indices
        valid = [idx for idx in src_indices if 0 <= idx < total]
        if len(valid) < frame_count:
            return None, target_fps  # not enough frames

        frames = vr.get_batch(valid).asnumpy()  # [T, H, W, 3]
        return frames, target_fps
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: av
    try:
        import av
        container = av.open(video_path)
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate) if stream.average_rate else target_fps
        total = stream.frames or 10000

        # Map target-fps frame indices to source-fps indices
        if abs(native_fps - target_fps) < 0.5:
            src_indices = list(range(start_frame, start_frame + frame_count))
        else:
            src_indices = []
            for i in range(start_frame, start_frame + frame_count):
                t_sec = i / target_fps
                src_idx = round(t_sec * native_fps)
                src_indices.append(src_idx)

        # Check bounds
        if any(idx >= total for idx in src_indices) or any(idx < 0 for idx in src_indices):
            container.close()
            return None, target_fps

        max_idx = max(src_indices)
        index_set = set(src_indices)
        all_frames = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in index_set:
                all_frames[i] = frame.to_ndarray(format="rgb24")
            if i >= max_idx:
                break
        container.close()

        collected = [all_frames[i] for i in src_indices if i in all_frames]
        if len(collected) < frame_count:
            return None, target_fps

        frames = np.stack(collected, axis=0)
        return frames, target_fps
    except ImportError:
        pass
    except Exception:
        pass

    raise RuntimeError(
        f"Could not load frames from {video_path}: "
        "neither decord nor av available"
    )


def resize_and_center_crop(frames: np.ndarray, target_h: int, target_w: int):
    """Resize + center-crop frames to (target_h, target_w).

    Matches precompute_infinitetalk_data.py::resize_and_center_crop().

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
    )

    crop_top = (new_h - target_h) // 2
    crop_left = (new_w - target_w) // 2
    t_tensor = t_tensor[:, :, crop_top: crop_top + target_h,
                        crop_left: crop_left + target_w]

    t_tensor = (t_tensor / 255.0 - 0.5) * 2.0
    t_tensor = t_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
    return t_tensor


# ===================================================================
# VAE encoding
# ===================================================================

@torch.no_grad()
def encode_future_anchors(vae, video_tensor, device):
    """VAE-encode 25 future-context frames → 5 anchor latent frames.

    Args:
        vae: WanVAE encoder instance.
        video_tensor: [C, 25, H, W] float32 in [-1, 1]
            5 overlap frames (clip tail) + 20 future frames.
        device: torch device for encoding.

    Returns:
        anchor_latents: [16, 5, H_lat, W_lat] bf16
            5 future latent frames (overlap context discarded).
    """
    assert video_tensor.shape[1] == 25, (
        f"Expected 25 pixel frames, got {video_tensor.shape[1]}"
    )

    vae_dtype = getattr(vae, "dtype", None) or next(vae.model.parameters()).dtype
    video = video_tensor.to(device=device, dtype=vae_dtype)

    # VAE encode: 25 frames → 7 latent frames
    # (1st frame → 1 latent, 24 remaining → 6 latents)
    latents = vae.encode([video])  # list of [C_lat, T_lat, H_lat, W_lat]
    latents = latents[0]  # [16, 7, H_lat, W_lat]

    assert latents.shape[1] == 7, (
        f"Expected 7 latent frames from 25 pixel frames, got {latents.shape[1]}"
    )

    # Discard first 2 latent frames (overlap temporal context),
    # keep last 5 (future anchors)
    anchor_latents = latents[:, 2:, :, :]  # [16, 5, H_lat, W_lat]
    anchor_latents = anchor_latents.to(torch.bfloat16).cpu()

    del video, latents
    torch.cuda.empty_cache()

    return anchor_latents


# ===================================================================
# Sample directory parsing
# ===================================================================

def parse_sample_dir_name(basename: str):
    """Parse a sample directory basename into (video_id, start_frame, end_frame).

    Expected format: data_{VIDEOID}_{VIDEOID}_{START}_{END}
    The VIDEOID may contain hyphens, dots, etc. The pattern is that it appears
    twice, separated by underscore.

    Returns:
        (video_id, start_frame, end_frame) or None if parsing fails.
    """
    # The basename looks like: data_XXXX_XXXX_123_456
    # where XXXX is repeated. We need to find the split point.
    #
    # Strategy: strip "data_" prefix, then find a pattern where the remaining
    # string is "{VID}_{VID}_{START}_{END}" with VID repeated.
    if not basename.startswith("data_"):
        return None

    rest = basename[5:]  # strip "data_"

    # The rest should be: {VIDEOID}_{VIDEOID}_{START}_{END}
    # START and END are integers. Work backwards from the end.
    parts = rest.rsplit("_", 2)  # split into [everything_before, START?, END?]
    if len(parts) != 3:
        return None

    prefix, start_str, end_str = parts
    try:
        start_frame = int(start_str)
        end_frame = int(end_str)
    except ValueError:
        return None

    # prefix should be "{VIDEOID}_{VIDEOID}"
    # Find the split: try all positions where we can split prefix into two
    # equal halves separated by underscore
    # The VIDEOID itself may contain underscores, so we look for the split
    # where the two halves are identical.
    for i in range(1, len(prefix)):
        if prefix[i] == '_':
            left = prefix[:i]
            right = prefix[i + 1:]
            if left == right:
                return (left, start_frame, end_frame)

    # If no clean split found (shouldn't happen with well-formed data),
    # fall back: assume no underscores in VIDEOID, the first half is the ID
    return None


def resolve_video_path(raw_data_root: str, video_id: str, start_frame: int, end_frame: int):
    """Resolve the source video path for a sample.

    Tries in order:
      1. {raw_data_root}/data/{VIDEOID}/{VIDEOID}_{START}_{END}.mp4 (clip-level)
      2. {raw_data_root}/videos/{VIDEOID}.mp4 (full video)

    Returns:
        (video_path, is_clip_file) or (None, None) if not found.
        is_clip_file: True if the path is a trimmed clip file.
    """
    # Try clip-level file first
    clip_path = os.path.join(
        raw_data_root, "data", video_id,
        f"{video_id}_{start_frame}_{end_frame}.mp4"
    )
    if os.path.isfile(clip_path):
        return clip_path, True

    # Try full video
    full_path = os.path.join(raw_data_root, "videos", f"{video_id}.mp4")
    if os.path.isfile(full_path):
        return full_path, False

    return None, None


# ===================================================================
# Atomic save
# ===================================================================

def _save_atomic(tensor, path):
    """Save tensor atomically: write to .tmp then rename."""
    tmp_path = path + ".tmp"
    torch.save(tensor, tmp_path)
    os.rename(tmp_path, path)  # atomic on same filesystem


# ===================================================================
# Main processing
# ===================================================================

def process_sample(
    sample_dir: str,
    raw_data_root: str,
    vae,
    device: str,
    quarter_res: bool,
    target_fps: float = 25.0,
):
    """Process a single sample: compute and save future anchor latents.

    Returns:
        'done'     — anchor latents saved successfully
        'skipped'  — output already exists
        'no_video' — source video not found
        'short'    — video too short (not enough future frames)
        'error'    — unexpected error (already logged)
    """
    vae_suffix = "_quarter" if quarter_res else ""
    output_name = f"future_anchor_latents{vae_suffix}.pt"
    output_path = os.path.join(sample_dir, output_name)

    if os.path.isfile(output_path):
        return "skipped"

    # Parse sample directory name
    basename = os.path.basename(sample_dir)
    parsed = parse_sample_dir_name(basename)
    if parsed is None:
        logger.warning("Cannot parse sample dir name: %s", basename)
        return "error"

    video_id, clip_start, clip_end = parsed

    # Resolve source video path
    video_path, is_clip = resolve_video_path(
        raw_data_root, video_id, clip_start, clip_end
    )
    if video_path is None:
        logger.warning(
            "Source video not found for %s (tried data/%s/ and videos/)",
            basename, video_id,
        )
        return "no_video"

    # Determine which frames to read:
    # We need 5 overlap frames from the clip tail + 20 future frames = 25 total.
    # Overlap starts at (clip_end - 5) in absolute video coords.
    needed_frames = 25  # 5 overlap + 20 future

    if is_clip:
        # Clip file spans [clip_start, clip_end) in its own frame space (0-indexed).
        # The clip has (clip_end - clip_start) frames.
        clip_length = clip_end - clip_start
        # Overlap starts at (clip_length - 5) within the clip file.
        # BUT the clip file only has clip_length frames, so we cannot read
        # frames past clip_length. Clip-level files are already trimmed.
        # We need frames from (clip_length - 5) to (clip_length - 5 + 25 - 1),
        # but the clip only has up to clip_length - 1. So clip files cannot
        # provide the future frames — skip to full video if possible.
        #
        # Actually, the task says: "If the clip-level file exists, it's already
        # trimmed to [START, END), so the overlap + future frames start at
        # offset (END - START) - 5 within the clip file."
        # This only works if the clip file has MORE than (END - START) frames
        # (i.e. it was pre-cut with extra padding). Let's try it, and fall back
        # to the full video if the clip doesn't have enough frames.

        read_start = clip_length - 5
        frames, _ = load_video_frames_range(
            video_path, read_start, needed_frames, target_fps
        )
        if frames is None:
            # Clip file too short — try full video
            full_path = os.path.join(
                raw_data_root, "videos", f"{video_id}.mp4"
            )
            if os.path.isfile(full_path):
                video_path = full_path
                is_clip = False
            else:
                logger.warning(
                    "Clip file too short and no full video for %s", basename
                )
                return "short"

    if not is_clip:
        # Full video: frames are in absolute coords.
        # Read from (clip_end - 5) to (clip_end - 5 + 25 - 1).
        read_start = clip_end - 5
        frames, _ = load_video_frames_range(
            video_path, read_start, needed_frames, target_fps
        )
        if frames is None:
            logger.warning(
                "Video too short for future anchors: %s "
                "(need frame %d, video_id=%s)",
                basename, read_start + needed_frames - 1, video_id,
            )
            return "short"

    assert frames.shape[0] == needed_frames, (
        f"Expected {needed_frames} frames, got {frames.shape[0]}"
    )

    # Determine target spatial resolution from existing vae_latents
    # (read shape from the precomputed latents to match exactly)
    existing_latents_path = os.path.join(
        sample_dir, f"vae_latents{vae_suffix}.pt"
    )
    if os.path.isfile(existing_latents_path):
        existing_shape = torch.load(
            existing_latents_path, map_location="cpu", weights_only=True
        ).shape
        # existing_shape: [16, T_lat, H_lat, W_lat]
        # Recover pixel spatial dims: H_pixel = H_lat * 8, W_pixel = W_lat * 8
        target_h = existing_shape[2] * VAE_STRIDE[1]
        target_w = existing_shape[3] * VAE_STRIDE[2]
    else:
        # Fallback: detect from video frame shape and aspect ratio bucket
        src_h, src_w = frames.shape[1], frames.shape[2]
        ratio = src_h / src_w
        buckets = ASPECT_RATIO_627_QUARTER if quarter_res else ASPECT_RATIO_627
        closest_bucket = sorted(
            buckets.keys(), key=lambda x: abs(float(x) - ratio)
        )[0]
        target_h, target_w = buckets[closest_bucket]

    # Preprocess: resize + center crop + normalize to [-1, 1]
    video_tensor = resize_and_center_crop(frames, target_h, target_w)
    # video_tensor: [C, 25, target_h, target_w] float32 in [-1, 1]

    # VAE encode → 5 future anchor latents
    anchor_latents = encode_future_anchors(vae, video_tensor, device)
    # anchor_latents: [16, 5, H_lat, W_lat] bf16

    # Save atomically
    _save_atomic(anchor_latents, output_path)

    return "done"


def main():
    parser = argparse.ArgumentParser(
        description="Precompute future anchor latents for DF training"
    )
    parser.add_argument(
        "--sample_list", type=str, required=True,
        help="Path to text file listing sample directories (one per line), "
             "OR a directory to scan for sample subdirs.",
    )
    parser.add_argument(
        "--raw_data_root", type=str, required=True,
        help="Root directory of raw TalkVid dataset (contains data/ and/or videos/).",
    )
    parser.add_argument(
        "--weights_dir", type=str, required=True,
        help="InfiniteTalk weights dir (Wan2.1-I2V-14B-480P/) containing Wan2.1_VAE.pth",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device for VAE encoding (default: cuda:0)",
    )
    parser.add_argument(
        "--quarter_res", action="store_true", default=False,
        help="Use quarter resolution (half spatial dims). "
             "Saves as future_anchor_latents_quarter.pt.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=0,
        help="Process only first N samples (0 = all)",
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index in sample list (for multi-GPU sharding)",
    )
    parser.add_argument(
        "--end_idx", type=int, default=-1,
        help="End index in sample list, exclusive (-1 = all)",
    )
    args = parser.parse_args()

    # ---- Add InfiniteTalk to path ----
    it_root = _add_infinitetalk_to_path()
    logger.info("InfiniteTalk source root: %s", it_root)

    # ---- Build sample directory list ----
    if os.path.isfile(args.sample_list):
        # Text file with one sample dir per line
        with open(args.sample_list, "r") as f:
            sample_dirs = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d sample dirs from %s", len(sample_dirs), args.sample_list)
    elif os.path.isdir(args.sample_list):
        # Directory — scan for subdirectories starting with "data_"
        sample_dirs = sorted([
            os.path.join(args.sample_list, d)
            for d in os.listdir(args.sample_list)
            if os.path.isdir(os.path.join(args.sample_list, d))
            and d.startswith("data_")
        ])
        logger.info(
            "Found %d sample dirs in %s", len(sample_dirs), args.sample_list
        )
    else:
        logger.error("--sample_list must be a file or directory: %s", args.sample_list)
        sys.exit(1)

    total = len(sample_dirs)

    # Apply sharding / subsetting
    if args.start_idx > 0 or args.end_idx >= 0:
        end = args.end_idx if args.end_idx >= 0 else len(sample_dirs)
        sample_dirs = sample_dirs[args.start_idx:end]
    if args.num_samples > 0:
        sample_dirs = sample_dirs[:args.num_samples]

    logger.info(
        "Processing %d / %d samples (range [%d:%s])",
        len(sample_dirs), total,
        args.start_idx,
        args.end_idx if args.end_idx >= 0 else "end",
    )

    # ---- Load VAE ----
    logger.info("Loading VAE to %s ...", args.device)
    t0 = time.time()
    vae = load_vae(args.weights_dir, args.device)
    logger.info("VAE loaded in %.1fs", time.time() - t0)

    # ---- Process samples ----
    counts = {"done": 0, "skipped": 0, "no_video": 0, "short": 0, "error": 0}
    t_start = time.time()

    for idx, sample_dir in enumerate(sample_dirs):
        basename = os.path.basename(sample_dir)
        t_sample = time.time()

        try:
            status = process_sample(
                sample_dir=sample_dir,
                raw_data_root=args.raw_data_root,
                vae=vae,
                device=args.device,
                quarter_res=args.quarter_res,
            )
        except Exception:
            status = "error"
            logger.error(
                "  ERROR processing %s:\n%s",
                basename, traceback.format_exc(),
            )

        counts[status] += 1
        elapsed = time.time() - t_sample

        if status == "done":
            logger.info(
                "[%d/%d] %s — encoded in %.1fs",
                idx + 1, len(sample_dirs), basename, elapsed,
            )
        elif status == "skipped":
            logger.info(
                "[%d/%d] %s — already exists, skipped",
                idx + 1, len(sample_dirs), basename,
            )
        else:
            logger.warning(
                "[%d/%d] %s — %s (%.1fs)",
                idx + 1, len(sample_dirs), basename, status, elapsed,
            )

        # Periodic memory cleanup
        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    total_time = time.time() - t_start
    logger.info(
        "Finished in %.1fs. done=%d  skipped=%d  no_video=%d  short=%d  error=%d  total=%d",
        total_time,
        counts["done"], counts["skipped"], counts["no_video"],
        counts["short"], counts["error"], len(sample_dirs),
    )


if __name__ == "__main__":
    main()
