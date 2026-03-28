# SPDX-License-Identifier: Apache-2.0
"""
WandbCallback subclass that muxes source audio into generated video samples.

For InfiniteTalk talking-face generation, visual-only videos are insufficient
for verifying lip-sync quality. This callback detects 'audio_path' in the
data batch (provided by InfiniteTalkDataset when audio_data_root is set)
and muxes the corresponding audio segment into wandb video logs.
"""

import os
import subprocess
import tempfile
from typing import Optional, Callable

import torch
import wandb

from fastgen.callbacks.wandb import WandbCallback, to_wandb
from fastgen.utils.distributed import rank0_only
import fastgen.utils.logging_utils as logger


def _to_wandb_with_audio(
    tensor: torch.Tensor,
    audio_path: str,
    fps: int = 25,
    vid_format: str = "mp4",
    caption: Optional[str] = None,
) -> wandb.Video:
    """Convert video tensor to wandb.Video with muxed audio from source file."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return to_wandb(tensor, fps=fps, vid_format=vid_format, caption=caption)

    # tensor is [B, T, C, H, W] uint8 at this point (from to_wandb preprocessing)
    # But we receive it BEFORE to_wandb conversion, as [B, C, T, H, W] float [-1,1]
    if tensor.ndim == 5:
        vid = tensor[0]  # [C, T, H, W]
    else:
        vid = tensor  # [C, T, H, W] or [T, C, H, W]

    # Ensure [C, T, H, W] → [T, H, W, C]
    if vid.shape[0] == 3:  # C first
        frames = vid.permute(1, 2, 3, 0)
    else:  # T first
        frames = vid.permute(0, 2, 3, 1)

    frames = ((frames + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
    num_frames, h, w, _ = frames.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        noaudio_path = os.path.join(tmpdir, f"noaudio.{vid_format}")
        output_path = os.path.join(tmpdir, f"with_audio.{vid_format}")

        cmd = [
            ffmpeg_exe, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            noaudio_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for i in range(num_frames):
            proc.stdin.write(frames[i].tobytes())
        proc.stdin.close()
        proc.wait()

        duration = num_frames / fps
        mux_cmd = [
            ffmpeg_exe, "-y",
            "-i", noaudio_path, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-t", f"{duration:.3f}", "-shortest",
            output_path,
        ]
        result = subprocess.run(mux_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return wandb.Video(output_path, fps=fps, format=vid_format, caption=caption)

    # Fallback: no audio
    return to_wandb(tensor, fps=fps, vid_format=vid_format, caption=caption)


class InfiniteTalkWandbCallback(WandbCallback):
    """WandbCallback that muxes source audio into generated video samples.

    When the data batch contains 'audio_path' (string path to source .wav),
    all logged video samples get the corresponding audio muxed in via ffmpeg.
    """

    def __init__(self, *args, audio_fps: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_fps = audio_fps

    def get_sample_map(self, model, data_batch, output_batch):
        """Override to mux audio into video samples."""
        # Guard: if gen_rand is not in output_batch, return empty (no visual to log)
        if "gen_rand" not in output_batch:
            logger.debug("No 'gen_rand' in output_batch, skipping sample map")
            return {}

        # Get audio path from data batch (string, not collated as tensor)
        audio_path = data_batch.get("audio_path")
        if isinstance(audio_path, (list, tuple)):
            audio_path = audio_path[0]
        if not isinstance(audio_path, str) or not os.path.exists(str(audio_path)):
            audio_path = None

        # Run base get_sample_map but intercept the to_wandb calls
        # by temporarily replacing the module-level to_wandb function
        if audio_path is None:
            return super().get_sample_map(model, data_batch, output_batch)

        # With audio: run the base logic but use our audio-muxing to_wandb
        import fastgen.callbacks.wandb as wandb_module
        original_to_wandb = wandb_module.to_wandb

        def _to_wandb_audio_wrapper(tensor, **kwargs):
            # Only mux audio for video tensors (5D), not images (4D)
            if tensor.ndim == 5:
                kwargs.setdefault("fps", self.audio_fps)
                return _to_wandb_with_audio(tensor, audio_path, **kwargs)
            return original_to_wandb(tensor, **kwargs)

        wandb_module.to_wandb = _to_wandb_audio_wrapper
        try:
            result = super().get_sample_map(model, data_batch, output_batch)
        finally:
            wandb_module.to_wandb = original_to_wandb

        return result
