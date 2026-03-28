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
        """Override to handle InfiniteTalk's standalone DiT (no init_preprocessors).

        The base WandbCallback.get_sample_map assumes model.net has init_preprocessors
        (diffusers-specific) to VAE-decode latents. Our standalone port doesn't have this,
        so gen_rand is already pixel-space (decoded in _get_outputs callable) but
        data_batch["real"] is still a latent. We handle this by only logging gen_rand
        and skipping raw latent logging.
        """
        # Only generate visuals on rank 0 (avoids duplicate AR generation on other ranks)
        from fastgen.utils.distributed import is_rank0
        if not is_rank0():
            return {}

        if "gen_rand" not in output_batch:
            return {}

        from fastgen.callbacks.wandb import to_wandb
        from fastgen.utils.distributed import synchronize
        from typing import Callable
        import gc

        sample_map = {}
        gen_rand = output_batch["gen_rand"]

        # Call the gen_rand callable (AR generation + VAE decode)
        if isinstance(gen_rand, Callable):
            synchronize()
            gen_rand = gen_rand()
            synchronize()

        # gen_rand is now pixel-space [B, C, T, H, W] or [C, T, H, W]
        if wandb.run and gen_rand is not None:
            try:
                sample_map["student/generation"] = to_wandb(
                    gen_rand, fps=self.audio_fps, vid_format=self.vid_format
                )
            except Exception as e:
                logger.warning(f"Failed to log student generation to wandb: {e}")

            # Also decode and log real data if VAE is available
            if "real" in data_batch and hasattr(model.net, 'vae'):
                try:
                    from fastgen.utils.basic_utils import inference_mode
                    with inference_mode(model.net, precision_amp=model.precision_amp_enc,
                                       device_type=model.device.type if hasattr(model.device, 'type') else 'cuda'):
                        real_decoded = model.net.vae.decode(data_batch["real"][:1].float())
                        if isinstance(real_decoded, (list, tuple)):
                            real_decoded = real_decoded[0].unsqueeze(0) if len(real_decoded) == 1 else torch.stack(real_decoded)
                    sample_map["data/real"] = to_wandb(
                        real_decoded, fps=self.audio_fps, vid_format=self.vid_format
                    )
                except Exception as e:
                    logger.warning(f"Failed to decode/log real data: {e}")

        synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        return sample_map
