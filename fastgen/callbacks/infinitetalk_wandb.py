# SPDX-License-Identifier: Apache-2.0
"""
WandbCallback for InfiniteTalk — list-based validation video logging.

DDP constraint: never do expensive work (AR generation) between DDP-synced
validation forward passes. Store data during on_validation_step_end (instant),
generate videos in on_validation_end (after DDP loop).
"""

import gc
import os
import tempfile
from typing import Optional, Callable
from functools import partial

import numpy as np
import torch
import wandb

from fastgen.callbacks.wandb import WandbCallback, to_wandb
from fastgen.utils.distributed import synchronize, is_rank0
from fastgen.utils import basic_utils
import fastgen.utils.logging_utils as logger


class InfiniteTalkWandbCallback(WandbCallback):

    def __init__(self, *args, audio_fps: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_fps = audio_fps
        # Store val data for deferred video generation
        self._val_conditions: list[dict] = []
        self._val_real_data: list[torch.Tensor] = []
        self._val_audio_paths: list[Optional[str]] = []
        self._gt_uploaded = False
        self._model_ref = None  # set during validation

    @staticmethod
    def _mux_video_audio(video_uint8: np.ndarray, audio_path: str, fps: int) -> Optional[str]:
        """Write video+audio to a temp MP4 file via raw ffmpeg piping.

        Args:
            video_uint8: [T, C, H, W] uint8 numpy array (from _to_uint8_video)
        Returns:
            Path to muxed MP4, or None on failure. Caller must clean up.
        """
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None

        import subprocess

        # Convert [T, C, H, W] → [T, H, W, C] for raw piping
        if video_uint8.ndim == 4 and video_uint8.shape[1] == 3:
            frames = np.transpose(video_uint8, (0, 2, 3, 1))
        else:
            frames = video_uint8
        num_frames, h, w, _ = frames.shape

        tmp_dir = tempfile.mkdtemp()
        noaudio_path = os.path.join(tmp_dir, "noaudio.mp4")
        output_path = os.path.join(tmp_dir, "with_audio.mp4")

        try:
            # Encode raw frames → silent H.264 via stdin pipe
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

            # Mux audio
            duration = num_frames / fps
            mux_cmd = [
                ffmpeg_exe, "-y",
                "-i", noaudio_path, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac",
                "-t", f"{duration:.3f}", "-shortest",
                output_path,
            ]
            result = subprocess.run(mux_cmd, capture_output=True, timeout=30)

            # Clean up intermediate
            if os.path.exists(noaudio_path):
                os.unlink(noaudio_path)

            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception as e:
            logger.warning(f"Audio muxing failed: {e}")

        # Cleanup on failure
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    @staticmethod
    def _to_uint8_video(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, (list, tuple)):
            tensor = torch.stack(tensor) if len(tensor) > 1 else tensor[0].unsqueeze(0)
        t = tensor.permute(0, 2, 1, 3, 4)
        return t.mul(127.5).add(127.5).clamp(0, 255).to(torch.uint8).cpu()

    def on_dataloader_init_end(
        self, model, dataloader_train, dataloader_val, iteration: int = 0
    ) -> None:
        """All ranks enter (no @rank0_only). Only rank 0 uploads GT."""
        if dataloader_val is None:
            synchronize()
            return
        if not hasattr(model.net, "vae"):
            if is_rank0():
                logger.info("No VAE loaded — skipping GT val video upload")
            synchronize()
            return

        if is_rank0():
            logger.info("Uploading GT validation videos to wandb...")
            device = model.device
            try:
                gt_videos = []
                with torch.no_grad(), basic_utils.inference_mode(
                    precision_amp=model.precision_amp_enc, device_type=device.type
                ):
                    for step, data in enumerate(dataloader_val):
                        real = data["real"].to(device)
                        decoded = model.net.vae.decode(real[:1].float())
                        gt_videos.append(self._to_uint8_video(decoded))
                if wandb.run:
                    gt_list = [wandb.Video(v[0].numpy(), fps=self.audio_fps, format="mp4") for v in gt_videos]
                    wandb.log({"val_gt/videos": gt_list}, step=0)
                self._gt_uploaded = True
                logger.info(f"Uploaded {len(gt_videos)} GT validation videos")
            except Exception as e:
                logger.warning(f"Failed to upload GT val videos: {e}")
        synchronize()

    def on_validation_step_end(
        self, model, data_batch, output_batch, loss_dict, step=0, iteration=0, idx=0,
    ) -> None:
        """Store condition + real data for deferred generation. Instant on all ranks."""
        self.val_loss_dict_record.add(loss_dict)
        logger.info(f"[val_step_end] step={step}, rank0={is_rank0()}")

        if step % self.validation_logging_step != 0:
            return
        if not is_rank0():
            return

        self._model_ref = model

        # Deep-copy condition tensors to CPU (they'll be overwritten by next val sample)
        condition = {}
        for key in ("text_embeds", "first_frame_cond", "clip_features", "audio_emb"):
            if key in data_batch:
                condition[key] = data_batch[key][:1].detach().clone().cpu()
        self._val_conditions.append(condition)
        logger.info(f"[val_step_end] stored condition {step}, total={len(self._val_conditions)}")

        if "real" in data_batch:
            self._val_real_data.append(data_batch["real"][:1].detach().clone().cpu())

        # Store audio path for muxing (if available from dataloader)
        audio_path = data_batch.get("audio_path")
        if audio_path is not None:
            # audio_path comes as a list/tuple from collation
            self._val_audio_paths.append(audio_path[0] if isinstance(audio_path, (list, tuple)) else audio_path)
        else:
            self._val_audio_paths.append(None)

    def on_validation_end(self, model, iteration: int = 0, idx: int = 0) -> None:
        """Log val loss + generate all videos AFTER DDP loop completes."""
        self.log_stats(self.val_loss_dict_record, iteration=iteration, group=f"val{idx}")

        if not is_rank0() or not self._val_conditions:
            self._val_conditions = []
            self._val_real_data = []
            self._val_audio_paths = []
            self._model_ref = None
            return

        if not wandb.run or not hasattr(model.net, "vae"):
            self._val_conditions = []
            self._val_real_data = []
            self._val_audio_paths = []
            self._model_ref = None
            return

        # Generate videos from stored conditions
        logger.info(f"[val_end] Generating {len(self._val_conditions)} val videos (deferred)...")
        device = model.device
        gen_videos = []

        for i, cond in enumerate(self._val_conditions):
            logger.info(f"[val_end] generating video {i+1}/{len(self._val_conditions)}...")
            try:
                # Move condition to GPU
                cond_gpu = {k: v.to(device) for k, v in cond.items()}

                # Generate noise matching the input shape
                input_shape = list(model.config.input_shape)
                noise = torch.randn(1, *input_shape, device=device, dtype=model.precision)

                # Run AR generation + VAE decode
                with torch.no_grad():
                    latent = model.generator_fn(
                        net=model.net_inference,
                        noise=noise,
                        condition=cond_gpu,
                        student_sample_steps=model.config.student_sample_steps,
                        t_list=model.config.sample_t_cfg.t_list,
                        precision_amp=model.precision_amp_infer,
                    )
                    video = model.net.vae.decode(latent[:1].float())
                    if isinstance(video, (list, tuple)):
                        video = video[0].unsqueeze(0) if len(video) == 1 else torch.stack(video)

                gen_videos.append(self._to_uint8_video(video))
                del cond_gpu, noise, latent, video
            except Exception as e:
                logger.warning(f"Failed to generate val video: {e}")
                import traceback
                traceback.print_exc()

        if gen_videos:
            # Build wandb Video objects, muxing audio when available
            gen_list = []
            tmp_files = []
            for vi, v in enumerate(gen_videos):
                video_np = v[0].numpy()  # [T, C, H, W] uint8
                audio_path = self._val_audio_paths[vi] if vi < len(self._val_audio_paths) else None
                if audio_path and os.path.exists(audio_path):
                    muxed = self._mux_video_audio(video_np, audio_path, self.audio_fps)
                    if muxed:
                        gen_list.append(wandb.Video(muxed, fps=self.audio_fps, format="mp4"))
                        tmp_files.append(muxed)
                        continue
                gen_list.append(wandb.Video(video_np, fps=self.audio_fps, format="mp4"))

            log_dict = {f"val{idx}/generated": gen_list}

            # Decode GT on first validation if not uploaded at init
            if not self._gt_uploaded and self._val_real_data:
                try:
                    gt_videos = []
                    with torch.no_grad(), basic_utils.inference_mode(
                        precision_amp=model.precision_amp_enc, device_type=device.type
                    ):
                        for real in self._val_real_data:
                            decoded = model.net.vae.decode(real.to(device).float())
                            gt_videos.append(self._to_uint8_video(decoded))
                    gt_list = []
                    for vi, v in enumerate(gt_videos):
                        video_np = v[0].numpy()
                        audio_path = self._val_audio_paths[vi] if vi < len(self._val_audio_paths) else None
                        if audio_path and os.path.exists(audio_path):
                            muxed = self._mux_video_audio(video_np, audio_path, self.audio_fps)
                            if muxed:
                                gt_list.append(wandb.Video(muxed, fps=self.audio_fps, format="mp4"))
                                tmp_files.append(muxed)
                                continue
                        gt_list.append(wandb.Video(video_np, fps=self.audio_fps, format="mp4"))
                    log_dict[f"val{idx}/reconstructed"] = gt_list
                    self._gt_uploaded = True
                except Exception as e:
                    logger.warning(f"Failed to decode GT: {e}")

            wandb.log(log_dict, step=iteration)
            logger.info(f"Logged {len(gen_videos)} val videos at iteration {iteration}")

            # Clean up temp muxed files
            for tf in tmp_files:
                try:
                    os.unlink(tf)
                except OSError:
                    pass

        self._val_conditions = []
        self._val_real_data = []
        self._val_audio_paths = []
        self._model_ref = None
        gc.collect()
        torch.cuda.empty_cache()

    def get_sample_map(self, model, data_batch, output_batch):
        """Training sample map — no synchronize (would deadlock)."""
        if not is_rank0():
            return {}
        if "gen_rand" not in output_batch:
            return {}

        sample_map = {}
        gen_rand = output_batch["gen_rand"]
        if isinstance(gen_rand, Callable):
            gen_rand = gen_rand()

        if wandb.run and gen_rand is not None:
            try:
                sample_map["student/generation"] = to_wandb(
                    gen_rand, fps=self.audio_fps, vid_format=self.vid_format
                )
            except Exception as e:
                logger.warning(f"Failed to log student generation: {e}")

        gc.collect()
        torch.cuda.empty_cache()
        return sample_map
