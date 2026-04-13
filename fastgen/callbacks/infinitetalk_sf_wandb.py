# SPDX-License-Identifier: Apache-2.0
"""
WandbCallback for InfiniteTalk Self-Forcing training.

Logs SF-specific losses (total_loss, vsd_loss, fake_score_loss, gan_loss_gen)
every training step, generates AR sample videos at configurable intervals,
and handles validation video logging with audio muxing + ground-truth reconstruction,
matching the DF callback's feature set.

FSDP constraint: validation_step runs AR generation + VAE decode on all ranks
(FSDP requires synchronized forward passes). Rank 0 captures the decoded pixel
tensor in on_validation_step_end, then muxes audio and logs in on_validation_end.
"""

import gc
import os
import subprocess
import tempfile
import warnings
from typing import Optional, Callable

import numpy as np
import torch
import wandb

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import synchronize, is_rank0
import fastgen.utils.logging_utils as logger


class InfiniteTalkSFWandbCallback(Callback):
    """WandB callback tailored for InfiniteTalk Self-Forcing distillation.

    Responsibilities:
      - Log scalar losses on every training step (rank 0 only).
      - At ``sample_logging_iter`` intervals, call the ``gen_rand`` callable
        from ``output_batch`` to produce an AR sample video and log it.
      - Upload ground-truth validation videos once at training start.
      - Capture pre-decoded pixel tensors during ``on_validation_step_end``
        and log them (with audio muxing + GT reconstruction) in
        ``on_validation_end``.
    """

    def __init__(
        self,
        *args,
        sample_logging_iter: int = 100,
        validation_logging_step: int = 1,
        audio_fps: int = 25,
        vid_format: str = "mp4",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sample_logging_iter = sample_logging_iter
        self.validation_logging_step = validation_logging_step
        self.audio_fps = audio_fps
        self.vid_format = vid_format

        # Deferred validation storage (rank 0 only)
        self._val_gen_pixels: list[torch.Tensor] = []   # pre-decoded pixel videos [1,C,T,H,W]
        self._val_gen_rand_fns: list[Callable] = []     # for callable path (training samples)
        self._val_real_data: list[torch.Tensor] = []    # GT latents for reconstruction
        self._val_audio_paths: list[Optional[str]] = []
        self._val_loss_dicts: list[dict[str, torch.Tensor]] = []
        self._gt_uploaded = False

    def on_app_begin(self) -> None:
        """Initialize wandb run (rank 0 only)."""
        from fastgen.callbacks.wandb import init_wandb
        assert hasattr(self, "config"), "Missing config in InfiniteTalkSFWandbCallback."
        init_wandb(self.config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mux_video_audio(video_uint8: np.ndarray, audio_path: str, fps: int) -> Optional[str]:
        """Write video+audio to a temp MP4 file via raw ffmpeg piping.

        Args:
            video_uint8: [T, C, H, W] uint8 numpy array
            audio_path: path to audio file (wav/m4a/etc.)
            fps: video frame rate
        Returns:
            Path to muxed MP4, or None on failure. Caller must clean up.
        """
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None

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
        """Convert pixel tensor in [-1, 1] to uint8 in [B, T, C, H, W] layout.

        Accepts [B, C, T, H, W] (standard VAE decode output) or a list/tuple of
        [C, T, H, W] tensors (also a VAE decode return flavor).
        """
        if isinstance(tensor, (list, tuple)):
            tensor = torch.stack(tensor) if len(tensor) > 1 else tensor[0].unsqueeze(0)
        # [B, C, T, H, W] → [B, T, C, H, W]
        t = tensor.permute(0, 2, 1, 3, 4)
        return t.mul(127.5).add(127.5).clamp(0, 255).to(torch.uint8).cpu()

    def _build_wandb_video(
        self,
        video_uint8_btchw: torch.Tensor,
        audio_path: Optional[str],
    ) -> tuple[Optional[wandb.Video], Optional[str]]:
        """Build a wandb.Video from an already-uint8 [B, T, C, H, W] tensor.

        Muxes audio if ``audio_path`` is provided and exists; otherwise logs
        silent video. Returns (wandb.Video, tmp_path_to_cleanup_or_None).
        """
        if video_uint8_btchw is None:
            return None, None
        # Take first sample in batch → [T, C, H, W]
        video_np = video_uint8_btchw[0].numpy()
        if audio_path and os.path.exists(audio_path):
            muxed = self._mux_video_audio(video_np, audio_path, self.audio_fps)
            if muxed:
                return wandb.Video(muxed, fps=self.audio_fps, format=self.vid_format), muxed
        # Fallback: silent video — wandb.Video expects [T, C, H, W]
        return wandb.Video(video_np, fps=self.audio_fps, format=self.vid_format), None

    @staticmethod
    def _safe_item(val) -> float:
        """Extract a Python float from a tensor or scalar, safely."""
        if isinstance(val, torch.Tensor):
            return val.detach().float().item()
        return float(val)

    # ------------------------------------------------------------------
    # GT upload at training start
    # ------------------------------------------------------------------

    def on_dataloader_init_end(
        self, model, dataloader_train, dataloader_val, iteration: int = 0,
    ) -> None:
        """Upload ground-truth validation videos at training start (rank 0 only).

        Lazy-loads the VAE via the model's ``_ensure_vae_loaded`` helper so we
        can decode the precomputed ``real`` latents. If the VAE can't load,
        defers GT upload to the first validation pass.
        """
        if dataloader_val is None:
            synchronize()
            return

        # Lazy-load VAE now so we can decode GT. All ranks attempt (each rank has its own VAE)
        # so that validation_step on non-rank-0 ranks doesn't trigger a second load.
        if hasattr(model, "_ensure_vae_loaded"):
            try:
                model._ensure_vae_loaded()
            except Exception as e:
                logger.warning(f"_ensure_vae_loaded during init failed: {e}")

        if not hasattr(model.net, "vae"):
            if is_rank0():
                logger.info("No VAE loaded — deferring GT val video upload to first validation")
            synchronize()
            return

        if is_rank0():
            logger.info("Uploading GT validation videos to wandb...")
            device = model.device
            try:
                gt_videos_uint8: list[torch.Tensor] = []
                audio_paths: list[Optional[str]] = []
                with torch.no_grad():
                    for step, data in enumerate(dataloader_val):
                        real = data["real"].to(device)
                        decoded = model.net.vae.decode(real[:1].float())
                        gt_videos_uint8.append(self._to_uint8_video(decoded))
                        ap = data.get("audio_path")
                        if ap is not None and isinstance(ap, (list, tuple)):
                            ap = ap[0]
                        audio_paths.append(ap)

                if wandb.run:
                    gt_list: list[wandb.Video] = []
                    tmp_files: list[str] = []
                    for vi, v in enumerate(gt_videos_uint8):
                        vid_obj, tmp = self._build_wandb_video(v, audio_paths[vi])
                        if vid_obj is not None:
                            gt_list.append(vid_obj)
                        if tmp:
                            tmp_files.append(tmp)
                    if gt_list:
                        wandb.log({"val_gt/videos": gt_list}, step=0)
                    for tf in tmp_files:
                        try:
                            os.unlink(tf)
                        except OSError:
                            pass
                self._gt_uploaded = True
                logger.info(f"Uploaded {len(gt_videos_uint8)} GT validation videos")
            except Exception as e:
                logger.warning(f"Failed to upload GT val videos: {e}")
                import traceback
                traceback.print_exc()
        synchronize()

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------

    def on_training_step_end(
        self,
        model,
        data_batch,
        output_batch,
        loss_dict,
        iteration: int = 0,
    ) -> None:
        """Log SF losses and optionally generate a sample video."""
        if not is_rank0():
            return

        # --- Scalar loss logging (every step) ---
        if loss_dict and wandb.run:
            log_payload = {}
            for key in ("total_loss", "vsd_loss", "fake_score_loss", "gan_loss_gen"):
                if key in loss_dict:
                    log_payload[f"train/{key}"] = self._safe_item(loss_dict[key])
            # Also log any other losses present
            for key, val in loss_dict.items():
                train_key = f"train/{key}"
                if train_key not in log_payload:
                    log_payload[train_key] = self._safe_item(val)
            if log_payload:
                wandb.log(log_payload, step=iteration)

        # --- Sample video logging at configured interval ---
        if iteration > 0 and iteration % self.sample_logging_iter == 0:
            self._log_train_sample(data_batch, output_batch, iteration)

    def _log_train_sample(self, data_batch: dict, output_batch: dict, iteration: int) -> None:
        """Generate and log a training sample video from output_batch['gen_rand']."""
        if not is_rank0() or not wandb.run:
            return
        if "gen_rand" not in output_batch:
            return

        tmp_path: Optional[str] = None
        try:
            gen_rand = output_batch["gen_rand"]
            if callable(gen_rand):
                with torch.no_grad():
                    gen_rand = gen_rand()

            if gen_rand is None:
                return

            uint8_vid = self._to_uint8_video(gen_rand)
            # Audio path for muxing (best-effort)
            audio_path = data_batch.get("audio_path") if isinstance(data_batch, dict) else None
            if audio_path is not None and isinstance(audio_path, (list, tuple)):
                audio_path = audio_path[0]

            video_obj, tmp_path = self._build_wandb_video(uint8_vid, audio_path)
            if video_obj is not None:
                wandb.log({"train_media/student_generation": video_obj}, step=iteration)
                logger.info(f"Logged training sample video at iteration {iteration}")
        except Exception as e:
            warnings.warn(
                f"InfiniteTalkSFWandbCallback: failed to log training sample "
                f"at iteration {iteration}: {e}",
                stacklevel=2,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def on_validation_step_end(
        self,
        model,
        data_batch,
        output_batch,
        loss_dict,
        step: int = 0,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        """Capture pre-decoded pixel tensors + GT + audio path for deferred logging.

        Runs on ALL ranks so losses can be aggregated. Only rank 0 stores
        visualization data.
        """
        # Accumulate loss on all ranks (needed for distributed average later)
        if loss_dict:
            self._val_loss_dicts.append(
                {k: v.detach().clone() for k, v in loss_dict.items()}
            )

        if step % self.validation_logging_step != 0:
            return
        if not is_rank0():
            return

        # --- Store generated pixel video / callable ---
        if "gen_rand" in output_batch:
            gen_rand = output_batch["gen_rand"]
            if isinstance(gen_rand, torch.Tensor):
                # Already-decoded pixel video [1, C, T, H, W] from validation_step
                self._val_gen_pixels.append(gen_rand.detach().clone().cpu())
            elif callable(gen_rand):
                # Callable path (unused for current SF validation, kept for safety)
                self._val_gen_rand_fns.append(gen_rand)

        # --- Store GT latent for reconstruction (only until GT is uploaded) ---
        if not self._gt_uploaded and "real" in data_batch:
            self._val_real_data.append(data_batch["real"][:1].detach().clone().cpu())

        # --- Store audio path for muxing ---
        audio_path = data_batch.get("audio_path")
        if audio_path is not None and isinstance(audio_path, (list, tuple)):
            audio_path = audio_path[0]
        self._val_audio_paths.append(audio_path)

    def on_validation_end(
        self,
        model,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        """Log validation losses, mux audio into generated videos, decode GT."""
        # --- Aggregate and log validation losses ---
        if self._val_loss_dicts and is_rank0() and wandb.run:
            try:
                avg_losses: dict[str, float] = {}
                counts: dict[str, int] = {}
                for ld in self._val_loss_dicts:
                    for k, v in ld.items():
                        avg_losses[k] = avg_losses.get(k, 0.0) + self._safe_item(v)
                        counts[k] = counts.get(k, 0) + 1
                log_payload = {
                    f"val{idx}/{k}": avg_losses[k] / counts[k]
                    for k in avg_losses
                }
                wandb.log(log_payload, step=iteration)
            except Exception as e:
                warnings.warn(
                    f"InfiniteTalkSFWandbCallback: failed to log val losses: {e}",
                    stacklevel=2,
                )
        self._val_loss_dicts = []

        # Early-out on non-rank-0 or when nothing to log
        if not is_rank0():
            self._reset_val_buffers()
            return
        if not wandb.run:
            self._reset_val_buffers()
            return
        if not self._val_gen_pixels and not self._val_gen_rand_fns:
            self._reset_val_buffers()
            return

        logger.info(
            f"[val_end] Logging {len(self._val_gen_pixels) + len(self._val_gen_rand_fns)} "
            f"val videos at iteration {iteration}..."
        )

        device = model.device
        gen_list: list[wandb.Video] = []
        tmp_files: list[str] = []

        # --- Pre-decoded pixel videos (SF's standard path via validation_step) ---
        for i, pixel_video in enumerate(self._val_gen_pixels):
            try:
                uint8_vid = self._to_uint8_video(pixel_video)
                audio_path = self._val_audio_paths[i] if i < len(self._val_audio_paths) else None
                vid_obj, tmp = self._build_wandb_video(uint8_vid, audio_path)
                if vid_obj is not None:
                    gen_list.append(vid_obj)
                if tmp:
                    tmp_files.append(tmp)
            except Exception as e:
                warnings.warn(
                    f"InfiniteTalkSFWandbCallback: failed to build gen video {i}: {e}",
                    stacklevel=2,
                )

        # --- Callable path (unused by current SF but supported for parity) ---
        pixel_offset = len(self._val_gen_pixels)
        for i, gen_fn in enumerate(self._val_gen_rand_fns):
            try:
                with torch.no_grad():
                    result = gen_fn()
                if result is None:
                    continue
                uint8_vid = self._to_uint8_video(result)
                audio_idx = pixel_offset + i
                audio_path = (
                    self._val_audio_paths[audio_idx]
                    if audio_idx < len(self._val_audio_paths) else None
                )
                vid_obj, tmp = self._build_wandb_video(uint8_vid, audio_path)
                if vid_obj is not None:
                    gen_list.append(vid_obj)
                if tmp:
                    tmp_files.append(tmp)
            except Exception as e:
                warnings.warn(
                    f"InfiniteTalkSFWandbCallback: failed to generate val video {i}: {e}",
                    stacklevel=2,
                )

        log_dict: dict = {}
        if gen_list:
            log_dict[f"val{idx}/generated"] = gen_list

        # --- GT reconstruction (first-time fallback if not uploaded at init) ---
        if not self._gt_uploaded and self._val_real_data:
            try:
                if not hasattr(model.net, "vae") and hasattr(model, "_ensure_vae_loaded"):
                    model._ensure_vae_loaded()
                if hasattr(model.net, "vae"):
                    gt_videos_uint8: list[torch.Tensor] = []
                    with torch.no_grad():
                        for real in self._val_real_data:
                            decoded = model.net.vae.decode(real.to(device).float())
                            gt_videos_uint8.append(self._to_uint8_video(decoded))
                    gt_list: list[wandb.Video] = []
                    for vi, v in enumerate(gt_videos_uint8):
                        audio_path = (
                            self._val_audio_paths[vi]
                            if vi < len(self._val_audio_paths) else None
                        )
                        vid_obj, tmp = self._build_wandb_video(v, audio_path)
                        if vid_obj is not None:
                            gt_list.append(vid_obj)
                        if tmp:
                            tmp_files.append(tmp)
                    if gt_list:
                        log_dict[f"val{idx}/reconstructed"] = gt_list
                    self._gt_uploaded = True
                else:
                    logger.warning("Failed to decode GT: VAE not available")
            except Exception as e:
                logger.warning(f"Failed to decode GT: {e}")
                import traceback
                traceback.print_exc()

        # --- Upload to wandb ---
        if log_dict:
            try:
                wandb.log(log_dict, step=iteration)
                logger.info(f"Logged {len(gen_list)} val videos at iteration {iteration}")
            except Exception as e:
                warnings.warn(
                    f"InfiniteTalkSFWandbCallback: failed to upload val videos: {e}",
                    stacklevel=2,
                )

        # Clean up temp muxed files
        for tf in tmp_files:
            try:
                os.unlink(tf)
            except OSError:
                pass

        self._reset_val_buffers()

    def _reset_val_buffers(self) -> None:
        self._val_gen_pixels = []
        self._val_gen_rand_fns = []
        self._val_real_data = []
        self._val_audio_paths = []
        gc.collect()
        torch.cuda.empty_cache()
