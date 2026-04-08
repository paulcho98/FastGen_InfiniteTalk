# SPDX-License-Identifier: Apache-2.0
"""
WandbCallback for InfiniteTalk Self-Forcing training.

Logs SF-specific losses (total_loss, vsd_loss, fake_score_loss, gan_loss_gen)
every training step, generates AR sample videos at configurable intervals,
and handles deferred validation video generation to avoid deadlocks in DDP.

DDP constraint: never do expensive work (AR generation) between DDP-synced
validation forward passes. Store data during on_validation_step_end (instant),
generate videos in on_validation_end (after DDP loop).
"""

import gc
import warnings
from typing import Optional, Callable

import numpy as np
import torch
import wandb

from fastgen.callbacks.callback import Callback
from fastgen.utils.distributed import is_rank0
import fastgen.utils.logging_utils as logger


class InfiniteTalkSFWandbCallback(Callback):
    """WandB callback tailored for InfiniteTalk Self-Forcing distillation.

    Responsibilities:
      - Log scalar losses on every training step (rank 0 only).
      - At ``sample_logging_iter`` intervals, call the ``gen_rand`` callable
        from ``output_batch`` to produce an AR sample video and log it.
      - Collect validation ``gen_rand`` callables during ``on_validation_step_end``
        and generate / log videos in ``on_validation_end`` (deferred generation
        to stay safe under DDP synchronization).
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

        # Deferred validation storage
        self._val_gen_rand_fns: list[Callable] = []
        self._val_loss_dicts: list[dict[str, torch.Tensor]] = []

    def on_app_begin(self) -> None:
        """Initialize wandb run (rank 0 only)."""
        from fastgen.callbacks.wandb import init_wandb
        assert hasattr(self, "config"), "Missing config in InfiniteTalkSFWandbCallback."
        init_wandb(self.config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_wandb_video(
        tensor: torch.Tensor,
        fps: int = 25,
        vid_format: str = "mp4",
    ) -> Optional[wandb.Video]:
        """Convert a generated latent/pixel tensor to a ``wandb.Video``.

        Expects tensor of shape ``[B, C, T, H, W]`` or ``[B, T, C, H, W]``
        in ``[-1, 1]`` range.  Normalises to ``[0, 255]`` uint8,
        reorders to ``[T, H, W, C]`` numpy, and returns a ``wandb.Video``.
        """
        if tensor is None:
            return None

        if isinstance(tensor, (list, tuple)):
            tensor = torch.stack(tensor) if len(tensor) > 1 else tensor[0].unsqueeze(0)

        # Ensure float for arithmetic
        t = tensor.detach().float()

        # Guard: reject raw latents (16ch) — VAE decode must have failed
        if t.ndim == 5 and t.shape[1] != 3 and t.shape[2] != 3:
            warnings.warn(
                f"_tensor_to_wandb_video: got shape {list(t.shape)} which looks "
                f"like raw latents (expected 3 channels for pixel video). "
                f"VAE decode likely failed — skipping video logging.",
                stacklevel=2,
            )
            return None

        # Determine layout: [B, C, T, H, W] vs [B, T, C, H, W]
        # Heuristic: if dim 1 == 3, it is C-first
        if t.ndim == 5 and t.shape[1] == 3:
            # [B, C, T, H, W] -> [B, T, C, H, W]
            t = t.permute(0, 2, 1, 3, 4)

        # Take first sample in batch
        t = t[0]  # [T, C, H, W]

        # Normalise [-1, 1] -> [0, 255]
        t = t.mul(127.5).add(127.5).clamp(0, 255).to(torch.uint8).cpu()

        # [T, C, H, W] -> [T, H, W, C]
        video_np = t.permute(0, 2, 3, 1).numpy()

        return wandb.Video(video_np, fps=fps, format=vid_format)

    @staticmethod
    def _safe_item(val) -> float:
        """Extract a Python float from a tensor or scalar, safely."""
        if isinstance(val, torch.Tensor):
            return val.detach().float().item()
        return float(val)

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
            self._log_train_sample(output_batch, iteration)

    def _log_train_sample(self, output_batch: dict, iteration: int) -> None:
        """Generate and log a training sample video from output_batch['gen_rand']."""
        if not is_rank0() or not wandb.run:
            return
        if "gen_rand" not in output_batch:
            return

        try:
            gen_rand = output_batch["gen_rand"]
            if callable(gen_rand):
                with torch.no_grad():
                    gen_rand = gen_rand()

            if gen_rand is None:
                return

            video_obj = self._tensor_to_wandb_video(
                gen_rand, fps=self.audio_fps, vid_format=self.vid_format
            )
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
        """Collect validation gen_rand callables for deferred generation.

        This method runs inside the DDP-synced validation loop on ALL ranks,
        so it must be instant (no generation, no synchronize).
        """
        # Accumulate loss on all ranks (needed for distributed average)
        if loss_dict:
            self._val_loss_dicts.append(
                {k: v.detach().clone() for k, v in loss_dict.items()}
            )

        if step % self.validation_logging_step != 0:
            return
        if not is_rank0():
            return

        # Store the gen_rand callable/tensor for deferred generation
        if "gen_rand" in output_batch:
            gen_rand = output_batch["gen_rand"]
            if callable(gen_rand):
                # Store the callable itself; we will invoke it in on_validation_end
                self._val_gen_rand_fns.append(gen_rand)
            elif isinstance(gen_rand, torch.Tensor):
                # Wrap tensor in a lambda so on_validation_end has a uniform interface
                captured = gen_rand.detach().clone()
                self._val_gen_rand_fns.append(lambda _t=captured: _t)

    def on_validation_end(
        self,
        model,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        """Log validation losses and generate deferred sample videos."""
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

        # --- Generate and log deferred validation videos ---
        if not is_rank0() or not self._val_gen_rand_fns:
            self._val_gen_rand_fns = []
            return

        if not wandb.run:
            self._val_gen_rand_fns = []
            return

        logger.info(
            f"[val_end] Generating {len(self._val_gen_rand_fns)} "
            f"deferred val videos at iteration {iteration}..."
        )

        gen_videos: list[wandb.Video] = []
        for i, gen_fn in enumerate(self._val_gen_rand_fns):
            try:
                with torch.no_grad():
                    result = gen_fn()
                if result is None:
                    continue
                video_obj = self._tensor_to_wandb_video(
                    result, fps=self.audio_fps, vid_format=self.vid_format
                )
                if video_obj is not None:
                    gen_videos.append(video_obj)
            except Exception as e:
                warnings.warn(
                    f"InfiniteTalkSFWandbCallback: failed to generate val video "
                    f"{i}: {e}",
                    stacklevel=2,
                )

        if gen_videos:
            try:
                wandb.log(
                    {f"val{idx}/generated": gen_videos},
                    step=iteration,
                )
                logger.info(
                    f"Logged {len(gen_videos)} val videos at iteration {iteration}"
                )
            except Exception as e:
                warnings.warn(
                    f"InfiniteTalkSFWandbCallback: failed to upload val videos: {e}",
                    stacklevel=2,
                )

        self._val_gen_rand_fns = []
        gc.collect()
        torch.cuda.empty_cache()
