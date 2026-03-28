# SPDX-License-Identifier: Apache-2.0
"""
WandbCallback for InfiniteTalk — validation videos as lists, GT uploaded once.

Follows the OmniAvatar pattern:
  - on_dataloader_init_end: decode + upload GT val videos once
  - on_validation_step_end: AR-generate student video, collect into list
  - on_validation_end: log all val videos as a wandb list
  - get_sample_map: handles standalone DiT (no init_preprocessors)
"""

import gc
import os
from typing import Optional, Callable

import torch
import wandb

from fastgen.callbacks.wandb import WandbCallback, to_wandb
from fastgen.utils.distributed import rank0_only, synchronize, is_rank0
from fastgen.utils import basic_utils
import fastgen.utils.logging_utils as logger


class InfiniteTalkWandbCallback(WandbCallback):
    """WandbCallback with list-based validation logging and audio muxing."""

    def __init__(self, *args, audio_fps: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_fps = audio_fps
        self._val_gen_videos: list[torch.Tensor] = []
        self._val_gt_videos: list[torch.Tensor] = []
        self._gt_uploaded = False

    @staticmethod
    def _to_uint8_video(tensor: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, T, H, W] float [-1,1] → [B, T, C, H, W] uint8 CPU."""
        if isinstance(tensor, (list, tuple)):
            tensor = torch.stack(tensor) if len(tensor) > 1 else tensor[0].unsqueeze(0)
        t = tensor.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] → [B, T, C, H, W]
        return t.mul(127.5).add(127.5).clamp(0, 255).to(torch.uint8).cpu()

    def on_dataloader_init_end(
        self, model, dataloader_train, dataloader_val, iteration: int = 0
    ) -> None:
        """Upload GT validation videos once at the start.

        NOTE: NOT decorated with @rank0_only — all ranks must enter this method
        so they all hit synchronize() at the end. Only rank 0 does the actual
        VAE decode + wandb upload work. Without this, @rank0_only would cause
        non-rank0 processes to skip the barrier and desync NCCL.
        """
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
                    gt_list = [
                        wandb.Video(v[0].numpy(), fps=self.audio_fps, format="mp4")
                        for v in gt_videos
                    ]
                    wandb.log({"val_gt/videos": gt_list}, step=0)
                self._gt_uploaded = True
                logger.info(f"Uploaded {len(gt_videos)} GT validation videos to wandb")
            except Exception as e:
                logger.warning(f"Failed to upload GT val videos: {e}")
        synchronize()

    def on_validation_step_end(
        self,
        model,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor | Callable],
        loss_dict: dict[str, torch.Tensor],
        step: int = 0,
        iteration: int = 0,
        idx: int = 0,
    ) -> None:
        """Collect val loss + generate student video per sample."""
        self.val_loss_dict_record.add(loss_dict)

        if step % self.validation_logging_step != 0:
            return
        if not is_rank0():
            return
        if not hasattr(model.net, "vae"):
            return

        # AR-generate student video
        gen_rand = output_batch.get("gen_rand")
        if gen_rand is not None and isinstance(gen_rand, Callable):
            synchronize()
            gen_rand = gen_rand()
            synchronize()

        if gen_rand is None:
            return

        # gen_rand is pixel-space [B, C, T, H, W] from our _generate_and_decode callable
        self._val_gen_videos.append(self._to_uint8_video(gen_rand))

        # Also decode GT if not yet uploaded (first validation)
        if not self._gt_uploaded:
            device = model.device
            with torch.no_grad(), basic_utils.inference_mode(
                precision_amp=model.precision_amp_enc, device_type=device.type
            ):
                gt_decoded = model.net.vae.decode(data_batch["real"][:1].to(device).float())
            self._val_gt_videos.append(self._to_uint8_video(gt_decoded))

        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_end(self, model, iteration: int = 0, idx: int = 0) -> None:
        """Log all collected val videos as lists + val loss."""
        self.log_stats(self.val_loss_dict_record, iteration=iteration, group=f"val{idx}")

        if wandb.run and self._val_gen_videos and is_rank0():
            gen_list = [
                wandb.Video(v[0].numpy(), fps=self.audio_fps, format="mp4")
                for v in self._val_gen_videos
            ]
            log_dict = {f"val{idx}/generated": gen_list}

            if self._val_gt_videos:
                gt_list = [
                    wandb.Video(v[0].numpy(), fps=self.audio_fps, format="mp4")
                    for v in self._val_gt_videos
                ]
                log_dict[f"val{idx}/reconstructed"] = gt_list
                self._gt_uploaded = True

            wandb.log(log_dict, step=iteration)
            logger.info(f"Logged {len(self._val_gen_videos)} val videos at iteration {iteration}")

        self._val_gen_videos = []
        self._val_gt_videos = []

    def get_sample_map(self, model, data_batch, output_batch):
        """Training sample map — only called for sample_logging_iter.

        Returns empty since validation handles all visual logging.
        """
        if not is_rank0():
            return {}
        if "gen_rand" not in output_batch:
            return {}

        sample_map = {}
        gen_rand = output_batch["gen_rand"]

        if isinstance(gen_rand, Callable):
            synchronize()
            gen_rand = gen_rand()
            synchronize()

        if wandb.run and gen_rand is not None:
            try:
                sample_map["student/generation"] = to_wandb(
                    gen_rand, fps=self.audio_fps, vid_format=self.vid_format
                )
            except Exception as e:
                logger.warning(f"Failed to log student generation: {e}")

        synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        return sample_map
