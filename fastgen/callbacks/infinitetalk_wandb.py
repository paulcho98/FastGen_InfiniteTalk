# SPDX-License-Identifier: Apache-2.0
"""
WandbCallback for InfiniteTalk — list-based validation video logging.

DDP constraint: never do expensive work (AR generation) between DDP-synced
validation forward passes. Store data during on_validation_step_end (instant),
generate videos in on_validation_end (after DDP loop).
"""

import gc
import os
from typing import Optional, Callable
from functools import partial

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
        self._gt_uploaded = False
        self._model_ref = None  # set during validation

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

        if "real" in data_batch:
            self._val_real_data.append(data_batch["real"][:1].detach().clone().cpu())

    def on_validation_end(self, model, iteration: int = 0, idx: int = 0) -> None:
        """Log val loss + generate all videos AFTER DDP loop completes."""
        self.log_stats(self.val_loss_dict_record, iteration=iteration, group=f"val{idx}")

        if not is_rank0() or not self._val_conditions:
            self._val_conditions = []
            self._val_real_data = []
            self._model_ref = None
            return

        if not wandb.run or not hasattr(model.net, "vae"):
            self._val_conditions = []
            self._val_real_data = []
            self._model_ref = None
            return

        # Generate videos from stored conditions
        logger.info(f"Generating {len(self._val_conditions)} val videos (deferred)...")
        device = model.device
        gen_videos = []

        for cond in self._val_conditions:
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
            gen_list = [wandb.Video(v[0].numpy(), fps=self.audio_fps, format="mp4") for v in gen_videos]
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
                    log_dict[f"val{idx}/reconstructed"] = [
                        wandb.Video(v[0].numpy(), fps=self.audio_fps, format="mp4") for v in gt_videos
                    ]
                    self._gt_uploaded = True
                except Exception as e:
                    logger.warning(f"Failed to decode GT: {e}")

            wandb.log(log_dict, step=iteration)
            logger.info(f"Logged {len(gen_videos)} val videos at iteration {iteration}")

        self._val_conditions = []
        self._val_real_data = []
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
