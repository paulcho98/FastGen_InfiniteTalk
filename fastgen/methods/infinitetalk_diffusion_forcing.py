# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InfiniteTalk Diffusion Forcing model for Stage 1 initialization.

Alternative to ODE-based KD (InfiniteTalkKDModel). Instead of pre-computing
ODE trajectories from the teacher, this adds Gaussian noise to real data at
inhomogeneous block-wise timesteps and trains the student to denoise with L2 loss.
No teacher model or ODE generation needed.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, Callable
from functools import partial

import torch
import torch.nn.functional as F

from fastgen.methods.knowledge_distillation.KD import KDModel
from fastgen.methods.distribution_matching.causvid import CausVidModel
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.config import BaseModelConfig as ModelConfig


class InfiniteTalkDiffusionForcingModel(KDModel):
    """Diffusion Forcing on real data -- alternative to ODE KD for Stage 1.

    Adds noise to real data at inhomogeneous block-wise timesteps.
    Student denoises -> L2 loss vs clean data. No teacher ODE needed.

    Inheritance: InfiniteTalkDiffusionForcingModel -> KDModel -> FastGenModel
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _build_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build InfiniteTalk condition dict from data batch.

        Args:
            data: Batch from InfiniteTalkDataset.

        Returns:
            Condition dict for InfiniteTalk networks.
        """
        condition = {
            "text_embeds": data["text_embeds"],
            "first_frame_cond": data["first_frame_cond"],
            "clip_features": data["clip_features"],
            "audio_emb": data["audio_emb"],
        }
        return condition

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        noise = torch.randn_like(gen_data, dtype=self.precision)
        context_noise = getattr(self.config, "context_noise", 0)
        gen_rand_func = partial(
            CausVidModel.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            context_noise=context_noise,
            precision_amp=self.precision_amp_infer,
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """Single training step using diffusion forcing on real data.

        Instead of gathering from pre-computed ODE trajectories (as in CausalKDModel),
        this adds Gaussian noise to real data at inhomogeneous block-wise timesteps.
        """
        real_data = data["real"]  # [B, 16, 21, H, W]
        condition = self._build_condition(data)

        batch_size, num_frames = real_data.shape[0], real_data.shape[2]
        chunk_size = self.net.chunk_size

        # Sample inhomogeneous block-wise timesteps
        t_inhom, _ = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            num_frames,
            chunk_size,
            sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            device=self.device,
            dtype=real_data.dtype,
        )  # [B, T]

        # Diffusion forcing: add noise to real data at sampled timesteps
        eps = torch.randn_like(real_data)
        t_inhom_expanded = t_inhom[:, None, :, None, None]  # [B, 1, T, 1, 1]
        noisy_data = self.net.noise_scheduler.forward_process(real_data, eps, t_inhom_expanded)

        # Student denoise
        gen_data = self.gen_data_from_net(noisy_data, t_inhom, condition=condition)

        # L2 loss
        loss = 0.5 * F.mse_loss(gen_data, real_data, reduction="mean")

        loss_map = {"total_loss": loss, "recon_loss": loss}
        outputs = self._get_outputs(gen_data, condition=condition)

        return loss_map, outputs
