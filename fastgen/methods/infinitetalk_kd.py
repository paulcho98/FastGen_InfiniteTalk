# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InfiniteTalk Causal KD model for Stage 1 (ODE initialization).

Overrides single_train_step to build InfiniteTalk-specific condition dicts
from the precomputed ODE trajectory data.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, Callable
from functools import partial

import torch
import torch.nn.functional as F

from fastgen.methods.knowledge_distillation.KD import CausalKDModel
from fastgen.methods.distribution_matching.causvid import CausVidModel
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.config import BaseModelConfig as ModelConfig


class InfiniteTalkKDModel(CausalKDModel):
    """Causal KD for InfiniteTalk -- Stage 1 of Self-Forcing pipeline.

    Trains the causal student to match ODE trajectories from the teacher.
    Overrides single_train_step to handle InfiniteTalk's condition dict format
    (text embeddings, first frame conditioning, CLIP features, audio embeddings).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def build_model(self):
        """Build model then re-apply LoRA freeze.

        FastGenModel.build_model() calls requires_grad_(True) on the entire
        network, overriding freeze_base() from the constructor. Re-apply here.
        """
        super().build_model()
        from fastgen.networks.InfiniteTalk.lora import freeze_base
        freeze_base(self.net)

    def _build_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build InfiniteTalk condition dict from data batch.

        Args:
            data: Batch from InfiniteTalkDataset with ODE paths.

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
        """Single training step for InfiniteTalk causal KD.

        Builds the InfiniteTalk condition dict before running the standard
        CausalKDModel training logic (gather from ODE path, L2 loss).
        """
        denoise_path = data["path"]  # [B, num_steps, 16, 21, H, W]
        denoised_data = data["real"]  # [B, 16, 21, H, W]
        condition = self._build_condition(data)

        batch_size, num_frames = denoise_path.shape[0], denoise_path.shape[3]
        chunk_size = self.net.chunk_size

        # Sample inhomogeneous timesteps for causal training
        t_inhom, ids = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            num_frames,
            chunk_size,
            sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            device=self.device,
            dtype=denoise_path.dtype,
        )

        # Gather noisy data from path at sampled timesteps
        expand_shape = [ids.shape[0], 1, 1, ids.shape[1]] + [1] * max(0, denoise_path.ndim - 4)
        ids = ids.view(expand_shape).expand(-1, -1, *denoise_path.shape[2:])
        denoise_path_all = torch.cat([denoise_path, denoised_data.unsqueeze(1)], dim=1)
        noisy_data = torch.gather(denoise_path_all, 1, ids).squeeze(1)

        # Student forward
        gen_data = self.gen_data_from_net(noisy_data, t_inhom, condition=condition)

        # L2 loss
        loss = 0.5 * F.mse_loss(gen_data, denoised_data, reduction="mean")

        loss_map = {"total_loss": loss, "recon_loss": loss}
        outputs = self._get_outputs(gen_data, condition=condition)

        return loss_map, outputs
