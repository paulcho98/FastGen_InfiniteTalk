# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method config for InfiniteTalk Diffusion Forcing (Stage 1 alternative to ODE KD).

Inherits from base KD config (like OmniAvatar's DF pattern). Uses
InfiniteTalkDiffusionForcingModel which adds noise to real data at
inhomogeneous block-wise timesteps -- no teacher ODE trajectories needed.
"""

import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.configs.config import BaseConfig, BaseModelConfig
from fastgen.methods.infinitetalk_diffusion_forcing import InfiniteTalkDiffusionForcingModel
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
)


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    context_noise: float = 0.0


@attrs.define(slots=False)
class Config(BaseConfig):
    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(InfiniteTalkDiffusionForcingModel)(
        config=None,
    )


def create_config():
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    config.dataloader_train.batch_size = 1
    config.model.student_sample_steps = 4
    config.model.net_scheduler.warm_up_steps = [0]

    return config
