# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method config for InfiniteTalk Causal KD (Stage 1 of Self-Forcing pipeline).

Inherits from config_kd_causal (CausalKDModel config) which provides:
- ModelConfig with context_noise field
- student_sample_steps = 4 default
"""

import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.configs.methods.config_kd_causal import (
    Config as CausalKDConfig,
)
from fastgen.methods.infinitetalk_kd import InfiniteTalkKDModel
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
)


@attrs.define(slots=False)
class Config(CausalKDConfig):
    model_class: DictConfig = L(InfiniteTalkKDModel)(
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
