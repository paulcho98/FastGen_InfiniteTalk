# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Method config for InfiniteTalk Self-Forcing distillation.

Extends the base Self-Forcing config with InfiniteTalk-specific fields:
- text_guide_scale: guidance scale for text conditioning (default 5.0)
- audio_guide_scale: guidance scale for audio conditioning (default 4.0)
- fake_score_net: separate fake_score network config (allows LoRA-based fake_score)

These enable the 3-call CFG used in InfiniteTalk's inference:
    output = uncond + text_scale * (cond - drop_text) + audio_scale * (drop_text - uncond)
"""

import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from typing import Optional

from fastgen.configs.methods.config_self_forcing import (
    Config as SFConfig,
    ModelConfig as SFModelConfig,
)
from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    ParamCount_CALLBACK,
    TrainProfiler_CALLBACK,
    GPUStats_CALLBACK,
    EMA_CALLBACK,
)


@attrs.define(slots=False)
class InfiniteTalkSFModelConfig(SFModelConfig):
    # Separate text and audio guidance scales for 3-call CFG
    text_guide_scale: float = 5.0
    audio_guide_scale: float = 4.0

    # Separate fake_score config (allows LoRA-based 14B fake_score distinct from teacher)
    fake_score_net: Optional[DictConfig] = None


@attrs.define(slots=False)
class Config(SFConfig):
    model: InfiniteTalkSFModelConfig = attrs.field(factory=InfiniteTalkSFModelConfig)
    model_class: DictConfig = L(InfiniteTalkSelfForcingModel)(
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
            **EMA_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    config.dataloader_train.batch_size = 1
    config.model.student_sample_steps = 4
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
