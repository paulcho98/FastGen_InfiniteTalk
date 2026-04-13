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
from fastgen.callbacks.infinitetalk_sf_wandb import InfiniteTalkSFWandbCallback


@attrs.define(slots=False)
class InfiniteTalkSFModelConfig(SFModelConfig):
    # Separate text and audio guidance scales for 3-call CFG
    text_guide_scale: float = 5.0
    audio_guide_scale: float = 4.0

    # Separate fake_score config (allows LoRA-based 14B fake_score distinct from teacher)
    fake_score_net: Optional[DictConfig] = None

    # Sequential FSDP init: build each model → FSDP-shard → free CPU → build next.
    # Required for 3x 14B models that exceed CPU RAM if built simultaneously.
    fsdp_sequential_init: bool = False

    # Student first-frame anchor mode:
    #   False (default): student anchors frame 0 always (I2V style)
    #   True: student anchors only in eval mode (inference/validation).
    #         During training rollout, no anchor → student learns frame 0 via VSD gradient.
    student_anchor_eval_only: bool = False

    # Fake score first-frame anchor mode:
    #   False (default): fake_score always anchors frame 0
    #   True: fake_score anchors only in eval mode — matches student's soft conditioning
    #         so it tracks the student's actual distribution (no clean frame 0 during training).
    #         Teacher always anchors regardless (represents target distribution).
    fake_score_anchor_eval_only: bool = False

    # Teacher first-frame anchor disable:
    #   False (default): teacher always anchors frame 0 (original I2V target distribution)
    #   True: teacher's _enable_first_frame_anchor is permanently set to False,
    #         so teacher never anchors — even during training VSD forwards.
    #
    # Typical use: combine with student_anchor_eval_only=True and
    # fake_score_anchor_eval_only=True to produce a train-time anchor-free
    # pipeline — no model anchors during training, and only the student anchors
    # during validation / inference. This keeps the VSD target distribution
    # consistent with the student's actual (non-anchored) training distribution,
    # while still producing I2V-style outputs at eval.
    #
    # Note: teacher is always in eval mode (frozen). Using _anchor_eval_only on it
    # would paradoxically keep it anchoring, so we use the hard-disable path instead.
    teacher_anchor_disabled: bool = False

    # Gradient accumulation rounds (mirrored from trainer config for combined step scaling)
    grad_accum_rounds: int = 1


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
            # Note: standard WANDB_CALLBACK replaced by InfiniteTalk-specific callback
            # that handles SF loss keys (vsd_loss, fake_score_loss) and AR video logging.
            "infinitetalk_sf_wandb": L(InfiniteTalkSFWandbCallback)(
                sample_logging_iter=100,
            ),
        }
    )

    config.dataloader_train.batch_size = 1
    config.model.student_sample_steps = 4
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.fake_score_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
