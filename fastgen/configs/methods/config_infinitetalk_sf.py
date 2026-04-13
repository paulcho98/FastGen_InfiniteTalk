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

    # Lookahead attention sink (Feature 1):
    # When True, the sink K/V (stored at buffer positions [0, sink_size) in
    # frames) is rotated at attention read-time with a RoPE temporal position
    # equal to F_window - 1 + lookahead_distance, placing it "in the future"
    # relative to the current generating block.
    # Requires use_dynamic_rope=True on the student network (the static-RoPE
    # path cannot retroactively shift cached-key positions).
    # Only takes effect in chunks past the sink (when k_win has a sink slab).
    lookahead_sink_enabled: bool = False

    # Distance in frames for lookahead sink. Only meaningful when
    # lookahead_sink_enabled=True. Typical values 1..8. Must be >= 1 when
    # lookahead_sink_enabled=True; validated at network construction time.
    # Used as the DETERMINISTIC distance when lookahead_distance_min /
    # lookahead_distance_max are both 0 (default). When the range is set AND
    # self.training=True, the model samples a fresh distance per forward and
    # ignores this field; when self.training=False (eval/inference), the fixed
    # lookahead_distance is used regardless of the range for reproducibility.
    lookahead_distance: int = 0

    # Stochastic lookahead distance range [min, max] inclusive. When both > 0,
    # trains with per-forward sampling of the sink's temporal position — makes
    # the model robust to lookahead distance variations at inference time.
    # Defaults (0, 0) disable sampling; training uses the fixed
    # lookahead_distance. Validated at network construction time:
    #   both > 0, min <= max, max+total_num_frames <= freqs[0] capacity.
    lookahead_distance_min: int = 0
    lookahead_distance_max: int = 0

    # Model-generated sink cache (Feature 2):
    # When True, the last denoise step of the sink chunk (cur_start_frame=0)
    # runs with apply_anchor=False, and its output is used as the input to a
    # subsequent cache-prefill forward pass so the cached sink K/V is computed
    # from the student's OWN frame-0 prediction (not the reference-image
    # overwrite). The displayed video still has the reference image at frame 0
    # (anchor applied manually outside forward). Inference-time only; no effect
    # during training because anchoring is gated on self.training.
    model_sink_cache_enabled: bool = False

    # Skip clean-cache pass (Feature 3):
    # When True, the separate cache-prefill forward pass after each chunk's
    # denoise loop is skipped. Instead, the last denoise step runs with
    # store_kv=True, so the K/V cached for that chunk comes from the (slightly
    # noisy) input to the last denoise step (at t=t_list[-2]).
    # Saves 1/(sample_steps+1) of inference forwards per chunk.
    # For the sink chunk, model_sink_cache_enabled=True overrides this setting
    # and keeps the separate cache pass alive (otherwise F2 cannot function).
    skip_clean_cache_pass: bool = False

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
