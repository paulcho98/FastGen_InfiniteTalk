# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config for InfiniteTalk Self-Forcing distillation (Stage 2).

14B teacher (bidirectional, frozen) -> 14B student (causal, LoRA) using Self-Forcing.
Also uses a 14B fake_score (bidirectional, LoRA) for VSD loss.

Unlike OmniAvatar (14B->1.3B), InfiniteTalk uses the same 14B architecture for
all three roles. LoRA adapters make training feasible:
  - Teacher:    base Wan I2V + InfiniteTalk ckpt + LoRA merged, fully frozen
  - Student:    causal variant with LoRA adapters (trainable)
  - Fake score: bidirectional with LoRA adapters (trainable)

3-call CFG with separate text (5.0) and audio (4.0) guidance scales.
"""

import os
import fastgen.configs.methods.config_infinitetalk_sf as config_sf_default

from fastgen.utils import LazyCall as L

from fastgen.networks.InfiniteTalk.network import InfiniteTalkWan
from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader

# ---- Paths (override via CLI or env) ----
WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
BASE_MODEL_PATHS = ",".join([
    f"{WEIGHTS_DIR}/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
    for i in range(1, 8)
])
INFINITETALK_CKPT = os.environ.get("INFINITETALK_CKPT", "")
TEACHER_LORA_CKPT = os.environ.get("INFINITETALK_TEACHER_LORA_CKPT", "")
STUDENT_LORA_CKPT = os.environ.get("INFINITETALK_STUDENT_LORA_CKPT", "")
DATA_ROOT = os.environ.get("INFINITETALK_DATA_ROOT", "")

# ---- Network configs ----

# Teacher: 14B bidirectional, frozen (base + InfiniteTalk ckpt + LoRA merged)
InfiniteTalk_14B_Teacher: dict = L(InfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path=TEACHER_LORA_CKPT,
    lora_rank=32,
    lora_alpha=32,
    apply_lora_adapters=False,  # Teacher: merge LoRA, freeze all
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)

# Fake score: 14B bidirectional with runtime LoRA adapters (trainable)
InfiniteTalk_14B_FakeScore: dict = L(InfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path="",  # No pre-trained LoRA for fake score
    lora_rank=32,
    lora_alpha=32,
    apply_lora_adapters=True,  # Fake score: runtime LoRA, trainable
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)

# Student: 14B causal with LoRA adapters (trainable)
CausalInfiniteTalk_14B_Student: dict = L(CausalInfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path=STUDENT_LORA_CKPT,
    lora_rank=32,
    lora_alpha=32,
    chunk_size=3,
    total_num_frames=21,
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)


def create_config():
    config = config_sf_default.create_config()

    # Learning rates
    config.model.net_optimizer.lr = 5e-6
    config.model.fake_score_optimizer.lr = 5e-6

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # Input shape: 640x640 @ 81 frames -> latent [16, 21, 80, 80]
    config.model.input_shape = [16, 21, 80, 80]
    config.model.fake_score_pred_type = "x0"

    # 3-call CFG with separate text and audio guidance scales
    # Note: guidance_scale is set to None to disable the base 2-call CFG path;
    # the 3-call CFG is handled entirely by InfiniteTalkSelfForcingModel._apply_classifier_free_guidance
    config.model.guidance_scale = 5.0  # Triggers CFG in base class; our override replaces the logic
    config.model.text_guide_scale = 5.0
    config.model.audio_guide_scale = 4.0

    # Networks: 14B teacher + 14B student (causal LoRA) + 14B fake_score (bidir LoRA)
    config.model.net = CausalInfiniteTalk_14B_Student
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.teacher = InfiniteTalk_14B_Teacher
    config.model.fake_score_net = InfiniteTalk_14B_FakeScore

    # GAN disabled by default to save VRAM (matching T2V 14B teacher config pattern)
    config.model.gan_loss_weight_gen = 0
    config.model.student_update_freq = 2

    # Student weights: let the network's own __init__ handle loading (base + ckpt + LoRA).
    # Do NOT copy teacher weights onto student (both are 14B but teacher is bidir, student is causal).
    config.model.load_student_weights = False

    # Timestep schedule -- shift=7.0 matches InfiniteTalk's 480p distribution
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 7.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Self-Forcing specific
    config.model.enable_gradient_in_rollout = True
    config.model.start_gradient_frame = 0
    config.model.same_step_across_blocks = True
    config.model.context_noise = 0.0

    # Dataloader
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=f"{DATA_ROOT}/sample_list.txt",
        batch_size=1,
        num_workers=4,
        neg_text_emb_path=os.environ.get("INFINITETALK_NEG_TEXT_EMB_PATH", None),
    )

    # Training
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "infinitetalk_sf"
    return config
