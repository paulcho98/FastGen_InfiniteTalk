# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config for InfiniteTalk Diffusion Forcing (Stage 1 alternative to ODE KD).

Trains the causal 14B student (with LoRA) on real data with inhomogeneous
block-wise timesteps. No pre-computed ODE trajectories needed.

Network: CausalInfiniteTalkWan (14B with LoRA adapters, causal attention)
"""

import os
import fastgen.configs.methods.config_infinitetalk_df as config_df_default

from fastgen.utils import LazyCall as L
from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

# ---- Paths (override via env vars) ----
WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
BASE_MODEL_PATHS = ",".join([
    f"{WEIGHTS_DIR}/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
    for i in range(1, 8)
])
INFINITETALK_CKPT = os.environ.get("INFINITETALK_CKPT", "")
LORA_CKPT = os.environ.get("INFINITETALK_LORA_CKPT", "")
DATA_ROOT = os.environ.get("INFINITETALK_DATA_ROOT", "")

# ---- Student network config ----
CausalInfiniteTalk_14B_Student: dict = L(CausalInfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path=LORA_CKPT,
    lora_rank=32,
    lora_alpha=32,
    chunk_size=3,
    total_num_frames=21,
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)


def create_config():
    config = config_df_default.create_config()

    # Precision — bf16 throughout for 14B model
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "bfloat16"

    # Input shape: 448x896 @ 81 frames -> latent [16, 21, 56, 112]
    # (Most TalkVid videos are 16:9, bucket [448, 896] from ASPECT_RATIO_627)
    config.model.input_shape = [16, 21, 56, 112]

    # Student network
    config.model.net = CausalInfiniteTalk_14B_Student
    config.model.net.total_num_frames = config.model.input_shape[1]

    # Timestep schedule -- shift=7.0 matches InfiniteTalk's 480p distribution
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 7.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # Diffusion forcing settings
    config.model.student_sample_steps = 4

    # Training
    config.dataloader_train.batch_size = 1
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "infinitetalk_df"
    return config
