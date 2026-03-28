# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config for InfiniteTalk Causal KD (Stage 1 of Self-Forcing pipeline).

Pre-trains the causal 14B student (with LoRA) on ODE trajectories from the teacher.

Network: CausalInfiniteTalkWan (14B with LoRA adapters, causal attention)
Dataloader: InfiniteTalkDataLoader with load_ode_path=True
"""

import os
import fastgen.configs.methods.config_infinitetalk_kd as config_kd_default

from fastgen.utils import LazyCall as L
from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader

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
    config = config_kd_default.create_config()

    # Precision
    config.model.precision = "bfloat16"
    config.model.precision_fsdp = "float32"

    # Input shape: 448x896 @ 81 frames -> latent [16, 21, 56, 112]
    config.model.input_shape = [16, 21, 56, 112]

    # Student network
    config.model.net = CausalInfiniteTalk_14B_Student
    config.model.net.total_num_frames = config.model.input_shape[1]

    # Timestep schedule -- shift=7.0 matches InfiniteTalk's 480p distribution
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 7.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    # t_list derived from shift=7.0: new_t = 7*t / (1 + 6*t) applied to linspace(1,0,5)
    config.model.sample_t_cfg.t_list = [0.999, 0.955, 0.875, 0.700, 0.0]

    # KD settings
    config.model.student_sample_steps = 4

    # Dataloader -- KD uses ODE trajectory data (includes "path" key)
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=f"{DATA_ROOT}/sample_list.txt",
        batch_size=1,
        num_workers=4,
        neg_text_emb_path=os.environ.get("INFINITETALK_NEG_TEXT_EMB_PATH", None),
        load_ode_path=True,
    )

    # Training
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 10
    config.trainer.save_ckpt_iter = 500

    config.log_config.group = "infinitetalk_kd"
    return config
