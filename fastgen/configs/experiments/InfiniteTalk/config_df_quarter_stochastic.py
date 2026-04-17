# SPDX-License-Identifier: Apache-2.0
"""
Quarter-resolution DF config with stochastic sliding-window attention.

Extends config_df_quarter_prod with stochastic attention configs matching
OmniAvatar's config_df_shift_5: each forward pass randomly samples a
(local_attn_size, sink_size) pair from a weighted list, preventing the model
from over-relying on a specific window size.

Usage:
    bash scripts/run_df_training_quarter_stochastic.sh
    # or directly:
    torchrun --nproc_per_node=2 train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_df_quarter_stochastic.py
"""

import os
from fastgen.configs.experiments.InfiniteTalk.config_df_quarter_prod import (
    create_config as create_quarter_config,
)


def create_config():
    config = create_quarter_config()

    # ── Stochastic sliding-window attention (same configs as OmniAvatar) ──
    # Each forward pass randomly samples one config (equal 20% weight each).
    # local_attn_size = sink_size + rolling_window_frames (sink included in budget).
    # All configs use sink >= 1 for identity anchoring.
    config.model.net.stochastic_attn_configs = [
        {"local_attn_size": 7,  "sink_size": 1, "weight": 0.2},   # sink=1, window=6
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.2},   # sink=1, window=9
        {"local_attn_size": 13, "sink_size": 1, "weight": 0.2},   # sink=1, window=12
        {"local_attn_size": 9,  "sink_size": 3, "weight": 0.2},   # sink=3, window=6
        {"local_attn_size": 12, "sink_size": 3, "weight": 0.2},   # sink=3, window=9
    ]

    # ── Training schedule ──
    config.trainer.max_iter = 10000

    # ── Logging ──
    config.log_config.group = "infinitetalk_df_quarter_stochastic"
    lora_rank = config.model.net.lora_rank
    grad_accum = config.trainer.grad_accum_rounds
    num_gpus = int(os.environ.get("NUM_GPUS", "2"))
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        import time
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"quarter_stoch_r{lora_rank}_bs4_accum{grad_accum}_{num_gpus}gpu_{timestamp}"
    config.log_config.name = run_name

    return config
