# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk Self-Forcing — Fixed Attention (window=9, sink=1)

Student uses causal sliding-window attention with 1 sink frame + 9-frame
rolling context (local_attn_size=10 total). Matches one of the five attention
configs used by config_df_quarter_stochastic.py:

    {"local_attn_size": 10, "sink_size": 1, "weight": 0.2}

Teacher and fake_score remain bidirectional (full attention).

Usage:
    bash scripts/run_sf_w9s1.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf import (
    create_config as create_sf_config,
)

# Student attention settings (fixed, non-stochastic)
STUDENT_LOCAL_ATTN_SIZE = 10   # total frames attended = sink_size + rolling window
STUDENT_SINK_SIZE = 1          # identity-anchor frame kept in window


def create_config():
    config = create_sf_config()

    # ---- Override student attention ----
    config.model.net.local_attn_size = STUDENT_LOCAL_ATTN_SIZE
    config.model.net.sink_size = STUDENT_SINK_SIZE

    # ---- Logging: distinct group/name so this experiment is isolated ----
    config.log_config.group = "infinitetalk_sf_w9s1"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        if config.model.student_anchor_eval_only and config.model.fake_score_anchor_eval_only:
            anchor_tag = "softboth"
        elif config.model.student_anchor_eval_only:
            anchor_tag = "softanchor"
        else:
            anchor_tag = "i2v"
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"sf_w9s1_{anchor_tag}_freq5_lr1e5_accum4_{timestamp}"
    config.log_config.name = run_name

    return config
