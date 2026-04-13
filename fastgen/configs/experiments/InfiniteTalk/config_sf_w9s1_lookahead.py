# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk Self-Forcing — w9s1 attention + Lookahead Sink + optional F2/F3.

Inherits config_sf_w9s1 (student local_attn_size=10, sink_size=1) and
flips the three feature gates:
  - use_dynamic_rope = True        (required by lookahead sink)
  - lookahead_sink_enabled = True
  - lookahead_distance = 4 (override via LOOKAHEAD_DISTANCE env var)
  - model_sink_cache_enabled via env MODEL_SINK_CACHE=1 (default off)
  - skip_clean_cache_pass via env SKIP_CLEAN_CACHE=1 (default off)

Usage:
    bash scripts/run_sf_w9s1_lookahead.sh
    LOOKAHEAD_DISTANCE=6 bash scripts/run_sf_w9s1_lookahead.sh
    MODEL_SINK_CACHE=1 SKIP_CLEAN_CACHE=1 bash scripts/run_sf_w9s1_lookahead.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1 import (
    create_config as create_w9s1_config,
)


def create_config():
    config = create_w9s1_config()

    # ---- Student network: enable dynamic RoPE (required for lookahead) ----
    config.model.net.use_dynamic_rope = True

    # ---- Feature 1: lookahead sink ----
    config.model.lookahead_sink_enabled = True
    config.model.lookahead_distance = int(os.environ.get("LOOKAHEAD_DISTANCE", "4"))

    # ---- Feature 2: model-generated sink cache ----
    config.model.model_sink_cache_enabled = bool(os.environ.get("MODEL_SINK_CACHE", ""))

    # ---- Feature 3: skip clean cache pass ----
    config.model.skip_clean_cache_pass = bool(os.environ.get("SKIP_CLEAN_CACHE", ""))

    # ---- Student network also needs matching lookahead kwargs passed at
    # ---- construction so _apply_anchor_config's runtime override has something
    # ---- compatible to work with (constructor will validate freqs+distance).
    config.model.net.lookahead_sink_enabled = config.model.lookahead_sink_enabled
    config.model.net.lookahead_distance = config.model.lookahead_distance

    # ---- Logging ----
    f2_tag = "_f2" if config.model.model_sink_cache_enabled else ""
    f3_tag = "_f3" if config.model.skip_clean_cache_pass else ""
    config.log_config.group = f"infinitetalk_sf_w9s1_lookahead{f2_tag}{f3_tag}"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        L = config.model.lookahead_distance
        run_name = f"sf_w9s1_la{L}{f2_tag}{f3_tag}_freq5_lr1e5_{timestamp}"
    config.log_config.name = run_name

    return config
