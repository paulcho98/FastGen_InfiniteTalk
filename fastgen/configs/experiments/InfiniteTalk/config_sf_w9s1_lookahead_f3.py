# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk Self-Forcing — w9s1 + Lookahead Sink (F1) + Skip Clean Cache (F3) baked.

Baked-in feature configuration (no env-var toggles for F2/F3):
  - use_dynamic_rope = True        (required by lookahead)
  - lookahead_sink_enabled = True  (F1 on)
  - model_sink_cache_enabled = False   (F2 OFF — sink K/V comes from clean
                                         first_frame_cond via the input anchor)
  - skip_clean_cache_pass = True       (F3 ON — training + validation + inference
                                         all skip the separate clean-cache pass
                                         and let the last denoise / exit step
                                         populate the KV cache directly)

All three anchors are ON (inherited from config_sf.py defaults):
  - student_anchor_eval_only = False
  - fake_score_anchor_eval_only = False
  - teacher_anchor_disabled = False

Env overrides retained:
  - LOOKAHEAD_DISTANCE (default 4)
  - LOOKAHEAD_DISTANCE_MIN / LOOKAHEAD_DISTANCE_MAX for stochastic distance

Usage:
    bash scripts/run_sf_w9s1_lookahead_f3.sh
    LOOKAHEAD_DISTANCE=6 bash scripts/run_sf_w9s1_lookahead_f3.sh
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

    # ---- Feature 1: lookahead sink (baked ON) ----
    config.model.lookahead_sink_enabled = True
    config.model.lookahead_distance = int(os.environ.get("LOOKAHEAD_DISTANCE", "4"))

    # Optional stochastic distance range (training only). Both env vars must be
    # set AND > 0 to activate; otherwise the fixed distance above is used for
    # every forward. Defaults (0, 0) = use fixed distance.
    config.model.lookahead_distance_min = int(os.environ.get("LOOKAHEAD_DISTANCE_MIN", "0"))
    config.model.lookahead_distance_max = int(os.environ.get("LOOKAHEAD_DISTANCE_MAX", "0"))

    # ---- Feature 2: model-sink-cache (baked OFF) ----
    # Sink K/V comes from the clean first_frame_cond latent (via the input
    # anchor firing on the sink chunk's exit step / cache pass). This is the
    # desired identity-preserving sink for anchors-on training.
    config.model.model_sink_cache_enabled = False

    # ---- Feature 3: skip-clean-cache-pass (baked ON) ----
    # Training: rollout_with_gradient stores KV during the exit step instead of
    # running a separate t=0 cache pass — saves one forward per block.
    # Validation / inference: _student_sample_loop and run_inference store KV
    # during the last denoise step at t=t_list[-2] (0.700 for the SF schedule).
    config.model.skip_clean_cache_pass = True

    # ---- Propagate lookahead settings to net constructor ----
    # The net's __init__ validates freqs+distance, so these must be present
    # even though _apply_anchor_config also stamps the runtime attrs.
    config.model.net.lookahead_sink_enabled = config.model.lookahead_sink_enabled
    config.model.net.lookahead_distance = config.model.lookahead_distance
    config.model.net.lookahead_distance_min = config.model.lookahead_distance_min
    config.model.net.lookahead_distance_max = config.model.lookahead_distance_max

    # ---- Logging ----
    stoch_tag = (
        f"_la{config.model.lookahead_distance_min}-{config.model.lookahead_distance_max}"
        if config.model.lookahead_distance_min > 0 and config.model.lookahead_distance_max > 0
        else f"_la{config.model.lookahead_distance}"
    )
    # All SF experiments share the canonical wandb group; run name carries tags.
    config.log_config.group = "infinitetalk_sf"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"sf_w9s1{stoch_tag}_f3_anchor_freq5_lr1e5_{timestamp}"
    config.log_config.name = run_name

    return config
