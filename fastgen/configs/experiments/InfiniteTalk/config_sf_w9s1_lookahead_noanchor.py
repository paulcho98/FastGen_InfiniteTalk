# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk Self-Forcing — w9s1 attention + Lookahead Sink + Train-Time Anchor-Free
(+ optional F2/F3).

Combines two orthogonal feature sets:

1. From config_sf_w9s1_lookahead.py:
   - Student attention: local_attn_size=10, sink_size=1 (inherited via config_sf_w9s1)
   - use_dynamic_rope=True (required by lookahead)
   - Lookahead sink enabled, distance=4 (tunable via LOOKAHEAD_DISTANCE)
   - F2 (model-generated sink cache) opt-in via MODEL_SINK_CACHE=1
   - F3 (skip clean cache pass) opt-in via SKIP_CLEAN_CACHE=1

2. From config_sf_w9s1_noanchor.py (train-time anchor-free on ALL three models):
   - student_anchor_eval_only=True — student anchors only at eval/inference
   - fake_score_anchor_eval_only=True — fake_score never anchors during training
   - teacher_anchor_disabled=True — teacher PERMANENTLY off (incl. training)

Effect per phase:
    TRAINING:
        Student:     NO anchor (generates frame 0 from noise; learns via VSD)
        Fake score:  NO anchor (tracks student's anchor-free distribution)
        Teacher:     NO anchor (VSD target is consistent with student distribution)
    VALIDATION / INFERENCE:
        Student:     ANCHOR (reference image replaces frame 0 → I2V output)

Usage:
    bash scripts/run_sf_w9s1_lookahead_noanchor.sh
    LOOKAHEAD_DISTANCE=6 bash scripts/run_sf_w9s1_lookahead_noanchor.sh
    MODEL_SINK_CACHE=1 SKIP_CLEAN_CACHE=1 bash scripts/run_sf_w9s1_lookahead_noanchor.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_lookahead import (
    create_config as create_lookahead_config,
)


def create_config():
    config = create_lookahead_config()

    # ---- Train-time anchor-free: no model anchors during training;
    #      only the student anchors during eval/inference ----
    config.model.student_anchor_eval_only = True
    config.model.fake_score_anchor_eval_only = True
    config.model.teacher_anchor_disabled = True

    # ---- Logging: distinct group/name to isolate this experiment ----
    f2_tag = "_f2" if config.model.model_sink_cache_enabled else ""
    f3_tag = "_f3" if config.model.skip_clean_cache_pass else ""
    # Distance tag reflects stochastic mode when range is set, else the fixed distance.
    if config.model.lookahead_distance_min > 0 and config.model.lookahead_distance_max > 0:
        la_tag = f"la{config.model.lookahead_distance_min}-{config.model.lookahead_distance_max}"
    else:
        la_tag = f"la{config.model.lookahead_distance}"
    config.log_config.group = f"infinitetalk_sf_w9s1_lookahead_noanchor{f2_tag}{f3_tag}"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"sf_w9s1_{la_tag}_noanchor{f2_tag}{f3_tag}_freq5_lr1e5_{timestamp}"
    config.log_config.name = run_name

    return config
