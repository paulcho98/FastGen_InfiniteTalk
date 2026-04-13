# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk Self-Forcing — Train-Time Anchor-Free + w9s1 attention

Inherits config_sf_w9s1 (student uses local_attn_size=10, sink_size=1) and
combines three anchor-mode flags:

    student_anchor_eval_only  = True   # student: no anchor in training,
                                       #          anchor in eval/inference
    fake_score_anchor_eval_only = True # fake_score: no anchor in training
                                       #             (not used in eval)
    teacher_anchor_disabled     = True # teacher: never anchors

Effect per phase:
    TRAINING:
        Student:    NO anchor (generates frame 0 from noise; learns via VSD gradient)
        Fake score: NO anchor (tracks student's anchor-free distribution)
        Teacher:    NO anchor (target distribution has anchor-free frame 0 too,
                               so VSD signal isn't contaminated by train/eval mismatch)
    VALIDATION / INFERENCE:
        Student:    ANCHOR (frame 0 replaced with reference image latent → I2V output)

Contrast with softboth:
    softboth        = student+fake_score eval-only; teacher still anchors always.
                      VSD target has clean frame 0 while student produces noisy
                      frame 0 in training → distribution mismatch.
    train-noanchor  = softboth + teacher_anchor_disabled. Teacher target is
                      consistent with student's anchor-free training output.

Usage:
    bash scripts/run_sf_w9s1_noanchor.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1 import (
    create_config as create_w9s1_config,
)


def create_config():
    config = create_w9s1_config()

    # ---- Train-time anchor-free: no model anchors during training;
    #      only the student anchors during eval/inference ----
    config.model.student_anchor_eval_only = True
    config.model.fake_score_anchor_eval_only = True
    config.model.teacher_anchor_disabled = True

    # ---- Logging: distinct group/name ----
    config.log_config.group = "infinitetalk_sf_w9s1_noanchor"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"sf_w9s1_noanchor_freq5_lr1e5_accum4_{timestamp}"
    config.log_config.name = run_name

    return config
