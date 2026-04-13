# SPDX-License-Identifier: Apache-2.0
"""
Validation-only variant of config_sf_w9s1_noanchor.py.

Runs validation at iter 0 and iter 1, then exits. No training updates.
Uses a 2-sample val list. Primary purpose: verify the train-time anchor-free
mode loads and behaves correctly end-to-end.

At startup, the logs must show exactly these lines (in order):
    [anchor] Student: eval-only (no anchor during training rollout)
    [anchor] Fake score: eval-only (tracks student distribution)
    [anchor] Teacher: DISABLED (anchor-free target distribution)

Note on expected validation output: during validation the student DOES anchor
(eval-only mode → anchor_active = not self.training = True), so frame 0 of
the generated video should closely match the reference image, just like
standard softanchor/softboth validation. The difference from baseline is
purely what happens during TRAINING, which this test does not exercise.
If frame 0 looks wildly off, that indicates a bug — student should still
anchor at eval.

Usage:
    bash scripts/run_sf_w9s1_noanchor_valtest.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_noanchor import (
    create_config as create_noanchor_config,
)


def create_config():
    config = create_noanchor_config()

    # ---- Validation-only schedule ----
    config.trainer.max_iter = 1
    config.trainer.skip_iter0_validation = False
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 2}]
    config.trainer.save_ckpt_iter = 99999

    # ---- Small val list ----
    val_list = os.environ.get(
        "INFINITETALK_VAL_LIST",
        "data/precomputed_talkvid/val_quarter_2.txt",
    )
    if hasattr(config, "dataloader_val") and config.dataloader_val is not None:
        config.dataloader_val.data_list_path = val_list

    # ---- Logging ----
    config.log_config.group = "infinitetalk_sf_w9s1_noanchor_valtest"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"valtest_w9s1_noanchor_{timestamp}"
    config.log_config.name = run_name
    config.log_config.wandb_mode = "online"

    return config
