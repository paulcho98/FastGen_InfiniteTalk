# SPDX-License-Identifier: Apache-2.0
"""
Validation-only variant of config_sf_w9s1.py for debugging the validation path.

Runs validation twice (at iter 0 and at max_iter=1) and exits. No gradient
updates occur — the training loop `range(iter_start+1, max_iter)` is empty
when iter_start=0 and max_iter=1, so all weights stay at the pretrained
state loaded from the DF checkpoint.

Uses a 2-sample val list (data/precomputed_talkvid/val_quarter_2.txt) and
MAX_VAL_STEPS=2 for fast turnaround. Total wall-clock ≈ FSDP init
(~5-8 min) + 2× validation (~1 min each) + final FSDP checkpoint save
(~2 min) ≈ 10-12 min end-to-end.

Usage:
    bash scripts/run_sf_w9s1_valtest.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1 import (
    create_config as create_w9s1_config,
)


def create_config():
    config = create_w9s1_config()

    # ---- Validation-only schedule ----
    config.trainer.max_iter = 1
    # Run validation at iter 0 (before any training) — overrides parent's True
    config.trainer.skip_iter0_validation = False
    # Process only 2 val samples per validation pass (matches val list size)
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 2}]
    # Disable per-iter checkpointing; trainer still saves once at max_iter
    config.trainer.save_ckpt_iter = 99999

    # ---- Small val list ----
    val_list = os.environ.get(
        "INFINITETALK_VAL_LIST",
        "data/precomputed_talkvid/val_quarter_2.txt",
    )
    if hasattr(config, "dataloader_val") and config.dataloader_val is not None:
        config.dataloader_val.data_list_path = val_list

    # ---- Logging: separate group to keep test runs out of production dashboards ----
    config.log_config.group = "infinitetalk_sf_w9s1_valtest"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"valtest_w9s1_{timestamp}"
    config.log_config.name = run_name
    # Online mode per test plan — verifies end-to-end wandb upload
    config.log_config.wandb_mode = "online"

    return config
