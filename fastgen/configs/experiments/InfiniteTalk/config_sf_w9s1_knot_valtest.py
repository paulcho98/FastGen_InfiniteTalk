# SPDX-License-Identifier: Apache-2.0
"""
Validation-only variant of config_sf_w9s1_knot_runahead.py.

Iter 0 + iter 1 validation, no training. Fast smoke test for the KF plumbing:
c+k denoise window, knot injection, Eq. 5 fusion, running-ahead advance,
last-frame reference swap.

Usage:
    bash scripts/run_sf_w9s1_knot_valtest.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_knot_runahead import (
    create_config as create_kf_config,
)


def create_config():
    config = create_kf_config()

    config.trainer.max_iter = 1
    config.trainer.skip_iter0_validation = False
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 2}]
    config.trainer.save_ckpt_iter = 99999

    val_list = os.environ.get(
        "INFINITETALK_VAL_LIST",
        "data/precomputed_talkvid/val_quarter_2.txt",
    )
    if hasattr(config, "dataloader_val") and config.dataloader_val is not None:
        config.dataloader_val.data_list_path = val_list

    # Shrink train list to val list + disable lazy caching (see
    # config_sf_w9s1_lookahead_valtest.py for rationale).
    train_list = os.environ.get(
        "INFINITETALK_TRAIN_LIST",
        "data/precomputed_talkvid/val_quarter_2.txt",
    )
    if hasattr(config, "dataloader_train") and config.dataloader_train is not None:
        config.dataloader_train.data_list_path = train_list
        config.dataloader_train.raw_data_root = None
        config.dataloader_train.csv_path = None
        config.dataloader_train.weights_dir = None
        config.dataloader_train.wav2vec_dir = None

    config.log_config.group = "infinitetalk_sf_w9s1_knot_valtest"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        k = config.model.knot_size
        s = config.model.running_ahead_step
        run_name = f"valtest_w9s1_knot_k{k}_s{s}_{timestamp}"
    config.log_config.name = run_name
    config.log_config.wandb_mode = "online"

    return config
