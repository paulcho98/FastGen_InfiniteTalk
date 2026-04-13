# SPDX-License-Identifier: Apache-2.0
"""
Validation-only variant of config_sf_w9s1_lookahead.py.

Iter 0 + iter 1 validation, no training.

Expected startup logs:
    [attn] Lookahead sink ENABLED, distance=4 frames
    [attn] Model-generated sink cache: <disabled|ENABLED>
    [attn] Skip clean cache pass: <disabled|ENABLED>

Because validation has cur_start_frame > 0 only on chunks past the sink, the
lookahead effect is only visible in chunks 1+. F2/F3 observable via wandb as
(a) frame-0 consistency with reference, (b) per-chunk forward count in logs.

Usage:
    bash scripts/run_sf_w9s1_lookahead_valtest.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_lookahead import (
    create_config as create_lookahead_config,
)


def create_config():
    config = create_lookahead_config()

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

    # Train dataloader is instantiated by the trainer even though we never
    # iterate it (max_iter=1, no training). Two things make this slow under
    # the default lookahead config:
    #  (1) InfiniteTalkDataLoader does a shape-check torch.load() on every
    #      sample's vae_latents.pt — with the full train list that's thousands
    #      of NFS reads. Fix: shrink the train list to the 2-sample val list.
    #  (2) Lazy caching (raw_data_root + csv_path + weights_dir + wav2vec_dir)
    #      loads 3 big encoders to CPU on every rank at init — VAE (~500 MB),
    #      CLIP (~GB), wav2vec (~400 MB) × 8 ranks. Takes ~20+ min over NFS.
    #      Fix: disable lazy caching for the valtest by setting raw_data_root
    #      to None (val samples have all precomputed files — no lazy needed).
    train_list = os.environ.get(
        "INFINITETALK_TRAIN_LIST",
        "data/precomputed_talkvid/val_quarter_2.txt",
    )
    if hasattr(config, "dataloader_train") and config.dataloader_train is not None:
        config.dataloader_train.data_list_path = train_list
        # Disable lazy caching — removes VAE/CLIP/wav2vec encoder loads at init.
        # InfiniteTalkDataLoader.__init__ gates lazy caching on `raw_data_root`
        # being non-None (see line 527 of infinitetalk_dataloader.py).
        config.dataloader_train.raw_data_root = None
        config.dataloader_train.csv_path = None
        config.dataloader_train.weights_dir = None
        config.dataloader_train.wav2vec_dir = None

    config.log_config.group = "infinitetalk_sf_w9s1_lookahead_valtest"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        L = config.model.lookahead_distance
        run_name = f"valtest_w9s1_la{L}_{timestamp}"
    config.log_config.name = run_name
    config.log_config.wandb_mode = "online"

    return config
