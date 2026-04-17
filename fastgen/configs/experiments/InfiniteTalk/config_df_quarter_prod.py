# SPDX-License-Identifier: Apache-2.0
"""
Quarter-resolution DF production config — auto-detects GPU count.

Spatial dims halved: 448×896 → 224×448 pixel, 56×112 → 28×56 latent.
Token count: 21 × 14 × 28 = 8,232 (vs 32,928 at full res).

Measured memory: 67.9 GB peak at BS=4 on A100-80GB (forward+backward+step).
  BS=4 is the max safe batch size per GPU.

With 8 GPUs: BS=4, accum=1 → effective BS 32 (no accum overhead!)
With 2 GPUs: BS=4, accum=4 → effective BS 32

Uses train/val split with 30 non-overlapping val videos, quarter_res=True
to load *_quarter.pt files. Lazy caching encodes at quarter-res on-the-fly.

Usage:
    bash scripts/run_df_training_quarter.sh
"""

import os
from fastgen.configs.experiments.InfiniteTalk.config_df_prod import create_config as create_prod_config


def create_config():
    config = create_prod_config()

    # ── Quarter-res latent shape ──
    config.model.input_shape = [16, 21, 28, 56]

    # ── Batch size: 4 per GPU (measured max on A100-80GB = 67.9GB peak) ──
    config.dataloader_train.batch_size = 4

    # ── Dataloaders: load *_quarter.pt files ──
    config.dataloader_train.quarter_res = True
    config.dataloader_train.expected_latent_shape = [16, 21, 28, 56]

    # Use the val30 split (30 samples from non-overlapping videos)
    val_list = os.environ.get(
        "INFINITETALK_VAL_LIST",
        "data/precomputed_talkvid/val_quarter_30.txt",
    )
    # Use training list with val videos excluded
    train_list = os.environ.get(
        "INFINITETALK_TRAIN_LIST",
        "data/precomputed_talkvid/train_excl_val30.txt",
    )
    config.dataloader_train.data_list_path = train_list

    if hasattr(config, "dataloader_val") and config.dataloader_val is not None:
        config.dataloader_val.quarter_res = True
        config.dataloader_val.expected_latent_shape = [16, 21, 28, 56]
        config.dataloader_val.data_list_path = val_list
        config.dataloader_val.batch_size = 4

    # ── Validation: 10 videos logged per validation round ──
    config.trainer.validation_iter = 200
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 10}]

    # ── Grad accumulation: auto-adjust for GPU count ──
    # Target effective batch size = 32
    # With N GPUs and BS=4: accum = 32 / (4 * N)
    # 8 GPUs: accum=1, 4 GPUs: accum=2, 2 GPUs: accum=4
    num_gpus = int(os.environ.get("NUM_GPUS", "2"))
    config.trainer.grad_accum_rounds = max(1, 32 // (4 * num_gpus))

    # ── Logging ──
    # Use FASTGEN_RUN_NAME env var for resume support (same name → same output dir → finds checkpoint).
    # If not set, generate a timestamped name for fresh runs.
    config.log_config.group = "infinitetalk_df_quarter"
    lora_rank = config.model.net.lora_rank
    grad_accum = config.trainer.grad_accum_rounds
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        import time
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"quarter_r{lora_rank}_bs4_accum{grad_accum}_{num_gpus}gpu_{timestamp}"
    config.log_config.name = run_name

    return config
