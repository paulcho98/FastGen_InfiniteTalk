# SPDX-License-Identifier: Apache-2.0
"""
Quarter-resolution Diffusion Forcing config for InfiniteTalk.

Spatial dims halved: 448×896 → 224×448 pixel, 56×112 → 28×56 latent.
Token count: 21 × 14 × 28 = 8,232 (vs 32,928 at full res).
Expected VRAM: ~55-58 GB on single A100-80GB (vs ~70-77 GB full res).

Inherits everything from config_df_prod (LoRA rank 128, 8-GPU DDP, etc.)
and only overrides input_shape + enables quarter_res in the dataloaders.

Usage (8 GPU):
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
    INFINITETALK_VAE_PATH=/.../Wan2.1_VAE.pth \
    torchrun --nproc_per_node=8 train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_df_quarter.py

Requires: precompute with --quarter_res first (saves *_quarter.pt files).
"""

from fastgen.configs.experiments.InfiniteTalk.config_df_prod import create_config as create_prod_config


def create_config():
    config = create_prod_config()

    # Quarter-res latent shape: 224×448 pixel → 28×56 latent (via VAE stride 8)
    config.model.input_shape = [16, 21, 28, 56]

    # Dataloader: load *_quarter.pt files, filter by quarter-res spatial dims
    config.dataloader_train.quarter_res = True
    config.dataloader_train.expected_latent_shape = [16, 21, 28, 56]

    if hasattr(config, "dataloader_val") and config.dataloader_val is not None:
        config.dataloader_val.quarter_res = True
        config.dataloader_val.expected_latent_shape = [16, 21, 28, 56]

    # Logging group
    config.log_config.group = "infinitetalk_df_quarter"

    return config
