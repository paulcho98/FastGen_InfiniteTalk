# SPDX-License-Identifier: Apache-2.0
"""
Smoke-test config for InfiniteTalk Self-Forcing distillation (Stage 2).

Runs 3 iterations to exercise both training code paths:
  - Iterations 0, 2: student update (rollout_with_gradient, VSD loss)
  - Iteration 1:     fake_score update only
All three should complete without OOM, NaN, or crash.

Uses LoRA rank 8 (smaller than production rank 32) to reduce memory footprint.
No checkpoint saving, no wandb -- stdout logging only.

Usage (2-GPU, matching production FSDP setup):
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
    INFINITETALK_DATA_ROOT=data/test_precomputed \
    torchrun --nproc_per_node=2 train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_sf_test.py

Single-GPU (no FSDP/DDP):
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
    INFINITETALK_DATA_ROOT=data/test_precomputed \
    python train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_sf_test.py
"""

import os
from fastgen.configs.experiments.InfiniteTalk.config_sf import create_config as create_base_config
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.utils import LazyCall as L
from fastgen.callbacks.callback import Callback


class StdoutLoggerCallback(Callback):
    """Simple callback that prints loss to stdout at each training step."""

    def on_training_step_end(self, model=None, data_batch=None, output_batch=None, loss_dict=None, iteration=None, **kwargs):
        if loss_dict:
            parts = []
            for k, v in loss_dict.items():
                try:
                    parts.append(f"{k}={float(v):.6f}")
                except (TypeError, ValueError):
                    pass
            print(f"  [iter {iteration}] {', '.join(parts)}")
        else:
            print(f"  [iter {iteration}] (no loss_dict)")


def create_config():
    config = create_base_config()

    # --- Training: 3 iterations, no checkpoint, log every iter ---
    config.trainer.max_iter = 3
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 999999  # effectively disable checkpoint saving
    config.trainer.validation_iter = 999999  # effectively disable validation
    config.trainer.skip_iter0_validation = True

    # --- Callbacks: stdout only, no wandb ---
    config.trainer.callbacks = {"stdout": L(StdoutLoggerCallback)()}

    # --- LoRA rank 8 (smaller than production rank 32) for all three networks ---
    config.model.net.lora_rank = 8
    config.model.net.lora_alpha = 8
    config.model.teacher.lora_rank = 8
    config.model.teacher.lora_alpha = 8
    config.model.fake_score_net.lora_rank = 8
    config.model.fake_score_net.lora_alpha = 8

    # --- Dataloader: precomputed test data ---
    DATA_LIST = os.environ.get("INFINITETALK_DATA_LIST", "data/test_precomputed/sample_list.txt")
    NEG_TEXT_EMB = os.environ.get("INFINITETALK_NEG_TEXT_EMB", "data/test_precomputed/neg_text_embeds.pt")
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=DATA_LIST,
        neg_text_emb_path=NEG_TEXT_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        num_workers=0,
    )

    # --- Reduce gradient accumulation for quick test ---
    config.trainer.grad_accum_rounds = 1

    return config
