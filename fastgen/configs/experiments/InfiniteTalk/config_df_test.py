# SPDX-License-Identifier: Apache-2.0
"""
Test config for InfiniteTalk Diffusion Forcing — 20 iterations, stdout logging.

Usage:
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
    INFINITETALK_DATA_ROOT=data/test_precomputed \
    python train.py --config fastgen/configs/experiments/InfiniteTalk/config_df_test.py
"""

import os
from fastgen.configs.experiments.InfiniteTalk.config_df import create_config as create_base_config
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.utils import LazyCall as L
from fastgen.callbacks.callback import Callback


class StdoutLoggerCallback(Callback):
    """Simple callback that prints loss to stdout at each training step."""

    def on_training_step_end(self, model=None, data_batch=None, output_batch=None, loss_dict=None, iteration=None, **kwargs):
        if loss_dict:
            loss_str = ", ".join(f"{k}={v:.6f}" for k, v in loss_dict.items() if isinstance(v, (int, float)))
            print(f"  [iter {iteration}] {loss_str}")
        else:
            print(f"  [iter {iteration}] (no loss_dict)")


def create_config():
    config = create_base_config()

    # Override for testing: 20 iterations
    config.trainer.max_iter = 20
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 10
    config.trainer.validation_iter = 999999  # effectively no validation

    # Use stdout logger, no wandb
    config.trainer.callbacks = {"stdout": L(StdoutLoggerCallback)()}

    # Single GPU — 14B with LoRA (294M trainable) fits in 80GB with grad checkpointing
    # Verified: forward 57GB, backward 68GB, optimizer 69GB — 11GB headroom
    config.trainer.fsdp = False
    config.trainer.ddp = False

    # Dataloader pointing to our precomputed test data (infinite iterator for trainer)
    DATA_LIST = os.environ.get("INFINITETALK_DATA_LIST", "data/test_precomputed/sample_list.txt")
    NEG_TEXT_EMB = os.environ.get("INFINITETALK_NEG_TEXT_EMB", "data/test_precomputed/neg_text_embeds.pt")
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=DATA_LIST,
        neg_text_emb_path=NEG_TEXT_EMB,
        batch_size=1,
        load_ode_path=False,
        num_workers=0,
    )

    # Reduce gradient accumulation for quick test
    config.trainer.grad_accum_rounds = 1

    return config
