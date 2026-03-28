# SPDX-License-Identifier: Apache-2.0
"""
2-GPU test config for InfiniteTalk DF — verifies DDP + wandb + validation + visual logging.

Usage:
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
    torchrun --nproc_per_node=2 train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_df_2gpu_test.py
"""

import os
from fastgen.configs.experiments.InfiniteTalk.config_df_prod import (
    create_config as create_prod_config,
)
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.utils import LazyCall as L


def create_config():
    config = create_prod_config()

    # Override for quick test
    config.trainer.max_iter = 10
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 10
    config.trainer.grad_accum_rounds = 2
    config.trainer.validation_iter = 5   # validate at iter 5

    # Visual sample at iter 5
    config.trainer.callbacks["wandb"].sample_logging_iter = 5

    # Use test data (3 samples for train, same 3 for val — just testing the pipeline)
    TEST_LIST = os.environ.get("INFINITETALK_DATA_LIST", "data/test_precomputed/sample_list.txt")
    NEG_EMB = os.environ.get("INFINITETALK_NEG_TEXT_EMB", "data/test_precomputed/neg_text_embeds.pt")

    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=TEST_LIST,
        neg_text_emb_path=NEG_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        num_workers=0,
    )
    AUDIO_ROOT = os.environ.get(
        "INFINITETALK_AUDIO_ROOT",
        "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data",
    )
    config.dataloader_val = L(InfiniteTalkDataLoader)(
        data_list_path=TEST_LIST,
        neg_text_emb_path=NEG_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        audio_data_root=AUDIO_ROOT,
        num_workers=0,
    )

    config.log_config.name = "2gpu_test"

    return config
