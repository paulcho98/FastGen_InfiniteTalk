# SPDX-License-Identifier: Apache-2.0
"""
2-GPU test: verify DDP + wandb + val visual generation doesn't hang.

Validation at iter 2 (after FlexAttention compile on iter 1).
Visual generation for 2 val samples with audio muxing.

Usage:
    NCCL_TIMEOUT=1800 \
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
    INFINITETALK_VAE_PATH=/.../Wan2.1_VAE.pth \
    torchrun --nproc_per_node=2 train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_df_2gpu_test.py
"""

import os
from fastgen.configs.experiments.InfiniteTalk.config_df import (
    create_config as create_base_config,
)
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.callbacks.infinitetalk_wandb import InfiniteTalkWandbCallback
from fastgen.callbacks.train_profiler import TrainProfilerCallback
from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.utils import LazyCall as L


WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
AUDIO_ROOT = os.environ.get(
    "INFINITETALK_AUDIO_ROOT",
    "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data",
)
TEST_LIST = os.environ.get("INFINITETALK_DATA_LIST", "data/test_precomputed/sample_list.txt")
NEG_EMB = os.environ.get("INFINITETALK_NEG_TEXT_EMB", "data/test_precomputed/neg_text_embeds.pt")


def create_config():
    config = create_base_config()

    # ── LoRA rank 128 ──
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64
    config.model.net_optimizer.lr = 1e-5

    # ── 2-GPU DDP ──
    config.trainer.fsdp = False
    config.trainer.ddp = True

    # ── 4 iterations: compile on iter 1, val+visual on iter 2 ──
    config.trainer.max_iter = 4
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 4
    config.trainer.grad_accum_rounds = 2

    # ── Validation at iter 2 (after compile), skip iter 0 ──
    config.trainer.validation_iter = 2
    config.trainer.skip_iter0_validation = True

    # ── Callbacks ──
    config.trainer.callbacks = {
        "wandb": L(InfiniteTalkWandbCallback)(
            sample_logging_iter=2,         # train visual at iter 2
            validation_logging_step=1,     # visual for each val sample
            audio_fps=25,
        ),
        "train_profiler": L(TrainProfilerCallback)(every_n=1),
        "grad_clip": L(GradClipCallback)(grad_norm=10.0, model_key="net"),
    }

    # ── Train dataloader ──
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=TEST_LIST,
        neg_text_emb_path=NEG_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        num_workers=0,
    )

    # ── Val dataloader (2 samples with audio) ──
    config.dataloader_val = L(InfiniteTalkDataLoader)(
        data_list_path=TEST_LIST,
        neg_text_emb_path=NEG_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        audio_data_root=AUDIO_ROOT,
        num_workers=0,
    )

    # ── Logging ──
    config.log_config.project = "DF_InfiniteTalk"
    config.log_config.group = "infinitetalk_df"
    config.log_config.name = "2gpu_val_test"
    config.log_config.wandb_mode = "online"

    return config
