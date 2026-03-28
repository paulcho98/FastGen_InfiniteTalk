# SPDX-License-Identifier: Apache-2.0
"""
2-GPU test config for InfiniteTalk DF — full pipeline test:
DDP + wandb loss + visual samples with audio.

Validation at iter 3 (after FlexAttention compile on iter 1).
No iter-0 validation (causes torch.compile hang with DDP).

Usage:
    NCCL_TIMEOUT=1800 \
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
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
VAE_PATH = os.path.join(WEIGHTS_DIR, "Wan2.1_VAE.pth") if WEIGHTS_DIR else ""
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

    # ── VAE for visual sample decoding (loaded lazily after torch.compile) ──
    config.model.vae_path = VAE_PATH

    # ── 2-GPU DDP ──
    config.trainer.fsdp = False
    config.trainer.ddp = True

    # ── 8 iterations: compile on iter 1, val+visual on iter 3 ──
    config.trainer.max_iter = 8
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 8
    config.trainer.grad_accum_rounds = 2

    # ── Validation at iter 3 only (no iter-0 val — no dataloader_val) ──
    # Iter-0 val with DDP causes torch.compile to hang.
    # Instead: train iters 1-2 (compile happens here), then val at iter 3.
    config.trainer.validation_iter = 3

    # ── Callbacks ──
    config.trainer.callbacks = {
        "wandb": L(InfiniteTalkWandbCallback)(
            sample_logging_iter=3,         # train visual at iter 3
            validation_logging_step=1,     # val visual for each sample
            audio_fps=25,
        ),
        "train_profiler": L(TrainProfilerCallback)(every_n=1),
        "grad_clip": L(GradClipCallback)(grad_norm=10.0, model_key="net"),
    }

    # ── Train dataloader (3 test samples) ──
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=TEST_LIST,
        neg_text_emb_path=NEG_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        num_workers=0,
    )

    # ── No dataloader_val — skips ALL validation (iter-0 val hangs torch.compile) ──
    # Visual samples come from training at sample_logging_iter=3 instead.
    # TODO: fix iter-0 val hang, then re-enable dataloader_val for production.

    # ── Logging ──
    config.log_config.project = "DF_InfiniteTalk"
    config.log_config.group = "infinitetalk_df"
    config.log_config.name = "2gpu_full_test"
    config.log_config.wandb_mode = "online"

    return config
