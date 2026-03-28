# SPDX-License-Identifier: Apache-2.0
"""
Production config for InfiniteTalk Diffusion Forcing — 8 GPU DDP, LoRA rank 128.

Usage (8 GPU):
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
    torchrun --nproc_per_node=8 train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_df_prod.py

Effective batch size: 8 GPUs × 1 sample/GPU × 8 grad_accum = 64
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


# ---- Paths (override via env vars) ----
WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
TRAIN_LIST = os.environ.get(
    "INFINITETALK_TRAIN_LIST",
    "data/precomputed_talkvid/train_list.txt",
)
VAL_LIST = os.environ.get(
    "INFINITETALK_VAL_LIST",
    "data/precomputed_talkvid/val_list.txt",
)
NEG_TEXT_EMB = os.environ.get(
    "INFINITETALK_NEG_TEXT_EMB",
    "data/precomputed_talkvid/neg_text_embeds.pt",
)
VAE_PATH = os.path.join(WEIGHTS_DIR, "Wan2.1_VAE.pth") if WEIGHTS_DIR else ""
AUDIO_DATA_ROOT = os.environ.get(
    "INFINITETALK_AUDIO_ROOT",
    "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data",
)


def create_config():
    config = create_base_config()

    # ── LoRA rank 128 ──
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    # ── VAE for visual sample logging ──
    config.model.vae_path = VAE_PATH

    # ── 8-GPU DDP (no FSDP — single-GPU fits in 77 GB) ──
    config.trainer.fsdp = False
    config.trainer.ddp = True

    # ── Gradient accumulation ──
    # BS=1/GPU × 8 accum × 8 GPUs = effective BS 64
    config.trainer.grad_accum_rounds = 8

    # ── Training dataloader — 2990 samples ──
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=TRAIN_LIST,
        neg_text_emb_path=NEG_TEXT_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        num_workers=4,
    )

    # ── Validation dataloader — 10 held-out samples with audio paths ──
    config.dataloader_val = L(InfiniteTalkDataLoader)(
        data_list_path=VAL_LIST,
        neg_text_emb_path=NEG_TEXT_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        audio_data_root=AUDIO_DATA_ROOT,
        num_workers=0,
    )

    # ── Training schedule ──
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 1        # log loss every step
    config.trainer.save_ckpt_iter = 500
    config.trainer.validation_iter = 100   # run val every 100 steps

    # ── Callbacks ──
    config.trainer.callbacks = {
        "wandb": L(InfiniteTalkWandbCallback)(
            sample_logging_iter=100,
            audio_fps=25,
        ),
        "train_profiler": L(TrainProfilerCallback)(every_n=100),
        "grad_clip": L(GradClipCallback)(grad_norm=10.0, model_key="net"),
    }

    # ── Logging ──
    lora_rank = config.model.net.lora_rank
    grad_accum = config.trainer.grad_accum_rounds
    config.log_config.project = "DF_InfiniteTalk"
    config.log_config.group = "infinitetalk_df"
    config.log_config.name = f"r{lora_rank}_a{config.model.net.lora_alpha}_accum{grad_accum}"
    config.log_config.wandb_mode = "online"

    return config
