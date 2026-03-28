# SPDX-License-Identifier: Apache-2.0
"""
Production config for InfiniteTalk Diffusion Forcing — 8 GPU DDP, LoRA rank 128.

Usage (8 GPU):
    INFINITETALK_WEIGHTS_DIR=/.../Wan2.1-I2V-14B-480P \
    INFINITETALK_CKPT=/.../infinitetalk.safetensors \
    INFINITETALK_VAE_PATH=/.../Wan2.1_VAE.pth \
    torchrun --nproc_per_node=8 train.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_df_prod.py

Effective batch size: 8 GPUs × 1 sample/GPU × 4 grad_accum = 32
Lazy caching: samples with text_embeds.pt are encoded on-the-fly (VAE/CLIP/audio).
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
    "data/precomputed_talkvid/all_viable_train.txt",
)
NEG_TEXT_EMB = os.environ.get(
    "INFINITETALK_NEG_TEXT_EMB",
    "data/precomputed_talkvid/neg_text_embeds.pt",
)
AUDIO_DATA_ROOT = os.environ.get(
    "INFINITETALK_AUDIO_ROOT",
    "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data",
)

# Lazy caching paths (for samples with only text_embeds.pt)
CSV_PATH = os.environ.get(
    "INFINITETALK_CSV_PATH",
    "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv",
)
RAW_DATA_ROOT = os.environ.get(
    "INFINITETALK_RAW_DATA_ROOT",
    "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/",
)
WAV2VEC_DIR = os.environ.get(
    "INFINITETALK_WAV2VEC_DIR",
    "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base",
)


def create_config():
    config = create_base_config()

    # ── LoRA rank 128 ──
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64

    # ── LR ──
    config.model.net_optimizer.lr = 1e-5

    # ── 8-GPU DDP (no FSDP — single-GPU fits in 77 GB) ──
    config.trainer.fsdp = False
    config.trainer.ddp = True

    # ── Gradient accumulation ──
    # BS=1/GPU × 4 accum × 8 GPUs = effective BS 32
    config.trainer.grad_accum_rounds = 4

    # ── Training dataloader — 147K samples with lazy caching ──
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=TRAIN_LIST,
        neg_text_emb_path=NEG_TEXT_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        num_video_frames=93,
        num_latent_frames=21,
        # Lazy caching: encode VAE/CLIP/audio on-the-fly for samples with only text_embeds.pt
        raw_data_root=RAW_DATA_ROOT,
        csv_path=CSV_PATH,
        weights_dir=WEIGHTS_DIR,
        wav2vec_dir=WAV2VEC_DIR,
        encode_device="cpu",  # CPU encoding avoids OOM (training uses ~77 GB of 80 GB)
        num_workers=0,  # required for lazy caching (encoders can't cross process boundaries)
    )

    # ── Validation disabled — DDP desync issues with visual generation ──
    # TODO: fix deferred val video generation to work with DDP
    # For now, training visuals at sample_logging_iter + training loss are sufficient.

    # ── Training schedule ──
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 1        # log loss every step
    config.trainer.save_ckpt_iter = 100
    config.trainer.validation_iter = 999999  # disabled (DDP desync with visual gen)
    config.trainer.skip_iter0_validation = True

    # ── Callbacks ──
    config.trainer.callbacks = {
        "wandb": L(InfiniteTalkWandbCallback)(
            sample_logging_iter=100,       # training visual every 100 steps
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
    config.log_config.name = f"r{lora_rank}_a{config.model.net.lora_alpha}_accum{grad_accum}_lr1e-5"
    config.log_config.wandb_mode = "online"

    return config
