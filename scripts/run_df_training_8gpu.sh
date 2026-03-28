#!/bin/bash
# InfiniteTalk Diffusion Forcing — 8-GPU DDP production training
#
# LoRA rank 128, alpha 64, grad_accum 8 → effective batch size 64
# Wandb logging to project DF_InfiniteTalk
#
# Usage:
#   bash scripts/run_df_training_8gpu.sh
#
# Override defaults via env vars:
#   INFINITETALK_DATA_LIST=... bash scripts/run_df_training_8gpu.sh

set -e

# ── Weights ──
export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"

# ── Data ──
export INFINITETALK_DATA_LIST="${INFINITETALK_DATA_LIST:-data/precomputed_talkvid/precompute_sample_list.txt}"
export INFINITETALK_NEG_TEXT_EMB="${INFINITETALK_NEG_TEXT_EMB:-data/precomputed_talkvid/neg_text_embeds.pt}"

# ── Wandb ──
export WANDB_ENTITY="paulhcho"
export WANDB_API_KEY="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"

# ── Auto-detect GPUs ──
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"
echo "Weights: $INFINITETALK_WEIGHTS_DIR"
echo "Checkpoint: $INFINITETALK_CKPT"
echo "Data list: $INFINITETALK_DATA_LIST"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_df_prod.py
