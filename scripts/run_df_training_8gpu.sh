#!/bin/bash
# InfiniteTalk Diffusion Forcing — 8-GPU DDP production training
#
# LoRA rank 128, alpha 64, LR 1e-5, grad_accum 4 → effective batch size 32
# 147K samples with lazy caching (T5 pre-computed, VAE/CLIP/audio on-the-fly)
# Wandb logging to project DF_InfiniteTalk
#
# Usage:
#   bash scripts/run_df_training_8gpu.sh
#
# Prerequisites:
#   1. T5 text embeddings pre-computed: bash scripts/run_precompute_t5.sh 0
#   2. Sample list built: python scripts/build_training_sample_list.py \
#        --csv_path /path/to/video_list.csv --output_dir data/precomputed_talkvid
#
# Override defaults via env vars:
#   INFINITETALK_TRAIN_LIST=... bash scripts/run_df_training_8gpu.sh

set -e

# ── Weights ──
export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

# ── Data ──
export INFINITETALK_TRAIN_LIST="${INFINITETALK_TRAIN_LIST:-data/precomputed_talkvid/all_viable_train.txt}"
export INFINITETALK_NEG_TEXT_EMB="${INFINITETALK_NEG_TEXT_EMB:-data/precomputed_talkvid/neg_text_embeds.pt}"
export INFINITETALK_CSV_PATH="${INFINITETALK_CSV_PATH:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv}"
export INFINITETALK_RAW_DATA_ROOT="${INFINITETALK_RAW_DATA_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/}"
export INFINITETALK_WAV2VEC_DIR="${INFINITETALK_WAV2VEC_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base}"
export INFINITETALK_AUDIO_ROOT="${INFINITETALK_AUDIO_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data}"

# ── Wandb ──
export WANDB_ENTITY="paulhcho"
export WANDB_API_KEY="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"

# ── DDP timeout (FlexAttention compile takes ~25 min on first iter) ──
export NCCL_TIMEOUT=1800

# ── Auto-detect GPUs ──
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "======================================"
echo "InfiniteTalk DF Training"
echo "======================================"
echo "GPUs:           $NUM_GPUS"
echo "Weights:        $INFINITETALK_WEIGHTS_DIR"
echo "Train list:     $INFINITETALK_TRAIN_LIST ($(wc -l < $INFINITETALK_TRAIN_LIST) samples)"
echo "Lazy caching:   enabled (raw_data_root=$INFINITETALK_RAW_DATA_ROOT)"
echo "NCCL timeout:   ${NCCL_TIMEOUT}s"
echo "Wandb project:  DF_InfiniteTalk"
echo "======================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_df_prod.py
