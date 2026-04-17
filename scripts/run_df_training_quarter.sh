#!/bin/bash
# InfiniteTalk Diffusion Forcing — Quarter Resolution Training
#
# Spatial dims halved: 448×896 → 224×448 pixel, 56×112 → 28×56 latent
# Token count: 8,232 (vs 32,928 full res) → ~4x less activation memory
#
# Auto-detects GPU count. Adjusts grad_accum to maintain effective BS=32.
#   2 GPUs: BS=1 × accum=16 × 2 = 32
#   8 GPUs: BS=1 × accum=4  × 8 = 32
#
# Training data uses lazy caching — missing quarter-res VAE files
# are encoded on-the-fly from raw video at quarter resolution.
#
# Val set: 30 samples from 30 non-overlapping videos.
#
# Usage:
#   bash scripts/run_df_training_quarter.sh

set -e

# ── Weights ──
export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

# ── Data ──
export INFINITETALK_TRAIN_LIST="${INFINITETALK_TRAIN_LIST:-data/precomputed_talkvid/train_excl_val30.txt}"
export INFINITETALK_VAL_LIST="${INFINITETALK_VAL_LIST:-data/precomputed_talkvid/val_quarter_30.txt}"
export INFINITETALK_NEG_TEXT_EMB="${INFINITETALK_NEG_TEXT_EMB:-data/precomputed_talkvid/neg_text_embeds.pt}"
export INFINITETALK_CSV_PATH="${INFINITETALK_CSV_PATH:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list_cleaned.csv}"
export INFINITETALK_RAW_DATA_ROOT="${INFINITETALK_RAW_DATA_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/}"
export INFINITETALK_WAV2VEC_DIR="${INFINITETALK_WAV2VEC_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base}"
export INFINITETALK_AUDIO_ROOT="${INFINITETALK_AUDIO_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data}"

# ── Wandb ──
export WANDB_ENTITY="paulhcho"
export WANDB_API_KEY="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"
# To resume: export WANDB_RUN_ID=<id> and WANDB_RESUME=must before running
# unset WANDB_RUN_ID

# ── CUDA ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800

# ── Auto-detect GPUs ──
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export NUM_GPUS

echo "======================================"
echo "InfiniteTalk DF Training — Quarter Res"
echo "======================================"
echo "GPUs:           $NUM_GPUS"
echo "Resolution:     224×448 px → 28×56 latent"
echo "Tokens:         8,232 (vs 32,928 full res)"
ACCUM=$((32 / (4 * NUM_GPUS)))
[ $ACCUM -lt 1 ] && ACCUM=1
echo "Batch size:     4/GPU (measured: 67.9GB peak on A100-80GB)"
echo "Effective BS:   32 (BS=4 × accum=${ACCUM} × ${NUM_GPUS}gpu)"
echo "Train list:     $INFINITETALK_TRAIN_LIST ($(wc -l < $INFINITETALK_TRAIN_LIST) samples)"
echo "Val list:       $INFINITETALK_VAL_LIST ($(wc -l < $INFINITETALK_VAL_LIST) samples)"
echo "Lazy caching:   enabled (quarter-res on-the-fly)"
echo "======================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_df_quarter_prod.py
