#!/bin/bash
# InfiniteTalk Self-Forcing Training — Quarter Resolution
#
# Stage 2: Self-Forcing distillation with 3x 14B models (teacher + student + fake_score).
# Loads Stage 1 (DF) checkpoint for student initialization.
#
# Architecture:
#   Teacher:    14B bidirectional, LoRA merged, frozen
#   Student:    14B causal, LoRA rank=128 (trainable)
#   Fake score: 14B bidirectional, LoRA rank=128 (trainable)
#
# Training loop:
#   1:5 ratio (critic updates every step, student every 5th)
#   Combined fake_score + student step on student iterations
#   4-step AR rollout with gradient at stochastic exit step
#
# Memory: requires FSDP for 3x 14B models (~80GB peak expected on A100-80GB)
#
# Usage:
#   bash scripts/run_sf_training_quarter.sh

set -e

# ── Weights ──
export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

# Teacher LoRA (rank=32, merged and frozen)
export INFINITETALK_TEACHER_LORA_CKPT="${INFINITETALK_TEACHER_LORA_CKPT:-}"
# Student LoRA (rank=128, loaded from DF checkpoint -- set if you want to override the DF ckpt's LoRA)
export INFINITETALK_STUDENT_LORA_CKPT="${INFINITETALK_STUDENT_LORA_CKPT:-}"

# ── Stage 1 (DF) checkpoint — auto-detect latest ──
DF_CKPT_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter/quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints"
if [ -z "${INFINITETALK_DF_CKPT:-}" ]; then
    INFINITETALK_DF_CKPT=$(ls -1 "$DF_CKPT_DIR"/*.pth 2>/dev/null | sort -V | tail -1)
    if [ -z "$INFINITETALK_DF_CKPT" ]; then
        echo "ERROR: No .pth files found in $DF_CKPT_DIR"
        exit 1
    fi
fi
export INFINITETALK_DF_CKPT

# ── Data ──
export INFINITETALK_TRAIN_LIST="${INFINITETALK_TRAIN_LIST:-data/precomputed_talkvid/train_excl_val30.txt}"
export INFINITETALK_VAL_LIST="${INFINITETALK_VAL_LIST:-data/precomputed_talkvid/val_quarter_30.txt}"
export INFINITETALK_NEG_TEXT_EMB="${INFINITETALK_NEG_TEXT_EMB:-data/precomputed_talkvid/neg_text_embeds.pt}"
export INFINITETALK_AUDIO_ROOT="${INFINITETALK_AUDIO_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data}"
export INFINITETALK_CSV_PATH="${INFINITETALK_CSV_PATH:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list_cleaned.csv}"
export INFINITETALK_RAW_DATA_ROOT="${INFINITETALK_RAW_DATA_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/}"
export INFINITETALK_WAV2VEC_DIR="${INFINITETALK_WAV2VEC_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base}"

# ── Wandb ──
export WANDB_ENTITY="paulhcho"
export WANDB_API_KEY="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"

# ── CUDA ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800

# ── Auto-detect GPUs ──
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export NUM_GPUS

echo "============================================"
echo "InfiniteTalk SF Training — Quarter Res"
echo "============================================"
echo "GPUs:           $NUM_GPUS"
echo "Resolution:     224x448 px -> 28x56 latent"
echo "Models:         3x 14B (teacher + student + fake_score)"
echo "LoRA rank:      128 (student + fake_score), 32 (teacher)"
echo "FSDP:           enabled"
echo "Update ratio:   1:5 (combined step)"
echo "Batch size:     1/GPU x accum=4 = effective BS ${NUM_GPUS}x4"
echo "DF checkpoint:  $INFINITETALK_DF_CKPT"
echo "Train list:     $INFINITETALK_TRAIN_LIST"
echo "Val list:       $INFINITETALK_VAL_LIST"
echo "============================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf.py
