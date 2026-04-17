#!/bin/bash
# Resume InfiniteTalk DF Quarter-Res Training from checkpoint
#
# Usage:
#   bash scripts/resume_df_training_quarter.sh

set -e
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk

# ── Run to resume (reuse same output dir so checkpointer finds the .pth) ──
export FASTGEN_RUN_NAME=quarter_r128_bs4_accum1_8gpu_0402_0836
export WANDB_RUN_ID=rl4fz7a0
export WANDB_RESUME=must

# ── Weights ──
export INFINITETALK_WEIGHTS_DIR=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P
export INFINITETALK_CKPT=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors
export INFINITETALK_VAE_PATH=${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth

# ── Data ──
export INFINITETALK_TRAIN_LIST=data/precomputed_talkvid/train_excl_val30.txt
export INFINITETALK_VAL_LIST=data/precomputed_talkvid/val_quarter_30.txt
export INFINITETALK_NEG_TEXT_EMB=data/precomputed_talkvid/neg_text_embeds.pt
export INFINITETALK_CSV_PATH=/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list_cleaned.csv
export INFINITETALK_RAW_DATA_ROOT=/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/
export INFINITETALK_WAV2VEC_DIR=/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base
export INFINITETALK_AUDIO_ROOT=/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data

# ── Wandb ──
export WANDB_ENTITY=paulhcho
export WANDB_API_KEY=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

# ── CUDA ──
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800

# ── Auto-detect GPUs ──
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export NUM_GPUS

# ── Verify checkpoint exists ──
CKPT_DIR="FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter/${FASTGEN_RUN_NAME}/checkpoints"
LATEST=$(ls -t "$CKPT_DIR"/*.pth 2>/dev/null | head -1)
echo "======================================"
echo "InfiniteTalk DF Quarter — RESUME"
echo "======================================"
echo "Run name:       $FASTGEN_RUN_NAME"
echo "Checkpoint:     ${LATEST:-NOT FOUND}"
echo "GPUs:           $NUM_GPUS"
echo "Wandb run ID:   $WANDB_RUN_ID"
echo "======================================"

if [ -z "$LATEST" ]; then
    echo "ERROR: No checkpoint found in $CKPT_DIR"
    exit 1
fi

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_df_quarter_prod.py
