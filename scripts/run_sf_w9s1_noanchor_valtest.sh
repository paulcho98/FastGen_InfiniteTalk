#!/bin/bash
# InfiniteTalk Self-Forcing — Validation-Only Test (w9s1, train-time anchor-free)
#
# Validates that:
#   - student_anchor_eval_only=True, fake_score_anchor_eval_only=True,
#     teacher_anchor_disabled=True all land on the model correctly at init
#   - startup logs print:
#       [anchor] Student: eval-only (no anchor during training rollout)
#       [anchor] Fake score: eval-only (tracks student distribution)
#       [anchor] Teacher: DISABLED (anchor-free target distribution)
#   - validation path still works and produces correctly-dimensioned wandb videos
#
# Since validation puts the student in eval mode (self.training=False), the
# student's eval-only anchor IS active during val. Frame 0 should still match
# the reference image closely — this is not a "broken frame 0" test.
#
# Usage:
#   bash scripts/run_sf_w9s1_noanchor_valtest.sh

set -e

# ── Weights ──
export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

export INFINITETALK_TEACHER_LORA_CKPT="${INFINITETALK_TEACHER_LORA_CKPT:-}"
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

# ── Data: tiny 2-sample val list ──
export INFINITETALK_TRAIN_LIST="${INFINITETALK_TRAIN_LIST:-data/precomputed_talkvid/train_excl_val30.txt}"
export INFINITETALK_VAL_LIST="${INFINITETALK_VAL_LIST:-data/precomputed_talkvid/val_quarter_2.txt}"
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

echo "============================================================"
echo "InfiniteTalk SF Val-Only Test — w9s1 + train-time anchor-free"
echo "============================================================"
echo "GPUs:             $NUM_GPUS"
echo "Student attn:     local_attn_size=10, sink_size=1"
echo "Anchor (train):   OFF for student, teacher, fake_score"
echo "Anchor (eval):    ON for student (what this test actually exercises)"
echo "Max iter:         1  (validation at iter 0 + iter 1, no training)"
echo "Val list:         $INFINITETALK_VAL_LIST"
echo "Val samples/iter: 2"
echo "Wandb group:      infinitetalk_sf_w9s1_noanchor_valtest"
echo "DF checkpoint:    $INFINITETALK_DF_CKPT"
echo "============================================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_noanchor_valtest.py
