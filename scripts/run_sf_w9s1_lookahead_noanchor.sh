#!/bin/bash
# InfiniteTalk SF Training — w9s1 + Lookahead Sink + Train-Time Anchor-Free
#
# Combines two experimental modes on top of the w9s1 attention config:
#   (a) Lookahead sink — sink K rotated at future RoPE position
#       F_window - 1 + lookahead_distance. Distance defaults to 4.
#   (b) Train-time anchor-free — no model (student, teacher, fake_score)
#       anchors frame 0 during training. Only the student anchors during
#       eval/inference. Teacher is hard-disabled (never anchors anywhere).
#
# Environment overrides:
#   LOOKAHEAD_DISTANCE=N   (default 4)    lookahead distance in frames
#   MODEL_SINK_CACHE       (default 1, ON)   F2: cache model-generated sink K/V
#   SKIP_CLEAN_CACHE       (default 1, ON)   F3: skip separate clean cache pass
#   Set MODEL_SINK_CACHE= or SKIP_CLEAN_CACHE= (empty) to disable.
#
# Usage:
#   bash scripts/run_sf_w9s1_lookahead_noanchor.sh                  # F2+F3 on (default)
#   LOOKAHEAD_DISTANCE=6 bash scripts/run_sf_w9s1_lookahead_noanchor.sh
#   MODEL_SINK_CACHE= SKIP_CLEAN_CACHE= bash scripts/run_sf_w9s1_lookahead_noanchor.sh   # disable both

set -e

# ── F2/F3 baked-in defaults (env can still override to 0/empty to disable) ──
export MODEL_SINK_CACHE="${MODEL_SINK_CACHE:-1}"       # F2: model-generated sink K/V
export SKIP_CLEAN_CACHE="${SKIP_CLEAN_CACHE:-1}"       # F3: skip separate clean-cache pass

# ── Weights ──
export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

export INFINITETALK_TEACHER_LORA_CKPT="${INFINITETALK_TEACHER_LORA_CKPT:-}"
export INFINITETALK_STUDENT_LORA_CKPT="${INFINITETALK_STUDENT_LORA_CKPT:-}"

# ── Stage 1 (DF) checkpoint — auto-detect latest ──
# Source: the STOCHASTIC DF run. Its config sampled 5 attention configs incl.
# local_attn_size=10, sink_size=1 (our exact w9s1 setup) with 20% weight. So the
# student weights are already partially adapted to short-window causal attention.
# Non-stochastic DF was trained with a single (different) attention config and
# would be a worse initialization for this experiment.
DF_CKPT_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter_stochastic/quarter_stoch_r128_bs4_accum1_8gpu_0409_2248/checkpoints"
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

echo "================================================================"
echo "InfiniteTalk SF Training — w9s1 + Lookahead + Train-Anchor-Free"
echo "================================================================"
echo "GPUs:               $NUM_GPUS"
echo "Student attn:       local_attn_size=10, sink_size=1"
echo "Dynamic RoPE:       ON (required by lookahead)"
echo "Lookahead distance: ${LOOKAHEAD_DISTANCE:-4} frames"
echo "Anchor (training):  OFF for student, teacher, fake_score"
echo "Anchor (eval/inf):  ON for student only"
echo "F2 (model-sink):    ${MODEL_SINK_CACHE:+ON} ${MODEL_SINK_CACHE:-off}"
echo "F3 (skip cache):    ${SKIP_CLEAN_CACHE:+ON} ${SKIP_CLEAN_CACHE:-off}"
echo "DF checkpoint:      $INFINITETALK_DF_CKPT"
echo "Train list:         $INFINITETALK_TRAIN_LIST"
echo "Val list:           $INFINITETALK_VAL_LIST"
echo "================================================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_noanchor.py
