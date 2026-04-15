#!/bin/bash
# InfiniteTalk Self-Forcing Training — w9s1 + Lookahead Sink (F1) + F3 baked.
#
# Stage 2: Self-Forcing distillation with causal student.
#
# Baked feature configuration (NOT overridable via env):
#   - w9s1 attention   (sink_size=1, rolling=9, local_attn_size=10)
#   - F1 lookahead     (dynamic RoPE, lookahead_sink_enabled=True)
#   - F2 OFF           (sink K/V = clean first_frame_cond via input anchor)
#   - F3 ON            (skip separate clean-cache pass in training, validation,
#                       and inference — KV stored during the last denoise step)
#   - Anchors ON       (student + fake_score + teacher all anchor, input-side
#                       and output-side, on every forward — matches InfiniteTalk
#                       training distribution)
#
# Teacher and fake_score remain bidirectional.
#
# Optional env-var overrides:
#   LOOKAHEAD_DISTANCE=N            lookahead frames (default: 4)
#   LOOKAHEAD_DISTANCE_MIN=N        stochastic distance range (default: 0=off)
#   LOOKAHEAD_DISTANCE_MAX=N        stochastic distance range (default: 0=off)
#
# Usage:
#   bash scripts/run_sf_w9s1_lookahead_f3.sh
#   LOOKAHEAD_DISTANCE=6 bash scripts/run_sf_w9s1_lookahead_f3.sh
#   LOOKAHEAD_DISTANCE_MIN=2 LOOKAHEAD_DISTANCE_MAX=6 bash scripts/run_sf_w9s1_lookahead_f3.sh

set -e

# ── Weights ──
export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

# Teacher LoRA (rank=32, merged and frozen)
export INFINITETALK_TEACHER_LORA_CKPT="${INFINITETALK_TEACHER_LORA_CKPT:-}"
# Student LoRA (rank=128, loaded from DF checkpoint — set to override)
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

# ── Resolve display values ──
_LA_DIST="${LOOKAHEAD_DISTANCE:-4}"
_LA_MIN="${LOOKAHEAD_DISTANCE_MIN:-0}"
_LA_MAX="${LOOKAHEAD_DISTANCE_MAX:-0}"
if [ "$_LA_MIN" -gt 0 ] && [ "$_LA_MAX" -gt 0 ]; then
    _LA_DISPLAY="stochastic [${_LA_MIN}, ${_LA_MAX}]"
else
    _LA_DISPLAY="fixed ${_LA_DIST}"
fi

echo "============================================"
echo "InfiniteTalk SF Training — w9s1 + Lookahead + F3"
echo "============================================"
echo "GPUs:             $NUM_GPUS"
echo "Resolution:       224x448 px -> 28x56 latent"
echo "Student attn:     local_attn_size=10, sink_size=1 (1 sink + 9 rolling)"
echo "Dynamic RoPE:     ON (required for lookahead sink)"
echo "Lookahead dist:   ${_LA_DISPLAY}  (F1 baked ON)"
echo "F2 sink cache:    OFF   (baked — sink uses clean first_frame_cond)"
echo "F3 skip clean:    ON    (baked — training + validation + inference)"
echo "Anchors:          ON    (student + fake_score + teacher, input + output)"
echo "Teacher/fake:     bidirectional (full attention)"
echo "DF checkpoint:    $INFINITETALK_DF_CKPT"
echo "Train list:       $INFINITETALK_TRAIN_LIST"
echo "Val list:         $INFINITETALK_VAL_LIST"
echo "============================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_f3.py
