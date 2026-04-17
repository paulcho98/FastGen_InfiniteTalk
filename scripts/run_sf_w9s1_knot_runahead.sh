#!/bin/bash
# InfiniteTalk Self-Forcing Training — w9s1 + Knot Forcing + Running-Ahead + Last-frame ref.
#
# Paper: Xiao et al., "Knot Forcing: Taming Autoregressive Video Diffusion Models for
#        Real-time Infinite Interactive Portrait Animation", arXiv:2512.21734v2.
#
# Stage 2: Self-Forcing distillation with causal student, with Knot Forcing extensions.
#
# Baked feature configuration:
#   - w9s1 attention      (sink_size=1, rolling=9, local_attn_size=10)
#   - F1 lookahead sink   OFF (replaced by running-ahead — mutually exclusive)
#   - F2 sink cache       OFF
#   - F3 skip clean cache OFF (KF uses separate cache pass for committed c frames only)
#   - Temporal Knot       ON, k=1 (paper default; env KNOT_SIZE to override)
#   - Running-Ahead       ON, s=4, init_n=8 (env RUNNING_AHEAD_STEP / RUNNING_AHEAD_INIT_N)
#   - Last-frame ref      ON (dataloader uses vae_latents[:, :, -1])
#   - Anchors             ON (student + fake_score + teacher, input + output)
#
# Teacher and fake_score remain bidirectional.
#
# Optional env-var overrides:
#   KNOT_SIZE=N                        knot length (default 1, paper uses 1)
#   RUNNING_AHEAD_STEP=N               s value (default 4)
#   RUNNING_AHEAD_INIT_N=N             initial n (default 8)
#   INFINITETALK_DF_CKPT=<path>        DF checkpoint path (defaults to latest in DF_CKPT_DIR)
#
# Usage:
#   bash scripts/run_sf_w9s1_knot_runahead.sh
#   KNOT_SIZE=2 bash scripts/run_sf_w9s1_knot_runahead.sh
#   RUNNING_AHEAD_STEP=6 RUNNING_AHEAD_INIT_N=10 bash scripts/run_sf_w9s1_knot_runahead.sh

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
# Uses the same anchor-ON stochastic DF init as the F3 run (proven checkpoint).
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
_K="${KNOT_SIZE:-1}"
_S="${RUNNING_AHEAD_STEP:-4}"
_N="${RUNNING_AHEAD_INIT_N:-8}"

echo "============================================"
echo "InfiniteTalk SF Training — w9s1 + KF + RA + Last-ref"
echo "============================================"
echo "GPUs:             $NUM_GPUS"
echo "Resolution:       224x448 px -> 28x56 latent"
echo "Student attn:     local_attn_size=10, sink_size=1 (1 sink + 9 rolling)"
echo "Dynamic RoPE:     ON (required for running-ahead)"
echo "Temporal Knot:    ON (k=${_K}, commit=3, denoise=4)"
echo "Running Ahead:    ON (s=${_S}, init_n=${_N})"
echo "Last-frame ref:   ON (dataloader uses vae_latents[:, -1])"
echo "F1/F2/F3:         OFF (baked — F1 incompatible with RA; KF uses separate cache pass)"
echo "Anchors:          ON (student + fake_score + teacher, input + output)"
echo "Teacher/fake:     bidirectional (full attention)"
echo "DF checkpoint:    $INFINITETALK_DF_CKPT"
echo "Train list:       $INFINITETALK_TRAIN_LIST"
echo "Val list:         $INFINITETALK_VAL_LIST"
echo "============================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_knot_runahead.py
