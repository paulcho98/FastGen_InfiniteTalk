#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ─────────────────────────────────────────────────────────────────────────────
# ODE trajectory extraction for InfiniteTalk FastGen KD (Stage 1)
#
# Runs the 14B InfiniteTalk teacher with 3-call CFG (text + audio guidance)
# to generate deterministic ODE trajectories for each training sample.
#
# Output: ode_path.pt in each sample directory, shape [4, 16, 21, H, W] bf16
#
# Environment variables (override defaults):
#   WEIGHTS_DIR          — Wan2.1-I2V-14B-480P safetensor shards directory
#   INFINITETALK_CKPT    — infinitetalk.safetensors checkpoint
#   LORA_CKPT            — (optional) external LoRA to merge into base weights
#   DATA_LIST            — text file listing sample directories (one per line)
#   NEG_TEXT_EMB         — precomputed negative text embedding .pt
#   NUM_GPUS             — number of GPUs for distributed run (default: 8)
#   BATCH_SIZE           — per-GPU batch size (default: 1, safe for 80GB)
#   NUM_STEPS            — ODE solver steps (default: 40)
#   TEXT_GUIDE_SCALE     — text CFG scale (default: 5.0)
#   AUDIO_GUIDE_SCALE    — audio CFG scale (default: 4.0)
#   SHIFT                — RF schedule shift (default: 7.0)
#   SEED                 — base random seed (default: 42)
#
# Usage:
#   # Full run (8 GPUs, all samples):
#   WEIGHTS_DIR=/path/to/Wan2.1-I2V-14B-480P \
#   INFINITETALK_CKPT=/path/to/infinitetalk.safetensors \
#   DATA_LIST=/path/to/sample_list.txt \
#   NEG_TEXT_EMB=/path/to/neg_text_embeds.pt \
#     bash scripts/run_ode_extraction.sh
#
#   # Quick test (1 GPU, 3 samples):
#   NUM_GPUS=1 \
#   WEIGHTS_DIR=/path/to/Wan2.1-I2V-14B-480P \
#   INFINITETALK_CKPT=/path/to/infinitetalk.safetensors \
#   DATA_LIST=data/test_precomputed/sample_list.txt \
#   NEG_TEXT_EMB=data/test_precomputed/neg_text_embeds.pt \
#     bash scripts/run_ode_extraction.sh --max_samples 3
#
#   # Resume after interruption (skips completed samples):
#   ... same as above, --skip_existing is on by default
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Navigate to project root (scripts/ is one level down)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Required paths ──
WEIGHTS_DIR="${WEIGHTS_DIR:?Set WEIGHTS_DIR to Wan2.1-I2V-14B-480P directory}"
INFINITETALK_CKPT="${INFINITETALK_CKPT:-}"
DATA_LIST="${DATA_LIST:?Set DATA_LIST to path of sample_list.txt}"
NEG_TEXT_EMB="${NEG_TEXT_EMB:?Set NEG_TEXT_EMB to path of neg_text_embeds.pt}"

# ── Optional overrides ──
LORA_CKPT="${LORA_CKPT:-}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-40}"
TEXT_GUIDE_SCALE="${TEXT_GUIDE_SCALE:-5.0}"
AUDIO_GUIDE_SCALE="${AUDIO_GUIDE_SCALE:-4.0}"
SHIFT="${SHIFT:-7.0}"
SEED="${SEED:-42}"

# ── Build command ──
CMD_ARGS=(
    scripts/generate_infinitetalk_ode_pairs.py
    --weights_dir "${WEIGHTS_DIR}"
    --data_list_path "${DATA_LIST}"
    --neg_text_emb_path "${NEG_TEXT_EMB}"
    --text_guide_scale "${TEXT_GUIDE_SCALE}"
    --audio_guide_scale "${AUDIO_GUIDE_SCALE}"
    --num_steps "${NUM_STEPS}"
    --shift "${SHIFT}"
    --t_list 0.999 0.937 0.833 0.624 0.0
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --skip_existing
)

# Optional arguments
if [ -n "${INFINITETALK_CKPT}" ]; then
    CMD_ARGS+=(--infinitetalk_ckpt "${INFINITETALK_CKPT}")
fi

if [ -n "${LORA_CKPT}" ]; then
    CMD_ARGS+=(--lora_ckpt "${LORA_CKPT}")
fi

# Append any extra CLI args passed to this script
CMD_ARGS+=("$@")

# ── Launch ──
echo "============================================================"
echo "InfiniteTalk ODE Trajectory Extraction"
echo "============================================================"
echo "  Project root:     ${PROJECT_ROOT}"
echo "  Weights dir:      ${WEIGHTS_DIR}"
echo "  InfiniteTalk ckpt: ${INFINITETALK_CKPT:-none}"
echo "  LoRA ckpt:        ${LORA_CKPT:-none}"
echo "  Data list:        ${DATA_LIST}"
echo "  Neg text emb:     ${NEG_TEXT_EMB}"
echo "  GPUs:             ${NUM_GPUS}"
echo "  Batch size:       ${BATCH_SIZE}"
echo "  ODE steps:        ${NUM_STEPS}"
echo "  CFG:              text=${TEXT_GUIDE_SCALE}, audio=${AUDIO_GUIDE_SCALE}"
echo "  Shift:            ${SHIFT}"
echo "  Seed:             ${SEED}"
echo "============================================================"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Launching distributed (${NUM_GPUS} GPUs)..."
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        "${CMD_ARGS[@]}"
else
    echo "Launching single-GPU..."
    python "${CMD_ARGS[@]}"
fi
