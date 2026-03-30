#!/usr/bin/env bash
# Run causal InfiniteTalk inference with default weight paths.
#
# Usage:
#   ./scripts/inference/run_inference_causal.sh \
#       --image /path/to/reference.png \
#       --audio /path/to/audio.wav \
#       --output_path /path/to/output.mp4
#
# Or with pre-computed:
#   ./scripts/inference/run_inference_causal.sh \
#       --precomputed_dir data/precomputed_talkvid/data_xxx \
#       --output_path /path/to/output.mp4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTGEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- Default paths ----------
IT_WEIGHTS="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights}"

BASE_SHARDS=""
for i in $(seq 1 7); do
    shard="${IT_WEIGHTS}/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-0000${i}-of-00007.safetensors"
    if [ -n "$BASE_SHARDS" ]; then BASE_SHARDS="${BASE_SHARDS},"; fi
    BASE_SHARDS="${BASE_SHARDS}${shard}"
done

IT_CKPT="${IT_WEIGHTS}/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH="${IT_WEIGHTS}/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth"
WAV2VEC_PATH="${IT_WEIGHTS}/InfiniteTalk/single/wav2vec2-base-960h-zh"
CLIP_PATH="${IT_WEIGHTS}/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
T5_PATH="${IT_WEIGHTS}/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth"
CKPT_PATH="${INFINITETALK_CKPT_PATH:-}"

# ---------- Run ----------
cd "$FASTGEN_ROOT"
python scripts/inference/inference_causal.py \
    --base_model_paths "$BASE_SHARDS" \
    --infinitetalk_ckpt "$IT_CKPT" \
    --vae_path "$VAE_PATH" \
    --wav2vec_path "$WAV2VEC_PATH" \
    --clip_path "$CLIP_PATH" \
    --t5_path "$T5_PATH" \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"} \
    "$@"
