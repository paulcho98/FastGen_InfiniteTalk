#!/usr/bin/env bash
# Causal InfiniteTalk inference matching the w9s1 + Knot Forcing + Running-Ahead recipe.
#
# Paper: Xiao et al., "Knot Forcing" arXiv:2512.21734v2.
#
# Baked CLI flags (mirroring config_sf_w9s1_knot_runahead.py):
#   --local_attn_size 10 --sink_size 1           (w9s1 attention)
#   --use_dynamic_rope                            (required by running-ahead)
#   --use_temporal_knot --knot_size 1             (KF on, k=1)
#   --use_running_ahead                           (RA on)
#   --running_ahead_step 4 --running_ahead_init_n 8
#   --use_last_frame_reference                    (KF training convention)
#   (no --lookahead_sink                           — F1 mutually exclusive with RA)
#   (no --skip_clean_cache_pass                    — F3 off under KF)
#   (no --model_sink_cache                         — F2 off)
#
# Required args (must be passed via "$@"):
#   --image PATH        or  --precomputed_dir PATH
#   --audio PATH        or  --source_audio PATH
#   --output_path PATH  or  --output_dir PATH
#
# Optional env overrides:
#   INFINITETALK_CKPT_PATH=<path to consolidated KF-SF net .pth>
#   KNOT_SIZE=N                       (default 1)
#   RUNNING_AHEAD_STEP=N              (default 4)
#   RUNNING_AHEAD_INIT_N=N            (default 8)
#   NUM_LATENT_FRAMES=N               (default auto from audio duration)
#   CHUNK_SIZE=N                      (default 3)
#
# Examples:
#   # Single sample
#   bash scripts/inference/run_inference_knot_runahead.sh \
#       --image ref.png --audio speech.wav --output_path out.mp4
#
#   # Batch from precomputed list
#   bash scripts/inference/run_inference_knot_runahead.sh \
#       --precomputed_list data/precomputed_talkvid/val_quarter_30.txt \
#       --output_dir out/val_knot_runahead

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTGEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- Default weight paths ----------
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

# ---------- Feature toggles (from env or defaults) ----------
K="${KNOT_SIZE:-1}"
S="${RUNNING_AHEAD_STEP:-4}"
N="${RUNNING_AHEAD_INIT_N:-8}"
CHUNK_SIZE_VAL="${CHUNK_SIZE:-3}"

# ---------- Optional args ----------
NUM_LATENT_ARG=()
if [ -n "${NUM_LATENT_FRAMES:-}" ]; then
    NUM_LATENT_ARG=(--num_latent_frames "${NUM_LATENT_FRAMES}")
fi

# ---------- Display ----------
echo "============================================"
echo "InfiniteTalk Inference — w9s1 + KF + RA + Last-ref"
echo "============================================"
echo "Student attn:     local_attn_size=10, sink_size=1 (w9s1)"
echo "Dynamic RoPE:     ON (required for running-ahead)"
echo "Temporal Knot:    ON (k=${K}, c+k=$((CHUNK_SIZE_VAL + K)))"
echo "Running Ahead:    ON (s=${S}, init_n=${N})"
echo "Last-frame ref:   ON"
echo "F1/F2/F3:         OFF (F1 ↔ RA exclusive; KF uses separate cache pass)"
echo "Anchor first:     ON (existing anchor is no-op for iter > 0 under KF)"
echo "Chunk size:       ${CHUNK_SIZE_VAL}"
echo "Checkpoint:       ${CKPT_PATH:-<none — using InfiniteTalk base>}"
echo "============================================"
echo ""

# ---------- Run ----------
cd "$FASTGEN_ROOT"
python scripts/inference/inference_causal.py \
    --base_model_paths "$BASE_SHARDS" \
    --infinitetalk_ckpt "$IT_CKPT" \
    --vae_path "$VAE_PATH" \
    --wav2vec_path "$WAV2VEC_PATH" \
    --clip_path "$CLIP_PATH" \
    --t5_path "$T5_PATH" \
    --local_attn_size 10 \
    --sink_size 1 \
    --use_dynamic_rope \
    --use_temporal_knot \
    --knot_size "$K" \
    --use_running_ahead \
    --running_ahead_step "$S" \
    --running_ahead_init_n "$N" \
    --use_last_frame_reference \
    --chunk_size "$CHUNK_SIZE_VAL" \
    "${NUM_LATENT_ARG[@]}" \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"} \
    "$@"
