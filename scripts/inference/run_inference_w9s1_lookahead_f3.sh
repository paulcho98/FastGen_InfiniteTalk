#!/usr/bin/env bash
# Causal InfiniteTalk inference matching the w9s1 + Lookahead + F3 training recipe.
#
# Baked CLI flags (mirroring config_sf_w9s1_lookahead_f3.py):
#   --local_attn_size 10 --sink_size 1     (w9s1 attention)
#   --use_dynamic_rope                     (required by F1)
#   --lookahead_sink                       (F1 on)
#   --lookahead_distance 4                 (or override via env)
#   --skip_clean_cache_pass                (F3 on)
#   (no --model_sink_cache                  — F2 off)
#   (no --no_anchor_first_frame             — anchor on, matches training)
#
# Required args (must be passed via "$@"):
#   --image PATH        or
#   --precomputed_dir PATH
#   --audio PATH        or
#   --source_audio PATH (implied by --precomputed_dir)
#   --output_path PATH  or
#   --output_dir PATH
#
# Optional env overrides:
#   INFINITETALK_CKPT_PATH=<path to consolidated SF net .pth>  (student LoRA-merged)
#   LOOKAHEAD_DISTANCE=N    (default 4)
#   NUM_LATENT_FRAMES=N     (default auto from audio duration)
#   CHUNK_SIZE=N            (default 3)
#
# Examples:
#   # Single sample
#   bash scripts/inference/run_inference_w9s1_lookahead_f3.sh \
#       --image samples/ref.png \
#       --audio samples/speech.wav \
#       --output_path out/sample.mp4
#
#   # Batch from precomputed dir
#   bash scripts/inference/run_inference_w9s1_lookahead_f3.sh \
#       --precomputed_list data/precomputed_talkvid/val_quarter_30.txt \
#       --output_dir out/val_w9s1_lookahead_f3
#
#   # Override lookahead distance
#   LOOKAHEAD_DISTANCE=6 bash scripts/inference/run_inference_w9s1_lookahead_f3.sh \
#       --image ref.png --audio audio.wav --output_path out.mp4

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
LA_DIST="${LOOKAHEAD_DISTANCE:-4}"
CHUNK_SIZE_VAL="${CHUNK_SIZE:-3}"

# ---------- Optional args ----------
NUM_LATENT_ARG=()
if [ -n "${NUM_LATENT_FRAMES:-}" ]; then
    NUM_LATENT_ARG=(--num_latent_frames "${NUM_LATENT_FRAMES}")
fi

# ---------- Display ----------
echo "============================================"
echo "InfiniteTalk Inference — w9s1 + Lookahead + F3"
echo "============================================"
echo "Student attn:     local_attn_size=10, sink_size=1 (w9s1)"
echo "Dynamic RoPE:     ON (required for lookahead)"
echo "Lookahead dist:   ${LA_DIST}"
echo "F2 sink cache:    OFF  (sink = clean first_frame_cond via anchor)"
echo "F3 skip clean:    ON   (KV stored on last denoise step)"
echo "Anchor first:     ON   (matches training)"
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
    --lookahead_sink \
    --lookahead_distance "$LA_DIST" \
    --skip_clean_cache_pass \
    --chunk_size "$CHUNK_SIZE_VAL" \
    "${NUM_LATENT_ARG[@]}" \
    ${CKPT_PATH:+--ckpt_path "$CKPT_PATH"} \
    "$@"
