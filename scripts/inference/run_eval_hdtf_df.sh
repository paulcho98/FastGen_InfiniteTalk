#!/bin/bash
# Evaluate DF checkpoints on HDTF testset (33 videos, 81 frames each).
# Runs inference for the latest checkpoint + every 1000th iteration.
#
# Usage:
#   bash scripts/inference/run_eval_hdtf_df.sh
#   CHECKPOINTS="0001000 0002000" bash scripts/inference/run_eval_hdtf_df.sh

set -e

# ── Paths ──
FASTGEN_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
VAE_PATH="${INFINITETALK_VAE_PATH:-${WEIGHTS_DIR}/Wan2.1_VAE.pth}"
WAV2VEC_PATH="${WAV2VEC_PATH:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base}"
CLIP_PATH="${CLIP_PATH:-${WEIGHTS_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth}"

CKPT_DIR="${DF_CKPT_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter/quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints}"
VIDEO_DIR="${HDTF_VIDEO_DIR:-/data/karlo-research_715/workspace/kinemaar/datasets/HDTF_original_testset_81frames/videos_cfr}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTGEN_DIR}/EVAL_OUTPUT/hdtf_df}"

# Base model shards
BASE_SHARDS=$(ls "${WEIGHTS_DIR}"/diffusion_pytorch_model-*.safetensors | tr '\n' ',' | sed 's/,$//')

# ── LoRA config (must match DF training) ──
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# ── Generation params ──
CHUNK_SIZE="${CHUNK_SIZE:-3}"
NUM_LATENT_FRAMES="${NUM_LATENT_FRAMES:-21}"
SEED="${SEED:-42}"
CONTEXT_NOISE="${CONTEXT_NOISE:-0.0}"
TARGET_H="${TARGET_H:-224}"
TARGET_W="${TARGET_W:-448}"

# ── Select checkpoints: latest + multiples of 1000 ──
if [ -z "${CHECKPOINTS:-}" ]; then
    LATEST=$(ls -1 "$CKPT_DIR"/*.pth 2>/dev/null | sort -V | tail -1)
    THOUSANDS=$(ls -1 "$CKPT_DIR"/*.pth 2>/dev/null | grep -E '000[0-9]+000\.pth$' | sort -V)
    # Combine, deduplicate, sort
    ALL_CKPTS=$(echo -e "${THOUSANDS}\n${LATEST}" | sort -V | uniq)
else
    # User-specified: space-separated iteration numbers (e.g., "0001000 0002000")
    ALL_CKPTS=""
    for iter_num in $CHECKPOINTS; do
        ckpt="${CKPT_DIR}/${iter_num}.pth"
        if [ -f "$ckpt" ]; then
            ALL_CKPTS="${ALL_CKPTS}${ckpt}\n"
        else
            echo "WARNING: checkpoint not found: $ckpt"
        fi
    done
    ALL_CKPTS=$(echo -e "$ALL_CKPTS" | sort -V | uniq)
fi

NUM_CKPTS=$(echo "$ALL_CKPTS" | grep -c '\.pth$' || true)
echo "============================================"
echo "HDTF Evaluation — DF Checkpoints"
echo "============================================"
echo "Video dir:      $VIDEO_DIR"
echo "Checkpoints:    $NUM_CKPTS"
echo "Output root:    $OUTPUT_ROOT"
echo "LoRA:           rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "Latent frames:  $NUM_LATENT_FRAMES (chunk=$CHUNK_SIZE)"
echo "Anchor 1st frm: ON (default)"
echo "============================================"
echo ""
echo "Checkpoints to evaluate:"
echo "$ALL_CKPTS" | while read -r c; do
    [ -n "$c" ] && echo "  $(basename "$c")"
done
echo ""

cd "$FASTGEN_DIR"

for CKPT_PATH in $ALL_CKPTS; do
    [ -z "$CKPT_PATH" ] && continue
    ITER_NAME=$(basename "$CKPT_PATH" .pth)
    OUT_DIR="${OUTPUT_ROOT}/iter_${ITER_NAME}"

    # Skip if already complete
    EXISTING=$(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l)
    TOTAL=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        echo ">>> Skipping iter ${ITER_NAME} — already complete (${EXISTING}/${TOTAL} videos)"
        continue
    fi

    echo ">>> Evaluating iter ${ITER_NAME} → ${OUT_DIR}"

    python scripts/inference/inference_causal.py \
        --video_dir "$VIDEO_DIR" \
        --output_dir "$OUT_DIR" \
        --ckpt_path "$CKPT_PATH" \
        --base_model_paths "$BASE_SHARDS" \
        --infinitetalk_ckpt "$INFINITETALK_CKPT" \
        --vae_path "$VAE_PATH" \
        --wav2vec_path "$WAV2VEC_PATH" \
        --clip_path "$CLIP_PATH" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --chunk_size "$CHUNK_SIZE" \
        --num_latent_frames "$NUM_LATENT_FRAMES" \
        --seed "$SEED" \
        --context_noise "$CONTEXT_NOISE" \
        --target_h "$TARGET_H" \
        --target_w "$TARGET_W"

    echo ">>> Done iter ${ITER_NAME}: $(ls "$OUT_DIR"/*.mp4 2>/dev/null | wc -l) videos saved"
    echo ""
done

echo "All evaluations complete. Results in: $OUTPUT_ROOT"
