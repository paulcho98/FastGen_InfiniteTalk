#!/bin/bash
# =============================================================================
# Run SF inference with --profile_timing on the
# sf_w9s1_la4_f3_anchor_freq5_lr1e5_0415_2326 run.
#
# Mirrors the flags used in InfiniteTalk/monitor_sf_w9s1_la4_f3.sh, adding
# --profile_timing and writing to a separate EVAL_OUTPUT subdir so existing
# iter_<N> outputs aren't clobbered.
#
# Usage:
#   bash scripts/inference/profile_sf_w9s1_la4_f3.sh            # defaults to iter 700
#   bash scripts/inference/profile_sf_w9s1_la4_f3.sh 500        # iter 500
#   ITER=500 bash scripts/inference/profile_sf_w9s1_la4_f3.sh   # env-var form
#   LIST=data/precomputed_talkvid/val_quarter_1.txt bash scripts/inference/profile_sf_w9s1_la4_f3.sh
# =============================================================================
set -u
set -o pipefail

# ── Resolve iter (positional arg wins; otherwise env ITER; otherwise 700) ──
ITER="${1:-${ITER:-700}}"
ITER_PADDED=$(printf "%07d" "$ITER")

# ── Paths ──
FASTGEN_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk"
RUN_NAME="sf_w9s1_la4_f3_anchor_freq5_lr1e5_0415_2326"
SF_CKPT_DIR="${FASTGEN_DIR}/FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/${RUN_NAME}/checkpoints"
CKPT_PATH="${SF_CKPT_DIR}/${ITER_PADDED}_net_consolidated.pth"

# ── Model weights ──
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
INFINITETALK_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH="${WEIGHTS_DIR}/Wan2.1_VAE.pth"
WAV2VEC_PATH="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"
CLIP_PATH="${WEIGHTS_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
BASE_SHARDS=$(ls "${WEIGHTS_DIR}"/diffusion_pytorch_model-*.safetensors | tr '\n' ',' | sed 's/,$//')

# ── Inference knobs (match training config) ──
LORA_RANK=128
LORA_ALPHA=64
CHUNK_SIZE=3
NUM_LATENT_FRAMES=21
SEED=42
TARGET_H=224
TARGET_W=448
LOOKAHEAD_DISTANCE=4

# ── Data ──
PRECOMPUTED_LIST="${LIST:-${FASTGEN_DIR}/data/precomputed_talkvid/val_quarter_30.txt}"
AUDIO_DATA_ROOT="/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data"

# ── Outputs (separate from the monitor's iter_<N> dirs to avoid clobber) ──
OUTPUT_DIR="${FASTGEN_DIR}/EVAL_OUTPUT/talkvid_val_${RUN_NAME}_profile/iter_${ITER_PADDED}"
LOG_FILE="${FASTGEN_DIR}/EVAL_OUTPUT/talkvid_val_${RUN_NAME}_profile/iter_${ITER_PADDED}_profile.log"
mkdir -p "$OUTPUT_DIR"

# ── Sanity checks ──
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: consolidated checkpoint not found: $CKPT_PATH"
    echo "       (did the monitor consolidate this iter? check ${SF_CKPT_DIR})"
    exit 1
fi
if [ ! -f "$PRECOMPUTED_LIST" ]; then
    echo "ERROR: precomputed list not found: $PRECOMPUTED_LIST"
    exit 1
fi

echo "============================================"
echo "SF Profile Timing — ${RUN_NAME}"
echo "============================================"
echo "Iter:             ${ITER_PADDED}"
echo "Checkpoint:       ${CKPT_PATH}"
echo "Precomputed list: ${PRECOMPUTED_LIST} ($(wc -l < "$PRECOMPUTED_LIST") samples)"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Log file:         ${LOG_FILE}"
echo "Lookahead dist:   ${LOOKAHEAD_DISTANCE} (fixed)"
echo "============================================"
echo ""

cd "$FASTGEN_DIR"

python scripts/inference/inference_causal.py \
    --precomputed_list "$PRECOMPUTED_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --base_model_paths "$BASE_SHARDS" \
    --infinitetalk_ckpt "$INFINITETALK_CKPT" \
    --vae_path "$VAE_PATH" \
    --wav2vec_path "$WAV2VEC_PATH" \
    --clip_path "$CLIP_PATH" \
    --audio_data_root "$AUDIO_DATA_ROOT" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --chunk_size "$CHUNK_SIZE" \
    --num_latent_frames "$NUM_LATENT_FRAMES" \
    --seed "$SEED" \
    --target_h "$TARGET_H" \
    --target_w "$TARGET_W" \
    --quarter_res \
    --use_dynamic_rope \
    --lookahead_sink \
    --lookahead_distance "$LOOKAHEAD_DISTANCE" \
    --skip_clean_cache_pass \
    --profile_timing \
    2>&1 | tee "$LOG_FILE"
