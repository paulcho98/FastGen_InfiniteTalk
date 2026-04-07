#!/bin/bash
# =============================================================================
# Evaluate checkpoints on TalkVid validation set (30 precomputed samples).
# Runs inference (using precomputed VAE/CLIP/audio) then eval metrics + VBench.
#
# Usage:
#   # DF checkpoints (latest + every 1000th):
#   bash scripts/inference/run_eval_talkvid_val.sh
#
#   # Specific checkpoints:
#   CHECKPOINTS="0001000 0003000" bash scripts/inference/run_eval_talkvid_val.sh
#
#   # SF checkpoint (consolidated):
#   CKPT_DIR=/path/to/sf/checkpoints CKPT_SUFFIX="_net_consolidated" \
#     bash scripts/inference/run_eval_talkvid_val.sh
# =============================================================================
set -euo pipefail

# ── Paths ──
FASTGEN_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_METRICS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics"
SHAPE_PREDICTOR="${EVAL_METRICS_DIR}/shape_predictor_68_face_landmarks.dat"
FFMPEG_PATH=$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || echo ffmpeg)

# ── Model weights ──
WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
VAE_PATH="${WEIGHTS_DIR}/Wan2.1_VAE.pth"
WAV2VEC_PATH="${WAV2VEC_PATH:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base}"
CLIP_PATH="${WEIGHTS_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
BASE_SHARDS=$(ls "${WEIGHTS_DIR}"/diffusion_pytorch_model-*.safetensors | tr '\n' ',' | sed 's/,$//')

# ── Checkpoint config ──
CKPT_DIR="${CKPT_DIR:-${FASTGEN_DIR}/FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter/quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints}"
CKPT_SUFFIX="${CKPT_SUFFIX:-}"  # e.g., "_net_consolidated" for SF checkpoints
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# ── TalkVid val set ──
PRECOMPUTED_LIST="${FASTGEN_DIR}/data/precomputed_talkvid/val_quarter_30.txt"
AUDIO_DATA_ROOT="${AUDIO_DATA_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data}"
GT_RESIZED="${FASTGEN_DIR}/EVAL_OUTPUT/talkvid_val_gt_448x224"

# ── Generation params ──
CHUNK_SIZE="${CHUNK_SIZE:-3}"
NUM_LATENT_FRAMES="${NUM_LATENT_FRAMES:-21}"
SEED="${SEED:-42}"
TARGET_H="${TARGET_H:-224}"
TARGET_W="${TARGET_W:-448}"

# ── Output ──
OUTPUT_ROOT="${OUTPUT_ROOT:-${FASTGEN_DIR}/EVAL_OUTPUT/talkvid_val}"
EVAL_ROOT="${EVAL_ROOT:-${FASTGEN_DIR}/EVAL_OUTPUT/talkvid_val_eval}"

# ── Select checkpoints ──
if [ -z "${CHECKPOINTS:-}" ]; then
    LATEST=$(ls -1 "$CKPT_DIR"/*${CKPT_SUFFIX}.pth 2>/dev/null | sort -V | tail -1)
    THOUSANDS=$(ls -1 "$CKPT_DIR"/*${CKPT_SUFFIX}.pth 2>/dev/null | grep -E '000[0-9]+000'"${CKPT_SUFFIX}"'\.pth$' | sort -V)
    ALL_CKPTS=$(echo -e "${THOUSANDS}\n${LATEST}" | sort -V | uniq | grep '\.pth$' || true)
else
    ALL_CKPTS=""
    for iter_num in $CHECKPOINTS; do
        ckpt="${CKPT_DIR}/${iter_num}${CKPT_SUFFIX}.pth"
        if [ -f "$ckpt" ]; then
            ALL_CKPTS="${ALL_CKPTS}${ckpt}"$'\n'
        else
            echo "WARNING: checkpoint not found: $ckpt"
        fi
    done
    ALL_CKPTS=$(echo "$ALL_CKPTS" | sort -V | uniq | grep '\.pth$' || true)
fi

NUM_CKPTS=$(echo "$ALL_CKPTS" | grep -c '\.pth$' || true)

echo "============================================"
echo "TalkVid Val Evaluation"
echo "============================================"
echo "Precomputed list: $PRECOMPUTED_LIST ($(wc -l < "$PRECOMPUTED_LIST") samples)"
echo "GT (resized):     $GT_RESIZED"
echo "Checkpoints:      $NUM_CKPTS from $CKPT_DIR"
echo "Inference out:    $OUTPUT_ROOT"
echo "Eval out:         $EVAL_ROOT"
echo "LoRA:             rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "============================================"
echo ""

cd "$FASTGEN_DIR"

for CKPT_PATH in $ALL_CKPTS; do
    [ -z "$CKPT_PATH" ] && continue
    CKPT_BASENAME=$(basename "$CKPT_PATH" .pth)
    ITER_NAME=$(echo "$CKPT_BASENAME" | sed "s/${CKPT_SUFFIX}$//")
    INFER_DIR="${OUTPUT_ROOT}/iter_${ITER_NAME}"
    EVAL_DIR="${EVAL_ROOT}/iter_${ITER_NAME}"

    echo "================================================================"
    echo "Processing iter ${ITER_NAME}"
    echo "================================================================"

    # ── Phase 1: Inference ──
    TOTAL_GT=$(find "$GT_RESIZED" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)
    EXISTING=$(find "$INFER_DIR" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)

    if [ "$EXISTING" -ge "$TOTAL_GT" ] && [ "$TOTAL_GT" -gt 0 ]; then
        echo "[infer] Skipping — already complete (${EXISTING}/${TOTAL_GT})"
    else
        echo "[infer] Running inference (${EXISTING:-0}/${TOTAL_GT} existing)..."
        mkdir -p "$INFER_DIR"

        python scripts/inference/inference_causal.py \
            --precomputed_list "$PRECOMPUTED_LIST" \
            --output_dir "$INFER_DIR" \
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
            --quarter_res

        GENERATED=$(find "$INFER_DIR" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)
        echo "[infer] Done: ${GENERATED}/${TOTAL_GT} videos"
    fi

    # ── Phase 2: Standard eval metrics ──
    if [ -f "$EVAL_DIR/all_metrics.log" ]; then
        echo "[eval] Skipping standard metrics — already complete"
    else
        echo "[eval] Running standard metrics..."
        mkdir -p "$EVAL_DIR"
        cd "$EVAL_METRICS_DIR"
        PYTHONPATH="${EVAL_METRICS_DIR}:${PYTHONPATH:-}" \
        PYTHONUNBUFFERED=1 \
        bash eval/run_metrics.sh \
            --real_videos_dir "$GT_RESIZED" \
            --fake_videos_dir "$INFER_DIR" \
            --shape_predictor_path "$SHAPE_PREDICTOR" \
            --output_dir "$EVAL_DIR" \
            --log_path "$EVAL_DIR/all_metrics.log" \
            --fallback_detection_confidence 0.2 \
            --ffmpeg_path "$FFMPEG_PATH" \
            --all || echo "[eval] WARNING: some metrics failed"
        cd "$FASTGEN_DIR"
    fi

    # ── Phase 3: VBench (per-dimension to avoid OOM) ──
    if ls "$EVAL_DIR/vbench/composited/"*results* >/dev/null 2>&1; then
        echo "[vbench] Skipping — already complete"
    else
        echo "[vbench] Running VBench..."
        DIMS=(subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality)
        cd "$EVAL_METRICS_DIR"
        for dim in "${DIMS[@]}"; do
            echo "[vbench]   $dim"
            PYTHONPATH="${EVAL_METRICS_DIR}:${PYTHONPATH:-}" \
            PYTHONUNBUFFERED=1 \
            CUDA_VISIBLE_DEVICES=0 \
            python eval/eval_vbench.py \
                --fake_videos_dir "$INFER_DIR" \
                --real_videos_dir "$GT_RESIZED" \
                --output_dir "$EVAL_DIR/vbench" \
                --device cuda:0 \
                --skip_face_crops \
                --dimensions "$dim" 2>&1 | tail -3
            sleep 2
        done
        cd "$FASTGEN_DIR"
    fi

    echo ""
done

echo "============================================"
echo "All evaluations complete!"
echo "Inference: $OUTPUT_ROOT"
echo "Eval:      $EVAL_ROOT"
echo "============================================"
