#!/bin/bash
# =============================================================================
# Master evaluation pipeline: DF inference + eval metrics + VBench
# Processes all selected checkpoints, running inference then evaluation on each.
#
# Usage:
#   bash scripts/inference/run_eval_pipeline.sh
# =============================================================================
set -u
set -o pipefail

# ── Paths ──
FASTGEN_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_METRICS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics"
GT_RESIZED="${FASTGEN_DIR}/EVAL_OUTPUT/hdtf_gt_448x224"
SHAPE_PREDICTOR="${EVAL_METRICS_DIR}/shape_predictor_68_face_landmarks.dat"

# ── DF Checkpoint configuration ──
DF_CKPT_DIR="${DF_CKPT_DIR:-${FASTGEN_DIR}/FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter/quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints}"
DF_OUTPUT_ROOT="${FASTGEN_DIR}/EVAL_OUTPUT/hdtf_df"
DF_EVAL_ROOT="${FASTGEN_DIR}/EVAL_OUTPUT/hdtf_df_eval"

FFMPEG_PATH=$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || echo ffmpeg)

# ── Helper: run standard eval metrics ──
run_eval_metrics() {
    local fake_dir="$1"
    local output_dir="$2"
    local label="$3"

    echo "[eval] Running standard metrics for $label"
    echo "[eval]   fake: $fake_dir"
    echo "[eval]   real: $GT_RESIZED"
    echo "[eval]   output: $output_dir"

    cd "$EVAL_METRICS_DIR"

    PYTHONPATH="${EVAL_METRICS_DIR}:${PYTHONPATH:-}" \
    PYTHONUNBUFFERED=1 \
    bash eval/run_metrics.sh \
        --real_videos_dir "$GT_RESIZED" \
        --fake_videos_dir "$fake_dir" \
        --shape_predictor_path "$SHAPE_PREDICTOR" \
        --output_dir "$output_dir" \
        --log_path "$output_dir/all_metrics.log" \
        --fallback_detection_confidence 0.2 \
        --ffmpeg_path "$FFMPEG_PATH" \
        --all

    local rc=$?
    echo "[eval] Standard metrics exit=$rc for $label"
    cd "$FASTGEN_DIR"
    return $rc
}

# ── Helper: run VBench ──
run_vbench() {
    local fake_dir="$1"
    local output_dir="$2"
    local label="$3"

    echo "[vbench] Running VBench for $label"

    cd "$EVAL_METRICS_DIR"

    # Run each dimension separately to avoid GPU OOM on longer videos
    local DIMS=(subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality)

    for dim in "${DIMS[@]}"; do
        echo "[vbench] $label: $dim"
        PYTHONPATH="${EVAL_METRICS_DIR}:${PYTHONPATH:-}" \
        PYTHONUNBUFFERED=1 \
        CUDA_VISIBLE_DEVICES=0 \
        python eval/eval_vbench.py \
            --fake_videos_dir "$fake_dir" \
            --real_videos_dir "$GT_RESIZED" \
            --output_dir "$output_dir/vbench" \
            --device cuda:0 \
            --skip_face_crops \
            --dimensions "$dim" 2>&1 | tail -5

        if [ $? -ne 0 ]; then
            echo "[vbench] WARNING: $dim failed for $label"
        fi
        sleep 2
    done

    # Extract results from stdout captured in log
    echo "[vbench] VBench complete for $label"
    cd "$FASTGEN_DIR"
}

# ── Main: Process DF checkpoints ──
echo "============================================"
echo "DF Evaluation Pipeline"
echo "============================================"
echo "GT (resized):   $GT_RESIZED"
echo "DF checkpoints: $DF_CKPT_DIR"
echo "Inference out:  $DF_OUTPUT_ROOT"
echo "Eval out:       $DF_EVAL_ROOT"
echo "============================================"

# Select checkpoints: every 1000th + latest
LATEST=$(ls -1 "$DF_CKPT_DIR"/*.pth 2>/dev/null | sort -V | tail -1)
THOUSANDS=$(ls -1 "$DF_CKPT_DIR"/*.pth 2>/dev/null | grep -E '000[0-9]+000\.pth$' | sort -V)
ALL_CKPTS=$(echo -e "${THOUSANDS}\n${LATEST}" | sort -V | uniq | grep '\.pth$')

echo "Checkpoints to process:"
echo "$ALL_CKPTS" | while read -r c; do echo "  $(basename "$c")"; done
echo ""

for CKPT_PATH in $ALL_CKPTS; do
    [ -z "$CKPT_PATH" ] && continue
    ITER_NAME=$(basename "$CKPT_PATH" .pth)
    INFER_DIR="${DF_OUTPUT_ROOT}/iter_${ITER_NAME}"
    EVAL_DIR="${DF_EVAL_ROOT}/iter_${ITER_NAME}"

    echo "================================================================"
    echo "Processing DF iter ${ITER_NAME}"
    echo "================================================================"

    # --- Phase 1: Inference ---
    TOTAL_GT=$(ls "$GT_RESIZED"/*.mp4 2>/dev/null | wc -l)
    EXISTING=$(ls "$INFER_DIR"/*.mp4 2>/dev/null | wc -l)

    if [ "$EXISTING" -ge "$TOTAL_GT" ] && [ "$TOTAL_GT" -gt 0 ]; then
        echo "[infer] Skipping inference — already complete (${EXISTING}/${TOTAL_GT} videos)"
    else
        echo "[infer] Running inference (${EXISTING}/${TOTAL_GT} existing)..."
        CHECKPOINTS="$ITER_NAME" bash scripts/inference/run_eval_hdtf_df.sh
        if [ $? -ne 0 ]; then
            echo "[infer] ERROR: Inference failed for iter ${ITER_NAME}, skipping eval"
            continue
        fi
    fi

    # Verify inference output count
    GENERATED=$(ls "$INFER_DIR"/*.mp4 2>/dev/null | wc -l)
    echo "[infer] Generated: ${GENERATED}/${TOTAL_GT} videos"

    # --- Phase 2: Standard eval metrics ---
    if [ -f "$EVAL_DIR/all_metrics.log" ]; then
        echo "[eval] Skipping standard metrics — already complete"
    else
        mkdir -p "$EVAL_DIR"
        run_eval_metrics "$INFER_DIR" "$EVAL_DIR" "DF_iter_${ITER_NAME}" || true
    fi

    # --- Phase 3: VBench ---
    if [ -f "$EVAL_DIR/vbench/vbench_metrics.log" ] || ls "$EVAL_DIR/vbench/composited/"*results* >/dev/null 2>&1; then
        echo "[vbench] Skipping VBench — already complete"
    else
        run_vbench "$INFER_DIR" "$EVAL_DIR" "DF_iter_${ITER_NAME}" || true
    fi

    echo ""
done

echo "============================================"
echo "All DF evaluations complete!"
echo "Results in: $DF_EVAL_ROOT"
echo "============================================"
