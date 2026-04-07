#!/bin/bash
# =============================================================================
# Monitor SF i2v training checkpoints and run inference + evaluation on new ones.
#
# Polls the SF checkpoint directory every 5 minutes. When a new checkpoint
# appears, runs inference_causal.py then eval metrics + VBench.
#
# Usage:
#   bash scripts/inference/monitor_sf_eval.sh
# =============================================================================
set -u
set -o pipefail

# ── Paths ──
FASTGEN_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_METRICS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics"
GT_RESIZED="${FASTGEN_DIR}/EVAL_OUTPUT/hdtf_gt_448x224"
SHAPE_PREDICTOR="${EVAL_METRICS_DIR}/shape_predictor_68_face_landmarks.dat"
HDTF_VIDEO_DIR="/data/karlo-research_715/workspace/kinemaar/datasets/HDTF_original_testset_81frames/videos_cfr"
FFMPEG_PATH=$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || echo ffmpeg)

# ── SF Training Config ──
SF_CKPT_DIR="${SF_CKPT_DIR:-${FASTGEN_DIR}/FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/sf_quarter_i2v_freq5_lr1e5_accum4_0407_0013/checkpoints}"
SF_OUTPUT_ROOT="${FASTGEN_DIR}/EVAL_OUTPUT/hdtf_sf_i2v"
SF_EVAL_ROOT="${FASTGEN_DIR}/EVAL_OUTPUT/hdtf_sf_i2v_eval"

# ── Model weights (same as DF) ──
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
INFINITETALK_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH="${WEIGHTS_DIR}/Wan2.1_VAE.pth"
WAV2VEC_PATH="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"
CLIP_PATH="${WEIGHTS_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
BASE_SHARDS=$(ls "${WEIGHTS_DIR}"/diffusion_pytorch_model-*.safetensors | tr '\n' ',' | sed 's/,$//')

# LoRA config (same as DF/SF training)
LORA_RANK=128
LORA_ALPHA=64

# Polling interval
POLL_INTERVAL=${POLL_INTERVAL:-300}  # 5 minutes

# Track processed checkpoints
PROCESSED_FILE="${SF_EVAL_ROOT}/.processed_checkpoints"
mkdir -p "$SF_EVAL_ROOT"
touch "$PROCESSED_FILE"

run_sf_inference() {
    local ckpt_path="$1"
    local output_dir="$2"

    echo "[sf-infer] Running inference: $(basename "$ckpt_path")"
    cd "$FASTGEN_DIR"

    python scripts/inference/inference_causal.py \
        --video_dir "$HDTF_VIDEO_DIR" \
        --output_dir "$output_dir" \
        --ckpt_path "$ckpt_path" \
        --base_model_paths "$BASE_SHARDS" \
        --infinitetalk_ckpt "$INFINITETALK_CKPT" \
        --vae_path "$VAE_PATH" \
        --wav2vec_path "$WAV2VEC_PATH" \
        --clip_path "$CLIP_PATH" \
        --lora_rank "$LORA_RANK" \
        --lora_alpha "$LORA_ALPHA" \
        --chunk_size 3 \
        --num_latent_frames 21 \
        --seed 42 \
        --context_noise 0.0 \
        --target_h 224 \
        --target_w 448

    return $?
}

run_eval_metrics() {
    local fake_dir="$1"
    local output_dir="$2"
    local label="$3"

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
    cd "$FASTGEN_DIR"
}

run_vbench() {
    local fake_dir="$1"
    local output_dir="$2"
    local label="$3"

    cd "$EVAL_METRICS_DIR"
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
        sleep 2
    done
    cd "$FASTGEN_DIR"
}

echo "============================================"
echo "SF i2v Checkpoint Monitor"
echo "============================================"
echo "Watching:     $SF_CKPT_DIR"
echo "Poll interval: ${POLL_INTERVAL}s"
echo "Inference out: $SF_OUTPUT_ROOT"
echo "Eval out:      $SF_EVAL_ROOT"
echo "============================================"
echo ""

while true; do
    # Check if checkpoint directory exists yet
    if [ ! -d "$SF_CKPT_DIR" ]; then
        echo "[$(date +%H:%M:%S)] Checkpoint dir not yet created, waiting..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Find new checkpoints: detect .pth metadata files, consolidate FSDP shards if needed
    NEW_CKPTS=()
    for ckpt in $(ls -1 "$SF_CKPT_DIR"/*.pth 2>/dev/null | grep -v '_net_consolidated' | sort -V); do
        ckpt_name=$(basename "$ckpt")
        iter_name="${ckpt_name%.pth}"
        if ! grep -qF "$ckpt_name" "$PROCESSED_FILE" 2>/dev/null; then
            # SF checkpoints need FSDP consolidation before inference
            consolidated="${SF_CKPT_DIR}/${iter_name}_net_consolidated.pth"
            net_model_dir="${SF_CKPT_DIR}/${iter_name}.net_model"
            if [ ! -f "$consolidated" ] && [ -d "$net_model_dir" ]; then
                echo "[$(date +%H:%M:%S)] Consolidating FSDP shards for ${iter_name}..."
                python3 -c "
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
dcp_to_torch_save('${net_model_dir}', '${consolidated}')
print('Consolidation complete')
" 2>&1
                if [ ! -f "$consolidated" ]; then
                    echo "[$(date +%H:%M:%S)] WARNING: Consolidation failed for ${iter_name}, skipping"
                    echo "$ckpt_name" >> "$PROCESSED_FILE"
                    continue
                fi
            fi
            if [ -f "$consolidated" ]; then
                NEW_CKPTS+=("$consolidated")
            fi
        fi
    done

    if [ ${#NEW_CKPTS[@]} -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] No new checkpoints. Waiting ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    for CKPT_PATH in "${NEW_CKPTS[@]}"; do
        CKPT_BASENAME=$(basename "$CKPT_PATH")
        ITER_NAME=$(echo "$CKPT_BASENAME" | sed 's/_net_consolidated\.pth$//')
        INFER_DIR="${SF_OUTPUT_ROOT}/iter_${ITER_NAME}"
        EVAL_DIR="${SF_EVAL_ROOT}/iter_${ITER_NAME}"

        echo "================================================================"
        echo "[$(date)] Processing SF i2v iter ${ITER_NAME}"
        echo "================================================================"

        # Phase 1: Inference
        TOTAL_GT=$(ls "$GT_RESIZED"/*.mp4 2>/dev/null | wc -l)
        mkdir -p "$INFER_DIR"

        if ! run_sf_inference "$CKPT_PATH" "$INFER_DIR"; then
            echo "[sf-infer] ERROR: Inference failed for ${ITER_NAME}"
            echo "${ITER_NAME}.pth" >> "$PROCESSED_FILE"
            continue
        fi

        GENERATED=$(ls "$INFER_DIR"/*.mp4 2>/dev/null | wc -l)
        echo "[sf-infer] Generated: ${GENERATED}/${TOTAL_GT} videos"

        # Phase 2: Standard metrics
        mkdir -p "$EVAL_DIR"
        run_eval_metrics "$INFER_DIR" "$EVAL_DIR" "SF_i2v_${ITER_NAME}" || true

        # Phase 3: VBench
        run_vbench "$INFER_DIR" "$EVAL_DIR" "SF_i2v_${ITER_NAME}" || true

        # Mark as processed
        echo "$ITER_NAME.pth" >> "$PROCESSED_FILE"
        echo "[$(date)] SF i2v iter ${ITER_NAME} complete"
    done

    sleep "$POLL_INTERVAL"
done
