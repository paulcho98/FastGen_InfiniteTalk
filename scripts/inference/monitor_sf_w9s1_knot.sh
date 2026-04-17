#!/bin/bash
# =============================================================================
# Monitor SF w9s1+KF+RA+last-ref training run and run inference + eval
# on every checkpoint, on TalkVid val (30 samples).
#
# Training is on a separate instance, writing to the mounted NFS dir. Our
# single GPU is free — no conflict.
#
# Inference uses the KF flags that match the training config:
#   --use_dynamic_rope --use_temporal_knot --knot_size 1
#   --use_running_ahead --running_ahead_step 4 --running_ahead_init_n 8
#   --use_last_frame_reference
#   (anchors ON by default; don't pass --no_anchor_first_frame)
#   (F1 / F3 OFF — mutually exclusive with KF)
#
# Write-atomicity guard: only consolidate after the iter's `.pth` metadata file
# has been stable (mtime > STABILITY_SECONDS old). Training writes the DCP
# shards first, then the tiny `.pth` last — so mtime-stable `.pth` => shards
# are complete.
#
# VBench is SKIPPED (per user request). Only standard metrics.
# No deletions: consolidated checkpoints and eval outputs accumulate.
# =============================================================================
set -u
set -o pipefail

# ── Run config ──
RUN_NAME="sf_w9s1_knot_k1_ra_s4_n8_lastref"
SF_CKPT_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/${RUN_NAME}/checkpoints"
KNOT_SIZE=1
RUNNING_AHEAD_STEP=4
RUNNING_AHEAD_INIT_N=8

# ── Shared paths ──
FASTGEN_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk"
EVAL_METRICS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics"
GT_DIR="${FASTGEN_DIR}/EVAL_OUTPUT/talkvid_val_gt_448x224"
SHAPE_PREDICTOR="${EVAL_METRICS_DIR}/shape_predictor_68_face_landmarks.dat"
FFMPEG_PATH=$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || echo ffmpeg)

# ── Outputs (scoped to this run) ──
OUTPUT_ROOT="${FASTGEN_DIR}/EVAL_OUTPUT/talkvid_val_${RUN_NAME}"
EVAL_ROOT="${FASTGEN_DIR}/EVAL_OUTPUT/talkvid_val_${RUN_NAME}_eval"

# ── Model weights ──
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
INFINITETALK_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH="${WEIGHTS_DIR}/Wan2.1_VAE.pth"
WAV2VEC_PATH="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"
CLIP_PATH="${WEIGHTS_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
BASE_SHARDS=$(ls "${WEIGHTS_DIR}"/diffusion_pytorch_model-*.safetensors | tr '\n' ',' | sed 's/,$//')

# ── Inference knobs (match training) ──
LORA_RANK=128
LORA_ALPHA=64
CHUNK_SIZE=3
NUM_LATENT_FRAMES=21
SEED=42
TARGET_H=224
TARGET_W=448

# ── Data ──
PRECOMPUTED_LIST="${FASTGEN_DIR}/data/precomputed_talkvid/val_quarter_30.txt"
AUDIO_DATA_ROOT="/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data"

# ── Polling / stability ──
POLL_INTERVAL=${POLL_INTERVAL:-300}       # 5 min
STABILITY_SECONDS=${STABILITY_SECONDS:-90} # require .pth mtime > 90s old

# Track processed checkpoints
PROCESSED_FILE="${EVAL_ROOT}/.processed_checkpoints"
mkdir -p "$EVAL_ROOT" "$OUTPUT_ROOT"
touch "$PROCESSED_FILE"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

consolidate_ckpt() {
    local iter_name="$1"
    local net_model_dir="${SF_CKPT_DIR}/${iter_name}.net_model"
    local consolidated="${SF_CKPT_DIR}/${iter_name}_net_consolidated.pth"

    if [ -f "$consolidated" ]; then
        log "[consolidate] ${iter_name} already consolidated"
        return 0
    fi
    if [ ! -d "$net_model_dir" ]; then
        log "[consolidate] ${iter_name}.net_model dir missing; skipping"
        return 1
    fi

    log "[consolidate] Consolidating FSDP shards for ${iter_name}..."
    python3 -c "
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
dcp_to_torch_save('${net_model_dir}', '${consolidated}')
print('Consolidation complete: ${consolidated}')
"
    if [ ! -f "$consolidated" ]; then
        log "[consolidate] WARNING: consolidation failed for ${iter_name}"
        return 1
    fi
    return 0
}

run_sf_inference() {
    local ckpt_path="$1"
    local output_dir="$2"

    log "[infer] Running SF inference: $(basename "$ckpt_path")"
    cd "$FASTGEN_DIR"

    python scripts/inference/inference_causal.py \
        --precomputed_list "$PRECOMPUTED_LIST" \
        --output_dir "$output_dir" \
        --ckpt_path "$ckpt_path" \
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
        --use_temporal_knot \
        --knot_size "$KNOT_SIZE" \
        --use_running_ahead \
        --running_ahead_step "$RUNNING_AHEAD_STEP" \
        --running_ahead_init_n "$RUNNING_AHEAD_INIT_N" \
        --use_last_frame_reference
    local rc=$?
    return $rc
}

run_eval_metrics() {
    local fake_dir="$1"
    local output_dir="$2"

    cd "$EVAL_METRICS_DIR"
    PYTHONPATH="${EVAL_METRICS_DIR}:${PYTHONPATH:-}" \
    PYTHONUNBUFFERED=1 \
    bash eval/run_metrics.sh \
        --real_videos_dir "$GT_DIR" \
        --fake_videos_dir "$fake_dir" \
        --shape_predictor_path "$SHAPE_PREDICTOR" \
        --output_dir "$output_dir" \
        --log_path "$output_dir/all_metrics.log" \
        --fallback_detection_confidence 0.2 \
        --ffmpeg_path "$FFMPEG_PATH" \
        --all
    cd "$FASTGEN_DIR"
}

echo "============================================"
echo "SF Monitor: $RUN_NAME"
echo "============================================"
echo "Watching:         $SF_CKPT_DIR"
echo "Poll interval:    ${POLL_INTERVAL}s"
echo "Stability guard:  ${STABILITY_SECONDS}s on .pth mtime"
echo "Temporal Knot:    k=${KNOT_SIZE}"
echo "Running Ahead:    s=${RUNNING_AHEAD_STEP}, init_n=${RUNNING_AHEAD_INIT_N}"
echo "Last-frame ref:   ON"
echo "Inference out:    $OUTPUT_ROOT"
echo "Eval out:         $EVAL_ROOT"
echo "VBench:           SKIPPED"
echo "============================================"
echo ""

while true; do
    if [ ! -d "$SF_CKPT_DIR" ]; then
        log "Checkpoint dir not yet created, waiting..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Find candidate checkpoints: real .pth (not _net_consolidated)
    NEW_CKPTS=()
    now_epoch=$(date +%s)
    for pth in $(ls -1 "$SF_CKPT_DIR"/*.pth 2>/dev/null | grep -v '_net_consolidated' | sort -V); do
        ckpt_name=$(basename "$pth")
        iter_name="${ckpt_name%.pth}"

        # Skip already-processed
        if grep -qF "$ckpt_name" "$PROCESSED_FILE" 2>/dev/null; then
            continue
        fi

        # Stability guard: .pth mtime must be older than STABILITY_SECONDS
        mtime=$(stat -c %Y "$pth")
        age=$(( now_epoch - mtime ))
        if [ "$age" -lt "$STABILITY_SECONDS" ]; then
            log "[stability] ${ckpt_name} age=${age}s < ${STABILITY_SECONDS}s, deferring"
            continue
        fi

        # Consolidate (idempotent)
        if consolidate_ckpt "$iter_name"; then
            consolidated="${SF_CKPT_DIR}/${iter_name}_net_consolidated.pth"
            NEW_CKPTS+=("$consolidated")
        else
            log "[consolidate] marking ${ckpt_name} as processed (consolidation failed)"
            echo "$ckpt_name" >> "$PROCESSED_FILE"
        fi
    done

    if [ ${#NEW_CKPTS[@]} -eq 0 ]; then
        log "No new checkpoints ready. Sleeping ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    for CKPT_PATH in "${NEW_CKPTS[@]}"; do
        CKPT_BASENAME=$(basename "$CKPT_PATH")
        ITER_NAME="${CKPT_BASENAME%_net_consolidated.pth}"
        INFER_DIR="${OUTPUT_ROOT}/iter_${ITER_NAME}"
        EVAL_DIR="${EVAL_ROOT}/iter_${ITER_NAME}"

        echo "================================================================"
        log "Processing iter ${ITER_NAME}"
        echo "================================================================"

        # Phase 1: Inference
        TOTAL_GT=$(ls "$GT_DIR"/*.mp4 2>/dev/null | wc -l)
        EXISTING=$(ls "$INFER_DIR"/*.mp4 2>/dev/null | wc -l)
        mkdir -p "$INFER_DIR"

        if [ "$EXISTING" -ge "$TOTAL_GT" ] && [ "$TOTAL_GT" -gt 0 ]; then
            log "[infer] Skipping — already complete (${EXISTING}/${TOTAL_GT})"
        else
            if ! run_sf_inference "$CKPT_PATH" "$INFER_DIR"; then
                log "[infer] ERROR: Inference failed for ${ITER_NAME}"
                echo "${ITER_NAME}.pth" >> "$PROCESSED_FILE"
                continue
            fi
            GENERATED=$(ls "$INFER_DIR"/*.mp4 2>/dev/null | wc -l)
            log "[infer] Generated: ${GENERATED}/${TOTAL_GT} videos"
        fi

        # Phase 2: Standard metrics
        mkdir -p "$EVAL_DIR"
        if [ -f "$EVAL_DIR/all_metrics.log" ]; then
            log "[eval] Skipping standard metrics — already complete"
        else
            log "[eval] Running standard metrics..."
            run_eval_metrics "$INFER_DIR" "$EVAL_DIR" || log "[eval] WARNING: some metrics failed"
        fi

        # Mark processed
        echo "${ITER_NAME}.pth" >> "$PROCESSED_FILE"
        log "iter ${ITER_NAME} DONE"
    done

    sleep "$POLL_INTERVAL"
done
