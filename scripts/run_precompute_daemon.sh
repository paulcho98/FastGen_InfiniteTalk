#!/bin/bash
# Continuous precompute daemon — watches for samples with text_embeds.pt
# but missing other modalities (VAE, CLIP, audio, motion_frame) and fills them in.
#
# Designed to run alongside training: training creates text_embeds.pt via the
# T5-only precompute, then this daemon fills in the rest on a spare GPU.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=7 bash scripts/run_precompute_daemon.sh [INTERVAL_SECS]
#
# The daemon:
#   1. Scans OUTPUT_DIR for sample dirs with text_embeds.pt but missing vae_latents.pt
#   2. Runs precompute on those samples (per-file skip handles partial completion)
#   3. Sleeps INTERVAL_SECS then repeats
#   4. Ctrl+C to stop
#
# NOTE: This processes ALL discovered samples regardless of original batch size.
#       For ODE extraction, use the snapshot sample list from run_precompute_8gpu.sh
#       to avoid processing daemon-added samples.

set -e

INTERVAL=${1:-300}  # Check every 5 minutes by default

CSV_PATH="/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv"
DATA_ROOT="/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/"
OUTPUT_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/data/precomputed_talkvid"
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"

echo "Precompute daemon starting (interval: ${INTERVAL}s)"
echo "Watching: $OUTPUT_DIR"
echo "Press Ctrl+C to stop"

while true; do
    # Find samples with text_embeds.pt but missing vae_latents.pt
    INCOMPLETE=$(find "$OUTPUT_DIR" -name "text_embeds.pt" -exec dirname {} \; | while read d; do
        [ ! -f "$d/vae_latents.pt" ] && echo "$d"
    done | wc -l)

    if [ "$INCOMPLETE" -gt 0 ]; then
        echo "[$(date)] Found $INCOMPLETE incomplete samples, processing..."
        python scripts/precompute_infinitetalk_data.py \
            --csv_path "$CSV_PATH" \
            --data_root "$DATA_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            --weights_dir "$WEIGHTS_DIR" \
            --wav2vec_dir "$WAV2VEC_DIR" \
            --device cuda:0 \
            2>&1 | tail -5
    else
        echo "[$(date)] No incomplete samples found"
    fi

    sleep "$INTERVAL"
done
