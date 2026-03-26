#!/bin/bash
# Step 1: Precompute T5 text embeddings for all samples.
# Auto-detects GPUs and distributes evenly.
#
# Usage:
#   bash scripts/run_precompute_t5.sh [NUM_SAMPLES]
#   NUM_SAMPLES: total samples to process (default: 3000, 0 = all)
#
# Timing: ~0.5s/sample. 3000 on 8 GPUs ≈ 3 min. 172K on 8 GPUs ≈ 3 hours.
# Resumable: skips samples that already have text_embeds.pt

set -e

NUM_SAMPLES=${1:-3000}

CSV_PATH="/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv"
OUTPUT_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/data/precomputed_talkvid"
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs"
echo "Processing $NUM_SAMPLES samples -> $OUTPUT_DIR"

NUM_SAMPLES_ARG=""
if [ "$NUM_SAMPLES" -gt 0 ]; then
    NUM_SAMPLES_ARG="--num_samples $NUM_SAMPLES"
fi

python scripts/precompute_t5_only.py \
    --csv_path "$CSV_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --weights_dir "$WEIGHTS_DIR" \
    $NUM_SAMPLES_ARG
