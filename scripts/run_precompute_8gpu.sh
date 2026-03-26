#!/bin/bash
# Parallel precompute across 8 GPUs.
# Each GPU loads all encoders independently and processes a disjoint shard.
#
# Usage:
#   bash scripts/run_precompute_8gpu.sh [NUM_SAMPLES]
#   NUM_SAMPLES: total samples to process (default: 3000)
#
# With 8 GPUs: ~13s/sample ÷ 8 = ~1.6s effective, 3000 samples in ~80 min.

set -e

NUM_SAMPLES=${1:-3000}
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs"
SHARD_SIZE=$(( (NUM_SAMPLES + NUM_GPUS - 1) / NUM_GPUS ))  # ceiling division

CSV_PATH="/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv"
DATA_ROOT="/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/"
OUTPUT_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/data/precomputed_talkvid"
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"

mkdir -p "$OUTPUT_DIR"

echo "Launching precompute on $NUM_GPUS GPUs for $NUM_SAMPLES samples (shard_size=$SHARD_SIZE)"
echo "Output: $OUTPUT_DIR"

PIDS=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((gpu * SHARD_SIZE))
    END=$(( (gpu + 1) * SHARD_SIZE ))
    if [ $END -gt $NUM_SAMPLES ]; then
        END=$NUM_SAMPLES
    fi
    if [ $START -ge $NUM_SAMPLES ]; then
        break
    fi

    echo "  GPU $gpu: samples [$START, $END)"
    CUDA_VISIBLE_DEVICES=$gpu python scripts/precompute_infinitetalk_data.py \
        --csv_path "$CSV_PATH" \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        --weights_dir "$WEIGHTS_DIR" \
        --wav2vec_dir "$WAV2VEC_DIR" \
        --num_samples "$NUM_SAMPLES" \
        --start_idx "$START" \
        --end_idx "$END" \
        --device cuda:0 \
        > "${OUTPUT_DIR}/log_gpu${gpu}.txt" 2>&1 &
    PIDS+=($!)
done

echo "All $NUM_GPUS processes launched. PIDs: ${PIDS[*]}"
echo "Logs: ${OUTPUT_DIR}/log_gpu*.txt"
echo "Waiting for completion..."

FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done

if [ $FAIL -eq 0 ]; then
    echo "All GPUs completed successfully."
else
    echo "WARNING: $FAIL GPU(s) failed. Check logs."
fi
