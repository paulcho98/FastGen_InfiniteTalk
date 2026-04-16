#!/bin/bash
# Precompute future anchor latents in parallel across 8 GPUs.
# Each GPU processes ~18.3k samples. Skip-if-exists is on by default,
# so this is safe to re-run if interrupted.
set -e

SAMPLE_LIST="data/precomputed_talkvid/train_excl_val30.txt"
RAW_DATA_ROOT="/data/karlo-research_715/workspace/kinemaar/datasets/train/TalkVid/"
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"

TOTAL=$(wc -l < "$SAMPLE_LIST")
NUM_GPUS=8
SHARD_SIZE=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))

echo "Precomputing future anchor latents (quarter res)"
echo "  Samples: $TOTAL"
echo "  GPUs: $NUM_GPUS"
echo "  Shard size: ~$SHARD_SIZE per GPU"
echo ""

# Also precompute val set on GPU 0 first (fast, 30 samples)
echo "Precomputing val set on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_future_anchors.py \
    --sample_list data/precomputed_talkvid/val_quarter_30.txt \
    --raw_data_root "$RAW_DATA_ROOT" \
    --weights_dir "$WEIGHTS_DIR" \
    --quarter_res

echo "Launching $NUM_GPUS parallel shards for training set..."
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((GPU_ID * SHARD_SIZE))
    END=$(( (GPU_ID + 1) * SHARD_SIZE ))
    if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

    echo "  GPU $GPU_ID: samples [$START, $END)"
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/precompute_future_anchors.py \
        --sample_list "$SAMPLE_LIST" \
        --raw_data_root "$RAW_DATA_ROOT" \
        --weights_dir "$WEIGHTS_DIR" \
        --quarter_res \
        --start_idx $START \
        --end_idx $END \
        > "logs/precompute_anchor_gpu${GPU_ID}.log" 2>&1 &
done

echo ""
echo "All shards launched. Logs: logs/precompute_anchor_gpu{0..7}.log"
echo "Monitor: tail -f logs/precompute_anchor_gpu*.log"
echo "Wait:    wait"

wait
echo "Done! All $NUM_GPUS shards complete."
