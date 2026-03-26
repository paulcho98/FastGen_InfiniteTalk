#!/bin/bash
# ODE trajectory extraction — distributed across all available GPUs via torchrun.
# Each GPU loads the 14B teacher independently and processes a disjoint shard.
#
# Usage:
#   bash scripts/run_ode_extraction_8gpu.sh [SAMPLE_LIST]
#
#   SAMPLE_LIST: Optional path to a text file listing sample directories.
#                If not provided, auto-discovers ALL samples with vae_latents.pt.
#                Use a fixed list to prevent processing lazily-cached samples
#                that weren't part of the original preprocessing batch.
#
# Auto-detects available GPUs. Timing: ~20 min/sample per GPU.
# With 8 GPUs and 3000 samples: 375 samples/GPU × 20 min ≈ 5.2 days.
#
# Prerequisites:
#   - Precomputed data in DATA_DIR (from run_precompute_8gpu.sh)
#   - Each sample dir must have: vae_latents.pt, first_frame_cond.pt,
#     clip_features.pt, audio_emb.pt, text_embeds.pt
#   - neg_text_embeds.pt in DATA_DIR root
# Resumable: --skip_existing skips samples that already have ode_path.pt

set -e

DATA_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/data/precomputed_talkvid"
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
IT_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"

# Auto-detect GPUs
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected $NUM_GPUS GPUs"

# Use provided sample list, or auto-discover
SAMPLE_LIST="${1:-}"
if [ -z "$SAMPLE_LIST" ]; then
    SAMPLE_LIST="${DATA_DIR}/ode_sample_list.txt"
    echo "No sample list provided — auto-discovering from $DATA_DIR"
    find "$DATA_DIR" -name "vae_latents.pt" -exec dirname {} \; | sort > "$SAMPLE_LIST"
else
    echo "Using provided sample list: $SAMPLE_LIST"
fi
NUM_SAMPLES=$(wc -l < "$SAMPLE_LIST")
echo "Found $NUM_SAMPLES samples"

torchrun --nproc_per_node=$NUM_GPUS scripts/generate_infinitetalk_ode_pairs.py \
    --data_list_path "$SAMPLE_LIST" \
    --neg_text_emb_path "${DATA_DIR}/neg_text_embeds.pt" \
    --weights_dir "$WEIGHTS_DIR" \
    --infinitetalk_ckpt "$IT_CKPT" \
    --num_steps 40 \
    --text_guide_scale 5.0 \
    --audio_guide_scale 4.0 \
    --batch_size 1 \
    --skip_existing \
    --seed 42
