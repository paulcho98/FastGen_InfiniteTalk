#!/bin/bash
# ODE trajectory extraction — 8-GPU distributed, uses torchrun.
# Each GPU loads the 14B teacher independently and processes a disjoint shard.
#
# Usage:
#   bash scripts/run_ode_extraction_8gpu.sh
#
# Timing: ~20 min/sample on 1 GPU. With 8 GPUs and 3000 samples:
#   375 samples/GPU × 20 min = ~5.2 days wall-clock
#
# Prerequisites:
#   - Precomputed data in DATA_DIR (from run_precompute_8gpu.sh)
#   - Each sample dir must have: vae_latents.pt, first_frame_cond.pt,
#     clip_features.pt, audio_emb.pt, text_embeds.pt
#   - neg_text_embeds.pt in DATA_DIR root

set -e

DATA_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/data/precomputed_talkvid"
WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
IT_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"

# Build sample list from precomputed data directories
SAMPLE_LIST="${DATA_DIR}/sample_list.txt"
find "$DATA_DIR" -name "vae_latents.pt" -exec dirname {} \; | sort > "$SAMPLE_LIST"
NUM_SAMPLES=$(wc -l < "$SAMPLE_LIST")
echo "Found $NUM_SAMPLES samples with precomputed data"

torchrun --nproc_per_node=8 scripts/generate_infinitetalk_ode_pairs.py \
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
