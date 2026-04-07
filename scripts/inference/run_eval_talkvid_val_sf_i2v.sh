#!/bin/bash
# Evaluate SF i2v checkpoint on TalkVid validation set (30 samples).
# Runs inference (precomputed) → standard metrics → VBench.
#
# Usage:
#   bash scripts/inference/run_eval_talkvid_val_sf_i2v.sh
#   CHECKPOINTS="0000100 0000200" bash scripts/inference/run_eval_talkvid_val_sf_i2v.sh

cd "$(dirname "$0")/../.."

CHECKPOINTS="${CHECKPOINTS:-0000300}" \
CKPT_DIR=FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/sf_quarter_i2v_freq5_lr1e5_accum4_0407_0100/checkpoints \
CKPT_SUFFIX="_net_consolidated" \
OUTPUT_ROOT=EVAL_OUTPUT/talkvid_val_sf_i2v \
EVAL_ROOT=EVAL_OUTPUT/talkvid_val_sf_i2v_eval \
  bash scripts/inference/run_eval_talkvid_val.sh
