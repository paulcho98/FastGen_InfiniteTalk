#!/bin/bash
# Build a training sample list from all samples that have text_embeds.pt.
# Used for lazy-caching mode: samples with text_embeds.pt but missing other
# modalities will be encoded on-the-fly during training.
#
# Usage:
#   bash scripts/build_training_sample_list.sh [OUTPUT_DIR]
#
# Output:
#   OUTPUT_DIR/all_viable_samples.txt — one sample directory per line
#   OUTPUT_DIR/all_viable_train.txt  — 90% for training
#   OUTPUT_DIR/all_viable_val.txt    — 10 samples for validation

set -e

OUTPUT_DIR="${1:-data/precomputed_talkvid}"

echo "Scanning for samples with text_embeds.pt in $OUTPUT_DIR..."

# Find all directories containing text_embeds.pt
find "$OUTPUT_DIR" -name "text_embeds.pt" -printf '%h\n' | sort > "${OUTPUT_DIR}/all_viable_samples.txt"

TOTAL=$(wc -l < "${OUTPUT_DIR}/all_viable_samples.txt")
echo "Found $TOTAL viable samples"

# Split: last 10 for val, rest for train
VAL_COUNT=10
TRAIN_COUNT=$((TOTAL - VAL_COUNT))

if [ "$TRAIN_COUNT" -lt 1 ]; then
    echo "ERROR: Not enough samples for train/val split (need > $VAL_COUNT)"
    exit 1
fi

head -${TRAIN_COUNT} "${OUTPUT_DIR}/all_viable_samples.txt" > "${OUTPUT_DIR}/all_viable_train.txt"
tail -${VAL_COUNT} "${OUTPUT_DIR}/all_viable_samples.txt" > "${OUTPUT_DIR}/all_viable_val.txt"

echo "Train: ${TRAIN_COUNT} samples → ${OUTPUT_DIR}/all_viable_train.txt"
echo "Val:   ${VAL_COUNT} samples → ${OUTPUT_DIR}/all_viable_val.txt"

# Stats: how many are fully precomputed vs lazy-eligible
FULL=$(find "$OUTPUT_DIR" -name "text_embeds.pt" -execdir test -f vae_latents.pt \; -printf '%h\n' 2>/dev/null | wc -l)
LAZY=$((TOTAL - FULL))
echo ""
echo "Fully precomputed: $FULL"
echo "Lazy-eligible (T5 only): $LAZY"
