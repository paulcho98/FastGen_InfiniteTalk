#!/usr/bin/env python3
"""Build a training sample list from CSV metadata + available text_embeds.pt.

Filters out:
  - Non-16:9 aspect ratio (would produce wrong spatial resolution)
  - Videos shorter than min_duration (not enough frames for 81-frame training)
  - Samples without text_embeds.pt (T5 must be pre-computed)

Usage:
    python scripts/build_training_sample_list.py \
        --csv_path /path/to/video_list.csv \
        --output_dir data/precomputed_talkvid \
        --min_duration 3.5 \
        --val_count 10
"""
import argparse
import csv
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory containing precomputed sample dirs")
    parser.add_argument("--min_duration", type=float, default=3.5,
                        help="Minimum video duration in seconds (81 frames @ 25fps = 3.24s)")
    parser.add_argument("--aspect_ratio_tolerance", type=float, default=0.15,
                        help="Tolerance for 16:9 aspect ratio check")
    parser.add_argument("--val_count", type=int, default=10,
                        help="Number of samples to hold out for validation")
    args = parser.parse_args()

    # Read CSV
    with open(args.csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"CSV: {len(rows)} total videos")

    # Filter by aspect ratio and duration
    target_ratio = 16 / 9
    viable_names = set()
    filtered_ratio = 0
    filtered_short = 0

    for row in rows:
        h, w = int(row["height"]), int(row["width"])
        fps = float(row["fps"])
        total_frames = int(row["total_frames"])
        duration = total_frames / fps if fps > 0 else 0

        # Aspect ratio check
        if h <= 0 or abs(w / h - target_ratio) > args.aspect_ratio_tolerance:
            filtered_ratio += 1
            continue

        # Duration check
        if duration < args.min_duration:
            filtered_short += 1
            continue

        # Build expected sample dir name (matches precompute script convention)
        # CSV video_path: "data/-0F1owya2oo/-0F1owya2oo_189664_194706.mp4"
        # Sample dir:     "data_-0F1owya2oo_-0F1owya2oo_189664_194706"
        sample_name = os.path.splitext(
            row["video_path"].replace("/", "_")
        )[0]
        viable_names.add(sample_name)

    print(f"After filtering: {len(viable_names)} viable")
    print(f"  Filtered (aspect ratio): {filtered_ratio}")
    print(f"  Filtered (too short): {filtered_short}")

    # Find which viable samples have text_embeds.pt on disk
    available = []
    fully_precomputed = 0

    for name in sorted(viable_names):
        sample_dir = os.path.join(args.output_dir, name)
        text_path = os.path.join(sample_dir, "text_embeds.pt")
        if os.path.exists(text_path):
            available.append(sample_dir)
            if os.path.exists(os.path.join(sample_dir, "vae_latents.pt")):
                fully_precomputed += 1

    lazy_only = len(available) - fully_precomputed
    print(f"\nAvailable (have text_embeds.pt): {len(available)}")
    print(f"  Fully precomputed: {fully_precomputed}")
    print(f"  Lazy-eligible (T5 only): {lazy_only}")

    if len(available) <= args.val_count:
        print(f"ERROR: Not enough samples ({len(available)}) for val split ({args.val_count})")
        return

    # Split train/val
    train = available[:-args.val_count]
    val = available[-args.val_count:]

    # Write lists
    all_path = os.path.join(args.output_dir, "all_viable_samples.txt")
    train_path = os.path.join(args.output_dir, "all_viable_train.txt")
    val_path = os.path.join(args.output_dir, "all_viable_val.txt")

    with open(all_path, "w") as f:
        f.write("\n".join(available) + "\n")
    with open(train_path, "w") as f:
        f.write("\n".join(train) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(val) + "\n")

    print(f"\nWritten:")
    print(f"  {all_path} ({len(available)} samples)")
    print(f"  {train_path} ({len(train)} train)")
    print(f"  {val_path} ({len(val)} val)")


if __name__ == "__main__":
    main()
