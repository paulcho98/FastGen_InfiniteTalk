#!/usr/bin/env python3
"""Run 4 HDTF inference combinations with a given checkpoint.

1. Precomputed crop (precomputed_hdtf_quarter/)
2. Precomputed pad (precomputed_hdtf_quarter_padded/)
3. On-the-fly crop (--video_dir --preprocess_mode crop)
4. On-the-fly pad (--video_dir --preprocess_mode pad)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/inference/run_hdtf_4way.py \
        --ckpt_path <consolidated.pth> --output_tag sf_softanchor_600
"""

import argparse
import os
import subprocess
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

WEIGHTS_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
IT_CKPT = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH = f"{WEIGHTS_DIR}/Wan2.1_VAE.pth"
CLIP_PATH = f"{WEIGHTS_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
WAV2VEC_PATH = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"
HDTF_VIDEO_DIR = "/data/karlo-research_715/workspace/kinemaar/datasets/HDTF_original_testset_81frames/videos_cfr"

BASE_SHARDS = ",".join(sorted([
    os.path.join(WEIGHTS_DIR, f)
    for f in os.listdir(WEIGHTS_DIR)
    if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
]))


def run_inference(output_dir, ckpt_path, extra_args):
    """Run inference_causal.py with given args."""
    cmd = [
        sys.executable, "-u",
        os.path.join(FASTGEN_ROOT, "scripts/inference/inference_causal.py"),
        "--output_dir", output_dir,
        "--ckpt_path", ckpt_path,
        "--base_model_paths", BASE_SHARDS,
        "--infinitetalk_ckpt", IT_CKPT,
        "--vae_path", VAE_PATH,
        "--lora_rank", "128", "--lora_alpha", "64",
        "--chunk_size", "3", "--num_latent_frames", "21",
        "--seed", "42", "--context_noise", "0.0",
        "--quarter_res",
    ] + extra_args

    print(f"  -> {output_dir}")
    result = subprocess.run(cmd, cwd=FASTGEN_ROOT)
    if result.returncode != 0:
        print(f"  WARNING: exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run 4 HDTF inference combinations")
    parser.add_argument("--ckpt_path", required=True, help="Consolidated checkpoint path")
    parser.add_argument("--output_tag", required=True, help="Tag for output dirs (e.g. sf_softanchor_600)")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt_path):
        print(f"ERROR: checkpoint not found: {args.ckpt_path}")
        sys.exit(1)

    base_out = os.path.join(FASTGEN_ROOT, "EVAL_OUTPUT")

    runs = [
        {
            "name": "1. Precomputed crop",
            "output_dir": f"{base_out}/hdtf_precomp_crop_{args.output_tag}",
            "args": [
                "--precomputed_list", os.path.join(FASTGEN_ROOT, "data/precomputed_hdtf_quarter/sample_list.txt"),
            ],
        },
        {
            "name": "2. Precomputed pad",
            "output_dir": f"{base_out}/hdtf_precomp_pad_{args.output_tag}",
            "args": [
                "--precomputed_list", os.path.join(FASTGEN_ROOT, "data/precomputed_hdtf_quarter_padded/sample_list.txt"),
            ],
        },
        {
            "name": "3. On-the-fly crop",
            "output_dir": f"{base_out}/hdtf_onthefly_crop_{args.output_tag}",
            "args": [
                "--video_dir", HDTF_VIDEO_DIR,
                "--clip_path", CLIP_PATH,
                "--wav2vec_path", WAV2VEC_PATH,
                "--target_h", "224", "--target_w", "448",
                "--preprocess_mode", "crop",
            ],
        },
        {
            "name": "4. On-the-fly pad",
            "output_dir": f"{base_out}/hdtf_onthefly_pad_{args.output_tag}",
            "args": [
                "--video_dir", HDTF_VIDEO_DIR,
                "--clip_path", CLIP_PATH,
                "--wav2vec_path", WAV2VEC_PATH,
                "--target_h", "224", "--target_w", "448",
                "--preprocess_mode", "pad",
            ],
        },
    ]

    for i, run in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"Run {run['name']}")
        print(f"{'='*60}")
        run_inference(run["output_dir"], args.ckpt_path, run["args"])

    print(f"\n{'='*60}")
    print("ALL 4 HDTF RUNS DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
