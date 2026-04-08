#!/usr/bin/env python3
"""Consolidate a single FSDP checkpoint and run TalkVid inference.

Usage:
    python scripts/inference/consolidate_and_infer.py \
        --run_dir FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/sf_quarter_softanchor_freq5_lr1e5_accum4_0408_0034 \
        --iter 300

    # With custom output tag (default: inferred from run dir name)
    python scripts/inference/consolidate_and_infer.py \
        --run_dir FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/sf_quarter_i2v_freq5_lr1e5_accum4_0407_0100 \
        --iter 700 \
        --output_tag sf_i2v

    # Skip consolidation if already done
    python scripts/inference/consolidate_and_infer.py \
        --run_dir ... --iter 400 --skip_consolidate
"""

import argparse
import gc
import os
import subprocess
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# ── Paths ──
WEIGHTS_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
IT_CKPT = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH = f"{WEIGHTS_DIR}/Wan2.1_VAE.pth"
AUDIO_ROOT = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data"
TALKVID_LIST = os.path.join(FASTGEN_ROOT, "data/precomputed_talkvid/val_quarter_30.txt")

BASE_SHARDS = ",".join(sorted([
    os.path.join(WEIGHTS_DIR, f)
    for f in os.listdir(WEIGHTS_DIR)
    if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
]))


def consolidate(ckpt_dir, iteration):
    """Consolidate FSDP sharded checkpoint to single .pth file."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    src = os.path.join(ckpt_dir, f"{iteration:07d}.net_model")
    dst = os.path.join(ckpt_dir, f"{iteration:07d}_net_consolidated.pth")

    if os.path.exists(dst):
        try:
            sd = torch.load(dst, map_location="cpu", weights_only=False)
            print(f"Already consolidated: {len(sd)} keys")
            del sd
            gc.collect()
            return dst
        except Exception:
            print("Corrupt file, reconsolidating...")
            os.remove(dst)

    if not os.path.isdir(src):
        print(f"ERROR: Shard dir not found: {src}")
        sys.exit(1)

    print(f"Consolidating {src} ...")
    dcp_to_torch_save(src, dst)
    sd = torch.load(dst, map_location="cpu", weights_only=False)
    lora_keys = [k for k in sd if "lora" in k]
    print(f"  -> {len(sd)} tensors, {len(lora_keys)} LoRA keys")
    del sd
    gc.collect()
    return dst


def run_inference(ckpt_path, output_dir):
    """Run inference_causal.py on TalkVid val."""
    cmd = [
        sys.executable, "-u",
        os.path.join(FASTGEN_ROOT, "scripts/inference/inference_causal.py"),
        "--precomputed_list", TALKVID_LIST,
        "--output_dir", output_dir,
        "--ckpt_path", ckpt_path,
        "--base_model_paths", BASE_SHARDS,
        "--infinitetalk_ckpt", IT_CKPT,
        "--vae_path", VAE_PATH,
        "--audio_data_root", AUDIO_ROOT,
        "--lora_rank", "128", "--lora_alpha", "64",
        "--chunk_size", "3", "--num_latent_frames", "21",
        "--seed", "42", "--context_noise", "0.0",
        "--quarter_res",
    ]

    print(f"Running inference -> {output_dir}")
    result = subprocess.run(cmd, cwd=FASTGEN_ROOT)
    if result.returncode != 0:
        print(f"WARNING: inference exited with code {result.returncode}")
    return result.returncode


def infer_output_tag(run_dir):
    """Infer a short output tag from the run directory name."""
    basename = os.path.basename(run_dir.rstrip("/"))
    # sf_quarter_softanchor_freq5_lr1e5_accum4_0408_0034 -> sf_softanchor
    # sf_quarter_i2v_freq5_lr1e5_accum4_0407_0100 -> sf_i2v
    # sf_quarter_freq5_lr1e5_accum4_0406_0833 -> sf_soft_cond
    parts = basename.split("_")
    if "softanchor" in parts:
        return "sf_softanchor"
    elif "i2v" in parts:
        return "sf_i2v"
    else:
        return "sf_soft_cond"


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate FSDP checkpoint + run TalkVid inference"
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the training run directory (contains checkpoints/)")
    parser.add_argument("--iter", type=int, required=True,
                        help="Iteration number to consolidate and evaluate")
    parser.add_argument("--output_tag", type=str, default=None,
                        help="Tag for output dir (default: auto from run name)")
    parser.add_argument("--skip_consolidate", action="store_true",
                        help="Skip consolidation (assume already done)")
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"ERROR: checkpoints dir not found: {ckpt_dir}")
        sys.exit(1)

    tag = args.output_tag or infer_output_tag(args.run_dir)
    output_dir = os.path.join(
        FASTGEN_ROOT,
        f"EVAL_OUTPUT/talkvid_{tag}/iter_{args.iter:07d}",
    )

    print("=" * 60)
    print(f"Run:       {os.path.basename(args.run_dir.rstrip('/'))}")
    print(f"Iter:      {args.iter}")
    print(f"Tag:       {tag}")
    print(f"Output:    {output_dir}")
    print("=" * 60)

    # Step 1: Consolidate
    if args.skip_consolidate:
        ckpt_path = os.path.join(ckpt_dir, f"{args.iter:07d}_net_consolidated.pth")
        if not os.path.isfile(ckpt_path):
            print(f"ERROR: --skip_consolidate but file not found: {ckpt_path}")
            sys.exit(1)
        print(f"Skipping consolidation, using: {ckpt_path}")
    else:
        print("\n[1/2] Consolidating...")
        ckpt_path = consolidate(ckpt_dir, args.iter)

    # Step 2: Inference
    print(f"\n[2/2] Running inference...")
    rc = run_inference(ckpt_path, output_dir)

    print(f"\nDone! Output: {output_dir}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
