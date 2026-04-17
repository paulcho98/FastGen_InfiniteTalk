#!/usr/bin/env python3
"""Auto-detect new checkpoints, consolidate, run inference, and evaluate.

Polls training output directories for new FSDP checkpoints. For each new
checkpoint: consolidate → TalkVid inference → run metrics.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/inference/auto_eval_new_checkpoints.py \
        --poll_interval 300  # check every 5 min

    # One-shot (no polling, just process what's new):
    CUDA_VISIBLE_DEVICES=0 python scripts/inference/auto_eval_new_checkpoints.py --once
"""

import argparse
import gc
import os
import subprocess
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(FASTGEN_ROOT)
sys.path.insert(0, FASTGEN_ROOT)

# ── Paths ──
WEIGHTS_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
IT_CKPT = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH = f"{WEIGHTS_DIR}/Wan2.1_VAE.pth"
AUDIO_ROOT = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data"
TALKVID_LIST = os.path.join(FASTGEN_ROOT, "data/precomputed_talkvid/val_quarter_30.txt")
EVAL_METRICS_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/eval_metrics"
GT_DIR = os.path.join(FASTGEN_ROOT, "EVAL_OUTPUT/talkvid_val_gt_pixel_correct")
SHAPE_PRED = os.path.join(EVAL_METRICS_DIR, "shape_predictor_68_face_landmarks.dat")

# HDTF precomputed lists for pad inference
HDTF_PAD_LIST = os.path.join(FASTGEN_ROOT, "data/precomputed_hdtf_quarter_padded/sample_list.txt")

BASE_SHARDS = ",".join(sorted([
    os.path.join(WEIGHTS_DIR, f)
    for f in os.listdir(WEIGHTS_DIR)
    if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
]))

# Training runs to monitor
RUNS = {
    "sf_i2v": os.path.join(
        FASTGEN_ROOT,
        "FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/"
        "sf_quarter_i2v_freq5_lr1e5_accum4_0407_0100/checkpoints",
    ),
    "sf_softanchor": os.path.join(
        FASTGEN_ROOT,
        "FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/"
        "sf_quarter_softanchor_freq5_lr1e5_accum4_0408_0034/checkpoints",
    ),
}

# Auto-discover softboth runs
SF_BASE = os.path.join(FASTGEN_ROOT, "FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf")
if os.path.isdir(SF_BASE):
    for d in os.listdir(SF_BASE):
        if "softboth" in d:
            ckpt_dir = os.path.join(SF_BASE, d, "checkpoints")
            if os.path.isdir(ckpt_dir):
                RUNS["sf_softboth"] = ckpt_dir


def get_available_iters(ckpt_dir):
    """Get list of iterations that have FSDP shards."""
    iters = set()
    if not os.path.isdir(ckpt_dir):
        return iters
    for name in os.listdir(ckpt_dir):
        if name.endswith(".net_model") and os.path.isdir(os.path.join(ckpt_dir, name)):
            try:
                it = int(name.split(".")[0])
                iters.add(it)
            except ValueError:
                pass
    return iters


def get_completed_evals(tag):
    """Get iterations that already have inference + eval."""
    eval_root = os.path.join(FASTGEN_ROOT, "EVAL_OUTPUT")
    completed = set()
    for d in os.listdir(eval_root) if os.path.isdir(eval_root) else []:
        if d.startswith(f"talkvid_{tag}/iter_"):
            iter_str = d.split("iter_")[1]
            try:
                it = int(iter_str)
                mp4_dir = os.path.join(eval_root, d)
                mp4s = [f for f in os.listdir(mp4_dir) if f.endswith(".mp4")]
                if len(mp4s) >= 28:  # at least 28/30 done
                    completed.add(it)
            except (ValueError, OSError):
                pass
    return completed


def consolidate(ckpt_dir, iteration):
    """Consolidate FSDP checkpoint."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    src = os.path.join(ckpt_dir, f"{iteration:07d}.net_model")
    dst = os.path.join(ckpt_dir, f"{iteration:07d}_net_consolidated.pth")

    if os.path.exists(dst):
        try:
            sd = torch.load(dst, map_location="cpu", weights_only=False)
            del sd; gc.collect()
            return dst
        except Exception:
            os.remove(dst)

    if not os.path.isdir(src):
        return None

    print(f"    Consolidating iter {iteration}...")
    dcp_to_torch_save(src, dst)
    sd = torch.load(dst, map_location="cpu", weights_only=False)
    print(f"    -> {len(sd)} tensors")
    del sd; gc.collect()
    return dst


def run_inference(ckpt_path, output_dir, precomputed_list=None):
    """Run inference with given precomputed list."""
    cmd = [
        sys.executable, "-u", "scripts/inference/inference_causal.py",
        "--precomputed_list", precomputed_list or TALKVID_LIST,
        "--output_dir", output_dir, "--ckpt_path", ckpt_path,
        "--base_model_paths", BASE_SHARDS,
        "--infinitetalk_ckpt", IT_CKPT,
        "--vae_path", VAE_PATH,
        "--audio_data_root", AUDIO_ROOT,
        "--lora_rank", "128", "--lora_alpha", "64",
        "--chunk_size", "3", "--num_latent_frames", "21",
        "--seed", "42", "--context_noise", "0.0", "--quarter_res",
    ]
    return subprocess.run(cmd, cwd=FASTGEN_ROOT).returncode


def run_eval(fake_dir, output_dir):
    """Run metrics evaluation."""
    cmd = [
        "bash", os.path.join(EVAL_METRICS_DIR, "eval/run_metrics.sh"),
        "--real_videos_dir", GT_DIR,
        "--fake_videos_dir", fake_dir,
        "--shape_predictor_path", SHAPE_PRED,
        "--output_dir", output_dir,
        "--log_path", os.path.join(output_dir, "all_metrics.log"),
        "--fallback_detection_confidence", "0.2",
        "--all",
    ]
    return subprocess.run(cmd, cwd=EVAL_METRICS_DIR).returncode


def process_new_checkpoints():
    """Find and process any new checkpoints across all runs.

    Processes newest checkpoints first (reverse order) so the latest
    results are available sooner. For each checkpoint, runs:
    1. TalkVid inference + eval
    2. HDTF padded inference (precomputed, no eval — visual comparison only)
    """
    processed_any = False

    for tag, ckpt_dir in RUNS.items():
        if not os.path.isdir(ckpt_dir):
            continue

        available = get_available_iters(ckpt_dir)
        completed = get_completed_evals(tag)
        new_iters = sorted(available - completed, reverse=True)  # newest first

        if not new_iters:
            continue

        print(f"\n{'='*60}")
        print(f"[{tag}] New checkpoints (newest first): {new_iters}")
        print(f"{'='*60}")

        for it in new_iters:
            print(f"\n  --- {tag} iter {it} ---")

            # Consolidate
            ckpt_path = consolidate(ckpt_dir, it)
            if not ckpt_path:
                print(f"    SKIP: consolidation failed")
                continue

            # TalkVid inference
            infer_dir = os.path.join(FASTGEN_ROOT, f"EVAL_OUTPUT/talkvid_{tag}/iter_{it:07d}")
            print(f"    Running TalkVid inference...")
            rc = run_inference(ckpt_path, infer_dir)
            if rc != 0:
                print(f"    WARNING: TalkVid inference failed (exit {rc})")
                continue

            # TalkVid eval
            eval_dir = os.path.join(FASTGEN_ROOT, f"EVAL_OUTPUT/talkvid_{tag}/iter_{it:07d}_eval")
            print(f"    Running TalkVid evaluation...")
            run_eval(infer_dir, eval_dir)

            # HDTF padded inference
            hdtf_dir = os.path.join(FASTGEN_ROOT, f"EVAL_OUTPUT/hdtf_pad_{tag}/iter_{it:07d}")
            if os.path.isfile(HDTF_PAD_LIST):
                existing_hdtf = len([f for f in os.listdir(hdtf_dir) if f.endswith(".mp4")]) if os.path.isdir(hdtf_dir) else 0
                if existing_hdtf < 30:
                    print(f"    Running HDTF padded inference...")
                    run_inference(ckpt_path, hdtf_dir, precomputed_list=HDTF_PAD_LIST)
                else:
                    print(f"    HDTF padded: already done ({existing_hdtf} videos)")

            processed_any = True

    return processed_any


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll_interval", type=int, default=300,
                        help="Seconds between checks (default: 300)")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (no polling)")
    args = parser.parse_args()

    print(f"Monitoring {len(RUNS)} training runs:")
    for tag, path in RUNS.items():
        exists = os.path.isdir(path)
        print(f"  {tag}: {path} ({'exists' if exists else 'NOT FOUND'})")

    if args.once:
        process_new_checkpoints()
        print("\nDone (one-shot mode).")
        return

    print(f"\nPolling every {args.poll_interval}s... (Ctrl+C to stop)")
    while True:
        try:
            process_new_checkpoints()
            print(f"\n[{time.strftime('%H:%M:%S')}] Sleeping {args.poll_interval}s...")
            time.sleep(args.poll_interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
