#!/usr/bin/env python3
"""Verify our ported ODE pipeline by running 40-step denoise, decoding with VAE,
and saving the result as a video with muxed audio.

Uses the same ODE loop as generate_infinitetalk_ode_pairs.py but instead of
saving trajectory states, decodes the final denoised latent to pixel space.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/verify_ode_video.py \
        --sample_dir data/test_precomputed/data_-0F1owya2oo_-0F1owya2oo_189664_194706 \
        --audio_path /data/.../TalkVid/data/-0F1owya2oo/-0F1owya2oo_189664_194706.wav \
        --weights_dir /data/.../Wan2.1-I2V-14B-480P \
        --infinitetalk_ckpt /data/.../infinitetalk.safetensors \
        --output_dir data/test_precomputed/visualizations
"""
import os
import sys
import time
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# Add InfiniteTalk root for WanVAE import
IT_ROOT = os.path.normpath(os.path.join(REPO_ROOT, "../InfiniteTalk"))
sys.path.insert(0, IT_ROOT)

# Mock xformers (not installed, wan/modules/attention.py imports it at module level)
import types
import importlib.machinery

class _MockModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None

for p in ["xformers", "xformers.ops", "xformers.ops.fmha", "xformers.ops.fmha.attn_bias"]:
    parts = p.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[:i+1])
        if partial not in sys.modules:
            m = _MockModule(partial)
            if i < len(parts) - 1:
                m.__path__ = []
            m.__spec__ = importlib.machinery.ModuleSpec(partial, None)
            sys.modules[partial] = m

def _memory_efficient_attention(q, k, v, attn_bias=None, op=None):
    from flash_attn import flash_attn_func
    return flash_attn_func(q, k, v)

sys.modules["xformers"].ops = sys.modules["xformers.ops"]
sys.modules["xformers.ops"].memory_efficient_attention = _memory_efficient_attention

class _MockBlockDiag:
    @staticmethod
    def from_seqlens(*a, **k):
        return None
sys.modules["xformers.ops.fmha.attn_bias"].BlockDiagonalMask = _MockBlockDiag
sys.modules["xformers.ops.fmha"].attn_bias = sys.modules["xformers.ops.fmha.attn_bias"]

import torch
import numpy as np
from PIL import Image

# Reuse ODE extraction functions
from scripts.generate_infinitetalk_ode_pairs import (
    build_teacher,
    load_sample,
    build_audio_windows,
    extract_ode_trajectory,
    get_shifted_timesteps,
)


def load_vae(weights_dir: str, device: str):
    """Load WanVAE for decoding."""
    from wan.modules.vae import WanVAE
    vae_path = os.path.join(weights_dir, "Wan2.1_VAE.pth")
    print(f"Loading VAE from {vae_path}")
    vae = WanVAE(vae_pth=vae_path, device=device)
    return vae


def decode_latent_to_frames(vae, latent):
    """Decode VAE latent [1, C, T, H, W] → numpy frames [T, H, W, 3] uint8."""
    with torch.no_grad():
        # WanVAE.decode expects [B, C, T, H, W] float32
        video = vae.decode(latent.float())
        # Output: [B, C, T, H, W] in [-1, 1]
        video = video[0].clamp(-1, 1)  # [C, T, H, W]
        # → [T, H, W, C] uint8
        frames = video.permute(1, 2, 3, 0)  # [T, H, W, C]
        frames = ((frames + 1) / 2 * 255).byte().cpu().numpy()
    return frames


def save_video_with_audio(frames, audio_path, output_path, fps=25):
    """Save numpy frames [T, H, W, 3] as MP4 with muxed audio."""
    import imageio_ffmpeg

    num_frames, h, w, _ = frames.shape
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Write raw frames → MP4 (no audio)
    noaudio_path = output_path.replace(".mp4", "_noaudio.mp4")
    writer_cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        noaudio_path,
    ]
    proc = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(num_frames):
        proc.stdin.write(frames[i].tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"  Video (no audio): {noaudio_path} ({num_frames} frames @ {fps}fps)")

    if audio_path and os.path.exists(audio_path):
        # Mux audio
        duration = num_frames / fps
        mux_cmd = [
            ffmpeg_exe, "-y",
            "-i", noaudio_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-t", f"{duration:.3f}",
            "-shortest",
            output_path,
        ]
        result = subprocess.run(mux_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Video with audio: {output_path}")
            os.remove(noaudio_path)
        else:
            print(f"  Audio mux failed: {result.stderr[:300]}")
            os.rename(noaudio_path, output_path)
    else:
        os.rename(noaudio_path, output_path)
        print(f"  Video (no audio source): {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Directory with precomputed .pt files")
    parser.add_argument("--audio_path", type=str, default="",
                        help="Source audio WAV for muxing")
    parser.add_argument("--neg_text_emb_path", type=str,
                        default="data/test_precomputed/neg_text_embeds.pt")
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Wan2.1-I2V-14B-480P directory")
    parser.add_argument("--infinitetalk_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="data/test_precomputed/visualizations")
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--shift", type=float, default=7.0)
    parser.add_argument("--text_guide_scale", type=float, default=5.0)
    parser.add_argument("--audio_guide_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load teacher ──
    print("=" * 60)
    print("ODE Video Verification — Our Ported Pipeline")
    print("=" * 60)
    t0 = time.time()
    teacher = build_teacher(
        weights_dir=args.weights_dir,
        infinitetalk_ckpt=args.infinitetalk_ckpt,
        shift=args.shift,
        device=device,
        dtype=dtype,
    )
    print(f"Teacher loaded in {time.time()-t0:.0f}s")

    # ── Load VAE for decoding ──
    vae = load_vae(args.weights_dir, args.device)

    # ── Load sample data ──
    print(f"\nLoading sample: {args.sample_dir}")
    neg_text_emb = torch.load(args.neg_text_emb_path, map_location="cpu", weights_only=True)

    data = load_sample(
        sample_dir=args.sample_dir,
        neg_text_emb=neg_text_emb,
        device=device,
        dtype=dtype,
        load_ode_path=False,
    )

    if data is None:
        print("ERROR: Could not load sample data")
        return

    # Build audio windows (4D → 5D)
    audio_emb = data["condition"]["audio_emb"]
    if audio_emb.dim() == 4:  # [B, T, 12, 768]
        audio_windowed = build_audio_windows(audio_emb, window_size=5)
        data["condition"]["audio_emb"] = audio_windowed
        data["neg_text_condition"]["audio_emb"] = audio_windowed
        # neg_condition keeps zero audio

    latent_shape = data["real"].shape[1:]  # [C, T, H, W]
    print(f"Latent shape: {list(latent_shape)}")

    # ── Run ODE denoise ──
    print(f"\nRunning {args.num_steps}-step ODE with 3-call CFG "
          f"(text={args.text_guide_scale}, audio={args.audio_guide_scale})...")

    # Use target_t_list with just 0.0 so we get the clean state
    t_start = time.time()
    # Actually, we want the FINAL denoised state. The extract_ode_trajectory
    # returns subsampled states. We need to run the full ODE and get the last x_t.
    # Easiest: use target_t_list that includes 0.0, or we just run the loop directly.

    # Run the full ODE loop ourselves for simplicity
    from scripts.generate_infinitetalk_ode_pairs import get_shifted_timesteps

    num_timesteps = 1000
    timesteps = get_shifted_timesteps(
        args.num_steps, shift=args.shift, num_timesteps=num_timesteps, device=device,
    )
    t_schedule = timesteps / num_timesteps

    # Init from noise
    torch.manual_seed(args.seed)
    noise = torch.randn(1, *latent_shape, device=device, dtype=torch.float32)
    x_t = noise

    # Motion frame
    condition = data["condition"]
    neg_text_condition = data["neg_text_condition"]
    neg_condition = data["neg_condition"]

    if "motion_frame" in condition:
        motion_frame = condition["motion_frame"].to(torch.float32)
    else:
        motion_frame = condition["first_frame_cond"][:, :, :1, :, :].to(torch.float32)

    # ODE loop
    for step_idx in range(args.num_steps):
        t_cur = t_schedule[step_idx]
        t_next = t_schedule[step_idx + 1]

        # Anchor first frame
        x_t[:, :, :1, :, :] = motion_frame

        t_tensor = torch.full((1,), t_cur.item(), device=device, dtype=dtype)
        x_t_model = x_t.to(dtype)

        # 3-call CFG
        v_cond = teacher(x_t_model, t_tensor, condition=condition)
        v_drop_text = teacher(x_t_model, t_tensor, condition=neg_text_condition)
        v_uncond = teacher(x_t_model, t_tensor, condition=neg_condition)

        v_guided = (
            v_uncond
            + args.text_guide_scale * (v_cond - v_drop_text)
            + args.audio_guide_scale * (v_drop_text - v_uncond)
        )

        # Euler step
        v_guided = -v_guided.float()
        dt = t_cur - t_next
        x_t = x_t + v_guided * dt

        # Anchor after step
        x_t[:, :, :1, :, :] = motion_frame

        if (step_idx + 1) % 10 == 0 or step_idx == 0:
            print(f"  Step {step_idx+1}/{args.num_steps} | "
                  f"t={t_cur.item():.4f}→{t_next.item():.4f} | "
                  f"x_t range: [{x_t.min():.3f}, {x_t.max():.3f}]")

    ode_time = time.time() - t_start
    print(f"ODE completed in {ode_time:.0f}s")

    # ── Decode with VAE ──
    print("\nDecoding latent with VAE...")
    t_dec = time.time()
    frames = decode_latent_to_frames(vae, x_t)
    print(f"Decoded in {time.time()-t_dec:.1f}s | frames shape: {frames.shape}")

    # ── Save sample frames as PNGs ──
    sample_name = os.path.basename(args.sample_dir)
    for t_idx in [0, 20, 40, 60, 80]:
        if t_idx >= frames.shape[0]:
            continue
        path = os.path.join(args.output_dir, f"ported_ode_{sample_name}_frame{t_idx}.png")
        Image.fromarray(frames[t_idx]).save(path)
        print(f"  Saved {path}")

    # ── Save full video with audio ──
    video_path = os.path.join(args.output_dir, f"ported_ode_{sample_name}.mp4")
    save_video_with_audio(frames, args.audio_path, video_path, fps=25)

    print("\nDone!")


if __name__ == "__main__":
    main()
