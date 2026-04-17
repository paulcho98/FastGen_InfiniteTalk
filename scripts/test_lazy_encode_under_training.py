#!/usr/bin/env python
"""Test: can lazy encoding survive alongside the 14B training model?

Simulates the real scenario:
1. Load the full 14B CausalInfiniteTalkWan model (same as training)
2. Run torch.compile warmup (FlexAttention)
3. Do a training forward pass on a precomputed sample
4. Then try lazy encoding on the same GPU
5. Report peak memory at each stage

This tells us whether the ~7.3GB VAE peak fits in the headroom.
"""
import os
import sys
import gc
import csv
import tempfile
import warnings

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
IT_CKPT = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
WAV2VEC_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"
CSV_PATH = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv"
RAW_DATA_ROOT = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/"

# A precomputed sample (for the training forward pass)
PRECOMPUTED_SAMPLE = "data/precomputed_talkvid/data_-0F1owya2oo_-0F1owya2oo_106956_111998"
# A lazy sample (only has text_embeds.pt)
LAZY_SAMPLE = "data/precomputed_talkvid/data_0IKddT_rzjw_0IKddT_rzjw_839891_844933"


def gpu_gb():
    return torch.cuda.memory_allocated() / 1024**3

def gpu_reserved_gb():
    return torch.cuda.memory_reserved() / 1024**3

def gpu_peak_gb():
    return torch.cuda.max_memory_allocated() / 1024**3

def report(label):
    print(f"  [{label}] allocated={gpu_gb():.1f} GB, reserved={gpu_reserved_gb():.1f} GB, peak={gpu_peak_gb():.1f} GB")


def main():
    device = "cuda:0"
    torch.cuda.reset_peak_memory_stats()

    # ── Step 1: Load the 14B model (same as training) ──
    print("Step 1: Loading 14B CausalInfiniteTalkWan...")
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    base_paths = ",".join([
        f"{WEIGHTS_DIR}/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
        for i in range(1, 8)
    ])

    net = CausalInfiniteTalkWan(
        base_model_paths=base_paths,
        infinitetalk_ckpt_path=IT_CKPT,
        lora_rank=128,
        lora_alpha=64,
        chunk_size=3,
        total_num_frames=21,
        net_pred_type="flow",
        schedule_type="rf",
        shift=7.0,
    )
    net = net.to(device=device, dtype=torch.bfloat16)
    report("After model load")

    # ── Step 2: Simulate training state (optimizer, grads) ──
    print("\nStep 2: Creating optimizer (simulates training memory)...")
    from fastgen.networks.InfiniteTalk.lora import freeze_base
    freeze_base(net)
    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    optimizer = torch.optim.AdamW(trainable, lr=1e-5)
    report("After optimizer creation")

    # ── Step 3: Do a forward pass with a precomputed sample ──
    print("\nStep 3: Forward pass on precomputed sample...")
    sample = torch.load(os.path.join(PRECOMPUTED_SAMPLE, "vae_latents.pt"), map_location="cpu")
    if isinstance(sample, dict):
        sample = next(v for v in sample.values() if isinstance(v, torch.Tensor))
    real = sample[:, :21].unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # [1, 16, 21, H, W]

    # Simple noise + denoise forward (no compile, just raw forward)
    noise = torch.randn_like(real)
    t = torch.tensor([[0.5] * 21], device=device, dtype=torch.bfloat16)

    # Build minimal condition
    ffc = torch.load(os.path.join(PRECOMPUTED_SAMPLE, "first_frame_cond.pt"), map_location="cpu")
    if isinstance(ffc, dict):
        ffc = next(v for v in ffc.values() if isinstance(v, torch.Tensor))
    ffc = ffc[:, :21].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    text = torch.load(os.path.join(PRECOMPUTED_SAMPLE, "text_embeds.pt"), map_location="cpu")
    if isinstance(text, dict):
        text = next(v for v in text.values() if isinstance(v, torch.Tensor))
    text = text.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    clip = torch.load(os.path.join(PRECOMPUTED_SAMPLE, "clip_features.pt"), map_location="cpu")
    if isinstance(clip, dict):
        clip = next(v for v in clip.values() if isinstance(v, torch.Tensor))
    clip = clip.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    audio = torch.load(os.path.join(PRECOMPUTED_SAMPLE, "audio_emb.pt"), map_location="cpu")
    if isinstance(audio, dict):
        audio = next(v for v in audio.values() if isinstance(v, torch.Tensor))
    audio = audio[:81].unsqueeze(0).to(device=device, dtype=torch.bfloat16)

    condition = {
        "text_embeds": text,
        "first_frame_cond": ffc,
        "clip_features": clip,
        "audio_emb": audio,
    }

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = net(noise, t, condition=condition)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    report("After training step")

    # Clean up training tensors but keep model + optimizer (as training would)
    del real, noise, t, ffc, text, clip, audio, condition, output, loss, sample
    gc.collect()
    torch.cuda.empty_cache()
    report("After cleanup (steady state)")

    steady_state_gb = gpu_gb()
    reserved_gb = gpu_reserved_gb()
    headroom_gb = 80.0 - reserved_gb

    print(f"\n{'='*60}")
    print(f"Steady-state allocated: {steady_state_gb:.1f} GB")
    print(f"Reserved by PyTorch:    {reserved_gb:.1f} GB")
    print(f"Headroom:               {headroom_gb:.1f} GB")
    print(f"VAE encode peak needed: ~7.3 GB (from memory test)")
    print(f"VAE params:             ~0.5 GB")
    print(f"CLIP params:            ~2.3 GB")
    print(f"wav2vec params:         ~0.4 GB")

    if headroom_gb < 8.0:
        print(f"\nHeadroom ({headroom_gb:.1f} GB) < 8 GB needed. Testing anyway...")
    else:
        print(f"\nHeadroom ({headroom_gb:.1f} GB) looks sufficient!")

    # ── Step 4: Try lazy encoding ──
    print("\nStep 4: Attempting lazy encode on same GPU...")
    torch.cuda.empty_cache()
    report("After empty_cache")

    from fastgen.datasets.infinitetalk_dataloader import (
        _add_infinitetalk_to_path, _load_vae, _load_clip, _load_wav2vec,
        _move_encoder_to, _encode_vae, _encode_clip, _encode_audio,
        _load_video_frames, _resize_and_center_crop, _resize_and_centercrop_pil,
        _load_audio_array, ASPECT_RATIO_627,
    )
    _add_infinitetalk_to_path()

    # Load encoders on CPU
    print("  Loading encoders on CPU...")
    vae = _load_vae(WEIGHTS_DIR, "cpu")
    clip_enc = _load_clip(WEIGHTS_DIR, "cpu")
    wav2vec_fe, audio_enc = _load_wav2vec(WAV2VEC_DIR, "cpu")
    report("Encoders loaded on CPU")

    # Load video + get CSV row
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = {os.path.splitext(r["video_path"].replace("/", "_"))[0]: r for r in reader}
    sample_name = os.path.basename(LAZY_SAMPLE)
    row = rows[sample_name]
    video_path = os.path.join(RAW_DATA_ROOT, row["video_path"])
    audio_path = os.path.join(RAW_DATA_ROOT, row["audio_path"])
    src_h, src_w = int(row["height"]), int(row["width"])
    ratio = src_h / src_w
    closest = sorted(ASPECT_RATIO_627.keys(), key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = ASPECT_RATIO_627[closest]

    from PIL import Image
    frames, fps = _load_video_frames(video_path, frame_count=93, target_fps=25.0)
    video_tensor = _resize_and_center_crop(frames, target_h, target_w)
    cond_image = _resize_and_centercrop_pil(Image.fromarray(frames[0]), target_h, target_w)

    # ── VAE encode (biggest spike) ──
    print("  VAE encode...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        _move_encoder_to(vae, device)
        latents, ffc, motion = _encode_vae(vae, video_tensor, device, cond_image)
        _move_encoder_to(vae, "cpu")
        torch.cuda.empty_cache()
        print(f"  VAE encode SUCCESS! latents={list(latents.shape)}")
        report("After VAE encode")
        del latents, ffc, motion
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  VAE encode OOM! {e}")
            torch.cuda.empty_cache()
        else:
            raise

    # ── CLIP encode ──
    print("  CLIP encode...")
    torch.cuda.empty_cache()
    try:
        _move_encoder_to(clip_enc, device)
        clip_features = _encode_clip(clip_enc, device, cond_image)
        _move_encoder_to(clip_enc, "cpu")
        torch.cuda.empty_cache()
        print(f"  CLIP encode SUCCESS! shape={list(clip_features.shape)}")
        del clip_features
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  CLIP encode OOM!")
        else:
            raise

    # ── Audio encode ──
    print("  Audio encode...")
    torch.cuda.empty_cache()
    try:
        _move_encoder_to(audio_enc, device)
        audio_array = _load_audio_array(audio_path)
        audio_emb = _encode_audio(wav2vec_fe, audio_enc, audio_array, num_video_frames=93, device=device)
        _move_encoder_to(audio_enc, "cpu")
        torch.cuda.empty_cache()
        print(f"  Audio encode SUCCESS! shape={list(audio_emb.shape)}")
        del audio_emb, audio_array
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  Audio encode OOM!")
        else:
            raise

    print(f"\n{'='*60}")
    print("DONE — check results above for OOM vs SUCCESS")


if __name__ == "__main__":
    main()
