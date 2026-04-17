#!/usr/bin/env python
"""Measure GPU memory used by lazy encoding — determines if it fits alongside training."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"
CSV_PATH = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv"
RAW_DATA_ROOT = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/"

# Use one of our lazy samples
SAMPLE_DIR = "data/precomputed_talkvid/data_0IKddT_rzjw_0IKddT_rzjw_766825_771866"

def gpu_mb():
    return torch.cuda.memory_allocated() / 1024**2

def gpu_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def main():
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU memory before loading encoders: {gpu_mb():.0f} MB")

    from fastgen.datasets.infinitetalk_dataloader import (
        _load_vae, _load_clip, _load_wav2vec, _add_infinitetalk_to_path,
        _load_video_frames, _resize_and_center_crop, _resize_and_centercrop_pil,
        _encode_vae, _encode_clip, _encode_audio, _load_audio_array,
        ASPECT_RATIO_627,
    )
    _add_infinitetalk_to_path()

    device = "cuda:0"

    # Load encoders one at a time, measure each
    vae = _load_vae(WEIGHTS_DIR, device)
    print(f"After VAE load:   {gpu_mb():.0f} MB (peak: {gpu_peak_mb():.0f} MB)")

    clip = _load_clip(WEIGHTS_DIR, device)
    print(f"After CLIP load:  {gpu_mb():.0f} MB (peak: {gpu_peak_mb():.0f} MB)")

    wav2vec_fe, audio_enc = _load_wav2vec(WAV2VEC_DIR, device)
    print(f"After wav2vec:    {gpu_mb():.0f} MB (peak: {gpu_peak_mb():.0f} MB)")

    total_encoder_mem = gpu_mb()
    print(f"\nTotal encoder params on GPU: {total_encoder_mem:.0f} MB")

    # Now encode a real sample and measure peak
    import csv
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = {os.path.splitext(r["video_path"].replace("/", "_"))[0]: r for r in reader}

    sample_name = os.path.basename(SAMPLE_DIR)
    row = rows[sample_name]
    video_path = os.path.join(RAW_DATA_ROOT, row["video_path"])
    audio_path = os.path.join(RAW_DATA_ROOT, row["audio_path"])
    src_h, src_w = int(row["height"]), int(row["width"])

    # Resolution bucket
    ratio = src_h / src_w
    closest = sorted(ASPECT_RATIO_627.keys(), key=lambda x: abs(float(x) - ratio))[0]
    target_h, target_w = ASPECT_RATIO_627[closest]
    print(f"\nSource: {src_w}x{src_h} (ratio={ratio:.4f})")
    print(f"Bucket: {closest} -> {target_w}x{target_h} (latent: {target_w//8}x{target_h//8})")

    # Load video frames (CPU)
    import numpy as np
    from PIL import Image
    frames, fps = _load_video_frames(video_path, frame_count=93, target_fps=25.0)
    print(f"Loaded {frames.shape[0]} frames at {fps}fps, shape={frames.shape}")
    video_tensor = _resize_and_center_crop(frames, target_h, target_w)
    print(f"Resized video tensor: {list(video_tensor.shape)} (pixel: {target_w}x{target_h})")
    cond_image = _resize_and_centercrop_pil(Image.fromarray(frames[0]), target_h, target_w)

    # VAE encode
    torch.cuda.reset_peak_memory_stats()
    before = gpu_mb()
    latents, ffc, motion = _encode_vae(vae, video_tensor, device, cond_image)
    vae_peak = gpu_peak_mb()
    after = gpu_mb()
    print(f"\nVAE encode: current={after:.0f} MB, peak={vae_peak:.0f} MB, delta_peak={vae_peak - total_encoder_mem:.0f} MB")
    print(f"  latents: {list(latents.shape)}")
    print(f"  first_frame_cond: {list(ffc.shape)}")
    del latents, ffc, motion
    torch.cuda.empty_cache()

    # CLIP encode
    torch.cuda.reset_peak_memory_stats()
    clip_feat = _encode_clip(clip, device, cond_image)
    clip_peak = gpu_peak_mb()
    print(f"CLIP encode: peak={clip_peak:.0f} MB, delta_peak={clip_peak - total_encoder_mem:.0f} MB")
    print(f"  clip_features: {list(clip_feat.shape)}")
    del clip_feat
    torch.cuda.empty_cache()

    # Audio encode
    torch.cuda.reset_peak_memory_stats()
    audio_array = _load_audio_array(audio_path)
    audio_emb = _encode_audio(wav2vec_fe, audio_enc, audio_array, num_video_frames=93, device=device)
    audio_peak = gpu_peak_mb()
    print(f"Audio encode: peak={audio_peak:.0f} MB, delta_peak={audio_peak - total_encoder_mem:.0f} MB")
    print(f"  audio_emb: {list(audio_emb.shape)}")

    max_peak = max(vae_peak, clip_peak, audio_peak)
    print(f"\n{'='*60}")
    print(f"Encoder params resident:     {total_encoder_mem:.0f} MB")
    print(f"Max peak during encoding:    {max_peak:.0f} MB")
    print(f"Training uses:               ~77,000 MB")
    print(f"GPU total:                   80,000 MB")
    print(f"Remaining for encoding:      ~{80000 - 77000:.0f} MB")
    print(f"Fits? {'YES' if max_peak < 3000 else 'NO — would OOM'}")


if __name__ == "__main__":
    main()
