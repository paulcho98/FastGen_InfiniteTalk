#!/usr/bin/env python3
"""Precompute InfiniteTalk conditioning tensors for HDTF test set.

Encodes each video's first frame (VAE + CLIP) and audio (wav2vec2),
saving .pt files that inference_causal.py can load via --precomputed_list.

Usage:
    python scripts/inference/precompute_hdtf.py \
        --video_dir /path/to/HDTF_testset/videos_cfr \
        --output_dir data/precomputed_hdtf_quarter \
        --vae_path /path/to/Wan2.1_VAE.pth \
        --clip_path /path/to/clip-model.pth \
        --wav2vec_path /path/to/chinese-wav2vec2-base \
        --target_h 224 --target_w 448 \
        --num_latent_frames 21 --chunk_size 3
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, FASTGEN_ROOT)

from fastgen.datasets.infinitetalk_dataloader import _add_infinitetalk_to_path
_add_infinitetalk_to_path()

from scripts.inference.inference_causal import (
    load_vae,
    load_clip,
    load_wav2vec,
    encode_reference_image,
    encode_clip,
    encode_audio_from_file,
    extract_first_frame_and_audio,
    compute_generation_length,
)


def main():
    parser = argparse.ArgumentParser(description="Precompute HDTF conditioning tensors")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True)
    parser.add_argument("--clip_path", type=str, required=True)
    parser.add_argument("--wav2vec_path", type=str, required=True)
    parser.add_argument("--target_h", type=int, default=224)
    parser.add_argument("--target_w", type=int, default=448)
    parser.add_argument("--num_latent_frames", type=int, default=21)
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted([
        os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir)
        if os.path.splitext(f)[1].lower() in video_exts
    ])
    print(f"Found {len(video_files)} videos in {args.video_dir}")

    # Load encoders
    print("Loading VAE...")
    vae = load_vae(args.vae_path, device="cpu")

    print("Loading CLIP...")
    clip_model = load_clip(args.clip_path, device=args.device)

    print("Loading wav2vec2...")
    wav2vec_fe, wav2vec_model = load_wav2vec(args.wav2vec_path, device=args.device)

    # Encode negative text embeddings (zeros)
    neg_text_embeds = torch.zeros(1, 1, 512, 4096, dtype=torch.bfloat16)

    sample_list = []

    for vi, vpath in enumerate(video_files):
        vname = os.path.splitext(os.path.basename(vpath))[0]
        sample_dir = os.path.join(args.output_dir, vname)

        # Skip if already complete
        if (os.path.isfile(os.path.join(sample_dir, "first_frame_cond.pt"))
                and os.path.isfile(os.path.join(sample_dir, "clip_features.pt"))
                and os.path.isfile(os.path.join(sample_dir, "audio_emb.pt"))
                and os.path.isfile(os.path.join(sample_dir, "text_embeds.pt"))):
            print(f"  [{vi+1}/{len(video_files)}] {vname}: already complete, skipping")
            sample_list.append(sample_dir)
            continue

        print(f"  [{vi+1}/{len(video_files)}] Encoding {vname}...")
        os.makedirs(sample_dir, exist_ok=True)

        try:
            pil_image, audio_path = extract_first_frame_and_audio(vpath)
            num_latent, num_video = compute_generation_length(
                audio_path, args.num_latent_frames, args.chunk_size, args.fps
            )

            # VAE encode reference frame
            ffc = encode_reference_image(
                vae, pil_image, num_latent,
                target_h=args.target_h, target_w=args.target_w,
                device=args.device,
            )
            torch.save(ffc.cpu(), os.path.join(sample_dir, "first_frame_cond.pt"))

            # CLIP encode
            cf = encode_clip(
                clip_model, pil_image, device=args.device,
                target_h=args.target_h, target_w=args.target_w,
            )
            torch.save(cf.cpu(), os.path.join(sample_dir, "clip_features.pt"))

            # Audio encode
            ae = encode_audio_from_file(
                wav2vec_fe, wav2vec_model, audio_path, num_video, device=args.device,
            )
            torch.save(ae.cpu(), os.path.join(sample_dir, "audio_emb.pt"))

            # Text (zeros) + negative text
            text_embeds = torch.zeros(1, 1, 512, 4096, dtype=torch.bfloat16)
            torch.save(text_embeds, os.path.join(sample_dir, "text_embeds.pt"))
            torch.save(neg_text_embeds.clone(), os.path.join(sample_dir, "neg_text_embeds.pt"))

            # Save audio path for muxing
            with open(os.path.join(sample_dir, "audio_path.txt"), "w") as f:
                f.write(audio_path)

            sample_list.append(sample_dir)
            print(f"    Saved to {sample_dir}")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        torch.cuda.empty_cache()

    # Write sample list file
    list_path = os.path.join(args.output_dir, "sample_list.txt")
    with open(list_path, "w") as f:
        for sd in sample_list:
            f.write(sd + "\n")
    print(f"\nDone! {len(sample_list)} samples precomputed.")
    print(f"Sample list: {list_path}")
    print(f"\nUse with inference_causal.py:")
    print(f"  --precomputed_list {list_path}")


if __name__ == "__main__":
    main()
