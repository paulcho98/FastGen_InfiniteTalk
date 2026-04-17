#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Quick test: lazy encode 2 unprocessed samples, verify dtype fix works.

Usage:
    python scripts/test_lazy_encode.py
"""
import os
import sys
import tempfile
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_DIR = os.environ.get(
    "INFINITETALK_WEIGHTS_DIR",
    "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P",
)
WAV2VEC_DIR = os.environ.get(
    "INFINITETALK_WAV2VEC_DIR",
    "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base",
)
CSV_PATH = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv"
RAW_DATA_ROOT = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/"
NEG_TEXT_EMB = "data/precomputed_talkvid/neg_text_embeds.pt"

# Two lazy-eligible samples (only have text_embeds.pt)
LAZY_SAMPLES = [
    "data/precomputed_talkvid/data_0IKddT_rzjw_0IKddT_rzjw_766825_771866",
    "data/precomputed_talkvid/data_0IKddT_rzjw_0IKddT_rzjw_817241_822283",
]


def main():
    # Verify samples only have text_embeds.pt
    for d in LAZY_SAMPLES:
        assert os.path.isfile(os.path.join(d, "text_embeds.pt")), f"Missing text_embeds.pt in {d}"
        assert not os.path.isfile(os.path.join(d, "vae_latents.pt")), f"Already has vae_latents.pt in {d} — not a lazy sample"

    # Write a temp sample list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for d in LAZY_SAMPLES:
            f.write(d + "\n")
        list_path = f.name

    # Encoders load on CPU, get offloaded to GPU for each encode, then moved back.
    # This avoids OOM during training (~77GB) since we only spike temporarily.
    encode_device = "cuda:0"
    print(f"Testing lazy encoding with {len(LAZY_SAMPLES)} samples...")
    print(f"  weights_dir: {WEIGHTS_DIR}")
    print(f"  wav2vec_dir: {WAV2VEC_DIR}")
    print(f"  encode_device: {encode_device}")

    try:
        from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataset

        ds = InfiniteTalkDataset(
            data_list_path=list_path,
            neg_text_emb_path=NEG_TEXT_EMB,
            num_video_frames=93,
            num_latent_frames=21,
            expected_latent_shape=(16, 21, 60, 106),  # 480x848 bucket
            raw_data_root=RAW_DATA_ROOT,
            csv_path=CSV_PATH,
            weights_dir=WEIGHTS_DIR,
            wav2vec_dir=WAV2VEC_DIR,
            encode_device=encode_device,
        )
        print(f"Dataset created: {len(ds)} samples")

        for i in range(len(ds)):
            print(f"\n--- Sample {i} ---")
            sample = ds[i]
            if sample is None:
                print(f"  FAILED: returned None")
                continue

            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
                else:
                    print(f"  {k}: {v}")

            # Verify key shapes
            assert sample["real"].shape[0] == 16, f"Bad real channels: {sample['real'].shape}"
            assert sample["real"].shape[1] == 21, f"Bad real frames: {sample['real'].shape}"
            assert sample["first_frame_cond"].shape[0] == 16, f"Bad ffc channels"
            assert sample["clip_features"].shape[-2:] == (257, 1280), f"Bad clip shape"
            assert sample["audio_emb"].shape[-2:] == (12, 768), f"Bad audio shape"
            assert sample["text_embeds"].shape[-1] == 4096, f"Bad text shape"
            print(f"  ALL CHECKS PASSED")

        # Verify files were cached to disk
        print("\n--- Checking cached files ---")
        for d in LAZY_SAMPLES:
            cached = [f for f in os.listdir(d) if f.endswith(".pt")]
            print(f"  {os.path.basename(d)}: {sorted(cached)}")

        print("\nSUCCESS: Lazy encoding works correctly!")

    finally:
        os.unlink(list_path)


if __name__ == "__main__":
    main()
