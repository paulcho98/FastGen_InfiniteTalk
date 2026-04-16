# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for future anchor latent loading in InfiniteTalkDataset._load_sample."""

import os
import tempfile

import pytest
import torch

from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataset


# Latent spatial dims for _quarter resolution
H_LAT, W_LAT = 30, 52  # representative values
NUM_LATENT_FRAMES = 21
C_LAT = 16


def _make_sample_dir(tmpdir: str, include_future_anchor: bool = False):
    """Create a minimal sample directory with required .pt files.

    Returns the sample directory path.
    """
    sample_dir = os.path.join(tmpdir, "sample_0")
    os.makedirs(sample_dir, exist_ok=True)

    # vae_latents_quarter.pt — [16, num_latent_frames, H_lat, W_lat]
    vae_latents = torch.randn(C_LAT, NUM_LATENT_FRAMES, H_LAT, W_LAT, dtype=torch.bfloat16)
    torch.save(vae_latents, os.path.join(sample_dir, "vae_latents_quarter.pt"))

    # first_frame_cond_quarter.pt — [16, 1, H_lat, W_lat]
    first_frame_cond = torch.randn(C_LAT, 1, H_LAT, W_LAT, dtype=torch.bfloat16)
    torch.save(first_frame_cond, os.path.join(sample_dir, "first_frame_cond_quarter.pt"))

    # clip_features.pt — [1, 257, 1280]
    clip_features = torch.randn(1, 257, 1280, dtype=torch.bfloat16)
    torch.save(clip_features, os.path.join(sample_dir, "clip_features.pt"))

    # audio_emb.pt — [T_pixel, 12, 768] where T_pixel >= _train_pixel_frames
    audio_emb = torch.randn(81, 12, 768, dtype=torch.bfloat16)
    torch.save(audio_emb, os.path.join(sample_dir, "audio_emb.pt"))

    # text_embeds.pt — [1, 512, 4096]
    text_embeds = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
    torch.save(text_embeds, os.path.join(sample_dir, "text_embeds.pt"))

    if include_future_anchor:
        # future_anchor_latents_quarter.pt — [16, 5, H_lat, W_lat]
        future_anchor = torch.randn(C_LAT, 5, H_LAT, W_LAT, dtype=torch.bfloat16)
        torch.save(future_anchor, os.path.join(sample_dir, "future_anchor_latents_quarter.pt"))

    return sample_dir


def _make_dataset_stub():
    """Create an InfiniteTalkDataset without calling __init__."""
    ds = InfiniteTalkDataset.__new__(InfiniteTalkDataset)
    ds.num_latent_frames = NUM_LATENT_FRAMES
    ds._train_pixel_frames = 81
    ds._vae_suffix = "_quarter"
    ds.neg_text_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)
    ds.load_ode_path = False
    ds.audio_data_root = None
    return ds


class TestFutureAnchorLoading:
    """Tests for future anchor latent loading in _load_sample."""

    def test_loads_future_anchor_when_present(self):
        """When future_anchor_latents_quarter.pt exists, it should be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = _make_sample_dir(tmpdir, include_future_anchor=True)
            ds = _make_dataset_stub()

            result = ds._load_sample(sample_dir)

            assert "future_anchor_latents" in result
            assert "has_future_anchor" in result
            assert result["has_future_anchor"] is True

            fa = result["future_anchor_latents"]
            assert fa.shape == (C_LAT, 5, H_LAT, W_LAT)
            assert fa.dtype == torch.bfloat16
            # Should not be all zeros (loaded real data)
            assert fa.abs().sum() > 0

    def test_zero_fallback_when_missing(self):
        """When file is missing, result should contain zero tensor and has_future_anchor=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = _make_sample_dir(tmpdir, include_future_anchor=False)
            ds = _make_dataset_stub()

            result = ds._load_sample(sample_dir)

            assert "future_anchor_latents" in result
            assert "has_future_anchor" in result
            assert result["has_future_anchor"] is False

            fa = result["future_anchor_latents"]
            assert fa.shape == (C_LAT, 5, H_LAT, W_LAT)
            assert fa.dtype == torch.bfloat16
            # Should be all zeros
            assert fa.abs().sum() == 0

    def test_loads_dict_wrapped_future_anchor(self):
        """When the .pt file contains a dict wrapping a tensor, it should unwrap it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = _make_sample_dir(tmpdir, include_future_anchor=False)
            ds = _make_dataset_stub()

            # Save as dict instead of raw tensor
            fa_tensor = torch.randn(C_LAT, 5, H_LAT, W_LAT, dtype=torch.float32)
            torch.save(
                {"future_anchor_latents": fa_tensor},
                os.path.join(sample_dir, "future_anchor_latents_quarter.pt"),
            )

            result = ds._load_sample(sample_dir)

            assert result["has_future_anchor"] is True
            fa = result["future_anchor_latents"]
            assert fa.shape == (C_LAT, 5, H_LAT, W_LAT)
            # Should be cast to bf16 even if saved as float32
            assert fa.dtype == torch.bfloat16

    def test_collate_compatible(self):
        """Both present and missing cases should be collatable together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_with = _make_sample_dir(
                os.path.join(tmpdir, "with"), include_future_anchor=True
            )
            dir_without = _make_sample_dir(
                os.path.join(tmpdir, "without"), include_future_anchor=False
            )
            # Rename inner dirs so they don't clash
            ds = _make_dataset_stub()

            result_with = ds._load_sample(dir_with)
            result_without = ds._load_sample(dir_without)

            # default_collate should handle a batch of these
            from torch.utils.data.dataloader import default_collate

            batch = default_collate([result_with, result_without])

            assert batch["future_anchor_latents"].shape == (2, C_LAT, 5, H_LAT, W_LAT)
            assert batch["has_future_anchor"].tolist() == [True, False]
