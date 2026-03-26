# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch Dataset for InfiniteTalk precomputed training data.

Each sample directory contains precomputed .pt files:
    - vae_latents.pt: [16, 21, H, W] — VAE-encoded 81-frame video
    - first_frame_cond.pt: [16, 21, H, W] — VAE-encoded reference frame + zero padding
    - clip_features.pt: [1, 257, 1280] — CLIP ViT-H/14 on reference frame
    - audio_emb.pt: [81, 12, 768] — wav2vec2 hidden states (all 12 layers)
    - text_embeds.pt: [1, 512, 4096] — T5 UMT5-XXL text embeddings
    - neg_text_embeds.pt: [1, 512, 4096] — negative text embedding (shared across samples)
    - ode_path.pt: [num_steps, 16, 21, H, W] — ODE trajectory (KD only, optional)

Audio windowing (5-frame sliding window) is NOT applied in the dataloader.
It is handled inside WanModel.forward() at inference time, matching InfiniteTalk's
original behavior (multitalk.py lines 523-533).
"""

import os
import warnings

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class InfiniteTalkDataset(Dataset):
    """
    Dataset for InfiniteTalk training data with precomputed tensors.

    Returns dict with:
        real: [16, 21, H, W] -- clean video latents (bf16)
        first_frame_cond: [16, 21, H, W] -- reference frame conditioning (bf16)
        clip_features: [1, 257, 1280] -- CLIP features (bf16)
        audio_emb: [81, 12, 768] -- wav2vec2 audio embeddings (bf16)
        text_embeds: [1, 512, 4096] -- T5 text embedding (bf16)
        neg_text_embeds: [1, 512, 4096] -- negative text embedding for CFG (bf16)
        path: [num_steps, 16, 21, H, W] -- ODE trajectory (bf16, optional, for KD)
    """

    def __init__(
        self,
        data_list_path: str,
        neg_text_emb_path: str = None,
        load_ode_path: bool = False,
        num_video_frames: int = 81,
        expected_latent_shape: tuple = None,
    ):
        """
        Args:
            data_list_path: Path to text file listing sample directories (one per line).
            neg_text_emb_path: Path to shared negative text embedding file.
                If None, uses zeros [1, 512, 4096].
            load_ode_path: If True, load ode_path.pt for KD training.
            num_video_frames: Number of video frames (for audio truncation).
            expected_latent_shape: If set, e.g. (16, 21, 56, 112), filter out samples
                whose vae_latents.pt has a different shape. This handles datasets with
                mixed aspect ratios — only samples matching the training resolution
                are kept.

                NOTE: For future multi-resolution training, replace this filter with
                aspect-ratio bucketed sampling (group by resolution so each batch
                has uniform spatial dims). The model's transformer is resolution-
                agnostic (RoPE adapts), but noise/loss shapes must match within a batch.
        """
        self.load_ode_path = load_ode_path
        self.num_video_frames = num_video_frames
        self.expected_latent_shape = tuple(expected_latent_shape) if expected_latent_shape else None

        # Read sample directories from text file
        with open(data_list_path) as f:
            all_dirs = [line.strip() for line in f if line.strip()]

        # Filter out samples missing required files
        valid_dirs = []
        required_files = [
            "vae_latents.pt",
            "first_frame_cond.pt",
            "clip_features.pt",
            "audio_emb.pt",
            "text_embeds.pt",
        ]
        missing_count = 0
        for d in all_dirs:
            missing = [fn for fn in required_files if not os.path.exists(os.path.join(d, fn))]
            if missing:
                warnings.warn(f"Skipping {d}: missing {missing}")
                missing_count += 1
            else:
                valid_dirs.append(d)

        # Filter by resolution if expected_latent_shape is set
        shape_mismatch_count = 0
        if self.expected_latent_shape is not None:
            self.dirs = []
            for d in valid_dirs:
                # Quick shape check: load only metadata (mmap), don't load full tensor
                try:
                    lat = torch.load(os.path.join(d, "vae_latents.pt"),
                                     map_location="cpu", weights_only=False)
                    if isinstance(lat, dict):
                        lat = next(v for v in lat.values() if isinstance(v, torch.Tensor))
                    if tuple(lat.shape) == self.expected_latent_shape:
                        self.dirs.append(d)
                    else:
                        shape_mismatch_count += 1
                    del lat
                except Exception:
                    shape_mismatch_count += 1
        else:
            self.dirs = valid_dirs

        skipped = missing_count + shape_mismatch_count
        if skipped > 0:
            print(
                f"[InfiniteTalkDataset] Kept {len(self.dirs)}/{len(all_dirs)} samples "
                f"(skipped: {missing_count} missing files, {shape_mismatch_count} resolution mismatch)"
            )

        # Load negative text embedding (shared across all samples, for CFG)
        if neg_text_emb_path is not None and os.path.exists(neg_text_emb_path):
            neg_emb = torch.load(neg_text_emb_path, map_location="cpu", weights_only=False)
            if isinstance(neg_emb, dict):
                # Handle dict format — grab the first tensor value
                neg_emb = next(v for v in neg_emb.values() if isinstance(v, torch.Tensor))
            self.neg_text_embeds = neg_emb.to(torch.bfloat16)
        else:
            self.neg_text_embeds = torch.zeros(1, 512, 4096, dtype=torch.bfloat16)

        # Ensure correct shape [1, 512, 4096]
        if self.neg_text_embeds.dim() == 2:
            self.neg_text_embeds = self.neg_text_embeds.unsqueeze(0)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx) -> dict:
        sample_dir = self.dirs[idx]

        try:
            # --- VAE latents (clean video) ---
            real = torch.load(
                os.path.join(sample_dir, "vae_latents.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(real, dict):
                # Handle dict format if needed
                real = next(v for v in real.values() if isinstance(v, torch.Tensor))
            real = real.to(torch.bfloat16)  # [16, 21, H, W]

            # --- First frame conditioning ---
            first_frame_cond = torch.load(
                os.path.join(sample_dir, "first_frame_cond.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(first_frame_cond, dict):
                first_frame_cond = next(
                    v for v in first_frame_cond.values() if isinstance(v, torch.Tensor)
                )
            first_frame_cond = first_frame_cond.to(torch.bfloat16)  # [16, 21, H, W]

            # --- CLIP features ---
            clip_features = torch.load(
                os.path.join(sample_dir, "clip_features.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(clip_features, dict):
                clip_features = next(
                    v for v in clip_features.values() if isinstance(v, torch.Tensor)
                )
            clip_features = clip_features.to(torch.bfloat16)  # [1, 257, 1280]

            # --- Audio embeddings ---
            audio_emb = torch.load(
                os.path.join(sample_dir, "audio_emb.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(audio_emb, dict):
                audio_emb = next(
                    v for v in audio_emb.values() if isinstance(v, torch.Tensor)
                )
            # Truncate to num_video_frames if longer
            audio_emb = audio_emb[: self.num_video_frames]  # [81, 12, 768]
            audio_emb = audio_emb.to(torch.bfloat16)

            # --- Text embeddings ---
            text_embeds = torch.load(
                os.path.join(sample_dir, "text_embeds.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(text_embeds, dict):
                text_embeds = next(
                    v for v in text_embeds.values() if isinstance(v, torch.Tensor)
                )
            text_embeds = text_embeds.to(torch.bfloat16)
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(0)  # [1, 512, 4096]

            result = {
                "real": real,
                "first_frame_cond": first_frame_cond,
                "clip_features": clip_features,
                "audio_emb": audio_emb,
                "text_embeds": text_embeds,
                "neg_text_embeds": self.neg_text_embeds.clone(),
            }

            # --- ODE trajectory (optional, for KD training) ---
            if self.load_ode_path:
                ode_path_file = os.path.join(sample_dir, "ode_path.pt")
                if os.path.exists(ode_path_file):
                    result["path"] = torch.load(
                        ode_path_file, map_location="cpu", weights_only=False
                    ).to(torch.bfloat16)
                else:
                    # Also check path.pth (alternative filename)
                    alt_path_file = os.path.join(sample_dir, "path.pth")
                    if os.path.exists(alt_path_file):
                        result["path"] = torch.load(
                            alt_path_file, map_location="cpu", weights_only=False
                        ).to(torch.bfloat16)

            return result

        except Exception as e:
            warnings.warn(f"Error loading sample {sample_dir}: {e}")
            # Return None; collate_fn should filter these out
            return None


class InfiniteTalkDataLoader:
    """Infinite-iterator DataLoader wrapper with DistributedSampler support.

    FastGen's trainer expects an infinite iterator. This class wraps a standard
    DataLoader and yields batches indefinitely, cycling through the dataset.
    """

    def __init__(
        self,
        data_list_path: str = None,
        datatags: list = None,
        batch_size: int = 1,
        num_workers: int = 4,
        load_ode_path: bool = False,
        **kwargs,
    ):
        # Support both data_list_path and datatags (list of paths)
        if data_list_path is None and datatags is not None:
            data_list_path = datatags[0] if isinstance(datatags, list) else datatags
        assert data_list_path is not None, "Must provide data_list_path or datatags"

        self.dataset = InfiniteTalkDataset(
            data_list_path=data_list_path,
            load_ode_path=load_ode_path,
            **kwargs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Use DistributedSampler for multi-GPU training
        if dist.is_initialized():
            self._sampler = DistributedSampler(self.dataset, shuffle=True)
            shuffle = False
        else:
            self._sampler = None
            shuffle = True

        def collate_fn(batch):
            """Filter out None samples from failed loads."""
            valid = [s for s in batch if s is not None]
            if not valid:
                return {}
            return torch.utils.data.default_collate(valid)

        self._dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self._sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def __iter__(self):
        """Infinite iterator -- cycles through the dataset."""
        epoch = 0
        while True:
            if self._sampler is not None:
                self._sampler.set_epoch(epoch)
            yield from self._dataloader
            epoch += 1

    def __len__(self):
        return len(self.dataset)


# Keep the simple function for backward compatibility
def create_infinitetalk_dataloader(
    data_list_path: str,
    batch_size: int = 1,
    num_workers: int = 4,
    load_ode_path: bool = False,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for InfiniteTalk training data (non-infinite, no DDP support).

    For training, prefer InfiniteTalkDataLoader which provides infinite iteration
    and DistributedSampler support.
    """
    dataset = InfiniteTalkDataset(
        data_list_path=data_list_path,
        load_ode_path=load_ode_path,
        **kwargs,
    )

    def collate_fn(batch):
        """Filter out None samples from failed loads."""
        valid = [s for s in batch if s is not None]
        if not valid:
            return {}
        return torch.utils.data.default_collate(valid)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
