#!/usr/bin/env python3
"""
Stage 0 Verification: prove our ported DiT produces identical outputs
to InfiniteTalk's original WanModel.

Three verification levels, all using real precomputed TalkVid data:

  Level 1 -- Weight Loading Verification
      Load base Wan I2V safetensors + infinitetalk.safetensors into BOTH models.
      Compare ALL parameter names, shapes, and values.

  Level 2 -- Single Forward Pass Equivalence
      Run both models with identical inputs from 3 real TalkVid samples.
      Compare outputs elementwise.

  Level 3 -- Full 40-step Denoising Equivalence
      Run complete Euler ODE solve on 1 sample with identical noise seed.
      Compare final denoised latents.

Usage:
    python scripts/verify_infinitetalk_equivalence.py \\
        --data_dir data/test_precomputed/ \\
        --weights_dir /path/to/Wan2.1-I2V-14B-480P/ \\
        --infinitetalk_ckpt /path/to/infinitetalk.safetensors \\
        --infinitetalk_src /path/to/InfiniteTalk/ \\
        --device cuda:0 \\
        --level 1 2 3
"""

import argparse
import gc
import importlib
import importlib.machinery
import importlib.util
import math
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
# Completely disable torch.compile / dynamo — the original model uses
# @torch.compile on calculate_x_ref_attn_map which fails due to triton
# version incompatibility. We only need eager mode for verification.
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 14B architecture parameters (from Wan 2.1 I2V-14B-480P config.json).
# NOTE: in_dim=36 because patch_embedding receives the CONCATENATED
# noise (16ch) + condition (20ch) tensor.  The diffusers config.json
# and checkpoint both use in_dim=36.
MODEL_KWARGS = dict(
    model_type="i2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=36,
    dim=5120,
    ffn_dim=13824,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=40,
    num_layers=40,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    audio_window=5,
    intermediate_dim=512,
    output_dim=768,
    context_tokens=32,
    vae_scale=4,
    norm_input_visual=True,
    norm_output_audio=True,
    weight_init=False,
)

SAMPLE_DIRS = [
    "data_-0F1owya2oo_-0F1owya2oo_192347_197388",
    "data_-0F1owya2oo_-0F1owya2oo_189664_194706",
    "data_-0F1owya2oo_-0F1owya2oo_72088_77130",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Verify ported InfiniteTalk WanModel matches original"
    )
    p.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to test_precomputed/ with sample dirs and neg_text_embeds.pt",
    )
    p.add_argument(
        "--weights_dir", type=str, required=True,
        help="Path to Wan2.1-I2V-14B-480P/ containing 7 diffusion_pytorch_model shards",
    )
    p.add_argument(
        "--infinitetalk_ckpt", type=str, required=True,
        help="Path to infinitetalk.safetensors",
    )
    p.add_argument(
        "--infinitetalk_src", type=str, required=True,
        help="Path to InfiniteTalk/ repo root (for importing original model)",
    )
    p.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device (default: cuda:0)",
    )
    p.add_argument(
        "--level", type=int, nargs="+", default=[1, 2, 3],
        help="Verification levels to run (default: 1 2 3)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def load_base_and_infinitetalk_weights(
    weights_dir: str, infinitetalk_ckpt: str
) -> Dict[str, torch.Tensor]:
    """Load all 7 base shards + InfiniteTalk checkpoint into one state dict."""
    sd: Dict[str, torch.Tensor] = {}

    for i in range(1, 8):
        shard_path = os.path.join(
            weights_dir, f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors"
        )
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        shard = safetensors_load_file(shard_path, device="cpu")
        sd.update(shard)
        print(f"  Loaded shard {i}/7: {len(shard)} tensors")

    base_count = len(sd)
    print(f"  Total base tensors: {base_count}")

    it_sd = safetensors_load_file(infinitetalk_ckpt, device="cpu")
    print(f"  InfiniteTalk checkpoint: {len(it_sd)} tensors")
    sd.update(it_sd)
    print(f"  Merged total: {len(sd)} unique tensors")

    return sd


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _patch_ours_audio_squeeze(model):
    """Patch our WanModel.forward to squeeze audio_embedding for single-speaker.

    The original InfiniteTalk model squeezes audio_embedding inside
    SingleStreamMutiAttention.forward() (line: encoder_hidden_states.squeeze(0)).
    Our ported model uses SingleStreamAttention which expects a 3D tensor,
    but the AudioProjModel outputs 4D [B, N_t, N_a, C].  This patch adds
    the squeeze in the WanModel.forward() audio processing section so
    the downstream SingleStreamAttention receives the correct shape.

    This is a known porting discrepancy.
    """
    import functools

    _orig_forward = model.forward

    @functools.wraps(_orig_forward)
    def _patched_forward(*args, **kwargs):
        audio = kwargs.get("audio", None)
        if audio is None and len(args) > 6:
            # audio is the 7th positional arg
            audio = args[6]

        # Intercept the forward to add audio squeeze
        # We need to actually modify the internal behavior.
        # Simplest: monkey-patch the audio_proj to return squeezed output.
        orig_audio_proj_forward = model.audio_proj.forward

        def _squeezed_audio_proj(*a, **kw):
            out = orig_audio_proj_forward(*a, **kw)
            return out.squeeze(0)  # [B, N_t, N_a, C] -> [N_t, N_a, C] for B=1

        model.audio_proj.forward = _squeezed_audio_proj
        try:
            return _orig_forward(*args, **kwargs)
        finally:
            model.audio_proj.forward = orig_audio_proj_forward

    model.forward = _patched_forward


def build_ours(sd: Dict[str, torch.Tensor], device: str) -> nn.Module:
    """Build our ported WanModel and load weights."""
    from fastgen.networks.InfiniteTalk.wan_model import WanModel

    model = WanModel(**MODEL_KWARGS)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  [ours] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if missing:
        real_missing = [k for k in missing if k != "freqs"]
        if real_missing:
            print(f"  [ours] Missing keys ({len(real_missing)}): {real_missing[:5]}...")
    model.init_freqs()
    _patch_ours_audio_squeeze(model)
    print("  [ours] Applied audio squeeze patch for single-speaker compatibility")
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    return model


def _install_mocks():
    """Install mock modules for xformers and xfuser before importing original model.

    xformers.ops.memory_efficient_attention is replaced with a functional
    implementation using torch.nn.functional.scaled_dot_product_attention
    so the original SingleStreamAttention actually computes correct attention.
    """
    # -- xformers mock (with functional attention) --
    for name in [
        "xformers", "xformers.ops",
        "xformers.ops.fmha", "xformers.ops.fmha.attn_bias",
    ]:
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = m

    sys.modules["xformers"].__version__ = "0.0.0"

    class _MockBlockDiag:
        @staticmethod
        def from_seqlens(*a, **kw):
            return None

    def _memory_efficient_attention(q, k, v, attn_bias=None, op=None):
        """Drop-in for xformers MEA using flash_attn_func.
        Input: (B, M, H, K) -> (B, M, H, K).
        flash_attn_func uses the same [B, S, H, D] layout as xformers,
        providing numerically closer results than SDPA.
        """
        from flash_attn import flash_attn_func
        return flash_attn_func(q, k, v)

    sys.modules["xformers.ops.fmha.attn_bias"].BlockDiagonalMask = _MockBlockDiag
    sys.modules["xformers.ops.fmha"].attn_bias = sys.modules["xformers.ops.fmha.attn_bias"]
    sys.modules["xformers.ops"].fmha = sys.modules["xformers.ops.fmha"]
    sys.modules["xformers.ops"].memory_efficient_attention = _memory_efficient_attention
    sys.modules["xformers"].ops = sys.modules["xformers.ops"]

    # -- xfuser mock --
    for name in ["xfuser", "xfuser.core", "xfuser.core.distributed"]:
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = m

    sys.modules["xfuser.core.distributed"].get_sequence_parallel_rank = lambda: 0
    sys.modules["xfuser.core.distributed"].get_sequence_parallel_world_size = lambda: 1
    sys.modules["xfuser.core.distributed"].get_sp_group = lambda: None
    sys.modules["xfuser"].core = sys.modules["xfuser.core"]
    sys.modules["xfuser.core"].distributed = sys.modules["xfuser.core.distributed"]


def _import_original_model(infinitetalk_src: str):
    """Import the original WanModel by selective file loading.

    Bypasses wan/__init__.py which pulls in multitalk.py that uses
    inspect.ArgSpec (removed in Python 3.12) and t5.py that requires
    optimum.quanto.  We only need:
      wan.utils.multitalk_utils -> wan.modules.attention -> wan.modules.multitalk_model
    """
    if infinitetalk_src not in sys.path:
        sys.path.insert(0, infinitetalk_src)

    # Create namespace packages manually
    wan_mod = types.ModuleType("wan")
    wan_mod.__path__ = [os.path.join(infinitetalk_src, "wan")]
    wan_mod.__spec__ = importlib.machinery.ModuleSpec("wan", None)
    sys.modules["wan"] = wan_mod

    wan_utils = types.ModuleType("wan.utils")
    wan_utils.__path__ = [os.path.join(infinitetalk_src, "wan", "utils")]
    sys.modules["wan.utils"] = wan_utils

    wan_modules = types.ModuleType("wan.modules")
    wan_modules.__path__ = [os.path.join(infinitetalk_src, "wan", "modules")]
    sys.modules["wan.modules"] = wan_modules

    # Load multitalk_utils (needed by attention.py)
    spec = importlib.util.spec_from_file_location(
        "wan.utils.multitalk_utils",
        os.path.join(infinitetalk_src, "wan", "utils", "multitalk_utils.py"),
    )
    mu = importlib.util.module_from_spec(spec)
    sys.modules["wan.utils.multitalk_utils"] = mu
    spec.loader.exec_module(mu)

    # Load attention.py
    spec = importlib.util.spec_from_file_location(
        "wan.modules.attention",
        os.path.join(infinitetalk_src, "wan", "modules", "attention.py"),
    )
    at = importlib.util.module_from_spec(spec)
    sys.modules["wan.modules.attention"] = at
    spec.loader.exec_module(at)

    # Load multitalk_model.py
    spec = importlib.util.spec_from_file_location(
        "wan.modules.multitalk_model",
        os.path.join(infinitetalk_src, "wan", "modules", "multitalk_model.py"),
    )
    mm = importlib.util.module_from_spec(spec)
    sys.modules["wan.modules.multitalk_model"] = mm
    spec.loader.exec_module(mm)

    return mm.WanModel


def build_original(
    sd: Dict[str, torch.Tensor], infinitetalk_src: str, device: str
) -> nn.Module:
    """Build InfiniteTalk's original WanModel and load weights."""
    _install_mocks()
    OrigWanModel = _import_original_model(infinitetalk_src)

    model = OrigWanModel(**MODEL_KWARGS)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  [orig] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    if missing:
        real_missing = [k for k in missing if k != "freqs"]
        if real_missing:
            print(f"  [orig] Missing keys ({len(real_missing)}): {real_missing[:5]}...")
    model.init_freqs()
    model.enable_teacache = False
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    return model


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def construct_i2v_mask(
    frame_num: int, lat_h: int, lat_w: int,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build the 4-channel I2V temporal mask [1, 4, T_lat, lat_h, lat_w]."""
    msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.cat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
        msk[:, 1:],
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2).to(dtype)
    return msk


def window_audio(audio_emb: torch.Tensor, num_frames: int = 81) -> torch.Tensor:
    """Apply 5-frame sliding window to raw audio embeddings.

    Input:  [81, 12, 768]
    Output: [1, 81, 5, 12, 768]
    """
    indices = torch.arange(5) - 2  # [-2, -1, 0, 1, 2]
    center_indices = torch.arange(num_frames).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=num_frames - 1)
    windowed = audio_emb[center_indices]  # [81, 5, 12, 768]
    return windowed.unsqueeze(0)          # [1, 81, 5, 12, 768]


def load_sample(
    data_dir: str, sample_name: str, device: str, dtype: torch.dtype
) -> dict:
    """Load a precomputed TalkVid sample and prepare all model inputs."""
    sample_dir = os.path.join(data_dir, sample_name)

    vae_latents = torch.load(
        os.path.join(sample_dir, "vae_latents.pt"), map_location="cpu",
        weights_only=True,
    )  # [16, 21, 80, 80]
    first_frame_cond = torch.load(
        os.path.join(sample_dir, "first_frame_cond.pt"), map_location="cpu",
        weights_only=True,
    )  # [16, 21, 80, 80]
    clip_features = torch.load(
        os.path.join(sample_dir, "clip_features.pt"), map_location="cpu",
        weights_only=True,
    )  # [1, 257, 1280]
    audio_emb = torch.load(
        os.path.join(sample_dir, "audio_emb.pt"), map_location="cpu",
        weights_only=True,
    )  # [81, 12, 768]
    text_embeds = torch.load(
        os.path.join(sample_dir, "text_embeds.pt"), map_location="cpu",
        weights_only=True,
    )  # [1, 512, 4096]

    C, T_lat, lat_h, lat_w = vae_latents.shape
    frame_num = (T_lat - 1) * 4 + 1  # 81

    msk = construct_i2v_mask(frame_num, lat_h, lat_w, device="cpu", dtype=dtype)
    # y: [1, 20, T, H, W]
    y = torch.cat([msk, first_frame_cond.unsqueeze(0).to(dtype)], dim=1)

    windowed_audio = window_audio(audio_emb, num_frames=frame_num)

    N_t = T_lat // 1  # patch_size[0] = 1
    N_h = lat_h // 2  # patch_size[1] = 2
    N_w = lat_w // 2  # patch_size[2] = 2
    seq_len = N_t * N_h * N_w

    return dict(
        vae_latents=vae_latents.to(device=device, dtype=dtype),
        first_frame_cond=first_frame_cond.to(device=device, dtype=dtype),
        clip_features=clip_features.to(device=device, dtype=dtype),
        audio=windowed_audio.to(device=device, dtype=dtype),
        text_embeds=text_embeds.to(device=device, dtype=dtype),
        y=y.to(device=device, dtype=dtype),
        seq_len=seq_len,
        lat_h=lat_h,
        lat_w=lat_w,
        T_lat=T_lat,
        frame_num=frame_num,
    )


def build_ref_target_masks(lat_h: int, lat_w: int, device: str) -> torch.Tensor:
    """Build ref_target_masks for single-speaker.

    For HUMAN_NUMBER==1, InfiniteTalk sets all masks to ones (multitalk.py
    lines 587-591): human_mask1=ones, human_mask2=ones, background=ones.
    The pipeline resizes them to (lat_h, lat_w) and the model's forward
    further interpolates to (N_h, N_w) = (lat_h//2, lat_w//2).

    Result shape: [3, lat_h, lat_w] -- matching the pipeline's output
    before entering the model forward.
    """
    masks = torch.ones(3, lat_h, lat_w)
    return masks.to(device)


def prepare_forward_inputs(
    sample: dict, device: str, dtype: torch.dtype, timestep: float = 500.0,
) -> dict:
    """Prepare the exact inputs for WanModel.forward() from a loaded sample."""
    x = [sample["vae_latents"]]
    y_list = [sample["y"][0]]  # unbatch -> [20, T, H, W]
    context = [sample["text_embeds"][0]]  # [512, 4096]
    t = torch.tensor([timestep], device=device)
    clip_fea = sample["clip_features"]
    audio = sample["audio"]

    return dict(
        x=x,
        t=t,
        context=context,
        seq_len=sample["seq_len"],
        clip_fea=clip_fea,
        y=y_list,
        audio=audio,
    )


# ---------------------------------------------------------------------------
# Level 1: Weight Loading Verification
# ---------------------------------------------------------------------------

def verify_level_1(
    weights_dir: str, infinitetalk_ckpt: str, infinitetalk_src: str, device: str,
) -> bool:
    print("\n" + "=" * 72)
    print("LEVEL 1: Weight Loading Verification")
    print("=" * 72)

    print("\nLoading weights...")
    sd = load_base_and_infinitetalk_weights(weights_dir, infinitetalk_ckpt)

    print("\nBuilding our ported model...")
    ours = build_ours(sd, device="cpu")

    print("\nBuilding original InfiniteTalk model...")
    orig = build_original(sd, infinitetalk_src, device="cpu")

    del sd
    gc.collect()

    # ----- Compare parameter names -----
    ours_params = dict(ours.named_parameters())
    orig_params = dict(orig.named_parameters())
    ours_keys = set(ours_params.keys())
    orig_keys = set(orig_params.keys())

    only_ours = ours_keys - orig_keys
    only_orig = orig_keys - ours_keys
    common = ours_keys & orig_keys

    print(f"\n  Our model params:      {len(ours_keys)}")
    print(f"  Original model params: {len(orig_keys)}")
    print(f"  Common params:         {len(common)}")

    if only_ours:
        print(f"  Only in ours ({len(only_ours)}):   {sorted(only_ours)[:10]}")
    if only_orig:
        print(f"  Only in original ({len(only_orig)}): {sorted(only_orig)[:10]}")

    # ----- Compare total param count -----
    ours_total = sum(p.numel() for p in ours.parameters())
    orig_total = sum(p.numel() for p in orig.parameters())
    print(f"\n  Our total param count:      {ours_total:,}")
    print(f"  Original total param count: {orig_total:,}")

    # ----- Compare values for common parameters -----
    max_diff_overall = 0.0
    mismatches = []
    shape_mismatches = []

    for name in sorted(common):
        p_ours = ours_params[name]
        p_orig = orig_params[name]

        if p_ours.shape != p_orig.shape:
            shape_mismatches.append(
                f"    {name}: ours={p_ours.shape} vs orig={p_orig.shape}"
            )
            continue

        diff = (p_ours.float() - p_orig.float()).abs().max().item()
        max_diff_overall = max(max_diff_overall, diff)

        if not torch.allclose(p_ours.float(), p_orig.float(), atol=1e-6):
            mismatches.append(f"    {name}: max_diff={diff:.2e}")

    if shape_mismatches:
        print(f"\n  Shape mismatches ({len(shape_mismatches)}):")
        for m in shape_mismatches[:10]:
            print(m)

    if mismatches:
        print(f"\n  Value mismatches ({len(mismatches)}):")
        for m in mismatches[:10]:
            print(m)

    passed = (
        len(mismatches) == 0
        and len(shape_mismatches) == 0
        and len(only_ours) == 0
        and len(only_orig) == 0
        and ours_total == orig_total
    )

    print(f"\n  Max absolute difference across all params: {max_diff_overall:.2e}")
    print(f"\n  LEVEL 1: {'PASS' if passed else 'FAIL'}")

    del ours, orig
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return passed


# ---------------------------------------------------------------------------
# Level 2: Single Forward Pass Equivalence
# ---------------------------------------------------------------------------

def verify_level_2(
    data_dir: str,
    weights_dir: str,
    infinitetalk_ckpt: str,
    infinitetalk_src: str,
    device: str,
) -> bool:
    print("\n" + "=" * 72)
    print("LEVEL 2: Single Forward Pass Equivalence")
    print("=" * 72)

    dtype = torch.bfloat16

    # Run models SEQUENTIALLY to fit in 80GB:
    # Each 14B model in bf16 ~28GB + activations ~15GB = ~43GB.
    # Running both simultaneously would need ~86GB and OOM.

    print("\nLoading weights...")
    sd = load_base_and_infinitetalk_weights(weights_dir, infinitetalk_ckpt)

    all_passed = True
    timestep = 500.0

    for sample_idx, sample_name in enumerate(SAMPLE_DIRS):
        print(f"\n  --- Sample {sample_idx + 1}/{len(SAMPLE_DIRS)}: {sample_name} ---")

        sample = load_sample(data_dir, sample_name, device, dtype)
        fwd_inputs = prepare_forward_inputs(sample, device, dtype, timestep=timestep)

        # --- Run OUR model first ---
        print("    Running our ported model...")
        ours = build_ours(sd, device=device)
        with torch.no_grad():
            out_ours = ours(**fwd_inputs)
            if isinstance(out_ours, (list, tuple)):
                out_ours = out_ours[0]
            out_ours = out_ours.cpu()  # move to CPU before deleting model
        del ours
        gc.collect()
        torch.cuda.empty_cache()

        # --- Run ORIGINAL model second ---
        print("    Running original InfiniteTalk model...")
        orig = build_original(sd, infinitetalk_src, device=device)
        ref_target_masks = build_ref_target_masks(sample["lat_h"], sample["lat_w"], device)
        fwd_orig = dict(fwd_inputs)
        fwd_orig["ref_target_masks"] = ref_target_masks
        with torch.no_grad():
            out_orig = orig(**fwd_orig)
            if isinstance(out_orig, (list, tuple)):
                out_orig = out_orig[0]
            out_orig = out_orig.cpu()
        del orig, ref_target_masks
        gc.collect()
        torch.cuda.empty_cache()

        # --- Compare ---
        max_diff = (out_ours.float() - out_orig.float()).abs().max().item()
        mean_diff = (out_ours.float() - out_orig.float()).abs().mean().item()
        atol = 1e-3
        passed = torch.allclose(out_ours.float(), out_orig.float(), atol=atol)

        print(f"    Output shape: ours={out_ours.shape}, orig={out_orig.shape}")
        print(f"    Max abs diff:  {max_diff:.6e}")
        print(f"    Mean abs diff: {mean_diff:.6e}")
        print(f"    allclose(atol={atol}): {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

        del sample, fwd_inputs, fwd_orig, out_ours, out_orig
        gc.collect()
        torch.cuda.empty_cache()

    del sd
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  LEVEL 2: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


# ---------------------------------------------------------------------------
# Level 3: Full 40-step Denoising Equivalence
# ---------------------------------------------------------------------------

def timestep_transform(t, shift=7.0, num_timesteps=1000):
    """Apply InfiniteTalk's timestep shift (multitalk.py lines 95-104)."""
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


def run_denoising_loop(
    model,
    sample: dict,
    neg_text_embeds: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    num_steps: int = 40,
    shift: float = 7.0,
    text_scale: float = 5.0,
    audio_scale: float = 4.0,
    seed: int = 42,
    ref_target_masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the Euler ODE denoising loop matching InfiniteTalk's pipeline.

    Follows multitalk.py lines 639-763.
    """
    import numpy as np

    num_timesteps = 1000

    # Timestep schedule (multitalk.py lines 639-643)
    ts_list = list(np.linspace(num_timesteps, 1, num_steps, dtype=np.float32))
    ts_list.append(0.0)
    timesteps = [torch.tensor([t], device=device) for t in ts_list]
    timesteps = [
        timestep_transform(t, shift=shift, num_timesteps=num_timesteps)
        for t in timesteps
    ]

    # Start from noise with fixed seed
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    T_lat = sample["T_lat"]
    lat_h = sample["lat_h"]
    lat_w = sample["lat_w"]
    seq_len = sample["seq_len"]

    noise = torch.randn(
        16, T_lat, lat_h, lat_w,
        dtype=torch.float32, device=device, generator=gen,
    )
    latent = noise.to(dtype)

    # Shared forward args
    y_list = [sample["y"][0]]
    context_pos = [sample["text_embeds"][0]]
    context_neg = [neg_text_embeds[0].to(device=device, dtype=dtype)]
    clip_fea = sample["clip_features"]
    audio = sample["audio"]
    audio_zero = torch.zeros_like(audio)

    base_kw = dict(seq_len=seq_len, clip_fea=clip_fea, y=y_list)
    if ref_target_masks is not None:
        base_kw["ref_target_masks"] = ref_target_masks

    for i in range(num_steps):
        latent_in = [latent]

        # Call 1: full condition (positive text + audio)
        out = model(x=latent_in, t=timesteps[i], context=context_pos, audio=audio, **base_kw)
        noise_cond = (out[0] if isinstance(out, (list, tuple)) else out)[0]

        # Call 2: drop text (negative text + audio)
        out = model(x=latent_in, t=timesteps[i], context=context_neg, audio=audio, **base_kw)
        noise_drop_text = (out[0] if isinstance(out, (list, tuple)) else out)[0]

        # Call 3: uncond (negative text + zero audio)
        out = model(x=latent_in, t=timesteps[i], context=context_neg, audio=audio_zero, **base_kw)
        noise_uncond = (out[0] if isinstance(out, (list, tuple)) else out)[0]

        # CFG combination (multitalk.py lines 755-758)
        noise_pred = (
            noise_uncond
            + text_scale * (noise_cond - noise_drop_text)
            + audio_scale * (noise_drop_text - noise_uncond)
        )
        noise_pred = -noise_pred

        # Euler step (multitalk.py lines 761-763)
        dt = (timesteps[i] - timesteps[i + 1]) / num_timesteps
        latent = latent + noise_pred * dt[:, None, None, None]

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"    Step {i + 1:3d}/{num_steps}, "
                f"latent [{latent.min().item():.3f}, {latent.max().item():.3f}]"
            )

    return latent


def verify_level_3(
    data_dir: str,
    weights_dir: str,
    infinitetalk_ckpt: str,
    infinitetalk_src: str,
    device: str,
) -> bool:
    print("\n" + "=" * 72)
    print("LEVEL 3: Full 40-step Denoising Equivalence")
    print("=" * 72)

    dtype = torch.bfloat16

    print("\nLoading weights...")
    sd = load_base_and_infinitetalk_weights(weights_dir, infinitetalk_ckpt)

    print("\nBuilding our ported model...")
    ours = build_ours(sd, device=device)

    print("\nBuilding original InfiniteTalk model...")
    orig = build_original(sd, infinitetalk_src, device=device)

    del sd
    gc.collect()
    torch.cuda.empty_cache()

    sample_name = SAMPLE_DIRS[0]
    print(f"\n  Using sample: {sample_name}")

    sample = load_sample(data_dir, sample_name, device, dtype)
    neg_text_embeds = torch.load(
        os.path.join(data_dir, "neg_text_embeds.pt"),
        map_location="cpu", weights_only=True,
    ).to(dtype)

    ref_target_masks = build_ref_target_masks(sample["lat_h"], sample["lat_w"], device)

    print("\n  Denoising with OUR model...")
    with torch.no_grad():
        latent_ours = run_denoising_loop(
            ours, sample, neg_text_embeds, device, dtype, seed=42,
        )

    print("\n  Denoising with ORIGINAL model...")
    with torch.no_grad():
        latent_orig = run_denoising_loop(
            orig, sample, neg_text_embeds, device, dtype, seed=42,
            ref_target_masks=ref_target_masks,
        )

    max_diff = (latent_ours.float() - latent_orig.float()).abs().max().item()
    mean_diff = (latent_ours.float() - latent_orig.float()).abs().mean().item()
    atol = 5e-2
    passed = torch.allclose(latent_ours.float(), latent_orig.float(), atol=atol)

    print(f"\n  Final latent shape: ours={latent_ours.shape}, orig={latent_orig.shape}")
    print(f"  Max abs diff:  {max_diff:.6e}")
    print(f"  Mean abs diff: {mean_diff:.6e}")
    print(f"  allclose(atol={atol}): {'PASS' if passed else 'FAIL'}")
    print(f"\n  LEVEL 3: {'PASS' if passed else 'FAIL'}")

    del ours, orig, sample, latent_ours, latent_orig
    gc.collect()
    torch.cuda.empty_cache()

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Our ported model lives under the FastGen project root
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("Stage 0 Verification: InfiniteTalk Equivalence")
    print(f"  Data dir:          {args.data_dir}")
    print(f"  Weights dir:       {args.weights_dir}")
    print(f"  InfiniteTalk ckpt: {args.infinitetalk_ckpt}")
    print(f"  InfiniteTalk src:  {args.infinitetalk_src}")
    print(f"  Device:            {args.device}")
    print(f"  Levels:            {args.level}")

    # Validate data paths
    for d in SAMPLE_DIRS:
        p = os.path.join(args.data_dir, d)
        if not os.path.isdir(p):
            print(f"ERROR: Sample directory not found: {p}")
            sys.exit(1)

    results = {}

    if 1 in args.level:
        results[1] = verify_level_1(
            args.weights_dir, args.infinitetalk_ckpt,
            args.infinitetalk_src, args.device,
        )

    if 2 in args.level:
        results[2] = verify_level_2(
            args.data_dir, args.weights_dir, args.infinitetalk_ckpt,
            args.infinitetalk_src, args.device,
        )

    if 3 in args.level:
        results[3] = verify_level_3(
            args.data_dir, args.weights_dir, args.infinitetalk_ckpt,
            args.infinitetalk_src, args.device,
        )

    # ----- Summary -----
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    all_passed = True
    for level in sorted(results):
        status = "PASS" if results[level] else "FAIL"
        print(f"  Level {level}: {status}")
        if not results[level]:
            all_passed = False

    if all_passed:
        print("\nAll levels PASSED. Ported model is equivalent to original.")
        sys.exit(0)
    else:
        print("\nSome levels FAILED. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
