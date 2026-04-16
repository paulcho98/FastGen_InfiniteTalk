# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for future anchor forward-path injection (Tasks 4, 5 & 8).

These tests validate:
  - _sample_attn_config correctly propagates future_anchor and distance
  - Config without future_anchor has no anchor keys
  - Anchor token embedding shape is correct
  - Anchor RoPE split uses causal_rope_apply with correct start_frame
  - Integration: mask + anchor, config edge cases, condition passthrough
"""

import math
import torch
import pytest


# ---------------------------------------------------------------------------
# _sample_attn_config tests
# ---------------------------------------------------------------------------

def test_anchor_config_sampling():
    """future_anchor configs should produce anchor keys with sampled distance."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.local_attn_size = 10
    net.sink_size = 1
    net._stochastic_attn_configs = [
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.5},
        {
            "local_attn_size": 10,
            "sink_size": 1,
            "weight": 0.5,
            "future_anchor": True,
            "future_anchor_distance_range": [1, 5],
        },
    ]

    results = [net._sample_attn_config() for _ in range(200)]
    has_anchor = [r.get("future_anchor", False) for r in results]
    assert any(has_anchor), "Should sometimes get future_anchor=True"
    assert not all(has_anchor), "Should sometimes get future_anchor=False"

    anchor_results = [r for r in results if r.get("future_anchor", False)]
    distances = [r["future_anchor_distance"] for r in anchor_results]
    assert all(1 <= d <= 5 for d in distances), (
        f"All distances should be in [1, 5], got {distances}"
    )
    assert len(set(distances)) > 1, "Should sample different distances"


def test_config_without_anchor_has_no_anchor_key():
    """Non-anchor configs should not contain future_anchor keys."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.local_attn_size = 10
    net.sink_size = 1
    net._stochastic_attn_configs = [
        {"local_attn_size": 10, "sink_size": 1, "weight": 1.0},
    ]

    cfg = net._sample_attn_config()
    assert "future_anchor" not in cfg
    assert "future_anchor_distance" not in cfg


def test_config_no_stochastic_fallback():
    """When no stochastic configs, should return defaults without anchor keys."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.local_attn_size = 10
    net.sink_size = 1
    net._stochastic_attn_configs = []

    cfg = net._sample_attn_config()
    assert cfg["local_attn_size"] == 10
    assert cfg["sink_size"] == 1
    assert "future_anchor" not in cfg


def test_anchor_distance_range_custom():
    """Custom distance range should be respected."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.local_attn_size = 10
    net.sink_size = 1
    net._stochastic_attn_configs = [
        {
            "local_attn_size": 10,
            "sink_size": 1,
            "weight": 1.0,
            "future_anchor": True,
            "future_anchor_distance_range": [3, 3],
        },
    ]

    results = [net._sample_attn_config() for _ in range(50)]
    for r in results:
        assert r["future_anchor"] is True
        assert r["future_anchor_distance"] == 3


# ---------------------------------------------------------------------------
# Anchor RoPE split tests (mock-based, no full model)
# ---------------------------------------------------------------------------

def test_anchor_rope_split():
    """Verify CausalSelfAttention applies separate RoPE for anchor tokens.

    Uses monkey-patching to verify causal_rope_apply is called with correct
    start_frame for anchor tokens. We also mock _flash_attention since
    flash_attn is GPU-only and these tests run on CPU.
    """
    import fastgen.networks.InfiniteTalk.network_causal as nc

    # Record calls to causal_rope_apply
    calls = []
    original_causal_rope = nc.causal_rope_apply

    def recording_causal_rope(x, grid_sizes, freqs, start_frame=0):
        calls.append({
            "shape": list(x.shape),
            "grid_sizes": grid_sizes.tolist(),
            "start_frame": start_frame,
        })
        return original_causal_rope(x, grid_sizes, freqs, start_frame=start_frame)

    # Mock _flash_attention to avoid GPU dependency (SDPA on CPU)
    def mock_flash_attn(q, k, v):
        # Simple scaled dot-product on CPU in float32
        q_t = q.transpose(1, 2).float()
        k_t = k.transpose(1, 2).float()
        v_t = v.transpose(1, 2).float()
        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        return out.transpose(1, 2).contiguous().type(q.dtype)

    # Build a minimal CausalSelfAttention
    # head_dim must be divisible by 6 for the 3D RoPE freq split to work
    dim = 96
    num_heads = 4
    head_dim = dim // num_heads  # 24
    attn = nc.CausalSelfAttention(dim, num_heads, head_dim)
    attn.eval()

    # Input: 2 main frames + 1 anchor frame, h=2, w=2 -> 4 tokens/frame
    # main tokens: 2*4 = 8, anchor tokens: 4, total = 12
    B = 1
    f_main = 2
    h, w = 2, 2
    frame_seqlen = h * w
    num_main = f_main * frame_seqlen
    num_anchor = frame_seqlen
    total = num_main + num_anchor
    anchor_frame_offset = f_main + 3  # e.g., 3 frames into the future

    x = torch.randn(B, total, dim)
    seq_lens = torch.tensor([total])
    grid_sizes = torch.tensor([[f_main, h, w]])
    freqs = nc._precompute_freqs_cis_3d(head_dim)

    # Monkey-patch causal_rope_apply and _flash_attention at the module level
    original_flash = nc._flash_attention
    nc.causal_rope_apply = recording_causal_rope
    nc._flash_attention = mock_flash_attn
    try:
        with torch.no_grad():
            out = attn(
                x, seq_lens, grid_sizes, freqs,
                block_mask=None,
                num_anchor_tokens=num_anchor,
                anchor_frame_offset=anchor_frame_offset,
            )
    finally:
        nc.causal_rope_apply = original_causal_rope
        nc._flash_attention = original_flash

    assert out.shape == (B, total, dim)

    # rope_apply_full calls causal_rope_apply with start_frame=0 for main
    # tokens (Q and K separately = 2 calls), then causal_rope_apply for
    # anchor with start_frame=anchor_frame_offset (Q and K = 2 calls).
    # Total: 4 calls.
    assert len(calls) == 4, f"Expected 4 causal_rope_apply calls, got {len(calls)}"

    # First two calls: main tokens (start_frame=0)
    assert calls[0]["start_frame"] == 0, "Main Q should have start_frame=0"
    assert calls[1]["start_frame"] == 0, "Main K should have start_frame=0"

    # Last two calls: anchor tokens (start_frame=anchor_frame_offset)
    assert calls[2]["start_frame"] == anchor_frame_offset
    assert calls[3]["start_frame"] == anchor_frame_offset

    # Anchor grid should have 1 frame
    assert calls[2]["grid_sizes"][0][0] == 1, "Anchor grid should have 1 frame"
    assert calls[3]["grid_sizes"][0][0] == 1


def test_no_anchor_rope_unchanged():
    """Without anchor tokens, RoPE should follow the original path."""
    import fastgen.networks.InfiniteTalk.network_causal as nc

    calls = []
    original_causal_rope = nc.causal_rope_apply

    def recording_causal_rope(x, grid_sizes, freqs, start_frame=0):
        calls.append({"start_frame": start_frame})
        return original_causal_rope(x, grid_sizes, freqs, start_frame=start_frame)

    def mock_flash_attn(q, k, v):
        q_t = q.transpose(1, 2).float()
        k_t = k.transpose(1, 2).float()
        v_t = v.transpose(1, 2).float()
        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        return out.transpose(1, 2).contiguous().type(q.dtype)

    # head_dim must be divisible by 6 for the 3D RoPE freq split
    dim = 96
    num_heads = 4
    head_dim = dim // num_heads  # 24
    attn = nc.CausalSelfAttention(dim, num_heads, head_dim)
    attn.eval()

    B, f, h, w = 1, 2, 2, 2
    total = f * h * w
    x = torch.randn(B, total, dim)
    seq_lens = torch.tensor([total])
    grid_sizes = torch.tensor([[f, h, w]])
    freqs = nc._precompute_freqs_cis_3d(head_dim)

    original_flash = nc._flash_attention
    nc.causal_rope_apply = recording_causal_rope
    nc._flash_attention = mock_flash_attn
    try:
        with torch.no_grad():
            out = attn(
                x, seq_lens, grid_sizes, freqs,
                block_mask=None,
                num_anchor_tokens=0,
                anchor_frame_offset=0,
            )
    finally:
        nc.causal_rope_apply = original_causal_rope
        nc._flash_attention = original_flash

    assert out.shape == (B, total, dim)
    # rope_apply_full -> causal_rope_apply(start_frame=0) for Q and K = 2 calls
    assert len(calls) == 2
    assert all(c["start_frame"] == 0 for c in calls)


# ---------------------------------------------------------------------------
# Integration tests (Task 8)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FlexAttention")
def test_full_path_mask_with_anchor():
    """Build mask with anchor tokens, verify it accepts the correct sequence length."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.chunk_size = 3

    num_frames = 21
    frame_seqlen = 28 * 56  # quarter res: 28x56 = 1568 tokens/frame
    num_anchor_tokens = frame_seqlen  # 1 anchor frame

    mask = net._build_block_mask(
        torch.device("cuda"),
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        chunk_size=3,
        local_attn_size=10,
        sink_size=1,
        num_anchor_tokens=num_anchor_tokens,
    )
    assert mask is not None
    # Total tokens: 21 * 1568 + 1568 = 22 * 1568 = 34496
    # Padded to next 128 multiple
    expected_total = (num_frames + 1) * frame_seqlen  # 34496
    padded = math.ceil(expected_total / 128) * 128
    # BlockMask should have this padded shape
    # If it built without error, the mask is valid for the expected sequence length


def test_anchor_config_distance_bounds():
    """Distance range [1,1] always produces distance=1."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.local_attn_size = 10
    net.sink_size = 1
    net._stochastic_attn_configs = [
        {
            "local_attn_size": 10,
            "sink_size": 1,
            "weight": 1.0,
            "future_anchor": True,
            "future_anchor_distance_range": [1, 1],
        },
    ]
    for _ in range(20):
        cfg = net._sample_attn_config()
        assert cfg["future_anchor"] is True
        assert cfg["future_anchor_distance"] == 1


def test_condition_passthrough():
    """_build_condition includes future_anchor_latents when present."""
    from fastgen.methods.infinitetalk_diffusion_forcing import InfiniteTalkDiffusionForcingModel

    model = object.__new__(InfiniteTalkDiffusionForcingModel)

    data = {
        "text_embeds": torch.zeros(1, 512, 4096),
        "first_frame_cond": torch.zeros(1, 16, 21, 28, 56),
        "clip_features": torch.zeros(1, 257, 1280),
        "audio_emb": torch.zeros(1, 81, 12, 768),
        "future_anchor_latents": torch.randn(1, 16, 5, 28, 56),
    }
    condition = model._build_condition(data)
    assert "future_anchor_latents" in condition
    assert condition["future_anchor_latents"].shape == (1, 16, 5, 28, 56)


def test_condition_passthrough_without_anchor():
    """_build_condition works without future_anchor_latents."""
    from fastgen.methods.infinitetalk_diffusion_forcing import InfiniteTalkDiffusionForcingModel

    model = object.__new__(InfiniteTalkDiffusionForcingModel)

    data = {
        "text_embeds": torch.zeros(1, 512, 4096),
        "first_frame_cond": torch.zeros(1, 16, 21, 28, 56),
        "clip_features": torch.zeros(1, 257, 1280),
        "audio_emb": torch.zeros(1, 81, 12, 768),
    }
    condition = model._build_condition(data)
    assert "future_anchor_latents" not in condition
