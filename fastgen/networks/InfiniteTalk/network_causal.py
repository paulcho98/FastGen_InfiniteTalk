# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Causal InfiniteTalk wrapper for the student network in Self-Forcing distillation.

This module implements the causal variant of InfiniteTalk's 14B DiT, which adds:
  1. Causal self-attention via FlexAttention block masks (full-sequence mode)
  2. KV cache management for chunk-by-chunk autoregressive generation (AR mode)
  3. Causal RoPE with frame offset for absolute positional encoding in AR mode
  4. Per-frame timestep embedding for inhomogeneous timesteps (KD/DF training)
  5. Audio caching for AR mode (process once, slice per chunk)

The causal model is structurally DIFFERENT from the bidirectional model -- it has
its own transformer block (``CausalDiTBlock``) with causal self-attention
(``CausalSelfAttention``), per-frame modulation, and KV cache support.  However,
the WEIGHTS are compatible: both variants share patch_embedding, text_embedding,
time_embedding, time_projection, head, img_emb, audio_proj, and per-block
q/k/v/o + cross-attn + ffn + audio_cross_attn parameters.

Architecture reference:
    ``Self-Forcing-OmniAvatar/Self-Forcing/wan/modules/causal_model.py``
    -- adapted for InfiniteTalk's I2V conditioning and audio architecture.

Weight loading pipeline:
    1. Base Wan 2.1 I2V-14B safetensor shards  ->  model internals
    2. InfiniteTalk checkpoint on top (audio modules + base weight overrides)
    3. (optional) Merge external LoRA into base weights
    4. Apply runtime LoRA adapters + freeze base (student is always LoRA-trainable)

Two forward modes:
    - ``_forward_full_sequence``: Full-sequence with chunk-wise causal mask
      (FlexAttention).  Used for KD and DF training with per-frame timesteps.
    - ``_forward_ar``: Chunk-by-chunk with KV cache.  Used for Self-Forcing.

Key differences from OmniAvatar causal wrapper:
    - I2V conditioning (20ch: 4 mask + 16 VAE ref) instead of V2V (65ch)
    - in_dim=36 (16 noise + 20 I2V cond) instead of 65
    - Audio via AudioProjModel + SingleStreamAttention per-block cross-attention
      (instead of OmniAvatar's AudioPack + additive injection)
    - CLIP image cross-attention (WanI2VCrossAttention) instead of T2V cross-attention
    - LoRA always applied (student is always trainable via LoRA + audio modules)
"""

import os
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file as safetensors_load_file

from fastgen.networks.network import CausalFastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.networks.InfiniteTalk.audio_modules import AudioProjModel, SingleStreamAttention
from fastgen.networks.InfiniteTalk.lora import (
    apply_lora,
    freeze_base,
    merge_lora_from_file,
)
import fastgen.utils.logging_utils as logger


# ---------------------------------------------------------------------------
# FlexAttention imports (optional -- falls back to SDP if unavailable)
# ---------------------------------------------------------------------------
_disable_flex_env = os.environ.get("FASTGEN_DISABLE_FLEX_ATTENTION", "0") == "1"
try:
    if _disable_flex_env:
        raise ImportError("FlexAttention disabled via env var")
    from torch.nn.attention.flex_attention import (
        create_block_mask,
        flex_attention as _flex_attention,
        BlockMask,
    )

    FLEX_ATTENTION_AVAILABLE = True

    try:
        import torch._dynamo as _dynamo
        _dynamo.config.optimize_ddp = False
    except Exception:
        pass

    _compile_mode = os.environ.get("TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")
    _disable_compile = (
        os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"
        or os.environ.get("FASTGEN_FLEX_COMPILE", "1") == "0"
    )
    if not _disable_compile:
        flex_attention = torch.compile(
            _flex_attention, dynamic=False, mode=_compile_mode
        )
    else:
        flex_attention = _flex_attention
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    create_block_mask = None  # type: ignore
    flex_attention = None  # type: ignore

    class BlockMask:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Flash attention (for KV cache path -- not FlexAttention)
# ---------------------------------------------------------------------------
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    flash_attn_interface = None
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def _flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Flash attention for KV-cache path.  Input format: [B, S, H, D]."""
    half_dtypes = (torch.float16, torch.bfloat16)
    b, lq, lk = q.shape[0], q.shape[1], k.shape[1]
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

    q_flat = half(q.flatten(0, 1))
    k_flat = half(k.flatten(0, 1))
    v_flat = half(v.flatten(0, 1))
    q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)

    if FLASH_ATTN_3_AVAILABLE:
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        out = flash_attn_interface.flash_attn_varlen_func(
            q=q_flat, k=k_flat, v=v_flat,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=lq, max_seqlen_k=lk,
        )
        y = out[0] if isinstance(out, (list, tuple)) else out
        return y.unflatten(0, (b, lq)).type(out_dtype)
    elif FLASH_ATTN_2_AVAILABLE:
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return flash_attn.flash_attn_varlen_func(
            q=q_flat, k=k_flat, v=v_flat,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=lq, max_seqlen_k=lk,
        ).unflatten(0, (b, lq)).type(out_dtype)
    else:
        # Fallback: scaled dot-product attention (only reached if flash_attn is unavailable)
        q_t = q.transpose(1, 2).to(torch.bfloat16)
        k_t = k.transpose(1, 2).to(torch.bfloat16)
        v_t = v.transpose(1, 2).to(torch.bfloat16)
        # Note: KV-cache path is not causal (attending to cached past + current chunk)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t)
        return out.transpose(1, 2).contiguous().type(out_dtype)


# ---------------------------------------------------------------------------
# Causal RoPE with frame offset
# ---------------------------------------------------------------------------

def _precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute 1D RoPE frequency table as complex exponentials."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim)
    )
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute 3D RoPE frequency tables for (F, H, W)."""
    f_freqs = _precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs = _precompute_freqs_cis(dim // 3, end, theta)
    w_freqs = _precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs, h_freqs, w_freqs


def causal_rope_apply(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    start_frame: int = 0,
) -> torch.Tensor:
    """Apply 3D RoPE with a temporal frame offset for causal generation.

    Args:
        x: [B, S, num_heads, head_dim]
        grid_sizes: [B, 3] -- (F, H, W) per sample
        freqs: tuple of 3 complex frequency tables (f, h, w)
        start_frame: temporal offset (number of frames already generated)

    Returns:
        Tensor same shape as x with RoPE applied.
    """
    n, c = x.size(2), x.size(3) // 2
    freq_f, freq_h, freq_w = freqs

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freq_f[start_frame: start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freq_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freq_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1).to(device=x_i.device)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # Append any padding tokens unchanged
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)

    return torch.stack(output).type_as(x)


def rope_apply_full(
    x: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply standard (non-causal) 3D RoPE -- start_frame=0."""
    return causal_rope_apply(x, grid_sizes, freqs, start_frame=0)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional embedding (computed in float64 for precision)."""
    assert dim % 2 == 0
    half = dim // 2
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(half, dtype=torch.float64, device=position.device).div(half),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# ---------------------------------------------------------------------------
# Norm layers (matching WanModel conventions)
# ---------------------------------------------------------------------------

class WanRMSNorm(nn.Module):
    """RMSNorm matching WanModel's WanRMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm matching WanModel's WanLayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(origin_dtype)
        return out


# ---------------------------------------------------------------------------
# CLIP projection (matching WanModel's MLPProj)
# ---------------------------------------------------------------------------

class MLPProj(nn.Module):
    """CLIP image embedding projection."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


# ---------------------------------------------------------------------------
# Causal Self-Attention with KV cache
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Causal self-attention with KV cache support and FlexAttention.

    In full-sequence mode (kv_cache=None): uses FlexAttention with block mask
    for causal attention over the entire sequence.

    In AR mode (kv_cache provided): uses flash attention with gradient-safe
    KV caching (detached writes, cat [detached_past | live_current]).

    Supports:
      - ``local_attn_size``: rolling local attention window (in frames).
        ``-1`` means attend to everything in the cache.
      - ``sink_size``: number of initial frames always kept in the window
        (attention-sink tokens, never dropped from view).
      - ``use_dynamic_rope``: cache raw K (without RoPE), apply window-local
        RoPE at attention time.  Required for correct position encoding
        when using rolling eviction (otherwise evicted-then-re-viewed
        tokens have stale absolute positions).

    Weight-compatible with ``WanSelfAttention`` from wan_model.py -- same
    q/k/v/o Linear layers and norm_q/norm_k RMSNorm layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_dynamic_rope: bool = False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_dynamic_rope = use_dynamic_rope

        # Layers -- same names as WanSelfAttention for weight compatibility
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        block_mask=None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        current_start: int = 0,
        store_kv: bool = True,
        cache_local_end_override: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            seq_lens: [B]
            grid_sizes: [B, 3] -- (F, H, W)
            freqs: 3D RoPE freq tables
            block_mask: FlexAttention block mask (for full-sequence causal mode)
            kv_cache: dict with 'k', 'v', 'global_end_index', 'local_end_index'
            current_start: token offset for causal RoPE in AR mode
            store_kv: if True, write to cache and update metadata
            cache_local_end_override: if set, use this as local_end instead of
                reading from cache (for gradient checkpointing determinism)
        """
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if kv_cache is None:
            # ----- Full-sequence mode (training / bidirectional eval) -----
            roped_q = rope_apply_full(q, grid_sizes, freqs).type_as(v)
            roped_k = rope_apply_full(k, grid_sizes, freqs).type_as(v)

            if block_mask is not None and FLEX_ATTENTION_AVAILABLE:
                # FlexAttention path -- chunk-wise causal mask
                pad_len = math.ceil(s / 128) * 128 - s
                if pad_len > 0:
                    pad_shape = (b, pad_len, n, d)
                    roped_q = torch.cat(
                        [roped_q, torch.zeros(pad_shape, device=q.device, dtype=v.dtype)],
                        dim=1,
                    )
                    roped_k = torch.cat(
                        [roped_k, torch.zeros(pad_shape, device=k.device, dtype=v.dtype)],
                        dim=1,
                    )
                    v_padded = torch.cat(
                        [v, torch.zeros(pad_shape, device=v.device, dtype=v.dtype)],
                        dim=1,
                    )
                else:
                    v_padded = v

                out = flex_attention(
                    query=roped_q.transpose(1, 2),
                    key=roped_k.transpose(1, 2),
                    value=v_padded.transpose(1, 2),
                    block_mask=block_mask,
                )
                if pad_len > 0:
                    out = out[:, :, :-pad_len]
                x = out.transpose(1, 2)  # [B, S, H, D]
            else:
                # Standard flash attention (no causal mask -- fully bidirectional)
                x = _flash_attention(roped_q, roped_k, v)
        else:
            # ----- AR mode with KV cache -----
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            num_new_tokens = q.shape[1]
            current_end = current_start + num_new_tokens
            kv_cache_size = kv_cache["k"].shape[1]
            sink_tokens = self.sink_size * frame_seqlen

            # -- 1. Prepare K for caching (mode-dependent) --
            if not self.use_dynamic_rope:
                # Original mode: apply RoPE before caching (absolute positions)
                roped_q = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)
                roped_k = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame
                ).type_as(v)
                k_to_cache = roped_k
            else:
                # Dynamic mode: cache raw keys; RoPE applied later on window
                k_to_cache = k

            # -- 2. Read cache metadata --
            # Use the explicit override (frozen before the block loop)
            # to ensure gradient-checkpointing recomputation sees the same state.
            if cache_local_end_override is not None:
                local_end = cache_local_end_override
                # global_end tracks the actual sequence position (not cache buffer position).
                # After sliding-window eviction, global_end > local_end.
                # current_start IS the global position of already-cached content.
                global_end = current_start
            else:
                global_end = kv_cache["global_end_index"].item()
                local_end = kv_cache["local_end_index"].item()

            # -- 3. Handle physical eviction (only when buffer overflows) --
            # This is a FALLBACK for inference with small cache buffers.
            # During training, cache_size == total_tokens, so this never fires.
            if not store_kv:
                # Read-only pass (denoising steps): compute where the current
                # chunk WOULD go (for correct window bounds) but don't modify
                # the cache buffer.  new_local_start == local_end means the
                # current tokens aren't in the buffer and will be concatenated
                # from k_to_cache.  new_local_end includes the current tokens
                # conceptually so the window math matches the store_kv=True path.
                new_local_end = local_end + num_new_tokens
                new_local_start = local_end
            elif (
                self.local_attn_size > 0
                and current_end > global_end
                and num_new_tokens + local_end > kv_cache_size
            ):
                num_evicted = num_new_tokens + local_end - kv_cache_size
                num_rolled = local_end - num_evicted - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled] = \
                    kv_cache["k"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled] = \
                    kv_cache["v"][:, sink_tokens + num_evicted:sink_tokens + num_evicted + num_rolled].clone()
                new_local_end = local_end + (current_end - global_end) - num_evicted
                new_local_start = new_local_end - num_new_tokens
            else:
                # No eviction: simple append
                new_local_end = local_end + max(0, current_end - global_end)
                new_local_start = new_local_end - num_new_tokens

            # -- 4. Write to cache (detached, only if store_kv) --
            if store_kv:
                kv_cache["k"][:, new_local_start:new_local_end] = k_to_cache.detach()
                kv_cache["v"][:, new_local_start:new_local_end] = v.detach()

            # -- 5. Build attention window --
            # Compute the max attention span in tokens
            if self.local_attn_size > 0:
                max_attn_tokens = self.local_attn_size * frame_seqlen
            else:
                max_attn_tokens = new_local_end  # attend to everything

            k_win_start = max(0, new_local_end - max_attn_tokens)

            if sink_tokens > 0 and k_win_start > sink_tokens:
                # Sink + rolling window (non-contiguous regions)
                available_rolling = max_attn_tokens - sink_tokens
                rolling_start = max(sink_tokens, new_local_end - available_rolling)

                with torch.no_grad():
                    k_past = torch.cat([
                        kv_cache["k"][:, :sink_tokens],
                        kv_cache["k"][:, rolling_start:new_local_start],
                    ], dim=1)
                    v_past = torch.cat([
                        kv_cache["v"][:, :sink_tokens],
                        kv_cache["v"][:, rolling_start:new_local_start],
                    ], dim=1)
                k_win = torch.cat([k_past, k_to_cache], dim=1)
                v_win = torch.cat([v_past, v], dim=1)
                query_offset_in_win = sink_tokens + (new_local_start - rolling_start)
            else:
                # Simple contiguous case (no sink gap)
                if new_local_start == 0:
                    k_win = k_to_cache
                    v_win = v
                else:
                    with torch.no_grad():
                        k_past = kv_cache["k"][:, k_win_start:new_local_start]
                        v_past = kv_cache["v"][:, k_win_start:new_local_start]
                    k_win = torch.cat([k_past, k_to_cache], dim=1)
                    v_win = torch.cat([v_past, v], dim=1)
                query_offset_in_win = new_local_start - k_win_start

            # -- 6. Apply RoPE (mode-dependent) --
            if not self.use_dynamic_rope:
                # Original mode: Q already rotated, k_win contains rotated keys
                roped_query = roped_q
                roped_key = k_win
            else:
                # Dynamic mode: apply window-local RoPE to the full window
                F_window = k_win.shape[1] // frame_seqlen
                k_grid = grid_sizes.clone()
                k_grid[:, 0] = F_window
                roped_key = causal_rope_apply(
                    k_win, k_grid, freqs, start_frame=0
                ).type_as(v)

                # Q position within the window
                q_frame_start = query_offset_in_win // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=q_frame_start
                ).type_as(v)

            # -- 7. Attention --
            x = _flash_attention(roped_query, roped_key, v_win)

            # -- 8. Update metadata (only if store_kv) --
            if store_kv:
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(new_local_end)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# I2V Cross-Attention (matching WanI2VCrossAttention for weight compat)
# ---------------------------------------------------------------------------

class CausalI2VCrossAttention(nn.Module):
    """I2V cross-attention with separate K/V for CLIP image tokens vs text tokens.

    Weight-compatible with ``WanI2VCrossAttention`` from wan_model.py:
    inherits q/k/v/o from self-attention base, adds k_img/v_img/norm_k_img.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # Same names as WanI2VCrossAttention (inheriting from WanSelfAttention)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Additional image cross-attention projections
        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] -- visual tokens
            context: [B, L_ctx, C] -- [CLIP_img(257) | text(512)] tokens
            context_lens: unused (kept for interface compat)
        """
        context_img = context[:, :257]
        context_txt = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # Q from visual
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        # K/V from text
        k = self.norm_k(self.k(context_txt)).view(b, -1, n, d)
        v = self.v(context_txt).view(b, -1, n, d)

        # K/V from CLIP image
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        # Separate attention for text and image
        img_x = _flash_attention(q, k_img, v_img)
        txt_x = _flash_attention(q, k, v)

        # Combine and project
        x = txt_x.flatten(2) + img_x.flatten(2)
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# Causal DiT Block (per-frame modulation + causal self-attention)
# ---------------------------------------------------------------------------

class CausalDiTBlock(nn.Module):
    """Transformer block for the causal model.

    Structurally matches ``WanAttentionBlock`` from wan_model.py but with:
    - CausalSelfAttention instead of WanSelfAttention (KV cache + FlexAttention)
    - CausalI2VCrossAttention instead of WanI2VCrossAttention
    - Per-frame AdaLN modulation (unflatten -> per-frame scale/shift -> flatten)
    - Audio cross-attention via SingleStreamAttention (same as bidirectional)

    IMPORTANT: norm ordering matches the original Wan convention:
        norm1 = self-attention, norm3 = cross-attention (learnable), norm2 = FFN

    Weight names match WanAttentionBlock exactly for weight loading compatibility.
    """

    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        output_dim: int = 768,
        norm_input_visual: bool = True,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_dynamic_rope: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Self-attention (causal variant with KV cache)
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            use_dynamic_rope=use_dynamic_rope,
        )

        # Cross-attention (I2V)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = CausalI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        # FFN
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # AdaLN modulation (6 modulation vectors -- same as WanAttentionBlock)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

        # Audio cross-attention (single-speaker, same as WanAttentionBlock)
        self.audio_cross_attn = SingleStreamAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qk_norm=False,
            qkv_bias=True,
            eps=eps,
            norm_layer=WanRMSNorm,
        )
        self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True) if norm_input_visual else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        context_lens: Optional[torch.Tensor],
        audio_embedding: Optional[torch.Tensor] = None,
        block_mask=None,
        kv_cache: Optional[Dict] = None,
        current_start: int = 0,
        store_kv: bool = True,
        cache_local_end_override: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            e: [B, F, 6, C] -- per-frame timestep modulation
            seq_lens: [B]
            grid_sizes: [B, 3]
            freqs: RoPE freq tables
            context: [B, L_ctx, C] -- [CLIP(257) | text(512)]
            context_lens: [B] or None
            audio_embedding: [B, N_t, N_a, C_audio] -- audio context tokens
            block_mask: FlexAttention mask (full-sequence mode only)
            kv_cache: self-attention KV cache dict (AR mode only)
            current_start: token offset for causal RoPE (AR mode only)
            store_kv: whether to update KV cache
            cache_local_end_override: for gradient checkpointing determinism
        """
        dtype = x.dtype
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames

        # AdaLN modulation -- 6 vectors per frame
        # e is [B, F, 6, C]; modulation is [1, 6, C] -> broadcast to [1, 1, 6, C]
        e_mod = (self.modulation.unsqueeze(0).unsqueeze(0) + e)
        # But OmniAvatar does: (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # e is [B, F, 6, C], modulation is [1, 6, C] -> need unsqueeze(1) for F dim
        # Actually: modulation is [1, 6, C], e is [B, F, 6, C]
        # broadcast: [1, 1, 6, C] + [B, F, 6, C] -> [B, F, 6, C]
        e_mod = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        # Each e_mod[i] is [B, F, 1, C]

        # --- Self-attention ---
        norm_x = self.norm1(x).float()
        norm_x = norm_x.unflatten(1, (num_frames, frame_seqlen))
        norm_x = (norm_x * (1 + e_mod[1]) + e_mod[0]).flatten(1, 2).type_as(x)

        y = self.self_attn(
            norm_x,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            store_kv=store_kv,
            cache_local_end_override=cache_local_end_override,
        )

        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + (
                y.unflatten(1, (num_frames, frame_seqlen)) * e_mod[2]
            ).flatten(1, 2)

        x = x.to(dtype)

        # --- Cross-attention (text + CLIP image) ---
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # --- Audio cross-attention ---
        if audio_embedding is not None:
            x_a = self.audio_cross_attn(
                self.norm_x(x),
                encoder_hidden_states=audio_embedding,
                shape=grid_sizes[0],
            )
            x = x + x_a

        # --- FFN with per-frame AdaLN modulation ---
        norm_x = self.norm2(x).float()
        norm_x = norm_x.unflatten(1, (num_frames, frame_seqlen))
        ff_out = self.ffn((norm_x * (1 + e_mod[4]) + e_mod[3]).flatten(1, 2).to(dtype))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + (
                ff_out.unflatten(1, (num_frames, frame_seqlen)) * e_mod[5]
            ).flatten(1, 2)

        x = x.to(dtype)
        return x


# ---------------------------------------------------------------------------
# Causal Head (per-frame modulation)
# ---------------------------------------------------------------------------

class CausalHead(nn.Module):
    """Output head with per-frame modulation.

    Weight-compatible with ``Head`` from wan_model.py -- same norm, head, and
    modulation parameter names.
    """

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size

        out_channels = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_channels)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]
            e: [B, F, 1, C] -- per-frame head modulation
        """
        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames

        # modulation is [1, 2, C], e is [B, F, 1, C]
        # broadcast: [1, 1, 2, C] + [B, F, 1, C] won't work directly
        # We need: modulation [1, 2, C] -> [1, 1, 2, C]
        e_mod = (self.modulation.unsqueeze(0) + e).chunk(2, dim=2)
        # Each e_mod[i] is [B, F, 1, C]

        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = self.head(
                self.norm(x).unflatten(1, (num_frames, frame_seqlen))
                * (1 + e_mod[1]) + e_mod[0]
            )
        return x


# ---------------------------------------------------------------------------
# 14B architecture defaults (same as bidirectional wrapper)
# ---------------------------------------------------------------------------

_14B_DEFAULTS = dict(
    dim=5120,
    ffn_dim=13824,
    freq_dim=256,
    num_heads=40,
    num_layers=40,
)

_COMMON_CFG = dict(
    model_type="i2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=36,   # 16 noise + 4 mask + 16 VAE ref (I2V concat before patch_embedding)
    out_dim=16,
    text_dim=4096,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)

# Audio module defaults (InfiniteTalk single-speaker)
_AUDIO_DEFAULTS = dict(
    audio_window=5,
    intermediate_dim=512,
    output_dim=768,
    context_tokens=32,
    vae_scale=4,
    norm_input_visual=True,
    norm_output_audio=True,
)


# ---------------------------------------------------------------------------
# CausalInfiniteTalkWan: the student network
# ---------------------------------------------------------------------------

def _maybe_apply_first_frame_anchor(
    out: torch.Tensor,
    net_module,
    cur_start_frame: int,
    condition,
    apply_anchor: bool = True,
) -> torch.Tensor:
    """Hard-anchor frame 0 to the clean reference when enabled.

    Modes (instance attributes on net_module):
      _enable_first_frame_anchor = True (default): anchor is active
      _enable_first_frame_anchor = False: anchor fully disabled
      _anchor_eval_only = True: anchor only in eval mode (not during training)

    The explicit ``apply_anchor=False`` argument overrides all of the above —
    used by the sampling loop's model-generated-sink-cache path to obtain the
    student's raw frame-0 prediction without the reference overwrite.
    """
    if not apply_anchor:
        return out
    if cur_start_frame != 0:
        return out
    if not isinstance(condition, dict) or "first_frame_cond" not in condition:
        return out
    anchor_active = getattr(net_module, "_enable_first_frame_anchor", True)
    if anchor_active and getattr(net_module, "_anchor_eval_only", False):
        anchor_active = not net_module.training
    if not anchor_active:
        return out
    first_frame_cond = condition["first_frame_cond"]
    out = out.clone()
    out[:, :, 0:1] = first_frame_cond[:, :, 0:1]
    return out


class CausalInfiniteTalkWan(CausalFastGenNetwork):
    """Causal InfiniteTalk DiT for use as the student in Self-Forcing distillation.

    This class implements the full causal model including:
      - CausalSelfAttention with KV cache + FlexAttention block mask
      - Causal 3D RoPE with frame offset
      - InfiniteTalk audio processing (AudioProjModel + SingleStreamAttention)
      - Per-frame AdaLN modulation
      - I2V conditioning (20ch: 4 mask + 16 VAE ref)
      - LoRA adapters for parameter-efficient training

    Two forward modes:
      - ``_forward_full_sequence``:  all frames at once with causal block mask
        (FlexAttention).  Used for KD and DF training.
      - ``_forward_ar``:  chunk-by-chunk with KV cache.  Used for Self-Forcing.

    The model loads the same weights as the bidirectional ``InfiniteTalkWan``
    wrapper, then applies LoRA adapters and freezes base weights.
    """

    def __init__(
        self,
        base_model_paths: str = "",
        infinitetalk_ckpt_path: str = "",
        lora_ckpt_path: str = "",
        lora_rank: int = 32,
        lora_alpha: int = 32,
        chunk_size: int = 3,
        total_num_frames: int = 21,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        shift: float = 7.0,
        disable_grad_ckpt: bool = False,
        # WanModel architecture params (defaults for 14B):
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        num_heads: int = 40,
        num_layers: int = 40,
        audio_window: int = 5,
        vae_scale: int = 4,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_dynamic_rope: bool = False,
        stochastic_attn_configs: Optional[list] = None,
        **kwargs,
    ):
        """
        Args:
            base_model_paths: Comma-separated safetensor shard paths for the base
                Wan 2.1 I2V-14B weights (7 shards).
            infinitetalk_ckpt_path: Path to ``infinitetalk.safetensors`` containing
                audio modules and base weight overrides.
            lora_ckpt_path: Path to external LoRA checkpoint to merge into base
                weights before applying runtime LoRA adapters.
            lora_rank: LoRA rank for both merge and runtime adapters.
            lora_alpha: LoRA alpha scaling.
            chunk_size: Number of latent frames per AR chunk (default 3).
            total_num_frames: Total latent frames in the full video (default 21).
            net_pred_type: Network prediction type (``"flow"`` for rectified flow).
            schedule_type: Noise schedule type (``"rf"`` for rectified flow).
            shift: Timestep shift for the RF noise schedule (7.0 for 480p).
            disable_grad_ckpt: If True, disable gradient checkpointing.
            dim: Hidden dimension (5120 for 14B).
            ffn_dim: FFN intermediate dimension (13824 for 14B).
            freq_dim: Sinusoidal embedding dimension (256).
            num_heads: Number of attention heads (40 for 14B).
            num_layers: Number of transformer blocks (40 for 14B).
            audio_window: Audio context window size (5).
            vae_scale: VAE temporal compression ratio (4).
            **kwargs: Additional kwargs passed to CausalFastGenNetwork.
        """
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            chunk_size=chunk_size,
            total_num_frames=total_num_frames,
            shift=shift,
            **kwargs,
        )

        self.base_model_paths = base_model_paths
        self.infinitetalk_ckpt_path = infinitetalk_ckpt_path
        self.lora_ckpt_path = lora_ckpt_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.shift = shift

        self._use_gradient_checkpointing = not disable_grad_ckpt

        # Store architecture params
        self._dim = dim
        self._ffn_dim = ffn_dim
        self._freq_dim = freq_dim
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._text_len = _COMMON_CFG["text_len"]
        self._patch_size = _COMMON_CFG["patch_size"]
        self._out_dim = _COMMON_CFG["out_dim"]
        self._in_dim = _COMMON_CFG["in_dim"]
        self._vae_scale = vae_scale
        self._audio_window = audio_window
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_dynamic_rope = use_dynamic_rope
        self._stochastic_attn_configs = stochastic_attn_configs

        eps = _COMMON_CFG["eps"]
        patch_size = _COMMON_CFG["patch_size"]
        text_dim = _COMMON_CFG["text_dim"]

        # I2V conditioning: in_dim = 16 (noise) + 20 (4 mask + 16 VAE ref) = 36
        in_dim_with_cond = 36

        # --- Embeddings ---
        self.patch_embedding = nn.Conv3d(
            in_dim_with_cond, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # --- CLIP image embedding projection ---
        self.img_emb = MLPProj(1280, dim)

        # --- Transformer blocks ---
        cross_attn_type = "i2v_cross_attn"
        self.blocks = nn.ModuleList([
            CausalDiTBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                _COMMON_CFG["window_size"], _COMMON_CFG["qk_norm"],
                _COMMON_CFG["cross_attn_norm"], eps,
                output_dim=_AUDIO_DEFAULTS["output_dim"],
                norm_input_visual=_AUDIO_DEFAULTS["norm_input_visual"],
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                use_dynamic_rope=use_dynamic_rope,
            )
            for _ in range(num_layers)
        ])

        # --- Output head ---
        self.head = CausalHead(dim, _COMMON_CFG["out_dim"], patch_size, eps)

        # --- RoPE frequencies (3D, as complex exponentials) ---
        head_dim = dim // num_heads
        self.freqs = _precompute_freqs_cis_3d(head_dim)

        # --- Audio adapter ---
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=_AUDIO_DEFAULTS["intermediate_dim"],
            output_dim=_AUDIO_DEFAULTS["output_dim"],
            context_tokens=_AUDIO_DEFAULTS["context_tokens"],
            norm_output_audio=_AUDIO_DEFAULTS["norm_output_audio"],
        )

        # --- FlexAttention block mask (lazily constructed) ---
        self.block_mask = None

        # --- KV caches (lazily allocated) ---
        self._kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None

        # --- Cached audio for AR mode ---
        self._cached_audio: Optional[torch.Tensor] = None

        # --- Load weights + apply LoRA ---
        if not self._is_in_meta_context():
            self._load_weights()
            self.to(torch.bfloat16)
        else:
            # Meta-init (non-rank-0): skip weight loading but still apply LoRA
            # adapters so model structure matches rank 0 for FSDP sync.
            # The causal student always has LoRA adapters (it's always trainable).
            apply_lora(self, rank=self.lora_rank, alpha=self.lora_alpha)
            freeze_base(self)

    # ------------------------------------------------------------------
    # Unpatchify
    # ------------------------------------------------------------------

    def _unpatchify(
        self,
        x: torch.Tensor,
        grid_sizes: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Reconstruct video tensors from patchified features.

        Args:
            x: [B, F, H*W, out_dim * prod(patch_size)]  -- batched with frame dim
            grid_sizes: [B, 3]

        Returns:
            List of tensors with shape [C_out, F*p_t, H*p_h, W*p_w]
        """
        c = self._out_dim
        p = self._patch_size
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *p, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, p)])
            out.append(u)
        return out

    # ------------------------------------------------------------------
    # Audio processing (same pipeline as WanModel.forward)
    # ------------------------------------------------------------------

    def _process_audio(
        self,
        audio: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Process raw audio through the InfiniteTalk audio pipeline.

        Replicates the audio preprocessing from WanModel.forward() (lines 721-741).

        Args:
            audio: [B, num_frames, audio_window, 12, 768]
            device: target device
            dtype: target dtype

        Returns:
            audio_embedding: [B, N_t, context_tokens, output_dim]
        """
        audio_cond = audio.to(device=device, dtype=dtype)

        # Split first-frame audio from latter frames
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]

        # Reshape latter frames by vae_scale
        latter_frame_audio_emb = rearrange(
            latter_frame_audio_emb,
            "b (n_t n) w s c -> b n_t n w s c", n=self._vae_scale
        )

        # Extract first/middle/last sub-windows
        middle_index = self._audio_window // 2

        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index + 1, ...]
        latter_first_frame_audio_emb = rearrange(
            latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )

        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last_frame_audio_emb = rearrange(
            latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )

        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index + 1, ...]
        latter_middle_frame_audio_emb = rearrange(
            latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
        )

        latter_frame_audio_emb_s = torch.concat(
            [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2
        )

        # Feed through AudioProjModel
        audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
        audio_embedding = audio_embedding.to(dtype)

        return audio_embedding

    # ------------------------------------------------------------------
    # I2V conditioning (same as bidirectional wrapper)
    # ------------------------------------------------------------------

    @staticmethod
    def _construct_i2v_mask(
        frame_num: int,
        lat_h: int,
        lat_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the 4-channel temporal mask for I2V conditioning.

        The mask marks frame 0 as the reference (value=1) and all subsequent
        frames as generation targets (value=0).
        """
        # Build per-frame spatial mask: first frame = 1, rest = 0
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
        msk[:, 1:] = 0

        # VAE temporal stride handling: replicate frame 0 four times
        msk = torch.cat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)

        # Reshape: [1, T_pixel+3, lat_h, lat_w] -> [1, 4, T_lat, lat_h, lat_w]
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2).to(dtype)  # [1, 4, T_lat, lat_h, lat_w]

        return msk

    def _build_y(
        self,
        condition: Dict[str, torch.Tensor],
        T: int,
        start_frame: int = 0,
    ) -> torch.Tensor:
        """Build the 20-channel I2V conditioning tensor, optionally sliced for a chunk.

        Concatenates the 4-channel temporal mask with the 16-channel
        VAE-encoded reference frame to produce ``y = [msk, first_frame_cond]``
        of shape ``[B, 20, T, H, W]``.

        In AR mode, the conditioning is sliced to [start_frame : start_frame + T].

        Args:
            condition: Dict containing ``"first_frame_cond"`` with shape
                ``[B, 16, T_full, H, W]``.
            T: Number of latent time steps for this call.
            start_frame: Starting latent frame index (0 for full sequence).

        Returns:
            y tensor of shape ``[B, 20, T, H, W]``.
        """
        first_frame_cond = condition["first_frame_cond"]  # [B, 16, T_full, H, W]
        B, _, T_full, H, W = first_frame_cond.shape

        # Pixel frame count from full latent frames
        frame_num = (T_full - 1) * 4 + 1

        msk = self._construct_i2v_mask(
            frame_num, H, W,
            device=first_frame_cond.device,
            dtype=first_frame_cond.dtype,
        )
        msk = msk.expand(B, -1, -1, -1, -1)  # [B, 4, T_full, H, W]

        # Concatenate: [B, 4, T_full, H, W] + [B, 16, T_full, H, W] -> [B, 20, T_full, H, W]
        y_full = torch.cat([msk, first_frame_cond], dim=1)

        # Slice to current chunk if needed
        if start_frame > 0 or T < T_full:
            y = y_full[:, :, start_frame:start_frame + T]
        else:
            y = y_full

        return y

    # ------------------------------------------------------------------
    # FlexAttention block mask
    # ------------------------------------------------------------------

    def _sample_attn_config(self) -> dict:
        """Sample an attention config from stochastic_attn_configs.

        Returns dict with 'local_attn_size' and 'sink_size' keys.
        Falls back to instance defaults if no stochastic configs.
        """
        if not self._stochastic_attn_configs:
            return {
                "local_attn_size": self.local_attn_size,
                "sink_size": self.sink_size,
            }

        import random
        weights = [c.get("weight", 1.0) for c in self._stochastic_attn_configs]
        chosen = random.choices(self._stochastic_attn_configs, weights=weights, k=1)[0]
        return {
            "local_attn_size": chosen.get("local_attn_size", self.local_attn_size),
            "sink_size": chosen.get("sink_size", self.sink_size),
        }

    def _build_block_mask(
        self,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        chunk_size: int = None,
        local_attn_size: int = -1,
        sink_size: int = 0,
    ) -> Optional[BlockMask]:
        """Build a chunk-wise causal attention mask for full-sequence mode.

        Tokens within the same chunk attend bidirectionally. Tokens can attend
        to all previous chunks (or a sliding window of previous frames if
        ``local_attn_size > 0``).

        Args:
            device: Device to create mask tensors on.
            num_frames: Number of video frames in the sequence.
            frame_seqlen: Number of tokens per frame.
            chunk_size: Chunk size in frames (defaults to ``self.chunk_size``).
            local_attn_size: Sliding window size in **frames**. Each chunk
                sees at most this many frames (including itself). ``-1`` means
                unlimited (full causal, same as original behaviour).
            sink_size: Number of leading frames that are always visible to
                every query token, regardless of the sliding window.
        """
        if not FLEX_ATTENTION_AVAILABLE:
            return None

        if chunk_size is None:
            chunk_size = self.chunk_size

        total_length = num_frames * frame_seqlen
        pad_len = math.ceil(total_length / 128) * 128 - total_length
        padded_length = total_length + pad_len

        ends = torch.zeros(padded_length, device=device, dtype=torch.long)
        starts = torch.zeros(padded_length, device=device, dtype=torch.long)

        # Build chunk boundaries -- front-load remainder into first chunk
        num_chunks = num_frames // chunk_size
        remaining_size = num_frames % chunk_size

        frame_counts = []
        if num_frames > 0:
            if num_chunks == 0:
                frame_counts.append(remaining_size)
            else:
                frame_counts.append(chunk_size + remaining_size)
                frame_counts.extend([chunk_size] * max(num_chunks - 1, 0))

        current_start = 0
        for frames_in_chunk in frame_counts:
            chunk_len_tokens = frames_in_chunk * frame_seqlen

            # --- sliding window lower bound (in tokens) ---
            if local_attn_size > 0:
                effective_window = local_attn_size - sink_size
                chunk_last_frame = (current_start // frame_seqlen) + frames_in_chunk
                window_start_frame = max(0, chunk_last_frame - effective_window)
                window_start_token = window_start_frame * frame_seqlen
            else:
                window_start_token = 0

            chunk_end = current_start + chunk_len_tokens
            ends[current_start: chunk_end] = chunk_end
            starts[current_start: chunk_end] = window_start_token
            current_start += chunk_len_tokens

        # Sink boundary (in tokens): first ``sink_size`` frames always visible
        sink_end = sink_size * frame_seqlen

        def attention_mask(b, h, q_idx, kv_idx):
            in_window = (kv_idx >= starts[q_idx]) & (kv_idx < ends[q_idx])
            is_sink = kv_idx < sink_end
            return in_window | is_sink | (q_idx == kv_idx)

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            _compile=False,
            device=device,
        )
        return block_mask

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def _init_caches(
        self,
        batch_size: int,
        total_tokens: int,
        frame_seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate KV caches for all transformer blocks.

        During training, always allocates the full sequence size to avoid
        in-place eviction — gradient checkpointing recomputes forward passes
        that read from cache, so the buffer must remain stable between forward
        and backward (see docs/kv-cache-eviction-gradient-checkpointing.md).

        During inference (model.eval()), uses the smaller
        ``local_attn_size * frame_seqlen`` cache with rolling eviction to
        save memory on long sequences.
        """
        n = self._num_heads
        d = self._dim // n

        if not self.training and self.local_attn_size > 0:
            cache_tokens = self.local_attn_size * frame_seqlen
        else:
            cache_tokens = total_tokens

        self._kv_caches = []
        for _ in self.blocks:
            self._kv_caches.append(
                {
                    "k": torch.zeros(batch_size, cache_tokens, n, d, device=device, dtype=dtype),
                    "v": torch.zeros(batch_size, cache_tokens, n, d, device=device, dtype=dtype),
                    "global_end_index": torch.tensor(0, device=device, dtype=torch.long),
                    "local_end_index": torch.tensor(0, device=device, dtype=torch.long),
                }
            )

    def clear_caches(self) -> None:
        """Clear all KV caches and cached audio between samples."""
        if self._kv_caches is not None:
            for cache in self._kv_caches:
                cache["k"].zero_()
                cache["v"].zero_()
                cache["global_end_index"].fill_(0)
                cache["local_end_index"].fill_(0)
        self._kv_caches = None
        self.block_mask = None
        self._cached_audio = None

    def fully_shard(self, **kwargs):
        """Apply FSDP2 sharding to transformer blocks.

        Can't shard ``self`` (ABC) or ``self.blocks`` (ModuleList has no forward).
        Shard each block and submodule individually with explicit
        ``reshard_after_forward=True`` to ensure params are freed after each
        module's forward pass. Without this, all-gathered params persist and
        cause ~40 GB memory leak per 14B model.
        """
        from torch.distributed._composable.fsdp import fully_shard

        # Force resharding after each module's forward
        kwargs["reshard_after_forward"] = True

        # Shard each transformer block independently
        for block in self.blocks:
            fully_shard(block, **kwargs)

        # Shard other submodules (embeddings, head) as leaf FSDP units
        for name, child in self.named_children():
            if name == "blocks":
                continue  # already sharded above
            if sum(p.numel() for p in child.parameters()) > 0:
                fully_shard(child, **kwargs)

    # ------------------------------------------------------------------
    # Internal forward (full-sequence mode)
    # ------------------------------------------------------------------

    def _forward_full_sequence(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Full-sequence forward -- like bidirectional but with causal block mask.

        Processes ALL frames at once. Used for KD and DF training with
        per-frame inhomogeneous timesteps, and for non-AR evaluation.

        Args:
            x: [B, 16, T, H, W] noisy latent
            timestep: [B] scalar or [B, T] per-frame timesteps (already rescaled)
            context: [B, 512, 4096] text embeddings
            clip_fea: [B, 257, 1280] CLIP features
            y: [B, 20, T, H, W] I2V conditioning
            audio: [B, num_audio_frames, audio_window, 12, 768]
            use_gradient_checkpointing: enable gradient checkpointing

        Returns:
            [B, 16, T, H, W] model output
        """
        device = self.patch_embedding.weight.device
        if self.freqs[0].device != device:
            self.freqs = tuple(f.to(device) for f in self.freqs)

        # Concatenate I2V conditioning
        if y is not None:
            x = torch.cat([x, y], dim=1)  # [B, 36, T, H, W]

        # Patch embedding
        x = self.patch_embedding(x)  # [B, dim, f, h, w]
        grid_sizes = torch.tensor(
            [list(x.shape[2:])], dtype=torch.long, device=device
        ).expand(x.shape[0], -1)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # [B, f*h*w, dim]
        seq_lens = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.long, device=device)

        # Time embedding -- supports both scalar [B] and per-frame [B, T_latent]
        if timestep.dim() == 2:
            # Inhomogeneous per-frame timesteps [B, T_latent]
            B_t, T_t = timestep.shape
            t_flat = timestep.reshape(-1)  # [B * T_latent]
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t_emb_flat = self.time_embedding(
                    sinusoidal_embedding_1d(self._freq_dim, t_flat).float()
                )  # [B * T_latent, dim]
            t_emb = t_emb_flat.reshape(B_t, T_t, -1)  # [B, T_latent, dim]
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t_mod_flat = self.time_projection(t_emb_flat)  # [B * T_latent, 6*dim]
            t_mod = t_mod_flat.reshape(B_t, T_t, 6, self._dim)  # [B, T_latent, 6, dim]
            # t_emb for head: [B, T_latent, dim] -> use first f frames
            t_emb = t_emb[:, :f, :]  # [B, F, dim]
            # t_mod for blocks: [B, T_latent, 6, dim] -> use first f frames
            t_mod = t_mod[:, :f, :, :]  # [B, F, 6, dim]
        else:
            # Scalar timestep [B]
            with torch.amp.autocast('cuda', dtype=torch.float32):
                t_emb = self.time_embedding(
                    sinusoidal_embedding_1d(self._freq_dim, timestep).float()
                )  # [B, dim]
                t_mod = self.time_projection(t_emb).unflatten(1, (6, self._dim))  # [B, 6, dim]
            # Expand to per-frame: [B, 6, dim] -> [B, F, 6, dim]
            t_mod = t_mod.unsqueeze(1).expand(-1, f, -1, -1)
            # t_emb for head: [B, dim] -> [B, F, dim]
            t_emb = t_emb.unsqueeze(1).expand(-1, f, -1)

        assert t_emb.dtype == torch.float32 and t_mod.dtype == torch.float32

        # Text embedding
        # context (text_embeds) can be [B, 1, 512, 4096] or [B, 512, 4096]
        context_lens = None
        if context.dim() == 4:
            context = context.squeeze(1)  # [B, 1, 512, 4096] -> [B, 512, 4096]
        # Pad to text_len and pass through text_embedding
        context = self.text_embedding(
            torch.stack([
                torch.cat([
                    u, u.new_zeros(self._text_len - u.size(0), u.size(1))
                ]) if u.size(0) < self._text_len else u[:self._text_len]
                for u in [context[i] for i in range(context.shape[0])]
            ])
        )

        # CLIP embedding
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        # Audio processing
        audio_embedding = self._process_audio(audio, device, x.dtype) if audio is not None else None

        # Build block mask (chunk-wise causal, optionally stochastic)
        if self.training and self._stochastic_attn_configs:
            attn_cfg = self._sample_attn_config()
            if FLEX_ATTENTION_AVAILABLE:
                frame_seqlen = h * w
                self.block_mask = self._build_block_mask(
                    device, f, frame_seqlen, self.chunk_size,
                    local_attn_size=attn_cfg["local_attn_size"],
                    sink_size=attn_cfg["sink_size"],
                )
        elif self.block_mask is None and FLEX_ATTENTION_AVAILABLE:
            frame_seqlen = h * w
            self.block_mask = self._build_block_mask(
                device, f, frame_seqlen, self.chunk_size,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size,
            )

        # Create custom forward for gradient checkpointing
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # Transformer blocks
        kwargs_block = dict(
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
            block_mask=self.block_mask,
        )

        for block_index, block in enumerate(self.blocks):
            block_kwargs = dict(
                e=t_mod,
                **kwargs_block,
            )

            if self.training and use_gradient_checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **block_kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **block_kwargs)

        # Head: t_emb is [B, dim], need [B, F, 1, dim]
        head_e = t_emb.unsqueeze(2)  # [B, F, 1, dim]
        x = self.head(x, head_e)

        # Unpatchify
        out = self._unpatchify(x, grid_sizes)
        return torch.stack(out).float()

    # ------------------------------------------------------------------
    # Internal forward (AR / chunk-based mode)
    # ------------------------------------------------------------------

    def _forward_ar(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        current_start: int = 0,
        store_kv: bool = True,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Chunk-based autoregressive forward with KV cache.

        Processes a chunk of frames while attending to all previously cached
        frames via the KV cache.

        Args:
            x: [B, 16, chunk_frames, H, W] noisy latent chunk
            timestep: [B] scalar timestep (already rescaled)
            context: [B, 512, 4096] text embeddings (full)
            clip_fea: [B, 257, 1280] CLIP features
            y: [B, 20, chunk_frames, H, W] I2V conditioning for this chunk
            audio: [B, num_audio_frames, audio_window, 12, 768] FULL audio
            current_start: token offset (= frame_offset * h * w)
            store_kv: whether to write to KV cache
            use_gradient_checkpointing: enable gradient checkpointing

        Returns:
            [B, 16, chunk_frames, H, W] model output for this chunk
        """
        device = self.patch_embedding.weight.device
        if self.freqs[0].device != device:
            self.freqs = tuple(f.to(device) for f in self.freqs)

        # Concatenate I2V conditioning
        if y is not None:
            x = torch.cat([x, y], dim=1)  # [B, 36, chunk_frames, H, W]

        # Patch embedding
        x = self.patch_embedding(x)  # [B, dim, f, h, w]
        grid_sizes = torch.tensor(
            [list(x.shape[2:])], dtype=torch.long, device=device
        ).expand(x.shape[0], -1)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # [B, f*h*w, dim]
        seq_lens = torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.long, device=device)

        # Allocate caches on first call
        frame_seqlen = h * w
        total_tokens = self.total_num_frames * frame_seqlen
        if self._kv_caches is None:
            self._init_caches(x.shape[0], total_tokens, frame_seqlen, device, x.dtype)

        # Time embedding (scalar timestep [B])
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t_emb = self.time_embedding(
                sinusoidal_embedding_1d(self._freq_dim, timestep).float()
            )  # [B, dim]
            t_mod = self.time_projection(t_emb).unflatten(1, (6, self._dim))  # [B, 6, dim]
        # Expand to per-frame: [B, 6, dim] -> [B, F, 6, dim]
        t_mod = t_mod.unsqueeze(1).expand(-1, f, -1, -1)

        assert t_emb.dtype == torch.float32 and t_mod.dtype == torch.float32

        # Text embedding
        # context (text_embeds) can be [B, 1, 512, 4096] or [B, 512, 4096]
        context_lens = None
        if context.dim() == 4:
            context = context.squeeze(1)  # [B, 1, 512, 4096] -> [B, 512, 4096]
        # Pad to text_len and pass through text_embedding
        context = self.text_embedding(
            torch.stack([
                torch.cat([
                    u, u.new_zeros(self._text_len - u.size(0), u.size(1))
                ]) if u.size(0) < self._text_len else u[:self._text_len]
                for u in [context[i] for i in range(context.shape[0])]
            ])
        )

        # CLIP embedding
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        # Audio processing: during training, compute fresh each call so gradients
        # flow correctly through AudioProjModel (SF exit steps need grad).
        # During inference, cache to avoid redundant computation across chunks.
        if self.training:
            full_audio = self._process_audio(audio, device, x.dtype) if audio is not None else None
        else:
            if self._cached_audio is None:
                self._cached_audio = self._process_audio(audio, device, x.dtype) if audio is not None else None
            full_audio = self._cached_audio

        # Slice audio to current chunk's frames
        audio_embedding = full_audio
        if audio_embedding is not None:
            current_frame_start = current_start // frame_seqlen
            current_frame_end = current_frame_start + f
            # audio_embedding is [B, N_t, context_tokens, output_dim]
            # SingleStreamAttention expects encoder_hidden_states shaped [B*N_t, N_a, C_audio]
            # But we need to pass only the current chunk's frames
            # The reshaping to per-frame happens inside SingleStreamAttention via grid_sizes
            # So we need audio_embedding for just the current chunk's N_t frames
            audio_embedding = audio_embedding[:, current_frame_start:current_frame_end]

        # Transformer blocks
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            # Compute cache read position (deterministic, gradient-checkpointing safe)
            kv_cache_i = self._kv_caches[block_index]
            # During training, cache is full-size so local == global (no eviction).
            # During inference with sliding window, cache is smaller and eviction
            # shifts data, so local != global. Use None to read from cache metadata.
            if self.training:
                cache_local_end = current_start  # no eviction: local == global
            else:
                cache_local_end = None  # let attention read from cache metadata

            kwargs = dict(
                e=t_mod,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                audio_embedding=audio_embedding,
                block_mask=None,  # No FlexAttention in AR mode
                kv_cache=kv_cache_i,
                current_start=current_start,
                store_kv=store_kv,
                cache_local_end_override=cache_local_end,
            )

            if self.training and use_gradient_checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # Head: t_emb is [B, dim], need [B, F, 1, dim]
        head_e = t_emb.unsqueeze(1).unsqueeze(2).expand(-1, f, -1, -1)
        x = self.head(x, head_e)

        # Unpatchify
        out = self._unpatchify(x, grid_sizes)
        return torch.stack(out).float()

    # ------------------------------------------------------------------
    # Forward (FastGen CausalFastGenNetwork interface)
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = True,
        use_gradient_checkpointing: Optional[bool] = None,
        apply_anchor: bool = True,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass -- dispatches to full-sequence or AR mode.

        Auto-dispatch logic:
          - If ``t.dim() == 2`` (per-frame timesteps from sample_t_inhom):
            always use full-sequence mode (needed for KD/DF training).
          - Else if ``is_ar=True``: use AR mode with KV cache.
          - Else: use full-sequence mode.

        Args:
            x_t: Noisy latent [B, 16, T, H, W].
            t: Timestep in [0, 1) range, shape [B] or [B, num_frames].
            condition: Dict with keys:
                - ``text_embeds``:      [B, 512, 4096] or [B, 1, 512, 4096]
                - ``first_frame_cond``: [B, 16, T, H, W]
                - ``clip_features``:    [B, 257, 1280]
                - ``audio_emb``:        [B, num_audio_frames, audio_window, 12, 768]
            cur_start_frame: Frame offset for AR generation.
            store_kv: Whether to update KV cache.
            is_ar: If True, use AR mode with KV cache; if False, full-sequence.
            use_gradient_checkpointing: Enable gradient checkpointing.
            apply_anchor: If True (default), apply the frame-0 anchor overwrite
                when cur_start_frame==0 and network anchor mode allows. If False,
                bypass the anchor entirely (used by model-generated-sink-cache).
            **fwd_kwargs: Additional kwargs.

        Returns:
            Model output tensor.
        """
        if feature_indices is None:
            feature_indices = set()

        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        elif fwd_pred_type not in NET_PRED_TYPES:
            raise ValueError(
                f"Unsupported fwd_pred_type '{fwd_pred_type}'. Supported: {NET_PRED_TYPES}"
            )

        # Gradient checkpointing: default to constructor setting
        if use_gradient_checkpointing is None:
            use_gradient_checkpointing = self._use_gradient_checkpointing

        # Unpack condition
        assert isinstance(condition, dict), f"condition must be a dict, got {type(condition)}"
        text_embeds = condition["text_embeds"]        # [B, 512, 4096] or [B, 1, 512, 4096]
        clip_features = condition["clip_features"]    # [B, 257, 1280]
        audio_emb = condition["audio_emb"]            # [B, num_frames, audio_window, 12, 768]

        # Handle text_embeds shape: [B, 1, 512, 4096] -> [B, 512, 4096]
        if text_embeds.dim() == 4:
            text_embeds = text_embeds.squeeze(1)

        # Handle clip_features shape: [B, 1, 257, 1280] -> [B, 257, 1280]
        if clip_features.dim() == 4:
            clip_features = clip_features.squeeze(1)

        # Apply 5-frame sliding window to audio if not already windowed.
        # Dataset provides [B, num_video_frames, 12, 768] (raw per-frame).
        # Model expects [B, num_video_frames, 5, 12, 768] (windowed).
        if audio_emb is not None and audio_emb.dim() == 4:
            # audio_emb: [B, num_frames, 12, 768] -> apply 5-frame window
            num_frames = audio_emb.shape[1]
            half_win = self._audio_window // 2  # 2 for window=5
            indices = torch.arange(self._audio_window, device=audio_emb.device) - half_win  # [-2,-1,0,1,2]
            center_indices = torch.arange(num_frames, device=audio_emb.device).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = center_indices.clamp(0, num_frames - 1)  # [num_frames, 5]
            audio_emb = audio_emb[:, center_indices]  # [B, num_frames, 5, 12, 768]

        B, C, T, H, W = x_t.shape

        # Build I2V conditioning y-tensor (sliced for chunk in AR mode)
        y = self._build_y(condition, T, start_frame=cur_start_frame)

        # Rescale timestep: FastGen uses t in [0, 1), model expects t * 1000
        timestep = self.noise_scheduler.rescale_t(t)

        # Compute token offset from frame offset
        p_h, p_w = self._patch_size[1], self._patch_size[2]
        h_patches, w_patches = H // p_h, W // p_w
        current_start = cur_start_frame * h_patches * w_patches

        # Auto-detect inhomogeneous per-frame timesteps -> full-sequence mode
        if timestep.dim() == 2:
            model_output = self._forward_full_sequence(
                x=x_t,
                timestep=timestep,
                context=text_embeds,
                clip_fea=clip_features,
                y=y,
                audio=audio_emb,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        elif is_ar:
            model_output = self._forward_ar(
                x=x_t,
                timestep=timestep,
                context=text_embeds,
                clip_fea=clip_features,
                y=y,
                audio=audio_emb,
                current_start=current_start,
                store_kv=store_kv,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
        else:
            model_output = self._forward_full_sequence(
                x=x_t,
                timestep=timestep,
                context=text_embeds,
                clip_fea=clip_features,
                y=y,
                audio=audio_emb,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

        # Convert prediction type
        # For inhomogeneous t [B, T_latent], expand to match model output [B, C, T, H, W]
        t_for_convert = t
        if t.dim() == 2:
            t_for_convert = t[:, None, :, None, None]  # [B, 1, T, 1, 1]
        out = self.noise_scheduler.convert_model_output(
            x_t, model_output, t_for_convert,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

        # Hard-anchor frame 0 to clean reference. Behavior controlled by the
        # apply_anchor kwarg and net-module attributes; see
        # _maybe_apply_first_frame_anchor for full semantics.
        out = _maybe_apply_first_frame_anchor(
            out, self, cur_start_frame, condition, apply_anchor=apply_anchor,
        )

        # Feature extraction -- return expected structure for DMD2 compatibility
        if return_features_early:
            return []

        if feature_indices is not None and len(feature_indices) > 0:
            if return_logvar:
                logvar = torch.zeros(out.shape[0], 1, device=out.device, dtype=out.dtype)
                return [out, []], logvar
            return [out, []]

        if return_logvar:
            logvar = torch.zeros(out.shape[0], 1, device=out.device, dtype=out.dtype)
            return out, logvar

        return out

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        """Multi-stage weight loading pipeline.

        Stage 1: Load base Wan I2V-14B safetensor shards.
                 Keys match our internal naming directly.

        Stage 2: Load InfiniteTalk checkpoint (audio modules + weight overrides).

        Stage 3: Merge external LoRA weights into base model.

        Stage 4: Apply runtime LoRA adapters and freeze base weights.
                 (Always done for student -- student is always LoRA-trainable.)

        Stage 5: Reinitialize RoPE frequencies.
        """
        # --- Stage 1: Base Wan I2V safetensor shards ---
        if self.base_model_paths:
            paths = [p.strip() for p in self.base_model_paths.split(",") if p.strip()]
            logger.info(
                f"[CausalInfiniteTalkWan] Loading base Wan I2V from {len(paths)} shard(s)"
            )

            base_sd: Dict[str, torch.Tensor] = {}
            for p in paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Base model shard not found: {p}")
                shard = safetensors_load_file(p, device="cpu")
                base_sd.update(shard)
                logger.info(
                    f"[CausalInfiniteTalkWan]   Loaded shard: {os.path.basename(p)} "
                    f"({len(shard)} tensors)"
                )

            logger.info(
                f"[CausalInfiniteTalkWan] Total base tensors: {len(base_sd)}"
            )

            # Handle patch_embedding expansion: base has in_dim=16, we have in_dim=36
            pe_key = "patch_embedding.weight"
            if pe_key in base_sd:
                model_pe = self.patch_embedding.weight
                if base_sd[pe_key].shape != model_pe.shape:
                    old_shape = base_sd[pe_key].shape
                    logger.info(
                        f"[CausalInfiniteTalkWan] Expanding patch_embedding: "
                        f"{list(old_shape)} -> {list(model_pe.shape)}"
                    )
                    new_pe = torch.zeros_like(model_pe.data)
                    slices = tuple(slice(0, s) for s in old_shape)
                    new_pe[slices] = base_sd[pe_key]
                    base_sd[pe_key] = new_pe

            missing, unexpected = self.load_state_dict(base_sd, strict=False)
            loaded = len(base_sd) - len(unexpected)
            logger.info(
                f"[CausalInfiniteTalkWan] Base weights: {loaded} loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

            if missing:
                audio_missing = [k for k in missing if "audio" in k.lower()]
                other_missing = [k for k in missing if "audio" not in k.lower()]
                if audio_missing:
                    logger.info(
                        f"[CausalInfiniteTalkWan] Expected missing (audio): "
                        f"{len(audio_missing)} keys"
                    )
                if other_missing:
                    logger.warning(
                        f"[CausalInfiniteTalkWan] Unexpected missing keys: "
                        f"{other_missing[:10]}{'...' if len(other_missing) > 10 else ''}"
                    )
            if unexpected:
                logger.warning(
                    f"[CausalInfiniteTalkWan] Unexpected keys in base shards: "
                    f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
                )

            del base_sd
        else:
            logger.info(
                "[CausalInfiniteTalkWan] No base_model_paths provided, using random init"
            )

        # --- Stage 2: InfiniteTalk checkpoint ---
        if self.infinitetalk_ckpt_path:
            if not os.path.isfile(self.infinitetalk_ckpt_path):
                raise FileNotFoundError(
                    f"InfiniteTalk checkpoint not found: {self.infinitetalk_ckpt_path}"
                )

            logger.info(
                f"[CausalInfiniteTalkWan] Loading InfiniteTalk checkpoint: "
                f"{self.infinitetalk_ckpt_path}"
            )
            it_sd = safetensors_load_file(self.infinitetalk_ckpt_path, device="cpu")
            logger.info(
                f"[CausalInfiniteTalkWan] InfiniteTalk checkpoint: {len(it_sd)} tensors"
            )

            # Handle patch_embedding expansion in InfiniteTalk ckpt too
            pe_key = "patch_embedding.weight"
            if pe_key in it_sd:
                model_pe = self.patch_embedding.weight
                if it_sd[pe_key].shape != model_pe.shape:
                    old_shape = it_sd[pe_key].shape
                    logger.info(
                        f"[CausalInfiniteTalkWan] Expanding IT patch_embedding: "
                        f"{list(old_shape)} -> {list(model_pe.shape)}"
                    )
                    new_pe = torch.zeros_like(model_pe.data)
                    slices = tuple(slice(0, s) for s in old_shape)
                    new_pe[slices] = it_sd[pe_key]
                    it_sd[pe_key] = new_pe

            missing, unexpected = self.load_state_dict(it_sd, strict=False)
            loaded = len(it_sd) - len(unexpected)
            logger.info(
                f"[CausalInfiniteTalkWan] InfiniteTalk weights: {loaded} loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

            del it_sd
        else:
            logger.info(
                "[CausalInfiniteTalkWan] No infinitetalk_ckpt_path provided, "
                "audio modules use random init"
            )

        # --- Stage 3: Merge external LoRA into base weights ---
        if self.lora_ckpt_path:
            if not os.path.isfile(self.lora_ckpt_path):
                raise FileNotFoundError(
                    f"LoRA checkpoint not found: {self.lora_ckpt_path}"
                )

            logger.info(
                f"[CausalInfiniteTalkWan] Merging LoRA from: {self.lora_ckpt_path}"
            )
            lora_sd = safetensors_load_file(self.lora_ckpt_path, device="cpu")
            applied = merge_lora_from_file(
                self,
                lora_sd,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                prefix="diffusion_model.",
            )
            logger.info(
                f"[CausalInfiniteTalkWan] LoRA merge complete: {applied} updates applied"
            )
            del lora_sd

        # --- Stage 4: Apply runtime LoRA adapters + freeze base ---
        # Student always gets LoRA adapters (it's always trainable)
        logger.info(
            f"[CausalInfiniteTalkWan] Applying runtime LoRA adapters "
            f"(rank={self.lora_rank}, alpha={self.lora_alpha})"
        )
        apply_lora(self, rank=self.lora_rank, alpha=self.lora_alpha)
        freeze_base(self)

        # --- Stage 5: Reinitialize RoPE frequencies ---
        head_dim = self._dim // self._num_heads
        self.freqs = _precompute_freqs_cis_3d(head_dim)
