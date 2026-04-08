# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Ported for FastGen: stripped TeaCache, VRAM management, quantization, xfuser,
# sageattn, multi-speaker logic, diffusers ModelMixin/ConfigMixin.
# Added: feature_indices, return_features_early, gradient checkpointing.
import math
from typing import List, Optional, Set

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .audio_modules import AudioProjModel, SingleStreamAttention

# ---------------------------------------------------------------------------
# Flash attention with SDPA fallback
# ---------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

__all__ = ['WanModel']


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim, position):
    """Sinusoidal positional embedding (always computed in float64 for precision)."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    """Pre-compute complex-valued RoPE frequencies."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """Apply 3D RoPE (temporal + spatial H + spatial W) to queries/keys."""
    s, n, c = x.size(1), x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)
        freqs_i = freqs_i.to(device=x_i.device)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).float()


# ---------------------------------------------------------------------------
# Attention helper (flash_attn with SDPA fallback)
# ---------------------------------------------------------------------------

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """
    Attention with flash_attn varlen backend; falls back to F.scaled_dot_product_attention.

    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    q_lens / k_lens: [B] actual lengths (optional).
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if FLASH_ATTN_AVAILABLE:
        # --- flash_attn varlen path ---
        from flash_attn import flash_attn_varlen_func

        # preprocess query
        if q_lens is None:
            q_flat = half(q.flatten(0, 1))
            q_lens_t = torch.tensor(
                [lq] * b, dtype=torch.int32).to(
                    device=q.device, non_blocking=True)
        else:
            q_flat = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
            q_lens_t = q_lens

        # preprocess key, value
        if k_lens is None:
            k_flat = half(k.flatten(0, 1))
            v_flat = half(v.flatten(0, 1))
            k_lens_t = torch.tensor(
                [lk] * b, dtype=torch.int32).to(
                    device=k.device, non_blocking=True)
        else:
            k_flat = half(torch.cat([u[:vv] for u, vv in zip(k, k_lens)]))
            v_flat = half(torch.cat([u[:vv] for u, vv in zip(v, k_lens)]))
            k_lens_t = k_lens

        q_flat = q_flat.to(v_flat.dtype)
        k_flat = k_flat.to(v_flat.dtype)

        if q_scale is not None:
            q_flat = q_flat * q_scale

        x = flash_attn_varlen_func(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=torch.cat([q_lens_t.new_zeros([1]), q_lens_t]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens_t.new_zeros([1]), k_lens_t]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

        return x.type(out_dtype)
    else:
        # --- SDPA fallback (no varlen masking) ---
        q_sdpa = half(q).transpose(1, 2)   # [B, N, Lq, C]
        k_sdpa = half(k).transpose(1, 2)
        v_sdpa = half(v).transpose(1, 2)

        if q_scale is not None:
            q_sdpa = q_sdpa * q_scale

        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        out = out.transpose(1, 2).contiguous()  # [B, Lq, N, C]
        return out.type(out_dtype)


# ---------------------------------------------------------------------------
# Norm layers
# ---------------------------------------------------------------------------

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """x: [B, L, C]"""
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps
        ).to(origin_dtype)
        return out


# ---------------------------------------------------------------------------
# Attention sub-modules
# ---------------------------------------------------------------------------

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        ).type_as(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """Cross-attention with separate K/V projections for CLIP image tokens vs text tokens."""

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 output_dim=768,
                 norm_input_visual=True):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # audio cross-attention (single-speaker only)
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
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        audio_embedding=None,
    ):
        dtype = x.dtype
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.to(e.device) + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x), seq_lens, grid_sizes,
            freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2]

        x = x.to(dtype)

        # cross-attention of text
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # cross-attention of audio
        x_a = self.audio_cross_attn(
            self.norm_x(x), encoder_hidden_states=audio_embedding,
            shape=grid_sizes[0])
        x = x + x_a

        # FFN with AdaLN modulation
        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype))
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[5]

        x = x.to(dtype)
        return x


# ---------------------------------------------------------------------------
# Output head
# ---------------------------------------------------------------------------

class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        """
        Args:
            x: [B, L1, C]
            e: [B, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.to(e.device) + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


# ---------------------------------------------------------------------------
# CLIP projection
# ---------------------------------------------------------------------------

class MLPProj(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim),
            nn.GELU(), nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        return self.proj(image_embeds)


# ---------------------------------------------------------------------------
# WanModel — 14B DiT backbone for InfiniteTalk (FastGen-compatible)
# ---------------------------------------------------------------------------

class WanModel(nn.Module):
    """
    Wan diffusion backbone for audio-driven talking-head generation (I2V).

    Ported from InfiniteTalk's multitalk_model.py with FastGen extensions:
      - feature_indices / return_features_early for GAN discriminator
      - gradient checkpointing support
      - No TeaCache, no multi-speaker, no VRAM management, pure nn.Module
    """

    def __init__(self,
                 model_type='i2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # audio params
                 audio_window=5,
                 intermediate_dim=512,
                 output_dim=768,
                 context_tokens=32,
                 vae_scale=4,
                 norm_input_visual=True,
                 norm_output_audio=True,
                 # FastGen params
                 use_gradient_checkpointing=False,
                 weight_init=True):
        super().__init__()

        assert model_type == 'i2v', 'InfiniteTalk model requires model_type == "i2v".'
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # transformer blocks
        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              output_dim=output_dim, norm_input_visual=norm_input_visual)
            for _ in range(num_layers)
        ])

        # output head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE frequencies (pre-computed, stored as buffer-like attribute)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # CLIP image embedding projection
        self.img_emb = MLPProj(1280, dim)

        # audio adapter
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        # initialize weights
        if weight_init:
            self.init_weights()

    def init_freqs(self):
        """Re-compute RoPE frequencies (e.g. after loading from checkpoint)."""
        d = self.dim // self.num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

    def init_weights(self):
        """Initialize model parameters using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # output layer
        nn.init.zeros_(self.head.head.weight)

    # -------------------------------------------------------------------
    # Unpatchify helpers
    # -------------------------------------------------------------------

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x: Batched tensor [B, seq_len, C_out * prod(patch_size)] (padded)
            grid_sizes: [B, 3] — (F_patches, H_patches, W_patches) per sample

        Returns:
            List[Tensor]: each with shape [C_out, F, H, W]
        """
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def _unpatchify_features(self, features: List[torch.Tensor], grid_sizes) -> List[torch.Tensor]:
        """
        Convert intermediate block features from [B, N, dim] back to [B, dim, T, H, W].

        Used by GAN discriminator for feature extraction.
        """
        results = []
        p_t, p_h, p_w = self.patch_size
        # Use first sample's grid sizes (assumes batch is homogeneous)
        f, h, w = grid_sizes[0].tolist()
        for feat in features:
            # feat is [B, padded_seq, dim]; truncate to actual tokens
            feat = feat[:, :f * h * w, :]
            feat = rearrange(
                feat, 'b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)',
                f=f, h=h, w=w, pt=p_t, ph=p_h, pw=p_w
            )
            results.append(feat)
        return results

    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------

    def forward(
        self,
        x,              # List[Tensor]: noise latents, each [C_in, T, H, W]
        t,              # Timestep tensor [B]
        context,        # Text embeddings: list of [seq_len, 4096] tensors
        seq_len,        # Max sequence length for padding
        clip_fea=None,  # CLIP features [B, 257, 1280]
        y=None,         # Condition: list of [C_cond, T, H, W] tensors (mask+VAE ref)
        audio=None,     # Audio: [B, num_frames, audio_window, 12, 768]
        use_gradient_checkpointing=None,  # Override constructor setting
        feature_indices: Optional[Set[int]] = None,  # Block indices for feature extraction
        return_features_early: bool = False,  # Return features immediately after last index
        **kwargs,
    ):
        assert clip_fea is not None and y is not None

        # Resolve gradient checkpointing
        grad_ckpt = use_gradient_checkpointing if use_gradient_checkpointing is not None else self.use_gradient_checkpointing

        _, T, H, W = x[0].shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        # Concatenate condition channels
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x[0] = x[0].to(context[0].dtype)

        # Patch embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # Time embeddings (float32 precision)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # CLIP embedding
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        # -----------------------------------------------------------------
        # Audio preprocessing (lines 654-668 of original)
        # Split first-frame audio, reshape latter frames by vae_scale,
        # extract first/middle/last sub-windows, feed to AudioProjModel.
        # -----------------------------------------------------------------
        audio_cond = audio.to(device=x.device, dtype=x.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        latter_frame_audio_emb = rearrange(
            latter_frame_audio_emb,
            "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index + 1, ...]
        latter_first_frame_audio_emb = rearrange(
            latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last_frame_audio_emb = rearrange(
            latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index + 1, ...]
        latter_middle_frame_audio_emb = rearrange(
            latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio_emb_s = torch.concat(
            [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2)
        audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
        # Single-speaker: human_num is always 1, just squeeze the batch-of-speakers dim
        audio_embedding = audio_embedding.to(x.dtype)

        # -----------------------------------------------------------------
        # Transformer blocks
        # -----------------------------------------------------------------
        kwargs_block = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            audio_embedding=audio_embedding,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        features = []
        for layer_i, block in enumerate(self.blocks):
            if self.training and grad_ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs_block,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs_block)

            # Feature extraction at requested block indices
            if feature_indices and layer_i in feature_indices:
                features.append(x)

            # Early exit: return features without running head
            if return_features_early and feature_indices and len(features) == len(feature_indices):
                return self._unpatchify_features(features, grid_sizes)

        # Output head
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)

        if features:
            return torch.stack(x).float(), self._unpatchify_features(features, grid_sizes)
        return torch.stack(x).float()
