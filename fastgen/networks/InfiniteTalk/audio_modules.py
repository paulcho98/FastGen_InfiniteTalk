"""
Audio modules ported from InfiniteTalk for FastGen integration.

Source files:
  - InfiniteTalk/wan/modules/multitalk_model.py  (AudioProjModel, lines 366-429)
  - InfiniteTalk/wan/modules/attention.py         (SingleStreamAttention)

Changes from source:
  - Removed ModelMixin/ConfigMixin inheritance and @register_to_config decorator
  - Replaced xformers.ops.memory_efficient_attention with torch SDPA
  - Removed sequence-parallel (xfuser) codepaths
  - Stripped multi-speaker logic (lives in SingleStreamMutiAttention, not ported)
"""

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from einops import rearrange


# ---------------------------------------------------------------------------
# AudioProjModel — projects per-frame Whisper embeddings into cross-attention
# context tokens for the DiT.
#
# Class defaults: seq_len=5, seq_len_vf=12
# Runtime overrides from WanModel.__init__:
#   seq_len_vf = audio_window + vae_scale - 1 = 5 + 4 - 1 = 8
#
# Actual runtime linear sizes:
#   proj1:    Linear(5 * 12 * 768 = 46080,  512)
#   proj1_vf: Linear(8 * 12 * 768 = 73728,  512)   (uses runtime seq_len_vf=8)
#   proj2:    Linear(512, 512)
#   proj3:    Linear(512, 32 * 768 = 24576)
# ---------------------------------------------------------------------------


class AudioProjModel(nn.Module):
    def __init__(
        self,
        seq_len=5,
        seq_len_vf=12,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        norm_output_audio=False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c * N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(
            batch_size_c * N_t, self.context_tokens, self.output_dim
        )

        # normalization and reshape
        with torch.amp.autocast('cuda', dtype=torch.float32):
            context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


# ---------------------------------------------------------------------------
# SingleStreamAttention — per-frame cross-attention from visual tokens (Q)
# to audio context tokens (KV).
#
# Forward flow:
#   1. Reshape visual sequence to per-frame: (B, N_t*S, C) -> (B*N_t, S, C)
#   2. Q from visual, K/V from audio encoder_hidden_states
#   3. Cross-attention via SDPA
#   4. Reshape back: (B*N_t, S, C) -> (B, N_t*S, C)
# ---------------------------------------------------------------------------


class SingleStreamAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: nn.Module,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, eps=eps) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)

        self.add_q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.add_k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shape=None,
    ) -> torch.Tensor:
        """
        Args:
            x: Visual hidden states, shape (B, N_t * S, C)
            encoder_hidden_states: Audio context tokens, shape (B * N_t, N_a, C_audio)
            shape: Tuple (N_t, N_h, N_w) — temporal and spatial dims
        Returns:
            Output tensor, same shape as x: (B, N_t * S, C)
        """
        N_t, N_h, N_w = shape

        # Reshape to per-frame: (B, N_t*S, C) -> (B*N_t, S, C)
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        # Q from visual hidden states
        B, N, C = x.shape
        q = self.q_linear(x)
        q_shape = (B, N, self.num_heads, self.head_dim)
        q = q.view(q_shape).permute(0, 2, 1, 3)  # (B, H, N, D)

        if self.qk_norm:
            q = self.q_norm(q)

        # KV from audio encoder_hidden_states
        # Handle both [B*N_t, N_a, C] (pre-reshaped) and [B, N_t, N_a, C] (from AudioProjModel)
        if encoder_hidden_states.dim() == 4:
            encoder_hidden_states = rearrange(
                encoder_hidden_states, "b n_t n_a c -> (b n_t) n_a c")
        _, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv_shape = (B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.view(encoder_kv_shape).permute(2, 0, 3, 1, 4)
        encoder_k, encoder_v = encoder_kv.unbind(0)  # each (B, H, N_a, D)

        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        # Cross-attention via SDPA
        # q:         (B, H, N,   D)
        # encoder_k: (B, H, N_a, D)
        # encoder_v: (B, H, N_a, D)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, encoder_k, encoder_v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        # x: (B, H, N, D)

        # Linear transform
        x_output_shape = (B, N, C)
        x = x.transpose(1, 2)  # (B, N, H, D)
        x = x.reshape(x_output_shape)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back: (B*N_t, S, C) -> (B, N_t*S, C)
        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x
