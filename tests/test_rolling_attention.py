"""Unit tests for CausalSelfAttention with rolling attention window."""

import math
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for flash_attn")

from fastgen.networks.InfiniteTalk.network_causal import (
    CausalSelfAttention,
    causal_rope_apply,
)
from fastgen.networks.InfiniteTalk.wan_model import rope_params

DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _make_freqs(head_dim):
    """Build 3D RoPE freq tables matching the model's init."""
    d = head_dim
    return tuple(
        rope_params(1024, size)
        for size in [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    )


def _make_kv_cache(batch, total_tokens, num_heads, head_dim):
    """Allocate a KV cache buffer on CUDA."""
    return {
        "k": torch.zeros(batch, total_tokens, num_heads, head_dim, device=DEVICE, dtype=DTYPE),
        "v": torch.zeros(batch, total_tokens, num_heads, head_dim, device=DEVICE, dtype=DTYPE),
        "global_end_index": torch.tensor(0, device=DEVICE, dtype=torch.long),
        "local_end_index": torch.tensor(0, device=DEVICE, dtype=torch.long),
    }


@pytest.fixture
def attn_config():
    """Small model config for fast tests."""
    return dict(dim=128, num_heads=4, eps=1e-6)


class TestDefaultBehaviorUnchanged:
    """With default params (local_attn_size=-1), behavior matches the old code."""

    def test_full_cache_attention(self, attn_config):
        """All cached tokens are attended to when local_attn_size=-1."""
        attn = CausalSelfAttention(**attn_config).to(DEVICE, dtype=DTYPE).eval()
        B, F, H, W = 1, 6, 4, 4  # 6 frames, 16 tokens each
        frame_seqlen = H * W
        total_tokens = F * frame_seqlen
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(head_dim)
        cache = _make_kv_cache(B, total_tokens, attn_config["num_heads"], head_dim)

        # Feed 3 chunks of 2 frames each
        for chunk_idx in range(3):
            start = chunk_idx * 2 * frame_seqlen
            x = torch.randn(B, 2 * frame_seqlen, attn_config["dim"], device=DEVICE, dtype=DTYPE)
            grid = torch.tensor([[2, H, W]], device=DEVICE)
            seq_lens = torch.tensor([2 * frame_seqlen], device=DEVICE)
            with torch.no_grad():
                attn(x, seq_lens, grid, freqs,
                     kv_cache=cache, current_start=start, store_kv=True)

        # After 3 chunks, cache should hold all 6 frames
        assert cache["global_end_index"].item() == total_tokens
        assert cache["local_end_index"].item() == total_tokens


class TestRollingWindow:
    """With local_attn_size > 0, attention window is limited."""

    def test_no_physical_eviction_with_large_buffer(self, attn_config):
        """When cache buffer >= total tokens, eviction never triggers."""
        attn = CausalSelfAttention(
            **attn_config, local_attn_size=2,  # attend to last 2 frames only
        ).to(DEVICE, dtype=DTYPE).eval()
        B, F, H, W = 1, 6, 4, 4
        frame_seqlen = H * W
        total_tokens = F * frame_seqlen
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(head_dim)
        # Full-sequence buffer (larger than window)
        cache = _make_kv_cache(B, total_tokens, attn_config["num_heads"], head_dim)

        for chunk_idx in range(3):
            start = chunk_idx * 2 * frame_seqlen
            x = torch.randn(B, 2 * frame_seqlen, attn_config["dim"], device=DEVICE, dtype=DTYPE)
            grid = torch.tensor([[2, H, W]], device=DEVICE)
            seq_lens = torch.tensor([2 * frame_seqlen], device=DEVICE)
            with torch.no_grad():
                attn(x, seq_lens, grid, freqs,
                     kv_cache=cache, current_start=start, store_kv=True)

        # All tokens stored (no eviction happened)
        assert cache["local_end_index"].item() == total_tokens
        # Old tokens are still in buffer (non-zero)
        assert cache["k"][:, :frame_seqlen].abs().sum() > 0, \
            "First frame should still be in buffer (no physical eviction)"

    def test_physical_eviction_with_small_buffer(self, attn_config):
        """When cache buffer < total tokens, eviction triggers."""
        local_attn = 4  # 4 frames window
        attn = CausalSelfAttention(
            **attn_config, local_attn_size=local_attn,
        ).to(DEVICE, dtype=DTYPE).eval()
        B, F, H, W = 1, 6, 4, 4
        frame_seqlen = H * W
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(head_dim)
        # Small buffer: only holds local_attn_size frames
        small_cache_tokens = local_attn * frame_seqlen
        cache = _make_kv_cache(B, small_cache_tokens, attn_config["num_heads"], head_dim)

        for chunk_idx in range(3):
            start = chunk_idx * 2 * frame_seqlen
            x = torch.randn(B, 2 * frame_seqlen, attn_config["dim"], device=DEVICE, dtype=DTYPE)
            grid = torch.tensor([[2, H, W]], device=DEVICE)
            seq_lens = torch.tensor([2 * frame_seqlen], device=DEVICE)
            with torch.no_grad():
                attn(x, seq_lens, grid, freqs,
                     kv_cache=cache, current_start=start, store_kv=True)

        # global_end tracks all 6 frames, but local_end fits in the small buffer
        assert cache["global_end_index"].item() == 6 * frame_seqlen
        assert cache["local_end_index"].item() <= small_cache_tokens

    def test_store_kv_false_does_not_write(self, attn_config):
        """store_kv=False reads the cache but does not modify it."""
        attn = CausalSelfAttention(**attn_config).to(DEVICE, dtype=DTYPE).eval()
        B, H, W = 1, 4, 4
        frame_seqlen = H * W
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(head_dim)
        cache = _make_kv_cache(B, 6 * frame_seqlen, attn_config["num_heads"], head_dim)

        # Write chunk 0
        x0 = torch.randn(B, 2 * frame_seqlen, attn_config["dim"], device=DEVICE, dtype=DTYPE)
        grid = torch.tensor([[2, H, W]], device=DEVICE)
        seq_lens = torch.tensor([2 * frame_seqlen], device=DEVICE)
        with torch.no_grad():
            attn(x0, seq_lens, grid, freqs,
                 kv_cache=cache, current_start=0, store_kv=True)

        saved_end = cache["global_end_index"].item()
        saved_k = cache["k"].clone()

        # Read-only forward (store_kv=False) at same position
        x1 = torch.randn(B, 2 * frame_seqlen, attn_config["dim"], device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            attn(x1, seq_lens, grid, freqs,
                 kv_cache=cache, current_start=0, store_kv=False)

        assert cache["global_end_index"].item() == saved_end, \
            "store_kv=False should not update metadata"
        assert torch.equal(cache["k"], saved_k), \
            "store_kv=False should not modify cache contents"


class TestDynamicRoPE:
    """With use_dynamic_rope=True, raw keys are cached and RoPE applied at attention time."""

    def test_dynamic_rope_runs_without_error(self, attn_config):
        """Dynamic RoPE mode completes without error."""
        attn = CausalSelfAttention(
            **attn_config, use_dynamic_rope=True, local_attn_size=4,
        ).to(DEVICE, dtype=DTYPE).eval()
        B, F, H, W = 1, 6, 4, 4
        frame_seqlen = H * W
        total_tokens = F * frame_seqlen
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(head_dim)
        cache = _make_kv_cache(B, total_tokens, attn_config["num_heads"], head_dim)

        for chunk_idx in range(3):
            start = chunk_idx * 2 * frame_seqlen
            x = torch.randn(B, 2 * frame_seqlen, attn_config["dim"], device=DEVICE, dtype=DTYPE)
            grid = torch.tensor([[2, H, W]], device=DEVICE)
            seq_lens = torch.tensor([2 * frame_seqlen], device=DEVICE)
            with torch.no_grad():
                out = attn(x, seq_lens, grid, freqs,
                           kv_cache=cache, current_start=start, store_kv=True)

        assert out.shape == (B, 2 * frame_seqlen, attn_config["dim"])
        assert cache["global_end_index"].item() == total_tokens


class TestGradientFlow:
    """Verify gradients flow correctly through the rolling window."""

    def test_gradients_flow_with_rolling_window_and_eviction(self, attn_config):
        """Simulates self-forcing pattern: store_kv=True (no_grad) then store_kv=False (grad).

        Uses a SMALL buffer so eviction actually fires during the no_grad cache updates.
        Then verifies gradients flow through the grad-enabled read-only forward.
        """
        local_attn = 2  # 2-frame window
        attn = CausalSelfAttention(
            **attn_config, local_attn_size=local_attn,
        ).to(DEVICE, dtype=DTYPE).train()
        B, H, W = 1, 4, 4
        frame_seqlen = H * W
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(head_dim)
        # Small buffer: only 2 frames, so eviction fires on 3rd chunk
        small_cache_tokens = local_attn * frame_seqlen
        cache = _make_kv_cache(B, small_cache_tokens, attn_config["num_heads"], head_dim)

        # Phase 1: Populate cache (mimics self-forcing cache-update passes)
        for chunk_idx in range(3):
            start = chunk_idx * 2 * frame_seqlen
            x = torch.randn(B, 2 * frame_seqlen, attn_config["dim"], device=DEVICE, dtype=DTYPE)
            grid = torch.tensor([[2, H, W]], device=DEVICE)
            seq_lens = torch.tensor([2 * frame_seqlen], device=DEVICE)
            with torch.no_grad():  # cache updates are always no_grad
                attn(x, seq_lens, grid, freqs,
                     kv_cache=cache, current_start=start, store_kv=True)

        # Eviction should have happened (6 frames into 2-frame buffer)
        assert cache["global_end_index"].item() == 6 * frame_seqlen

        # Phase 2: Grad-enabled forward (mimics exit step, store_kv=False)
        x_grad = torch.randn(B, 2 * frame_seqlen, attn_config["dim"], device=DEVICE, dtype=DTYPE)
        grid = torch.tensor([[2, H, W]], device=DEVICE)
        seq_lens = torch.tensor([2 * frame_seqlen], device=DEVICE)
        out = attn(x_grad, seq_lens, grid, freqs,
                   kv_cache=cache, current_start=4 * frame_seqlen, store_kv=False)

        loss = out.sum()
        loss.backward()

        # Verify gradients reached the model parameters
        assert attn.q.weight.grad is not None, "Q projection should have gradients"
        assert attn.q.weight.grad.abs().sum() > 0, "Gradients should be non-zero"
        assert attn.k.weight.grad is not None, "K projection should have gradients"
        assert attn.v.weight.grad is not None, "V projection should have gradients"
        assert attn.o.weight.grad is not None, "O projection should have gradients"
