# Rolling Attention Window with Decoupled Cache Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `local_attn_size`, `sink_size`, and `use_dynamic_rope` parameters to `CausalSelfAttention`, decoupling the KV cache buffer size (full sequence for training) from the attention window (rolling for inference-like behavior during training).

**Architecture:** The cache buffer is always sized to `total_num_frames * frame_seqlen` (set in `_init_caches`). When `local_attn_size > 0`, attention is restricted to a rolling window of the last N frames plus sink tokens — but no physical eviction occurs because the buffer holds everything. During inference, the caller can optionally set a smaller cache buffer to save VRAM, in which case the existing OmniAvatar-style physical eviction kicks in as a fallback.

**Tech Stack:** PyTorch, flash_attn, FlexAttention (torch.nn.attention.flex_attention)

---

## File Map

- Modify: `fastgen/networks/InfiniteTalk/network_causal.py`
  - `CausalSelfAttention` — add 3 new params, rewrite AR branch
  - `CausalDiTBlock` — pass new params through
  - `CausalInfiniteTalkWan.__init__` — accept and propagate new params
  - `CausalInfiniteTalkWan._init_caches` — keep full-sequence sizing (no change needed)
- Modify: `fastgen/configs/experiments/InfiniteTalk/config_df.py` — optionally expose new params
- Test: `tests/test_rolling_attention.py` — unit tests for the attention window logic

---

### Task 1: Add Parameters to CausalSelfAttention

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:337-360`

- [ ] **Step 1: Update `CausalSelfAttention.__init__` signature and storage**

```python
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
        when ``local_attn_size > 0`` (otherwise evicted-then-re-viewed
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
```

Note: `local_attn_size`, `sink_size`, `use_dynamic_rope` default to the current behavior (attend to everything, no sinks, pre-rotated cache). All existing call sites work unchanged.

---

### Task 2: Rewrite the AR Branch of CausalSelfAttention.forward

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:430-482`

This is the core change. The new AR branch handles:
- Pre-rotated RoPE mode (current behavior, `use_dynamic_rope=False`)
- Dynamic RoPE mode (cache raw K, apply window-local RoPE at attention time)
- Rolling attention window via slicing (not physical eviction)
- Physical eviction as fallback only when cache buffer is smaller than sequence
- Sink token handling in the attention window

- [ ] **Step 1: Replace the AR branch with the new implementation**

Replace lines 430-482 (the entire `else:` block after `if kv_cache is None:`) with:

```python
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
                global_end = cache_local_end_override
            else:
                global_end = kv_cache["global_end_index"].item()
                local_end = kv_cache["local_end_index"].item()

            # -- 3. Handle physical eviction (only when buffer overflows) --
            # This is a FALLBACK for inference with small cache buffers.
            # During training, cache_size == total_tokens, so this never fires.
            if (
                store_kv
                and self.local_attn_size > 0
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
```

- [ ] **Step 2: Verify the full-sequence branch is untouched**

The `if kv_cache is None:` branch (lines 393-429) does NOT need any changes — `local_attn_size` only affects the AR (kv_cache) path. Full-sequence mode uses the FlexAttention block mask for causality, which already handles the attention pattern correctly.

---

### Task 3: Thread Parameters Through CausalDiTBlock

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:586-611`

- [ ] **Step 1: Add params to CausalDiTBlock.__init__ and pass to CausalSelfAttention**

Change the `__init__` signature at line 586:

```python
class CausalDiTBlock(nn.Module):
    # ... (docstring unchanged) ...

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
        # ... (existing field assignments unchanged) ...

        # Self-attention (causal variant with KV cache)
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            use_dynamic_rope=use_dynamic_rope,
        )

        # ... (rest of __init__ unchanged) ...
```

The `CausalDiTBlock.forward` method does NOT need changes — it already passes `kv_cache`, `current_start`, `store_kv`, and `cache_local_end_override` to `self.self_attn.forward()`.

---

### Task 4: Thread Parameters Through CausalInfiniteTalkWan

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:843-864` (init signature)
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:950-962` (block construction)

- [ ] **Step 1: Add params to CausalInfiniteTalkWan.__init__**

Add after `vae_scale` (line 864):

```python
    def __init__(
        self,
        # ... existing params ...
        vae_scale: int = 4,
        local_attn_size: int = -1,
        sink_size: int = 0,
        use_dynamic_rope: bool = False,
        **kwargs,
    ):
```

Store them as instance attributes (after the existing `self._dim = dim` etc.):

```python
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_dynamic_rope = use_dynamic_rope
```

- [ ] **Step 2: Pass params to CausalDiTBlock construction**

Change the block construction at line 953:

```python
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
```

- [ ] **Step 3: Verify _init_caches stays full-sequence**

`_init_caches` (lines 1228-1253) already allocates `total_tokens` for the buffer. This is correct — the cache holds the full sequence. The rolling window is handled by `CausalSelfAttention`'s attention window slicing, not by cache sizing. **No change needed to `_init_caches`.**

---

### Task 5: Write Unit Tests

**Files:**
- Create: `tests/test_rolling_attention.py`

- [ ] **Step 1: Write tests for rolling attention window behavior**

```python
"""Unit tests for CausalSelfAttention with rolling attention window."""

import math
import pytest
import torch

# Adjust import path as needed
from fastgen.networks.InfiniteTalk.network_causal import (
    CausalSelfAttention,
    causal_rope_apply,
    rope_params,
)


def _make_freqs(dim, head_dim):
    """Build RoPE freq tables matching the model's init."""
    d = head_dim
    return tuple(
        rope_params(1024, size)
        for size in [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    )


def _make_kv_cache(batch, total_tokens, num_heads, head_dim, device="cpu", dtype=torch.bfloat16):
    """Allocate a full-sequence KV cache buffer."""
    return {
        "k": torch.zeros(batch, total_tokens, num_heads, head_dim, device=device, dtype=dtype),
        "v": torch.zeros(batch, total_tokens, num_heads, head_dim, device=device, dtype=dtype),
        "global_end_index": torch.tensor(0, device=device, dtype=torch.long),
        "local_end_index": torch.tensor(0, device=device, dtype=torch.long),
    }


@pytest.fixture
def attn_config():
    """Small model config for fast tests."""
    return dict(dim=128, num_heads=4, eps=1e-6)


class TestDefaultBehaviorUnchanged:
    """With default params (local_attn_size=-1), behavior matches the old code."""

    def test_full_cache_attention(self, attn_config):
        """All cached tokens are attended to when local_attn_size=-1."""
        attn = CausalSelfAttention(**attn_config).eval()
        B, F, H, W = 1, 6, 4, 4  # 6 frames, 16 tokens each
        frame_seqlen = H * W
        total_tokens = F * frame_seqlen
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(attn_config["dim"], head_dim)
        cache = _make_kv_cache(B, total_tokens, attn_config["num_heads"], head_dim)

        # Feed 3 chunks of 2 frames each
        for chunk_idx in range(3):
            start = chunk_idx * 2 * frame_seqlen
            x = torch.randn(B, 2 * frame_seqlen, attn_config["dim"])
            grid = torch.tensor([[2, H, W]])
            seq_lens = torch.tensor([2 * frame_seqlen])
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
        ).eval()
        B, F, H, W = 1, 6, 4, 4
        frame_seqlen = H * W
        total_tokens = F * frame_seqlen
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(attn_config["dim"], head_dim)
        # Full-sequence buffer (larger than window)
        cache = _make_kv_cache(B, total_tokens, attn_config["num_heads"], head_dim)

        for chunk_idx in range(3):
            start = chunk_idx * 2 * frame_seqlen
            x = torch.randn(B, 2 * frame_seqlen, attn_config["dim"])
            grid = torch.tensor([[2, H, W]])
            seq_lens = torch.tensor([2 * frame_seqlen])
            with torch.no_grad():
                attn(x, seq_lens, grid, freqs,
                     kv_cache=cache, current_start=start, store_kv=True)

        # All tokens stored (no eviction happened)
        assert cache["local_end_index"].item() == total_tokens
        # Old tokens are still in buffer (non-zero)
        assert cache["k"][:, :frame_seqlen].abs().sum() > 0, \
            "First frame should still be in buffer (no physical eviction)"

    def test_store_kv_false_does_not_write(self, attn_config):
        """store_kv=False reads the cache but does not modify it."""
        attn = CausalSelfAttention(**attn_config).eval()
        B, H, W = 1, 4, 4
        frame_seqlen = H * W
        head_dim = attn_config["dim"] // attn_config["num_heads"]
        freqs = _make_freqs(attn_config["dim"], head_dim)
        cache = _make_kv_cache(B, 6 * frame_seqlen, attn_config["num_heads"], head_dim)

        # Write chunk 0
        x0 = torch.randn(B, 2 * frame_seqlen, attn_config["dim"])
        grid = torch.tensor([[2, H, W]])
        seq_lens = torch.tensor([2 * frame_seqlen])
        with torch.no_grad():
            attn(x0, seq_lens, grid, freqs,
                 kv_cache=cache, current_start=0, store_kv=True)

        saved_end = cache["global_end_index"].item()
        saved_k = cache["k"].clone()

        # Read-only forward (store_kv=False) at same position
        x1 = torch.randn(B, 2 * frame_seqlen, attn_config["dim"])
        with torch.no_grad():
            attn(x1, seq_lens, grid, freqs,
                 kv_cache=cache, current_start=0, store_kv=False)

        assert cache["global_end_index"].item() == saved_end, \
            "store_kv=False should not update metadata"
        assert torch.equal(cache["k"], saved_k), \
            "store_kv=False should not modify cache contents"
```

- [ ] **Step 2: Run tests**

Run: `cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk && python -m pytest tests/test_rolling_attention.py -v`

Expected: All tests PASS.

---

### Task 6: Verify Backward Compatibility (Existing Configs)

**Files:**
- No file changes — verification only.

- [ ] **Step 1: Confirm default params produce identical behavior**

With `local_attn_size=-1, sink_size=0, use_dynamic_rope=False` (all defaults), the new code should produce the **exact same** attention output as the old code. The key check:

- `max_attn_tokens = new_local_end` (attend to everything) — same as before
- `k_win_start = 0` — same as before
- No eviction fires — same as before
- Pre-rotated RoPE path — same as before

Run the existing Stage 0 verification script to confirm:

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
python scripts/verify_infinitetalk_equivalence.py --level 2
```

Expected: Level 2 (forward pass equivalence) PASSES with max_diff=0.0.

- [ ] **Step 2: Smoke-test DF training with defaults**

```bash
python train.py --config fastgen/configs/experiments/InfiniteTalk/config_df_test.py
```

Expected: 20 iterations complete without error. Loss values similar to previous runs.

---

### Task 7: Add Config Knobs for Rolling Window

**Files:**
- Modify: `fastgen/configs/experiments/InfiniteTalk/config_df.py`
- Modify: `fastgen/configs/experiments/InfiniteTalk/config_sf.py`

- [ ] **Step 1: Expose params in the student network LazyCall**

In `config_df.py`, add to the `CausalInfiniteTalk_14B_Student` dict (after `total_num_frames=21`):

```python
CausalInfiniteTalk_14B_Student: dict = L(CausalInfiniteTalkWan)(
    # ... existing params ...
    chunk_size=3,
    total_num_frames=21,
    local_attn_size=-1,   # -1 = attend to everything (default, safe)
    sink_size=0,
    use_dynamic_rope=False,
    # ... rest unchanged ...
)
```

Same change in `config_sf.py` for the student network config.

These default to current behavior. To enable rolling window, a user changes `local_attn_size` to e.g. `7` and `use_dynamic_rope` to `True`.

---

## Self-Review Checklist

**1. Spec coverage:**
- Rolling attention window (slice, not evict): Task 2, step 5 — `k_win_start = max(0, new_local_end - max_attn_tokens)` slices the cache without eviction when buffer is large enough. ✓
- Physical eviction as fallback: Task 2, step 3 — fires only when `num_new_tokens + local_end > kv_cache_size`. ✓
- Sink tokens: Task 2, step 5 — sink + rolling branch builds non-contiguous window. ✓
- Dynamic RoPE: Task 2, step 6 — window-local RoPE with query offset. ✓
- `store_kv` guard: preserved from existing code (lines 461-463, 480-482). ✓
- `.detach()` on writes: preserved from existing code (line 462-463). ✓
- `cache_local_end_override`: preserved from existing code (lines 449-454). ✓
- Backward compat: Task 6 — defaults reproduce old behavior. ✓
- `_init_caches` unchanged: Task 4 step 3 confirms full-sequence sizing stays. ✓

**2. Placeholder scan:** No TBDs, TODOs, or vague steps. All code blocks complete.

**3. Type consistency:**
- `local_attn_size: int = -1` — consistent across CausalSelfAttention, CausalDiTBlock, CausalInfiniteTalkWan.
- `sink_size: int = 0` — consistent.
- `use_dynamic_rope: bool = False` — consistent.
- `causal_rope_apply` function signature unchanged (used in Task 2 step 1).
