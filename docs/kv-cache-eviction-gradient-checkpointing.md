# KV Cache Eviction + Gradient Checkpointing Incompatibility

## Date: 2026-04-08

## Summary

When sliding window attention is used in AR mode during Self-Forcing training,
KV cache eviction (in-place buffer modification) produces **silently incorrect
gradients** when combined with gradient checkpointing. PyTorch does not detect
this because the cache tensor is passed inside a dict, bypassing version tracking.

## Root Cause

Gradient checkpointing (`torch.utils.checkpoint.checkpoint` with `use_reentrant=False`)
saves input tensors and **recomputes the forward pass** during backward to regenerate
intermediate activations. The recomputed forward must produce identical results to the
original forward for correct gradient computation.

The KV cache is read during the forward pass under `torch.no_grad()` as a detached
constant context. It does **not** require gradients. However, it must remain
**unchanged** between the original forward and the recomputation.

### Self-Forcing rollout sequence:
```
1. Block 0: exit step forward (gradient-enabled, checkpointed) — reads cache
2. Block 0: cache update (no_grad, store_kv=True) — eviction modifies cache IN-PLACE
3. Block 1: exit step forward — reads modified cache
4. ...
5. Block 6: cache update
6. loss.backward() → checkpoint recomputes Block 0's forward
   → reads MODIFIED cache → different activations → WRONG gradients
```

### Why PyTorch doesn't catch it:
The cache tensor is passed inside a `**kwargs` dict (`kv_cache=dict`), not as a
direct tensor argument. PyTorch's tensor version tracking only applies to tensors
that are saved by the checkpoint mechanism. Dict-wrapped tensors bypass this check,
so in-place modifications go undetected and backward completes silently.

### Empirical verification:
A standalone test with realistic attention (Q attending to cached K/V) showed
**10,498% relative gradient error** when cache was modified between forward and
backward. See test at: `tests/test_kv_cache_sliding_window.py`

## When this is NOT a problem

- **No sliding window** (`local_attn_size=-1`): Cache is only appended to, never
  overwritten. Previous entries remain stable. This is why all prior SF training
  (full causal) worked correctly.

- **Inference / validation**: No gradient checkpointing (runs under `torch.no_grad()`),
  so no recomputation occurs. Eviction is safe.

- **DF training**: Uses full-sequence FlexAttention, no KV cache at all.

## Fix

Always allocate the KV cache for the **full sequence length** (`total_num_frames *
frame_seqlen`), regardless of `local_attn_size`. This prevents eviction from ever
triggering during training, since the cache can hold all frames without overflow.

The sliding window restriction is enforced at the **attention window construction**
step (reading a window of the cache for attention), not at the cache storage level.
The eviction code remains for potential future inference-only use where memory is
constrained.

### Memory impact:
Extra cache memory = `(total_frames - local_attn_size) * frame_seqlen * num_heads *
head_dim * 2(K+V) * num_blocks * dtype_bytes`

For 21 frames vs 7-frame window: ~4.8 GB/GPU additional — modest on H200 (140GB).

### Code changes:
- `network_causal.py:_init_caches()`: Always use `cache_tokens = total_tokens`
- `network_causal.py:CausalSelfAttention.forward()`: Fixed `global_end = current_start`
  (was `global_end = cache_local_end_override`, wrong after eviction)

### Files:
- Fix: `fastgen/networks/OmniAvatar/network_causal.py`
- Test: `tests/test_kv_cache_sliding_window.py`
- Doc: this file
