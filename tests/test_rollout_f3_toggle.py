"""Tests for F3 (skip_clean_cache_pass) plumbing in SelfForcingModel.rollout_with_gradient.

Verifies three things:
  1. Default (F3 off): behavior unchanged — exit step has store_kv=False,
     separate cache pass runs with store_kv=True.
  2. F3 on: exit step has store_kv=True, no separate cache pass runs.
  3. Gradients still flow through the rollout in both configurations.

Tests mirror the style of tests/test_post_step_pin_inference.py — use a
MockNet/FakeSFModel to exercise the loop logic without any 14B model load.
"""
import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from fastgen.methods.distribution_matching.self_forcing import SelfForcingModel
from fastgen.networks.network import CausalFastGenNetwork


@pytest.fixture(autouse=True)
def _pretend_cuda_unavailable():
    """rollout_with_gradient contains CUDA-specific code paths
    (empty_cache, reset_peak_memory_stats). Patch torch.cuda.is_available to
    False so those paths are skipped for CPU-only tests."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


class _SpyNet(nn.Module):
    """Minimal causal-network mock that records every forward call.

    Registered as a virtual subclass of CausalFastGenNetwork so the
    isinstance() assertion in rollout_with_gradient passes without needing
    to implement the full abstract interface.
    """

    def __init__(self, chunk_size=3, total_num_frames=6, skip_clean_cache=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.total_num_frames = total_num_frames
        self._skip_clean_cache_pass = skip_clean_cache
        self._model_sink_cache = False
        self._enable_first_frame_anchor = True
        self._anchor_eval_only = False
        # Learnable parameter so gradient flow is observable
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = []
        self.noise_scheduler = SimpleNamespace(
            forward_process=lambda x, eps, t: x * (1 - t.view(-1, 1, 1, 1, 1))
                                               + eps * t.view(-1, 1, 1, 1, 1),
            x0_to_eps=lambda xt, x0, t: torch.randn_like(xt),
            max_sigma=1.0,
            t_precision=torch.float32,
        )
        self._kv_caches = None

    def clear_caches(self):
        pass

    def forward(self, x, t, **kwargs):
        self.forward_calls.append({
            "x_shape": tuple(x.shape),
            "t": t.detach().clone(),
            "store_kv": kwargs.get("store_kv", False),
            "cur_start_frame": kwargs.get("cur_start_frame", 0),
            "apply_anchor": kwargs.get("apply_anchor", True),
            "apply_input_anchor": kwargs.get("apply_input_anchor", True),
            "requires_grad": x.requires_grad or any(
                p.requires_grad for p in self.parameters()
            ),
        })
        # Differentiable transform so gradient can flow through rollout
        return x * self.scale


# Register _SpyNet as a virtual subclass so isinstance(_SpyNet(), CausalFastGenNetwork) is True
CausalFastGenNetwork.register(_SpyNet)


def _build_fake_sf_model(net, *, last_step_only=True, context_noise=0.0, device="cpu"):
    """Construct a minimal SelfForcingModel-like instance that satisfies the
    method's dependencies without requiring the heavy config machinery."""

    config = SimpleNamespace(
        student_sample_steps=4,
        same_step_across_blocks=True,
        last_step_only=last_step_only,   # force exit at last step for deterministic testing
        context_noise=context_noise,
        student_sample_type="sde",
        enable_gradient_in_rollout=True,
        start_gradient_frame=0,
        sample_t_cfg=SimpleNamespace(
            t_list=[0.999, 0.955, 0.875, 0.700, 0.0],
        ),
    )

    # We instantiate a SimpleNamespace-based fake rather than instantiating
    # SelfForcingModel (which requires a full method hierarchy via FastGenModel).
    # Bind the method unbound so `self` is the fake.
    fake = SimpleNamespace(
        net=net,
        config=config,
        device=device,
    )
    # Attach the real methods under test
    fake.rollout_with_gradient = SelfForcingModel.rollout_with_gradient.__get__(fake)
    fake._sample_denoising_end_steps = SelfForcingModel._sample_denoising_end_steps.__get__(fake)
    return fake


def _build_condition(B=1, C=16, T=21, H=8, W=8, device="cpu", dtype=torch.float32):
    return {
        "text_embeds": torch.zeros(B, 512, 4096, device=device, dtype=dtype),
        "clip_features": torch.zeros(B, 257, 1280, device=device, dtype=dtype),
        "audio_emb": torch.zeros(B, 93, 5, 12, 768, device=device, dtype=dtype),
        "first_frame_cond": torch.full((B, C, T, H, W), 3.0, device=device, dtype=dtype),
    }


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------

def test_rollout_default_f3_off_behavior_preserved():
    """Default path (no F3): exit step has store_kv=False, separate cache pass
    runs with store_kv=True. Exactly the pre-F3-plumbing behavior."""
    net = _SpyNet(chunk_size=3, total_num_frames=6, skip_clean_cache=False)
    fake = _build_fake_sf_model(net)

    B, C, T, H, W = 1, 16, 6, 8, 8
    noise = torch.randn(B, C, T, H, W)
    condition = _build_condition(B=B, C=C, T=T, H=H, W=W)

    _ = fake.rollout_with_gradient(noise, condition, enable_gradient=True)

    # 2 chunks × (3 non-exit forwards + 1 exit forward + 1 cache pass) = 10 forwards
    chunk0 = [c for c in net.forward_calls if c["cur_start_frame"] == 0]
    chunk1 = [c for c in net.forward_calls if c["cur_start_frame"] == 3]

    # Exit step is last step (step=3, t=0.700 before forward_process, so actually t=t_list[3]=0.700)
    # Separate cache pass comes next with t=0 (no context_noise)
    assert any(c["store_kv"] for c in chunk0), "chunk 0 cache pass should have fired"
    assert any(c["store_kv"] for c in chunk1), "chunk 1 cache pass should have fired"

    # Exactly ONE store_kv per chunk (the separate cache pass)
    assert sum(c["store_kv"] for c in chunk0) == 1, \
        "F3 off: exactly one store_kv per chunk (separate pass)"
    assert sum(c["store_kv"] for c in chunk1) == 1


def test_rollout_f3_on_skips_separate_cache_pass():
    """F3 on: exit step has store_kv=True, separate cache pass is NOT called."""
    net = _SpyNet(chunk_size=3, total_num_frames=6, skip_clean_cache=True)
    fake = _build_fake_sf_model(net)

    B, C, T, H, W = 1, 16, 6, 8, 8
    noise = torch.randn(B, C, T, H, W)
    condition = _build_condition(B=B, C=C, T=T, H=H, W=W)

    _ = fake.rollout_with_gradient(noise, condition, enable_gradient=True)

    chunk0 = [c for c in net.forward_calls if c["cur_start_frame"] == 0]
    chunk1 = [c for c in net.forward_calls if c["cur_start_frame"] == 3]

    # Still exactly one store_kv per chunk, but it's during the exit step now
    assert sum(c["store_kv"] for c in chunk0) == 1, \
        "F3 on: exactly one store_kv per chunk (on exit step)"
    assert sum(c["store_kv"] for c in chunk1) == 1

    # Total forward calls: 4 per chunk (3 non-exit + 1 exit-with-store), 8 total
    # (Compare vs. F3 off: 5 per chunk × 2 = 10)
    assert len(net.forward_calls) == 8, \
        f"F3 on should produce 8 forwards total (4/chunk × 2), got {len(net.forward_calls)}"


def test_rollout_f3_off_gradient_flows():
    """Gradient must flow to net.scale with F3 off."""
    net = _SpyNet(chunk_size=3, total_num_frames=6, skip_clean_cache=False)
    fake = _build_fake_sf_model(net)

    B, C, T, H, W = 1, 16, 6, 8, 8
    noise = torch.randn(B, C, T, H, W)
    condition = _build_condition(B=B, C=C, T=T, H=H, W=W)
    target = torch.randn(B, C, T, H, W)

    out = fake.rollout_with_gradient(noise, condition, enable_gradient=True)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    assert net.scale.grad is not None and torch.isfinite(net.scale.grad), \
        "F3 off: gradient should flow to scale param"
    assert net.scale.grad.abs() > 0, "F3 off: gradient should be non-zero"


def test_rollout_f3_on_gradient_flows():
    """Gradient must flow to net.scale with F3 on — this is the critical check:
    store_kv=True combined with a gradient-enabled forward must not break autograd."""
    net = _SpyNet(chunk_size=3, total_num_frames=6, skip_clean_cache=True)
    fake = _build_fake_sf_model(net)

    B, C, T, H, W = 1, 16, 6, 8, 8
    noise = torch.randn(B, C, T, H, W)
    condition = _build_condition(B=B, C=C, T=T, H=H, W=W)
    target = torch.randn(B, C, T, H, W)

    out = fake.rollout_with_gradient(noise, condition, enable_gradient=True)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    assert net.scale.grad is not None and torch.isfinite(net.scale.grad), \
        "F3 on: gradient should flow to scale param despite store_kv=True on exit"
    assert net.scale.grad.abs() > 0, "F3 on: gradient should be non-zero"


def test_rollout_exit_step_is_last_in_test_config():
    """Sanity: when last_step_only=True, exit happens at the last step (step 3 of 4).
    This means the store_kv=True under F3 lives on the t_list[3]=0.700 timestep."""
    net = _SpyNet(chunk_size=3, total_num_frames=6, skip_clean_cache=True)
    fake = _build_fake_sf_model(net, last_step_only=True)

    B, C, T, H, W = 1, 16, 6, 8, 8
    noise = torch.randn(B, C, T, H, W)
    condition = _build_condition(B=B, C=C, T=T, H=H, W=W)

    _ = fake.rollout_with_gradient(noise, condition, enable_gradient=True)

    # Find the store_kv call on chunk 0
    chunk0_store = [c for c in net.forward_calls if c["cur_start_frame"] == 0 and c["store_kv"]]
    assert len(chunk0_store) == 1
    # Its timestep should be t_list[3] = 0.700 (the last denoise step, since last_step_only)
    t_val = chunk0_store[0]["t"].item()
    assert abs(t_val - 0.700) < 1e-5, \
        f"Expected store_kv at t=0.700 under last_step_only+F3, got t={t_val}"
