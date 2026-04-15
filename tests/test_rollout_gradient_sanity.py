"""Gradient-sanity tests for the new input-anchor plumbing.

The new code paths introduced in the input-anchor plan are:
  1. `_maybe_apply_input_anchor` helper (network_causal.py) — clone + slice-assign.
  2. `_maybe_apply_input_anchor_bidir` helper (network.py) — same pattern.
  3. Post-step pin block in `_student_sample_loop` (causvid.py) — clone + slice-assign.

All three share the same gradient-sensitive pattern: clone the tensor, then
in-place assign a slice. The clone isolates the mutation from the autograd
graph of the input, so gradients should flow back through the clone's
non-pinned positions to the original x_t, and gradients at the pinned
position flow from `first_frame_cond` instead.

We test the helpers directly with `requires_grad=True` tensors. This is the
minimum viable gradient verification — it covers the EXACT code paths we
added without dragging in unrelated pre-existing autograd-unsafe code
(e.g. `_student_sample_loop`'s in-place `x[:, :, start:end] = x_next`
pattern, which is fine under `torch.no_grad()` — its only use site — but
would fail here).

The actual production gradient path used by SF training is
`SelfForcingModel.rollout_with_gradient` (self_forcing.py), which constructs
a fresh list of denoised blocks and concatenates them rather than writing
in-place into a shared tensor. That loop calls `self.net(...)` for every
forward, and the new `apply_input_anchor=True` default propagates via the
forward method's kwarg default — so if the helper itself is gradient-safe
(verified here), the whole training rollout is too.

Fast (<5s), CPU-only, no 14B model loading.
"""
import torch
from types import SimpleNamespace


def _build_net_module(enable=True, eval_only=False, training=False):
    return SimpleNamespace(
        _enable_first_frame_anchor=enable,
        _anchor_eval_only=eval_only,
        training=training,
    )


def test_input_anchor_helper_gradient_flows_through_nonpinned_frames():
    """Gradients on x_t at frames 1+ should flow back unmodified through the clone."""
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.randn(1, 16, 3, 8, 8, requires_grad=True)
    ffc = torch.full((1, 16, 3, 8, 8), 3.0)
    condition = {"first_frame_cond": ffc}
    net = _build_net_module()

    out = _maybe_apply_input_anchor(
        x_t, net, cur_start_frame=0, condition=condition, apply_input_anchor=True,
    )
    # Loss touches only frames 1+ — gradient should flow through the clone,
    # through the non-pinned positions, back to x_t.
    loss = out[:, :, 1:].sum()
    loss.backward()

    # Gradient at frame 0 must be zero (the pin cut it)
    assert x_t.grad is not None
    assert x_t.grad[:, :, 0].eq(0.0).all(), \
        "Frame 0 gradient should be 0 — the input anchor cut the graph there"
    # Gradient at frames 1+ must be 1 (sum's derivative is ones)
    assert x_t.grad[:, :, 1:].eq(1.0).all(), \
        "Frames 1+ gradient should be 1 (sum) — autograd blocked somewhere"


def test_input_anchor_helper_gradient_flows_from_first_frame_cond():
    """When loss touches frame 0 of the output, gradient should flow to
    first_frame_cond (not x_t at frame 0)."""
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.randn(1, 16, 3, 8, 8, requires_grad=True)
    ffc = torch.full((1, 16, 3, 8, 8), 3.0, requires_grad=True)
    condition = {"first_frame_cond": ffc}
    net = _build_net_module()

    out = _maybe_apply_input_anchor(
        x_t, net, cur_start_frame=0, condition=condition, apply_input_anchor=True,
    )
    loss = out[:, :, 0].sum()  # only frame 0
    loss.backward()

    # x_t gradient at frame 0 must be zero — pin cut the graph
    assert x_t.grad[:, :, 0].eq(0.0).all()
    # first_frame_cond gradient at frame 0 must be 1 (sum) — the pin routed grad here
    assert ffc.grad is not None
    assert ffc.grad[:, :, 0].eq(1.0).all(), \
        "first_frame_cond frame-0 gradient should be 1 — routing broken"
    # first_frame_cond gradient at frames 1+ should be 0 — we sliced only [:, :, 0:1]
    assert ffc.grad[:, :, 1:].eq(0.0).all(), \
        "first_frame_cond frames 1+ should have zero gradient"


def test_input_anchor_helper_noop_preserves_gradient():
    """When apply_input_anchor=False, the output is the input tensor (not a clone),
    so gradients flow through unchanged."""
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.randn(1, 16, 3, 8, 8, requires_grad=True)
    ffc = torch.full((1, 16, 3, 8, 8), 3.0)
    condition = {"first_frame_cond": ffc}
    net = _build_net_module()

    out = _maybe_apply_input_anchor(
        x_t, net, cur_start_frame=0, condition=condition, apply_input_anchor=False,
    )
    # The no-op path returns x_t itself — verify gradient flows all the way
    out.sum().backward()
    assert x_t.grad is not None
    assert x_t.grad.eq(1.0).all(), \
        "apply_input_anchor=False should preserve full gradient flow"


def test_input_anchor_helper_noop_on_nonzero_chunk_preserves_gradient():
    """cur_start_frame != 0 returns x_t unchanged — gradients intact."""
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.randn(1, 16, 3, 8, 8, requires_grad=True)
    ffc = torch.full((1, 16, 21, 8, 8), 3.0)  # ffc covers all 21 latent frames
    condition = {"first_frame_cond": ffc}
    net = _build_net_module()

    out = _maybe_apply_input_anchor(
        x_t, net, cur_start_frame=3, condition=condition, apply_input_anchor=True,
    )
    out.sum().backward()
    assert x_t.grad.eq(1.0).all(), \
        "Non-zero chunk should skip pinning — full gradient should flow"


def test_bidirectional_input_anchor_helper_gradient_flow():
    """Mirror test for the bidirectional helper — same semantics, no cur_start_frame."""
    from fastgen.networks.InfiniteTalk.network import _maybe_apply_input_anchor_bidir

    x_t = torch.randn(1, 16, 21, 8, 8, requires_grad=True)
    ffc = torch.full((1, 16, 21, 8, 8), 3.0, requires_grad=True)
    condition = {"first_frame_cond": ffc}
    net = _build_net_module()

    out = _maybe_apply_input_anchor_bidir(
        x_t, net, condition=condition, apply_input_anchor=True,
    )
    # Sum over all frames — gradient routed: frames 1+ to x_t, frame 0 to ffc
    out.sum().backward()
    assert x_t.grad[:, :, 0].eq(0.0).all()
    assert x_t.grad[:, :, 1:].eq(1.0).all()
    assert ffc.grad[:, :, 0].eq(1.0).all()
    assert ffc.grad[:, :, 1:].eq(0.0).all()


def test_input_anchor_helper_6_iter_stability():
    """Six iterations of helper + backward — catches any autograd state leakage
    between iterations (e.g., if clone somehow wasn't isolating graphs)."""
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    net = _build_net_module()

    for i in range(6):
        x_t = torch.randn(1, 16, 3, 8, 8, requires_grad=True)
        ffc = torch.full((1, 16, 3, 8, 8), 3.0 + i * 0.1)
        condition = {"first_frame_cond": ffc}

        out = _maybe_apply_input_anchor(
            x_t, net, cur_start_frame=0, condition=condition, apply_input_anchor=True,
        )
        loss = (out ** 2).mean()
        loss.backward()

        assert torch.isfinite(loss), f"Iter {i}: non-finite loss"
        assert x_t.grad is not None, f"Iter {i}: no grad on x_t"
        assert torch.isfinite(x_t.grad).all(), f"Iter {i}: non-finite grad on x_t"
        # Frame 0 of x_t.grad must be zero (pin cuts the graph there)
        assert x_t.grad[:, :, 0].eq(0.0).all(), f"Iter {i}: frame-0 grad should be 0"
