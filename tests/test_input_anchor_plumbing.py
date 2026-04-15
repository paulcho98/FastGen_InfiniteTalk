"""Unit tests for input-side frame-0 anchor helper and plumbing."""
import torch
import pytest
from types import SimpleNamespace


# The helper under test (we import it in step 3 after implementation).
# For now, reference it by module path; the test fails on ImportError.
def test_input_anchor_helper_pins_frame_0_when_active():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(2, 16, 21, 28, 56)
    ffc = torch.ones(2, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}

    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True,
        _anchor_eval_only=False,
        training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=condition, apply_input_anchor=True,
    )
    # Frame 0 pinned
    assert out[:, :, 0].eq(7.0).all(), "frame 0 not pinned to first_frame_cond"
    # Other frames untouched
    assert out[:, :, 1:].eq(0.0).all(), "frames 1+ should be unchanged"
    # Original x_t is NOT mutated (function must clone)
    assert x_t[:, :, 0].eq(0.0).all(), "input x_t was mutated in place"


def test_input_anchor_helper_noop_when_apply_input_anchor_false():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 21, 28, 56)
    ffc = torch.ones(1, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True, _anchor_eval_only=False, training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=condition, apply_input_anchor=False,
    )
    assert out is x_t, "apply_input_anchor=False should return input unchanged"


def test_input_anchor_helper_noop_when_cur_start_nonzero():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 3, 28, 56)
    ffc = torch.ones(1, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True, _anchor_eval_only=False, training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=3, condition=condition, apply_input_anchor=True,
    )
    assert out is x_t, "non-zero cur_start_frame should skip pinning"


def test_input_anchor_helper_respects_hard_disable():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 21, 28, 56)
    ffc = torch.ones(1, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=False,  # hard-disabled (e.g., teacher_anchor_disabled)
        _anchor_eval_only=False,
        training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=condition, apply_input_anchor=True,
    )
    assert out is x_t, "_enable_first_frame_anchor=False should skip pinning"


def test_input_anchor_helper_eval_only_during_training():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 21, 28, 56)
    ffc = torch.ones(1, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True,
        _anchor_eval_only=True,   # eval-only
        training=True,            # but we are training
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=condition, apply_input_anchor=True,
    )
    assert out is x_t, "_anchor_eval_only + training=True should skip pinning"


def test_input_anchor_helper_no_condition():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 21, 28, 56)
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True, _anchor_eval_only=False, training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=None, apply_input_anchor=True,
    )
    assert out is x_t, "missing condition should skip pinning"
