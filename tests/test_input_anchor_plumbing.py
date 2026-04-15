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


def test_apply_input_anchor_kwarg_pins_forward_input(monkeypatch):
    """Verify CausalInfiniteTalkWan.forward applies input anchor before dispatch."""
    from fastgen.networks.InfiniteTalk import network_causal

    captured_x = {}

    def fake_forward_ar(self, x, timestep, context, clip_fea, y, audio,
                       current_start, store_kv, use_gradient_checkpointing):
        # Capture the x that gets passed to the AR forward
        captured_x["x"] = x.detach().clone()
        # Return a dummy tensor matching expected output shape
        B, C, T, H, W = x.shape
        return torch.zeros(B, C, T, H, W, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(
        network_causal.CausalInfiniteTalkWan,
        "_forward_ar",
        fake_forward_ar,
    )

    # Build a minimal-mocked instance
    net = SimpleNamespace(
        _enable_first_frame_anchor=True,
        _anchor_eval_only=False,
        training=False,
        _use_gradient_checkpointing=False,
        _audio_window=5,
        _patch_size=(1, 2, 2),
        _lookahead_sink_enabled=False,
        _lookahead_distance_min=0,
        _lookahead_distance_max=0,
        net_pred_type="flow",
        noise_scheduler=SimpleNamespace(
            rescale_t=lambda t: t * 1000,
            convert_model_output=lambda x_t, model_out, t, src_pred_type, target_pred_type: model_out,
        ),
        _build_y=lambda cond, T, start_frame: torch.zeros(cond["first_frame_cond"].shape[0], 20, T, cond["first_frame_cond"].shape[3], cond["first_frame_cond"].shape[4], dtype=cond["first_frame_cond"].dtype),
    )

    # x_t has noisy frame 0
    x_t = torch.randn(1, 16, 3, 28, 56)  # chunk of 3 frames
    ffc = torch.full((1, 16, 21, 28, 56), 5.0)  # clean ref = 5.0 everywhere
    condition = {
        "text_embeds": torch.zeros(1, 512, 4096),
        "clip_features": torch.zeros(1, 257, 1280),
        "audio_emb": torch.zeros(1, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t = torch.tensor([0.5])

    _ = network_causal.CausalInfiniteTalkWan.forward(
        net, x_t, t, condition=condition,
        cur_start_frame=0, is_ar=True,
        apply_input_anchor=True, apply_anchor=True,
    )

    # Verify that _forward_ar received x with frame 0 pinned to 5.0
    assert "x" in captured_x, "_forward_ar was not called"
    assert captured_x["x"][:, :, 0].eq(5.0).all(), \
        "Input anchor did NOT pin frame 0 before forward"
    # Non-frame-0 positions are unchanged (from original x_t, which was random)
    # We can't check exact values (random), but we check they weren't all 5.0
    assert not captured_x["x"][:, :, 1:].eq(5.0).all(), \
        "Non-frame-0 positions were unexpectedly pinned"


def test_apply_input_anchor_false_skips_pinning(monkeypatch):
    """Verify apply_input_anchor=False leaves x_t untouched."""
    from fastgen.networks.InfiniteTalk import network_causal

    captured_x = {}

    def fake_forward_ar(self, x, timestep, context, clip_fea, y, audio,
                       current_start, store_kv, use_gradient_checkpointing):
        captured_x["x"] = x.detach().clone()
        B, C, T, H, W = x.shape
        return torch.zeros(B, C, T, H, W, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(
        network_causal.CausalInfiniteTalkWan,
        "_forward_ar",
        fake_forward_ar,
    )

    net = SimpleNamespace(
        _enable_first_frame_anchor=True, _anchor_eval_only=False, training=False,
        _use_gradient_checkpointing=False, _audio_window=5, _patch_size=(1, 2, 2),
        _lookahead_sink_enabled=False, _lookahead_distance_min=0, _lookahead_distance_max=0,
        net_pred_type="flow",
        noise_scheduler=SimpleNamespace(
            rescale_t=lambda t: t * 1000,
            convert_model_output=lambda x_t, model_out, t, src_pred_type, target_pred_type: model_out,
        ),
        _build_y=lambda cond, T, start_frame: torch.zeros(cond["first_frame_cond"].shape[0], 20, T, cond["first_frame_cond"].shape[3], cond["first_frame_cond"].shape[4], dtype=cond["first_frame_cond"].dtype),
    )

    original_x = torch.randn(1, 16, 3, 28, 56)
    x_t = original_x.clone()
    ffc = torch.full((1, 16, 21, 28, 56), 5.0)
    condition = {
        "text_embeds": torch.zeros(1, 512, 4096),
        "clip_features": torch.zeros(1, 257, 1280),
        "audio_emb": torch.zeros(1, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t = torch.tensor([0.5])

    _ = network_causal.CausalInfiniteTalkWan.forward(
        net, x_t, t, condition=condition,
        cur_start_frame=0, is_ar=True,
        apply_input_anchor=False,  # disabled
        apply_anchor=True,
    )

    assert torch.equal(captured_x["x"], original_x), \
        "apply_input_anchor=False should leave x_t unchanged"
