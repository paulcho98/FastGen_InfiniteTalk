"""Unit test that apply_anchor=False suppresses the frame-0 overwrite.

Uses the extracted module-level helper _maybe_apply_first_frame_anchor
to avoid instantiating the 14B network.
"""
import torch


def test_apply_anchor_false_skips_overwrite():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_first_frame_anchor

    stub_output = torch.full((1, 16, 3, 28, 56), 7.0)
    first_frame_cond = torch.full((1, 16, 3, 28, 56), -3.0)

    class StubNet:
        training = False
        _enable_first_frame_anchor = True
        _anchor_eval_only = False

    out_anchored = _maybe_apply_first_frame_anchor(
        stub_output.clone(), StubNet(), cur_start_frame=0,
        condition={"first_frame_cond": first_frame_cond},
        apply_anchor=True,
    )
    out_not_anchored = _maybe_apply_first_frame_anchor(
        stub_output.clone(), StubNet(), cur_start_frame=0,
        condition={"first_frame_cond": first_frame_cond},
        apply_anchor=False,
    )

    assert out_anchored[:, :, 0:1].eq(-3.0).all(), "anchor=True should overwrite frame 0"
    assert out_not_anchored[:, :, 0:1].eq(7.0).all(), "anchor=False should preserve frame 0"


def test_apply_anchor_nonzero_start_frame_is_noop():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_first_frame_anchor

    stub_output = torch.full((1, 16, 3, 28, 56), 7.0)
    first_frame_cond = torch.full((1, 16, 3, 28, 56), -3.0)

    class StubNet:
        training = False
        _enable_first_frame_anchor = True
        _anchor_eval_only = False

    # cur_start_frame > 0 — anchor never applies regardless of apply_anchor flag
    out = _maybe_apply_first_frame_anchor(
        stub_output.clone(), StubNet(), cur_start_frame=3,
        condition={"first_frame_cond": first_frame_cond},
        apply_anchor=True,
    )
    assert out[:, :, 0:1].eq(7.0).all(), "cur_start_frame>0 should never anchor"


def test_apply_anchor_respects_enable_flag():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_first_frame_anchor

    stub_output = torch.full((1, 16, 3, 28, 56), 7.0)
    first_frame_cond = torch.full((1, 16, 3, 28, 56), -3.0)

    class StubNetDisabled:
        training = False
        _enable_first_frame_anchor = False  # hard-disabled
        _anchor_eval_only = False

    out = _maybe_apply_first_frame_anchor(
        stub_output.clone(), StubNetDisabled(), cur_start_frame=0,
        condition={"first_frame_cond": first_frame_cond},
        apply_anchor=True,
    )
    assert out[:, :, 0:1].eq(7.0).all(), "_enable_first_frame_anchor=False overrides apply_anchor=True"


def test_apply_anchor_respects_eval_only_flag_in_training():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_first_frame_anchor

    stub_output = torch.full((1, 16, 3, 28, 56), 7.0)
    first_frame_cond = torch.full((1, 16, 3, 28, 56), -3.0)

    class StubNetTraining:
        training = True
        _enable_first_frame_anchor = True
        _anchor_eval_only = True  # eval-only

    out = _maybe_apply_first_frame_anchor(
        stub_output.clone(), StubNetTraining(), cur_start_frame=0,
        condition={"first_frame_cond": first_frame_cond},
        apply_anchor=True,
    )
    assert out[:, :, 0:1].eq(7.0).all(), "anchor_eval_only=True during training should skip anchor"
