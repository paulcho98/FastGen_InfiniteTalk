"""Knot Forcing unit tests — config flags, anchor pin-source, knot injection,
fusion, running-ahead, dataloader last-frame reference."""
import pytest
import torch


def test_kf_config_flags_default_off():
    """KF flags must default to off — KF-off must be bit-exact to existing SF."""
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    assert cfg.use_temporal_knot is False
    assert cfg.knot_size == 1
    assert cfg.use_running_ahead is False
    assert cfg.running_ahead_step == 4
    assert cfg.running_ahead_init_n == 8
    assert cfg.use_last_frame_reference is False


def test_kf_running_ahead_incompatible_with_lookahead_sink():
    """Running-ahead and F1 lookahead_sink both modify sink RoPE — must not coexist."""
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    cfg.use_running_ahead = True
    cfg.lookahead_sink_enabled = True
    with pytest.raises(ValueError, match="running_ahead.*lookahead_sink"):
        cfg.validate_kf_flags()


def test_kf_validation_passes_when_only_one_is_on():
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    cfg.use_running_ahead = True
    cfg.lookahead_sink_enabled = False
    cfg.validate_kf_flags()  # should not raise


def test_kf_validation_knot_size_positive():
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    cfg.use_temporal_knot = True
    cfg.knot_size = 0
    with pytest.raises(ValueError, match="knot_size"):
        cfg.validate_kf_flags()


def test_kf_validation_running_ahead_step_positive():
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    cfg.use_running_ahead = True
    cfg.running_ahead_step = 0
    with pytest.raises(ValueError, match="running_ahead_step"):
        cfg.validate_kf_flags()


def test_kf_validation_all_off_passes():
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    cfg.validate_kf_flags()  # defaults: all off, should not raise


def test_build_y_knot_injection():
    """At iter > 0, _build_y injects knot at slice position 0 (= absolute
    start_frame) when condition has 'knot_latent_for_chunk_start'."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    # _build_y is essentially stateless — only uses _construct_i2v_mask (a staticmethod).
    # Create a minimal stub bypassing __init__.
    stub = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)

    B, C, T_full, H, W = 1, 16, 21, 4, 4
    ffc = torch.zeros(B, C, T_full, H, W)
    ffc[:, :, 0] = 7.0  # reference at position 0
    knot = torch.full((B, C, 1, H, W), -3.0)

    # Baseline: no knot. At iter > 0 (start_frame=3), position 0 of slice = zeros.
    y_no_knot = stub._build_y({"first_frame_cond": ffc}, T=4, start_frame=3)
    assert y_no_knot.shape == (B, 20, 4, H, W)
    assert torch.allclose(y_no_knot[:, 0:4, 0:1], torch.zeros(B, 4, 1, H, W))
    assert torch.allclose(y_no_knot[:, 4:, 0:1], torch.zeros(B, C, 1, H, W))

    # With knot at chunk start: mask=1 + VAE=knot at slice position 0
    y_knot = stub._build_y(
        {"first_frame_cond": ffc, "knot_latent_for_chunk_start": knot},
        T=4, start_frame=3,
    )
    assert torch.allclose(y_knot[:, 0:4, 0:1], torch.ones(B, 4, 1, H, W))
    assert torch.allclose(y_knot[:, 4:, 0:1], knot[:, :, 0:1])
    # Other slice positions unchanged (remain zeros from ffc padding + mask)
    assert torch.allclose(y_knot[:, 4:, 1:], y_no_knot[:, 4:, 1:])

    # At iter 0 (start_frame=0): knot key is a no-op; reference wins.
    y_iter0 = stub._build_y(
        {"first_frame_cond": ffc, "knot_latent_for_chunk_start": knot},
        T=4, start_frame=0,
    )
    assert torch.allclose(y_iter0[:, 4:, 0:1], ffc[:, :, 0:1])  # reference, not knot
    assert torch.allclose(y_iter0[:, 0:4, 0:1], torch.ones(B, 4, 1, H, W))
