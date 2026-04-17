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
