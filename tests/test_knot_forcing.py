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
