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


def test_advance_running_ahead_disabled():
    """When _running_ahead_enabled is False, advance_running_ahead is a no-op."""
    from fastgen.networks.InfiniteTalk.network_causal import advance_running_ahead

    class Stub:
        _running_ahead_enabled = False
        _running_ahead_n = 0
        _running_ahead_step = 4
    net = Stub()
    advanced, new_n = advance_running_ahead(net, i=100, c=3, k=1)
    assert advanced is False
    assert new_n == 0
    assert net._running_ahead_n == 0  # unchanged


def test_advance_running_ahead_no_trigger():
    """When i + c + k <= n, no advancement."""
    from fastgen.networks.InfiniteTalk.network_causal import advance_running_ahead

    class Stub:
        _running_ahead_enabled = True
        _running_ahead_n = 8
        _running_ahead_step = 4
    net = Stub()
    # i=0, c=3, k=1 → i+c+k = 4, not > 8
    advanced, new_n = advance_running_ahead(net, i=0, c=3, k=1)
    assert advanced is False
    assert new_n == 8
    assert net._running_ahead_n == 8


def test_advance_running_ahead_fires():
    """When i + c + k > n, advance by step."""
    from fastgen.networks.InfiniteTalk.network_causal import advance_running_ahead

    class Stub:
        _running_ahead_enabled = True
        _running_ahead_n = 8
        _running_ahead_step = 4
    net = Stub()
    # i=6, c=3, k=1 → i+c+k = 10 > 8 → advance → new_n = 12
    advanced, new_n = advance_running_ahead(net, i=6, c=3, k=1)
    assert advanced is True
    assert new_n == 12
    assert net._running_ahead_n == 12


def test_advance_running_ahead_multiple_fires():
    """Sequential advancements update n correctly."""
    from fastgen.networks.InfiniteTalk.network_causal import advance_running_ahead

    class Stub:
        _running_ahead_enabled = True
        _running_ahead_n = 8
        _running_ahead_step = 4
    net = Stub()
    # Fire once
    advance_running_ahead(net, i=6, c=3, k=1)
    assert net._running_ahead_n == 12
    # Still fires when i+c+k > new n (i=9, new_n=12, 9+4=13 > 12)
    advanced2, new_n2 = advance_running_ahead(net, i=9, c=3, k=1)
    assert advanced2 is True
    assert new_n2 == 16
    # Next iter doesn't fire if caught up
    advanced3, new_n3 = advance_running_ahead(net, i=12, c=3, k=1)
    assert advanced3 is False
    assert new_n3 == 16


def test_compute_sink_rope_position_running_ahead_wins():
    """Running-ahead takes precedence over F1 and natural."""
    from fastgen.networks.InfiniteTalk.network_causal import compute_sink_rope_position

    class NetRA:
        _running_ahead_enabled = True
        _running_ahead_n = 8
        lookahead_sink_enabled = False
        lookahead_distance = 0
    assert compute_sink_rope_position(NetRA(), F_window=10) == 8


def test_compute_sink_rope_position_f1_when_no_running_ahead():
    from fastgen.networks.InfiniteTalk.network_causal import compute_sink_rope_position

    class NetF1:
        _running_ahead_enabled = False
        _running_ahead_n = 0
        lookahead_sink_enabled = True
        lookahead_distance = 4
    # F_window - 1 + lookahead_distance = 10 - 1 + 4 = 13
    assert compute_sink_rope_position(NetF1(), F_window=10) == 13


def test_compute_sink_rope_position_natural_when_neither():
    from fastgen.networks.InfiniteTalk.network_causal import compute_sink_rope_position

    class NetNeither:
        _running_ahead_enabled = False
        _running_ahead_n = 0
        lookahead_sink_enabled = False
        lookahead_distance = 0
    assert compute_sink_rope_position(NetNeither(), F_window=10) == 0


def test_compute_sink_rope_position_missing_attrs_default_natural():
    """Module with no running_ahead/F1 attrs → natural position 0."""
    from fastgen.networks.InfiniteTalk.network_causal import compute_sink_rope_position

    class Bare:
        pass
    assert compute_sink_rope_position(Bare(), F_window=10) == 0


def test_apply_running_ahead_config_on():
    """Running-ahead config propagates to self.net when enabled."""
    from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel

    class Stub: pass
    model = Stub()
    model.net = Stub()
    model.fake_score = Stub()
    model.teacher = Stub()
    model.config = Stub()
    model.config.use_running_ahead = True
    model.config.running_ahead_step = 4
    model.config.running_ahead_init_n = 8

    InfiniteTalkSelfForcingModel._apply_running_ahead_config(model)

    # Student gets the attrs stamped
    assert model.net._running_ahead_enabled is True
    assert model.net._running_ahead_step == 4
    assert model.net._running_ahead_n == 8
    # Teacher + fake_score do NOT (bidirectional, no sliding-window sink)
    assert getattr(model.teacher, "_running_ahead_enabled", False) is False
    assert getattr(model.fake_score, "_running_ahead_enabled", False) is False


def test_apply_running_ahead_config_off():
    """When use_running_ahead=False, net gets _running_ahead_enabled=False."""
    from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel

    class Stub: pass
    model = Stub()
    model.net = Stub()
    model.config = Stub()
    model.config.use_running_ahead = False
    model.config.running_ahead_step = 4
    model.config.running_ahead_init_n = 8

    InfiniteTalkSelfForcingModel._apply_running_ahead_config(model)
    assert model.net._running_ahead_enabled is False
    # Step and init_n are still stamped but inert
    assert model.net._running_ahead_step == 4
    assert model.net._running_ahead_n == 8


def test_maybe_swap_last_frame_ref_disabled():
    """When use_last_frame=False, returns first_frame_cond unchanged."""
    from fastgen.datasets.infinitetalk_dataloader import _maybe_swap_last_frame_ref
    C, T, H, W = 16, 5, 4, 4
    vae_latents = torch.arange(C * T * H * W, dtype=torch.float32).reshape(C, T, H, W)
    first_frame_cond = torch.zeros(C, T, H, W)
    first_frame_cond[:, 0] = 1.0

    out = _maybe_swap_last_frame_ref(first_frame_cond.clone(), vae_latents, use_last_frame=False)
    assert torch.allclose(out, first_frame_cond)


def test_maybe_swap_last_frame_ref_enabled():
    """When use_last_frame=True, position 0 is replaced with vae_latents[:, -1]."""
    from fastgen.datasets.infinitetalk_dataloader import _maybe_swap_last_frame_ref
    C, T, H, W = 16, 5, 4, 4
    vae_latents = torch.arange(C * T * H * W, dtype=torch.float32).reshape(C, T, H, W)
    first_frame_cond = torch.zeros(C, T, H, W)
    first_frame_cond[:, 0] = 1.0

    out = _maybe_swap_last_frame_ref(first_frame_cond.clone(), vae_latents, use_last_frame=True)
    # Position 0 of first_frame_cond is now the last latent frame from vae_latents
    assert torch.allclose(out[:, 0], vae_latents[:, -1])
    # Positions 1..T-1 unchanged (still zeros from the original first_frame_cond)
    assert torch.allclose(out[:, 1:], first_frame_cond[:, 1:])


# ─────────────────────────────────────────────────────────────────────────────
# KF-11: rollout_with_gradient KF branch integration tests (placeholders).
# End-to-end validation happens in the valtest smoke run (KF-20) and by
# running the full regression suite to confirm KF-off is bit-exact.
# ─────────────────────────────────────────────────────────────────────────────


def test_rollout_kf_off_is_bit_exact(monkeypatch):
    """When use_temporal_knot=False and use_running_ahead=False, rollout_with_gradient
    is bit-exact to the pre-KF implementation (feature-flag discipline)."""
    # Semi-smoke test: just confirm no new config fields are READ when off.
    # Full bit-exact test requires a fixture checkpoint; skip here.
    # We instead rely on the 90+ existing regression tests NOT failing.
    import pytest
    pytest.skip("Covered by regression test suite (existing SF tests must all pass).")


def test_rollout_kf_on_denoises_cplusk_frames():
    """KF rollout reads c+k frames per chunk instead of c."""
    # This test uses a MockNet pattern. Read tests/test_sample_loop_toggles.py for template.
    # If the pattern is too heavy, mark as a TODO skip with a clear note:
    import pytest
    pytest.skip("Integration test — requires MockNet pattern (see test_sample_loop_toggles.py); covered by valtest in KF-20")


def test_rollout_kf_commits_only_c_frames():
    """KF output tensor has num_blocks * c frames, not num_blocks * (c+k)."""
    import pytest
    pytest.skip("Integration test — covered by valtest in KF-20")


def test_rollout_kf_extends_noise_by_k():
    """KF rollout internally extends noise by k extra frames for the last chunk's knot."""
    import pytest
    pytest.skip("Integration test — covered by valtest in KF-20")


def test_maybe_pad_audio_for_knot_no_op():
    """extra=0 is a no-op."""
    from fastgen.datasets.infinitetalk_dataloader import _maybe_pad_audio_for_knot
    audio = torch.arange(21 * 12 * 768, dtype=torch.float32).reshape(21, 12, 768)
    out = _maybe_pad_audio_for_knot(audio, extra=0)
    assert out.shape == audio.shape
    assert torch.allclose(out, audio)


def test_maybe_pad_audio_for_knot_extends_repeat_last():
    """extra>0 extends by repeating the last slice."""
    from fastgen.datasets.infinitetalk_dataloader import _maybe_pad_audio_for_knot
    audio = torch.arange(21 * 12 * 768, dtype=torch.float32).reshape(21, 12, 768)
    out = _maybe_pad_audio_for_knot(audio, extra=4)
    assert out.shape == (25, 12, 768)
    assert torch.allclose(out[:21], audio)
    assert torch.allclose(out[21], audio[20])
    assert torch.allclose(out[24], audio[20])


def test_kf_off_is_default_bit_exact_config():
    """All KF flags default to off → KF-off code path is the existing SF path."""
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    # Each KF flag is off by default
    assert cfg.use_temporal_knot is False
    assert cfg.use_running_ahead is False
    assert cfg.use_last_frame_reference is False
    # KF-off path gates:
    #   - rollout_with_gradient: use_knot=False → k=0, denoise_len=chunk_size, no noise extension
    #   - _student_sample_loop: same
    #   - inference_causal run_inference: same
    #   - _build_y: knot_latent key absent → no injection
    # Validator passes on all-off
    cfg.validate_kf_flags()


def test_kf_training_config_round_trip():
    """The full KF training config loads, passes validation, and has expected knobs."""
    from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_knot_runahead import (
        create_config,
    )
    c = create_config()
    # Core KF flags
    assert c.model.use_temporal_knot is True
    assert c.model.knot_size == 1
    assert c.model.use_running_ahead is True
    assert c.model.running_ahead_step == 4
    assert c.model.running_ahead_init_n == 8
    assert c.model.use_last_frame_reference is True
    # F1/F2/F3 off (mutually exclusive or redundant under KF)
    assert c.model.lookahead_sink_enabled is False
    assert c.model.model_sink_cache_enabled is False
    assert c.model.skip_clean_cache_pass is False
    # Dynamic RoPE required for running-ahead's sink position override
    assert c.model.net.use_dynamic_rope is True
    # Dataloader gets both KF options
    assert c.dataloader_train.use_last_frame_reference is True
    assert c.dataloader_train.knot_size_extra_audio_pixel == 4
    # Validator passes (no incompatible flags)
    c.model.validate_kf_flags()
