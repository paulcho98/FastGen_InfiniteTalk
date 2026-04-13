"""Unit tests for stochastic lookahead distance (F1b).

Verifies:
  - Config field defaults (min=0, max=0 = disabled)
  - CausalInfiniteTalkWan.__init__ validates the range at construction time
  - _apply_anchor_config stamps the range onto self.net + blocks
  - Forward-time sampling fires only when self.training=True AND range is set
  - Sampled distance is in [min, max] inclusive
  - Distance is consistent across all blocks within one forward
  - Eval mode (self.training=False) uses the fixed distance, not the range
"""
import random
import pytest


# ---------------- Config field presence ----------------

def test_config_has_stochastic_range_fields():
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    m = InfiniteTalkSFModelConfig()
    assert m.lookahead_distance_min == 0
    assert m.lookahead_distance_max == 0


# ---------------- Network constructor validation ----------------

def test_constructor_rejects_half_set_range():
    """If one of min/max is > 0, the other must be too."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    with pytest.raises(ValueError, match="lookahead_distance_min"):
        CausalInfiniteTalkWan(
            base_model_paths="", infinitetalk_ckpt_path="", lora_ckpt_path="",
            lora_rank=0, lora_alpha=0, apply_lora_adapters=False,
            chunk_size=3, total_num_frames=21,
            local_attn_size=10, sink_size=1,
            use_dynamic_rope=True,
            net_pred_type="flow", schedule_type="rf", shift=7.0,
            lookahead_sink_enabled=True, lookahead_distance=4,
            lookahead_distance_min=3, lookahead_distance_max=0,
        )


def test_constructor_rejects_inverted_range():
    """min > max should fail."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    with pytest.raises(ValueError, match="must be <="):
        CausalInfiniteTalkWan(
            base_model_paths="", infinitetalk_ckpt_path="", lora_ckpt_path="",
            lora_rank=0, lora_alpha=0, apply_lora_adapters=False,
            chunk_size=3, total_num_frames=21,
            local_attn_size=10, sink_size=1,
            use_dynamic_rope=True,
            net_pred_type="flow", schedule_type="rf", shift=7.0,
            lookahead_sink_enabled=True, lookahead_distance=4,
            lookahead_distance_min=5, lookahead_distance_max=2,
        )


# ---------------- _apply_anchor_config stamping ----------------

def _make_stub_model(cfg):
    """Build a stand-in for InfiniteTalkSelfForcingModel with the bound method."""
    class _StubSelfAttn:
        def __init__(self):
            self.lookahead_sink_enabled = False
            self.lookahead_distance = 0

    class _StubBlock:
        def __init__(self):
            self.self_attn = _StubSelfAttn()

    class _StubNet:
        training = False
        _enable_first_frame_anchor = True
        _anchor_eval_only = False

        def __init__(self):
            self.blocks = [_StubBlock(), _StubBlock()]

    class _Model:
        def __init__(self, c):
            self.config = c
            self.net = _StubNet()
            self.teacher = None
            self.fake_score = None

        from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel
        _apply_anchor_config = InfiniteTalkSelfForcingModel._apply_anchor_config

    return _Model(cfg)


class _Cfg:
    def __init__(self, **kw):
        self.student_anchor_eval_only = kw.get("student_anchor_eval_only", False)
        self.fake_score_anchor_eval_only = kw.get("fake_score_anchor_eval_only", False)
        self.teacher_anchor_disabled = kw.get("teacher_anchor_disabled", False)
        self.lookahead_sink_enabled = kw.get("lookahead_sink_enabled", False)
        self.lookahead_distance = kw.get("lookahead_distance", 0)
        self.lookahead_distance_min = kw.get("lookahead_distance_min", 0)
        self.lookahead_distance_max = kw.get("lookahead_distance_max", 0)
        self.model_sink_cache_enabled = kw.get("model_sink_cache_enabled", False)
        self.skip_clean_cache_pass = kw.get("skip_clean_cache_pass", False)


def test_apply_anchor_config_stamps_stochastic_range():
    m = _make_stub_model(_Cfg(
        lookahead_sink_enabled=True,
        lookahead_distance=4,
        lookahead_distance_min=1,
        lookahead_distance_max=5,
    ))
    m._apply_anchor_config()
    assert m.net._lookahead_distance_min == 1
    assert m.net._lookahead_distance_max == 5


def test_apply_anchor_config_stamps_zero_range_by_default():
    m = _make_stub_model(_Cfg(
        lookahead_sink_enabled=True, lookahead_distance=4,
    ))
    m._apply_anchor_config()
    assert m.net._lookahead_distance_min == 0
    assert m.net._lookahead_distance_max == 0


# ---------------- Forward-time sampling behavior ----------------
#
# We can't call the real CausalInfiniteTalkWan.forward without a loaded model,
# so we test the sampling logic as a standalone function by simulating the
# exact code path from CausalInfiniteTalkWan.forward lines 2039-2058.


def _simulate_sampling(net_stub):
    """Reproduce the lookahead-sampling block from CausalInfiniteTalkWan.forward."""
    if (
        net_stub.training
        and getattr(net_stub, "_lookahead_sink_enabled", False)
        and getattr(net_stub, "_lookahead_distance_min", 0) > 0
        and getattr(net_stub, "_lookahead_distance_max", 0) > 0
    ):
        sampled = random.randint(
            net_stub._lookahead_distance_min, net_stub._lookahead_distance_max,
        )
        for block in net_stub.blocks:
            if hasattr(block, "self_attn"):
                block.self_attn.lookahead_distance = sampled
        return sampled
    return None


class _SimNet:
    def __init__(self, training, lookahead_sink, d_min, d_max, n_blocks=3):
        self.training = training
        self._lookahead_sink_enabled = lookahead_sink
        self._lookahead_distance_min = d_min
        self._lookahead_distance_max = d_max
        self.blocks = [type("B", (), {"self_attn": type("A", (), {"lookahead_distance": 999})()})() for _ in range(n_blocks)]


def test_sampling_active_in_training_mode():
    net = _SimNet(training=True, lookahead_sink=True, d_min=1, d_max=5)
    random.seed(0)
    sampled = _simulate_sampling(net)
    assert sampled is not None
    assert 1 <= sampled <= 5
    # All blocks got the same sampled distance
    for block in net.blocks:
        assert block.self_attn.lookahead_distance == sampled


def test_sampling_skipped_in_eval_mode():
    net = _SimNet(training=False, lookahead_sink=True, d_min=1, d_max=5)
    sampled = _simulate_sampling(net)
    assert sampled is None
    # Blocks retain their original value (no overwrite)
    for block in net.blocks:
        assert block.self_attn.lookahead_distance == 999


def test_sampling_skipped_when_range_is_zero():
    net = _SimNet(training=True, lookahead_sink=True, d_min=0, d_max=0)
    sampled = _simulate_sampling(net)
    assert sampled is None


def test_sampling_range_boundary_inclusive():
    net = _SimNet(training=True, lookahead_sink=True, d_min=4, d_max=4)
    # min==max should always sample exactly that value
    for _ in range(20):
        sampled = _simulate_sampling(net)
        assert sampled == 4


def test_sampling_distribution_covers_range():
    """Over many samples, the sampled distance should hit every integer in [min, max]."""
    net = _SimNet(training=True, lookahead_sink=True, d_min=1, d_max=5)
    random.seed(42)
    seen = set()
    for _ in range(200):
        seen.add(_simulate_sampling(net))
    assert seen == {1, 2, 3, 4, 5}, f"Expected all values 1..5, got {sorted(seen)}"
