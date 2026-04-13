"""Verify _apply_anchor_config stamps the new F1/F2/F3 config fields onto
self.net and block.self_attn. Uses stub objects — no 14B model loading."""


class _StubSelfAttn:
    """Stand-in for a CausalSelfAttention module — just a dict-like attr holder."""
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
    """Stand-in for InfiniteTalkSelfForcingModel exposing only cfg + net + the
    method under test (bound from the real class)."""
    def __init__(self, cfg):
        self.config = cfg
        self.net = _StubNet()
        self.teacher = None
        self.fake_score = None

    from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel
    _apply_anchor_config = InfiniteTalkSelfForcingModel._apply_anchor_config


class _Cfg:
    def __init__(self, **kw):
        self.student_anchor_eval_only = kw.get("student_anchor_eval_only", False)
        self.fake_score_anchor_eval_only = kw.get("fake_score_anchor_eval_only", False)
        self.teacher_anchor_disabled = kw.get("teacher_anchor_disabled", False)
        self.lookahead_sink_enabled = kw.get("lookahead_sink_enabled", False)
        self.lookahead_distance = kw.get("lookahead_distance", 0)
        self.model_sink_cache_enabled = kw.get("model_sink_cache_enabled", False)
        self.skip_clean_cache_pass = kw.get("skip_clean_cache_pass", False)


def test_new_fields_stamped_onto_net_when_enabled():
    m = _Model(_Cfg(
        lookahead_sink_enabled=True,
        lookahead_distance=4,
        model_sink_cache_enabled=True,
        skip_clean_cache_pass=True,
    ))
    m._apply_anchor_config()
    assert m.net._lookahead_sink_enabled is True
    assert m.net._lookahead_distance == 4
    assert m.net._model_sink_cache is True
    assert m.net._skip_clean_cache_pass is True


def test_new_fields_default_off():
    m = _Model(_Cfg())
    m._apply_anchor_config()
    assert m.net._lookahead_sink_enabled is False
    assert m.net._lookahead_distance == 0
    assert m.net._model_sink_cache is False
    assert m.net._skip_clean_cache_pass is False


def test_lookahead_propagates_to_each_block_self_attn():
    m = _Model(_Cfg(lookahead_sink_enabled=True, lookahead_distance=6))
    m._apply_anchor_config()
    for block in m.net.blocks:
        assert block.self_attn.lookahead_sink_enabled is True
        assert block.self_attn.lookahead_distance == 6


def test_lookahead_disabled_clears_block_self_attn():
    """When lookahead is OFF in config, blocks' self_attn attrs are set to off."""
    m = _Model(_Cfg(lookahead_sink_enabled=False, lookahead_distance=0))
    # Pre-set to ON to verify the config overrides
    for block in m.net.blocks:
        block.self_attn.lookahead_sink_enabled = True
        block.self_attn.lookahead_distance = 99
    m._apply_anchor_config()
    for block in m.net.blocks:
        assert block.self_attn.lookahead_sink_enabled is False
        assert block.self_attn.lookahead_distance == 0
