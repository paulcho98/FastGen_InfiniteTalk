from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig


def test_new_config_fields_default():
    m = InfiniteTalkSFModelConfig()
    assert m.lookahead_sink_enabled is False
    assert m.lookahead_distance == 0
    assert m.model_sink_cache_enabled is False
    assert m.skip_clean_cache_pass is False


def test_new_config_fields_settable():
    m = InfiniteTalkSFModelConfig()
    m.lookahead_sink_enabled = True
    m.lookahead_distance = 4
    m.model_sink_cache_enabled = True
    m.skip_clean_cache_pass = True
    assert m.lookahead_sink_enabled is True
    assert m.lookahead_distance == 4
    assert m.model_sink_cache_enabled is True
    assert m.skip_clean_cache_pass is True
