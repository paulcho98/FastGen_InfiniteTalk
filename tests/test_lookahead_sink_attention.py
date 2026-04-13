"""Unit tests for lookahead-sink constructor params + validation.

These tests avoid constructing the full 14B CausalInfiniteTalkWan network.
They verify:
  - CausalSelfAttention instance has the new instance attrs as defaults
  - Lookahead validation raises ValueError for invalid configs
"""
import pytest


def test_self_attention_default_lookahead_fields():
    """CausalSelfAttention instance has lookahead fields defaulting to disabled."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalSelfAttention

    attn = CausalSelfAttention(
        dim=64, num_heads=4, local_attn_size=10, sink_size=1,
        use_dynamic_rope=True,
    )
    assert attn.lookahead_sink_enabled is False
    assert attn.lookahead_distance == 0


def test_self_attention_lookahead_fields_are_settable():
    """The attention module's lookahead fields can be set post-construction
    (this is how CausalInfiniteTalkWan.__init__ propagates the flag)."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalSelfAttention

    attn = CausalSelfAttention(
        dim=64, num_heads=4, local_attn_size=10, sink_size=1,
        use_dynamic_rope=True,
    )
    attn.lookahead_sink_enabled = True
    attn.lookahead_distance = 4
    assert attn.lookahead_sink_enabled is True
    assert attn.lookahead_distance == 4


def test_lookahead_validation_requires_dynamic_rope():
    """CausalInfiniteTalkWan.__init__ should raise ValueError when
    lookahead_sink_enabled=True AND use_dynamic_rope=False."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    with pytest.raises(ValueError, match="use_dynamic_rope=True"):
        # We call the validation-only code path by catching the error before any
        # heavy weight loading. The validation is checked immediately at the top
        # of __init__ (or as early as practical) — so weight loading won't happen.
        CausalInfiniteTalkWan(
            base_model_paths="",
            infinitetalk_ckpt_path="",
            lora_ckpt_path="",
            lora_rank=0,
            lora_alpha=0,
            apply_lora_adapters=False,
            chunk_size=3,
            total_num_frames=21,
            local_attn_size=10,
            sink_size=1,
            use_dynamic_rope=False,  # bad combination
            net_pred_type="flow",
            schedule_type="rf",
            shift=7.0,
            lookahead_sink_enabled=True,
            lookahead_distance=4,
        )


def test_lookahead_validation_requires_positive_distance():
    """CausalInfiniteTalkWan.__init__ should raise ValueError when
    lookahead_sink_enabled=True AND lookahead_distance < 1."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    with pytest.raises(ValueError, match="lookahead_distance >= 1"):
        CausalInfiniteTalkWan(
            base_model_paths="",
            infinitetalk_ckpt_path="",
            lora_ckpt_path="",
            lora_rank=0,
            lora_alpha=0,
            apply_lora_adapters=False,
            chunk_size=3,
            total_num_frames=21,
            local_attn_size=10,
            sink_size=1,
            use_dynamic_rope=True,
            net_pred_type="flow",
            schedule_type="rf",
            shift=7.0,
            lookahead_sink_enabled=True,
            lookahead_distance=0,  # bad value
        )
