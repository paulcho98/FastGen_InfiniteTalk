"""Unit tests for F2+F3 toggle logic in CausVidModel._student_sample_loop.

Uses a MockNet that records every forward call; no 14B model loading.
Tests the full F2+F3 matrix from the plan.
"""
import torch


class MockNoiseScheduler:
    def x0_to_eps(self, xt, x0, t):
        return xt - x0

    def forward_process(self, x, eps, t):
        return x  # identity — we're testing control flow, not diffusion math


class MockNet:
    """Records every forward call's kwargs for later assertion."""
    def __init__(self, chunk_size=3, total_frames=9,
                 model_sink_cache=False, skip_clean_cache=False):
        self.chunk_size = chunk_size
        self.total_num_frames = total_frames
        self.noise_scheduler = MockNoiseScheduler()
        self._kv_caches = None
        self.calls = []
        self._model_sink_cache = model_sink_cache
        self._skip_clean_cache_pass = skip_clean_cache
        self.vae = None

    def clear_caches(self):
        self._kv_caches = None

    def __call__(self, x, t, **kwargs):
        self.calls.append({
            "t": float(t[0].item()) if t.numel() else 0.0,
            "store_kv": kwargs.get("store_kv", False),
            "apply_anchor": kwargs.get("apply_anchor", True),
            "cur_start_frame": kwargs.get("cur_start_frame", 0),
        })
        return torch.zeros_like(x)


def _run_loop(mock_net, condition=None):
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    if condition is None:
        # Include first_frame_cond so F2's manual-anchor-for-display path runs.
        condition = {"first_frame_cond": torch.zeros(1, 16, 9, 28, 56)}

    x = torch.zeros(1, 16, mock_net.total_num_frames, 28, 56)
    t_list = torch.tensor([0.999, 0.955, 0.875, 0.700, 0.0])
    return CausVidModel._student_sample_loop(
        net=mock_net, x=x, t_list=t_list,
        condition=condition, student_sample_type="sde",
    )


def test_both_off_preserves_current_behavior():
    """F2 off, F3 off: 4 denoise (store_kv=False, apply_anchor=True) + 1 cache
    pass (store_kv=True, apply_anchor=True), per chunk."""
    net = MockNet(model_sink_cache=False, skip_clean_cache=False)
    _run_loop(net)
    # 3 chunks x 5 calls each = 15
    assert len(net.calls) == 15
    for chunk_idx in range(3):
        chunk_calls = net.calls[chunk_idx * 5:(chunk_idx + 1) * 5]
        for c in chunk_calls[:4]:  # 4 denoise
            assert c["store_kv"] is False
            assert c["apply_anchor"] is True
        cache = chunk_calls[4]
        assert cache["store_kv"] is True
        assert cache["apply_anchor"] is True


def test_f3_on_skips_cache_pass():
    """F3 on (F2 off): 4 denoise per chunk, last one has store_kv=True, no cache pass."""
    net = MockNet(model_sink_cache=False, skip_clean_cache=True)
    _run_loop(net)
    # 3 chunks x 4 calls = 12
    assert len(net.calls) == 12
    for chunk_idx in range(3):
        chunk_calls = net.calls[chunk_idx * 4:(chunk_idx + 1) * 4]
        for c in chunk_calls[:3]:
            assert c["store_kv"] is False
        # Last denoise writes K/V
        assert chunk_calls[3]["store_kv"] is True
        for c in chunk_calls:
            assert c["apply_anchor"] is True


def test_f2_on_suppresses_anchor_on_sink_chunk_only():
    """F2 on (F3 off): sink chunk (chunk 0) last denoise + cache pass have
    apply_anchor=False. Other chunks unchanged."""
    net = MockNet(model_sink_cache=True, skip_clean_cache=False)
    _run_loop(net)
    assert len(net.calls) == 15

    # Chunk 0 (sink chunk): last denoise (idx 3) and cache pass (idx 4) both apply_anchor=False
    chunk0 = net.calls[0:5]
    assert chunk0[0]["apply_anchor"] is True
    assert chunk0[1]["apply_anchor"] is True
    assert chunk0[2]["apply_anchor"] is True
    assert chunk0[3]["apply_anchor"] is False  # last denoise
    assert chunk0[3]["store_kv"] is False       # separate cache pass still runs
    assert chunk0[4]["apply_anchor"] is False   # cache pass
    assert chunk0[4]["store_kv"] is True

    # Chunks 1, 2: unchanged (cur_start_frame > 0 → not sink chunk → f2_active=False)
    for chunk_idx in [1, 2]:
        for c in net.calls[chunk_idx * 5:(chunk_idx + 1) * 5]:
            assert c["apply_anchor"] is True


def test_f2_and_f3_on_f2_overrides_for_sink_chunk():
    """F2 on + F3 on: chunk 0 keeps separate cache pass (F2 overrides F3),
    chunks > 0 skip cache pass."""
    net = MockNet(model_sink_cache=True, skip_clean_cache=True)
    _run_loop(net)
    # Chunk 0: 5 calls (F2 forces cache pass). Chunks 1, 2: 4 calls each. Total 13.
    assert len(net.calls) == 13

    # Chunk 0: last denoise apply_anchor=False, store_kv=False; cache pass apply_anchor=False, store_kv=True
    chunk0 = net.calls[0:5]
    assert chunk0[3]["apply_anchor"] is False
    assert chunk0[3]["store_kv"] is False
    assert chunk0[4]["apply_anchor"] is False
    assert chunk0[4]["store_kv"] is True

    # Chunk 1: 4 denoise, last one has store_kv=True (F3)
    chunk1 = net.calls[5:9]
    assert len(chunk1) == 4
    for c in chunk1[:3]:
        assert c["store_kv"] is False
    assert chunk1[3]["store_kv"] is True
    for c in chunk1:
        assert c["apply_anchor"] is True


def test_f2_display_output_gets_manual_anchor(monkeypatch):
    """When F2 fires on chunk 0, x[:, :, 0:3] should have frame 0 replaced with
    condition['first_frame_cond'][:, :, 0:1]."""
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    net = MockNet(model_sink_cache=True, skip_clean_cache=False)
    # Identifiable first-frame-cond values
    ref = torch.full((1, 16, 9, 28, 56), -4.2)
    condition = {"first_frame_cond": ref}
    x = torch.zeros(1, 16, 9, 28, 56)
    t_list = torch.tensor([0.999, 0.955, 0.875, 0.700, 0.0])

    result = CausVidModel._student_sample_loop(
        net=net, x=x, t_list=t_list, condition=condition, student_sample_type="sde",
    )

    # Chunk 0's frame 0 of the final output should equal ref's frame 0 (the manual anchor)
    assert torch.equal(result[:, :, 0:1], ref[:, :, 0:1]), (
        "F2's manual display anchor didn't land on the returned tensor"
    )


def test_debug_trace_logs_when_env_var_set(monkeypatch):
    """LOOKAHEAD_DEBUG_TRACE=1 should emit [sample_loop] lines per (chunk, step)."""
    from fastgen.utils import logging_utils

    monkeypatch.setenv("LOOKAHEAD_DEBUG_TRACE", "1")

    # Capture loguru output
    import io
    stream = io.StringIO()
    sink_id = logging_utils.logger.add(stream, format="{message}", level="INFO")

    try:
        net = MockNet(model_sink_cache=True, skip_clean_cache=True)
        _run_loop(net)
    finally:
        logging_utils.logger.remove(sink_id)

    output = stream.getvalue()
    assert "[sample_loop]" in output, f"Expected [sample_loop] trace, got:\n{output}"


def test_debug_trace_silent_when_env_var_unset(monkeypatch):
    """Without the env var, no [sample_loop] lines should appear."""
    from fastgen.utils import logging_utils

    monkeypatch.delenv("LOOKAHEAD_DEBUG_TRACE", raising=False)

    import io
    stream = io.StringIO()
    sink_id = logging_utils.logger.add(stream, format="{message}", level="INFO")

    try:
        net = MockNet(model_sink_cache=False, skip_clean_cache=False)
        _run_loop(net)
    finally:
        logging_utils.logger.remove(sink_id)

    output = stream.getvalue()
    assert "[sample_loop]" not in output
