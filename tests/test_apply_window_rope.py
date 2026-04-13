"""Unit tests for _apply_window_rope helper in network_causal.py.

Uses monkey-patched causal_rope_apply to spy on the positions passed to RoPE.
No model loading.
"""
import pytest
import torch


def _make_spy_rope(calls):
    def spy_rope(x, grid, freqs, start_frame=0):
        calls.append({
            "x_shape": tuple(x.shape),
            "start_frame": start_frame,
            "f": int(grid[0, 0]),
        })
        return x
    return spy_rope


def test_lookahead_enabled_splits_rope_positions():
    """When lookahead_sink_enabled=True, sink gets future RoPE position while
    rest keeps natural positions and Q stays at its natural position."""
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    B, frame_seqlen = 1, 4
    sink_frames = 1
    rest_frames = 5  # 4 rolling + 1 current
    F_window = sink_frames + rest_frames

    k_win = torch.zeros(B, F_window * frame_seqlen, 2, 16)
    q = torch.zeros(B, frame_seqlen, 2, 16)
    grid_sizes = torch.tensor([[sink_frames, 2, 2]], dtype=torch.long)
    dummy_freqs = (torch.zeros(64, 4, dtype=torch.complex64),) * 3

    calls = []
    original_rope = ncm.causal_rope_apply
    ncm.causal_rope_apply = _make_spy_rope(calls)
    try:
        _apply_window_rope(
            q=q, k_win=k_win, grid_sizes=grid_sizes, freqs=dummy_freqs,
            frame_seqlen=frame_seqlen, sink_tokens=sink_frames * frame_seqlen,
            query_offset_in_win=5 * frame_seqlen,  # Q at last frame (pos 5)
            lookahead_sink_enabled=True, lookahead_distance=4, use_dynamic_rope=True,
        )
    finally:
        ncm.causal_rope_apply = original_rope

    assert len(calls) == 3, f"Expected 3 rope calls (sink K, rest K, Q), got {len(calls)}"
    sink_call, rest_call, q_call = calls

    # Sink: 1 frame at position F_window - 1 + lookahead_distance = 5 + 4 = 9
    assert sink_call["f"] == sink_frames
    assert sink_call["start_frame"] == 9, f"Expected sink at pos 9, got {sink_call['start_frame']}"

    # Rest: 5 frames at start_frame=sink_frames_n=1 → positions [1, 2, 3, 4, 5]
    assert rest_call["f"] == rest_frames
    assert rest_call["start_frame"] == 1, f"Expected rest at pos 1, got {rest_call['start_frame']}"

    # Q: 1 frame at natural position query_offset_in_win // frame_seqlen = 5
    assert q_call["f"] == 1
    assert q_call["start_frame"] == 5, f"Expected Q at pos 5, got {q_call['start_frame']}"


def test_lookahead_disabled_uses_contiguous_positions():
    """When lookahead_sink_enabled=False, whole k_win is rotated with
    positions [0..F_window-1] and Q at its natural position — existing behavior."""
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    B, frame_seqlen = 1, 4
    k_win = torch.zeros(B, 6 * frame_seqlen, 2, 16)
    q = torch.zeros(B, frame_seqlen, 2, 16)
    grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)
    dummy_freqs = (torch.zeros(64, 4, dtype=torch.complex64),) * 3

    calls = []
    original_rope = ncm.causal_rope_apply
    ncm.causal_rope_apply = _make_spy_rope(calls)
    try:
        _apply_window_rope(
            q=q, k_win=k_win, grid_sizes=grid_sizes, freqs=dummy_freqs,
            frame_seqlen=frame_seqlen, sink_tokens=frame_seqlen,  # sink_tokens=4 (1 frame)
            query_offset_in_win=5 * frame_seqlen,
            lookahead_sink_enabled=False, lookahead_distance=0, use_dynamic_rope=True,
        )
    finally:
        ncm.causal_rope_apply = original_rope

    # Two calls: whole-window K (start_frame=0, f=6), Q (start_frame=5, f=1)
    assert len(calls) == 2
    k_call, q_call = calls
    assert k_call["f"] == 6 and k_call["start_frame"] == 0
    assert q_call["f"] == 1 and q_call["start_frame"] == 5


def test_lookahead_q_position_invariant():
    """Property: toggling lookahead on/off doesn't change Q's start_frame
    for the same inputs."""
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    B, frame_seqlen = 1, 4
    k_win = torch.zeros(B, 6 * frame_seqlen, 2, 16)
    q = torch.zeros(B, frame_seqlen, 2, 16)
    grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)
    dummy_freqs = (torch.zeros(64, 4, dtype=torch.complex64),) * 3

    def get_q_start(enabled):
        calls = []
        original_rope = ncm.causal_rope_apply
        ncm.causal_rope_apply = _make_spy_rope(calls)
        try:
            _apply_window_rope(
                q=q, k_win=k_win, grid_sizes=grid_sizes, freqs=dummy_freqs,
                frame_seqlen=frame_seqlen, sink_tokens=frame_seqlen,
                query_offset_in_win=5 * frame_seqlen,
                lookahead_sink_enabled=enabled, lookahead_distance=4, use_dynamic_rope=True,
            )
        finally:
            ncm.causal_rope_apply = original_rope
        q_call = calls[-1]
        return q_call["start_frame"]

    assert get_q_start(False) == get_q_start(True) == 5


def test_lookahead_noop_when_sink_is_whole_kwin():
    """Chunk 0 edge case: k_win is just the current chunk (no cached sink slab).
    Even if lookahead_sink_enabled=True, the trigger condition is False because
    sink_tokens == k_win.shape[1]. Should fall through to normal dynamic-RoPE path."""
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    B, frame_seqlen = 1, 4
    # k_win has 3 current-chunk frames; sink_tokens also covers whole thing
    k_win = torch.zeros(B, 3 * frame_seqlen, 2, 16)
    q = torch.zeros(B, 3 * frame_seqlen, 2, 16)
    grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)
    dummy_freqs = (torch.zeros(64, 4, dtype=torch.complex64),) * 3

    calls = []
    original_rope = ncm.causal_rope_apply
    ncm.causal_rope_apply = _make_spy_rope(calls)
    try:
        _apply_window_rope(
            q=q, k_win=k_win, grid_sizes=grid_sizes, freqs=dummy_freqs,
            frame_seqlen=frame_seqlen,
            sink_tokens=3 * frame_seqlen,  # == k_win.shape[1], so lookahead shouldn't fire
            query_offset_in_win=0,
            lookahead_sink_enabled=True, lookahead_distance=4, use_dynamic_rope=True,
        )
    finally:
        ncm.causal_rope_apply = original_rope

    # Exactly 2 calls (whole-K, Q) — lookahead's 3-call pattern would indicate the
    # trigger fired when it shouldn't
    assert len(calls) == 2, f"Expected 2 rope calls (lookahead no-op), got {len(calls)}"


def test_static_rope_mode_returns_prerotated():
    """When use_dynamic_rope=False, helper returns pre-rotated tensors unchanged."""
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    static_q = torch.ones(1, 4, 2, 16) * 3.0
    static_k = torch.ones(1, 20, 2, 16) * 7.0
    dummy_grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    dummy_freqs = (torch.zeros(64, 4, dtype=torch.complex64),) * 3

    calls = []
    original_rope = ncm.causal_rope_apply
    ncm.causal_rope_apply = _make_spy_rope(calls)
    try:
        roped_q, roped_k = _apply_window_rope(
            q=torch.zeros(1, 4, 2, 16), k_win=torch.zeros(1, 20, 2, 16),
            grid_sizes=dummy_grid, freqs=dummy_freqs,
            frame_seqlen=4, sink_tokens=4, query_offset_in_win=16,
            lookahead_sink_enabled=False, lookahead_distance=0, use_dynamic_rope=False,
            static_roped_q=static_q, static_k_win=static_k,
        )
    finally:
        ncm.causal_rope_apply = original_rope

    assert len(calls) == 0, "Static-RoPE mode shouldn't call causal_rope_apply"
    assert torch.equal(roped_q, static_q)
    assert torch.equal(roped_k, static_k)


def test_debug_trace_logs_when_env_var_set(monkeypatch):
    """When LOOKAHEAD_DEBUG_TRACE=1, a [lookahead] line is logged via the
    fastgen logger (loguru).

    Injects a temporary loguru sink to intercept messages since loguru's
    sys.stdout sink was bound at import time and bypasses capsys/capfd.
    """
    from loguru import logger as loguru_logger
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    monkeypatch.setenv("LOOKAHEAD_DEBUG_TRACE", "1")

    # Intercept via a loguru sink added just for this test
    messages = []
    sink_id = loguru_logger.add(lambda msg: messages.append(msg), level="INFO")

    B, frame_seqlen = 1, 4
    k_win = torch.zeros(B, 6 * frame_seqlen, 2, 16)
    q = torch.zeros(B, frame_seqlen, 2, 16)
    grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)
    dummy_freqs = (torch.zeros(64, 4, dtype=torch.complex64),) * 3

    original_rope = ncm.causal_rope_apply
    ncm.causal_rope_apply = _make_spy_rope([])
    try:
        _apply_window_rope(
            q=q, k_win=k_win, grid_sizes=grid_sizes, freqs=dummy_freqs,
            frame_seqlen=frame_seqlen, sink_tokens=frame_seqlen,
            query_offset_in_win=5 * frame_seqlen,
            lookahead_sink_enabled=True, lookahead_distance=4, use_dynamic_rope=True,
        )
    finally:
        ncm.causal_rope_apply = original_rope
        loguru_logger.remove(sink_id)

    combined = "".join(str(m) for m in messages)
    assert "[lookahead]" in combined, f"Expected [lookahead] trace. Captured: {combined!r}"


def test_debug_trace_silent_when_env_var_unset(monkeypatch):
    """When LOOKAHEAD_DEBUG_TRACE is unset, no [lookahead] line is printed."""
    from loguru import logger as loguru_logger
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    monkeypatch.delenv("LOOKAHEAD_DEBUG_TRACE", raising=False)

    messages = []
    sink_id = loguru_logger.add(lambda msg: messages.append(msg), level="INFO")

    B, frame_seqlen = 1, 4
    k_win = torch.zeros(B, 6 * frame_seqlen, 2, 16)
    q = torch.zeros(B, frame_seqlen, 2, 16)
    grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)
    dummy_freqs = (torch.zeros(64, 4, dtype=torch.complex64),) * 3

    original_rope = ncm.causal_rope_apply
    ncm.causal_rope_apply = _make_spy_rope([])
    try:
        _apply_window_rope(
            q=q, k_win=k_win, grid_sizes=grid_sizes, freqs=dummy_freqs,
            frame_seqlen=frame_seqlen, sink_tokens=frame_seqlen,
            query_offset_in_win=5 * frame_seqlen,
            lookahead_sink_enabled=True, lookahead_distance=4, use_dynamic_rope=True,
        )
    finally:
        ncm.causal_rope_apply = original_rope
        loguru_logger.remove(sink_id)

    combined = "".join(str(m) for m in messages)
    assert "[lookahead]" not in combined, f"Expected no [lookahead] trace. Captured: {combined!r}"
