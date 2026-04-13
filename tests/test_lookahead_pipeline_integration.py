"""Integration test for the F1/F2/F3 pipeline using MockNet + real helper code.

Verifies the full pattern across chunks by:
  1. Running _student_sample_loop with LOOKAHEAD_DEBUG_TRACE=1 and parsing
     [sample_loop] log lines per chunk.
  2. Simulating a multi-chunk attention sequence by calling _apply_window_rope
     with realistic (chunk_idx, query_offset_in_win) values and asserting the
     trigger fires starting at chunk 1 (not chunk 0).

No 14B model loading — all tests use stubs / mocks and complete in seconds.
"""
import io
import re
import torch

from fastgen.utils import logging_utils


class MockNoiseScheduler:
    def x0_to_eps(self, xt, x0, t):
        return xt - x0

    def forward_process(self, x, eps, t):
        return x


class MockNet:
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


def _capture_logs_during_sample_loop(net):
    """Run sample loop with LOOKAHEAD_DEBUG_TRACE=1 and return the captured log
    output as a single string."""
    from fastgen.methods.distribution_matching.causvid import CausVidModel
    import os
    os.environ["LOOKAHEAD_DEBUG_TRACE"] = "1"

    stream = io.StringIO()
    sink_id = logging_utils.logger.add(stream, format="{message}", level="INFO")
    try:
        x = torch.zeros(1, 16, net.total_num_frames, 28, 56)
        condition = {"first_frame_cond": torch.zeros_like(x)}
        t_list = torch.tensor([0.999, 0.955, 0.875, 0.700, 0.0])
        CausVidModel._student_sample_loop(
            net=net, x=x, t_list=t_list, condition=condition,
            student_sample_type="sde",
        )
    finally:
        logging_utils.logger.remove(sink_id)
        os.environ.pop("LOOKAHEAD_DEBUG_TRACE", None)
    return stream.getvalue()


def _parse_sample_loop_lines(log_output):
    """Extract [sample_loop] lines into a list of dicts."""
    pattern = re.compile(
        r"\[sample_loop\]\s+(?:chunk_idx=(\d+)\s+start=(\d+)\s+step=(\d+)\s+"
        r"is_last=(\w+)\s+store_kv=(\w+)\s+apply_anchor=(\w+)\s+f2_active=(\w+)"
        r"|cache_pass\s+chunk_idx=(\d+)\s+start=(\d+)\s+store_kv=(\w+)\s+"
        r"apply_anchor=(\w+)\s+f2_active=(\w+))"
    )
    entries = []
    for m in pattern.finditer(log_output):
        if m.group(1) is not None:
            entries.append({
                "kind": "denoise",
                "chunk_idx": int(m.group(1)),
                "start": int(m.group(2)),
                "step": int(m.group(3)),
                "is_last": m.group(4) == "True",
                "store_kv": m.group(5) == "True",
                "apply_anchor": m.group(6) == "True",
                "f2_active": m.group(7) == "True",
            })
        else:
            entries.append({
                "kind": "cache_pass",
                "chunk_idx": int(m.group(8)),
                "start": int(m.group(9)),
                "store_kv": m.group(10) == "True",
                "apply_anchor": m.group(11) == "True",
                "f2_active": m.group(12) == "True",
            })
    return entries


def test_sample_loop_logs_f2_on_f3_on_match_expected_matrix():
    """Full integration: run sample_loop with F2+F3 on, parse logs, assert the
    exact per-chunk pattern matches the spec matrix."""
    net = MockNet(model_sink_cache=True, skip_clean_cache=True, total_frames=9)
    output = _capture_logs_during_sample_loop(net)
    entries = _parse_sample_loop_lines(output)

    # 3 chunks. Chunk 0 (sink): 4 denoise + 1 cache_pass = 5 entries.
    # Chunks 1, 2 (non-sink): 4 denoise each (F3 active). Total = 5 + 4 + 4 = 13.
    assert len(entries) == 13, f"Expected 13 log entries, got {len(entries)}:\n{output}"

    # Chunk 0 last denoise: apply_anchor=False, store_kv=False, f2_active=True
    c0_last = [e for e in entries if e["kind"] == "denoise" and e["chunk_idx"] == 0 and e["is_last"]]
    assert len(c0_last) == 1
    assert c0_last[0]["apply_anchor"] is False
    assert c0_last[0]["store_kv"] is False
    assert c0_last[0]["f2_active"] is True

    # Chunk 0 cache pass: present, apply_anchor=False
    c0_cache = [e for e in entries if e["kind"] == "cache_pass" and e["chunk_idx"] == 0]
    assert len(c0_cache) == 1
    assert c0_cache[0]["apply_anchor"] is False
    assert c0_cache[0]["store_kv"] is True

    # Chunks 1, 2: no cache_pass, last denoise has store_kv=True (F3)
    for cidx in [1, 2]:
        chunk_cache = [e for e in entries if e["kind"] == "cache_pass" and e["chunk_idx"] == cidx]
        assert len(chunk_cache) == 0, f"F3 should skip cache pass for chunk {cidx}"
        last = [e for e in entries if e["kind"] == "denoise" and e["chunk_idx"] == cidx and e["is_last"]]
        assert last[0]["store_kv"] is True
        assert last[0]["apply_anchor"] is True
        assert last[0]["f2_active"] is False


def test_sample_loop_logs_all_combinations_consistent():
    """Verify the [sample_loop] log content is consistent with the forward calls
    recorded on the MockNet across all 4 (F2, F3) combinations."""
    for model_sink_cache, skip_clean_cache in [(False, False), (False, True), (True, False), (True, True)]:
        net = MockNet(model_sink_cache=model_sink_cache, skip_clean_cache=skip_clean_cache)
        output = _capture_logs_during_sample_loop(net)
        log_entries = _parse_sample_loop_lines(output)

        # The log should have one entry per net.calls entry
        assert len(log_entries) == len(net.calls), (
            f"Log entry count {len(log_entries)} != forward call count "
            f"{len(net.calls)} for (F2={model_sink_cache}, F3={skip_clean_cache})"
        )

        # Each denoise log entry's store_kv/apply_anchor matches the forward call's kwargs
        denoise_entries = [e for e in log_entries if e["kind"] == "denoise"]
        denoise_calls = [c for c in net.calls if c["t"] != net.calls[-1]["t"] or True]  # all calls
        # More robust: pair them by order
        for i, (entry, call) in enumerate(zip(log_entries, net.calls)):
            assert entry["store_kv"] == call["store_kv"], (
                f"Entry {i} store_kv mismatch: log {entry['store_kv']} vs call {call['store_kv']}"
            )
            assert entry["apply_anchor"] == call["apply_anchor"]


def test_apply_window_rope_fires_only_on_chunks_past_zero():
    """Simulate a realistic multi-chunk attention sequence: chunk 0 with no
    cached content (query_offset_in_win=0, k_win=current_only), then chunks
    1+ where cache is populated (query_offset_in_win > 0).

    Verify: lookahead fires EXACTLY on chunks 1+, NEVER on chunk 0."""
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    frame_seqlen = 4
    sink_frames = 1
    chunk_frames = 3
    sink_tokens = sink_frames * frame_seqlen

    # Simulate 4 chunks
    fire_status = []  # for each chunk, did lookahead fire? (= 3 rope calls vs 2)
    for chunk_idx in range(4):
        # Chunk 0: no cache; k_win = current only (3 frames).
        # Chunks 1+: cache has all prior chunks' K plus current chunk.
        if chunk_idx == 0:
            cached_frames = 0
            query_offset = 0
        else:
            cached_frames = chunk_idx * chunk_frames  # prior chunks contributed this many frames
            query_offset = cached_frames * frame_seqlen

        k_win_total_frames = cached_frames + chunk_frames
        k_win = torch.zeros(1, k_win_total_frames * frame_seqlen, 2, 16)
        q = torch.zeros(1, chunk_frames * frame_seqlen, 2, 16)
        grid_sizes = torch.tensor([[chunk_frames, 2, 2]], dtype=torch.long)
        dummy_freqs = (torch.zeros(64, 4, dtype=torch.complex64),) * 3

        calls = []
        def spy(x, grid, freqs, start_frame=0):
            calls.append({"start_frame": start_frame})
            return x

        original_rope = ncm.causal_rope_apply
        ncm.causal_rope_apply = spy
        try:
            _apply_window_rope(
                q=q, k_win=k_win, grid_sizes=grid_sizes, freqs=dummy_freqs,
                frame_seqlen=frame_seqlen, sink_tokens=sink_tokens,
                query_offset_in_win=query_offset,
                lookahead_sink_enabled=True, lookahead_distance=4, use_dynamic_rope=True,
            )
        finally:
            ncm.causal_rope_apply = original_rope
        # 3 calls = lookahead fired (sink+rest+Q). 2 calls = didn't fire (whole-K+Q).
        fire_status.append(len(calls) == 3)

    assert fire_status == [False, True, True, True], (
        f"Expected lookahead to fire only on chunks 1+, got per-chunk {fire_status}"
    )
