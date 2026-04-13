#!/usr/bin/env python3
"""Standalone verification of the lookahead + F2 + F3 pipeline.

Runs `_student_sample_loop` with a MockNet for each (F2, F3) combination and
prints the LOOKAHEAD_DEBUG_TRACE log output. Also exercises
`_apply_window_rope` across a 4-chunk simulated sequence to verify the
trigger fires exactly on chunks past zero.

Usage:
    python scripts/verify_lookahead_logic.py

Exits 0 on success, 1 on any mismatch. Useful as a final sanity check
before kicking off real training.
"""
import io
import os
import sys
import torch

# Ensure the project root is on sys.path so local imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastgen.utils import logging_utils
from fastgen.methods.distribution_matching.causvid import CausVidModel


class MockNoiseScheduler:
    def x0_to_eps(self, xt, x0, t):
        return xt - x0

    def forward_process(self, x, eps, t):
        return x


class MockNet:
    def __init__(self, model_sink_cache=False, skip_clean_cache=False):
        self.chunk_size = 3
        self.total_num_frames = 9
        self.noise_scheduler = MockNoiseScheduler()
        self._kv_caches = None
        self.calls = []
        self._model_sink_cache = model_sink_cache
        self._skip_clean_cache_pass = skip_clean_cache
        self.vae = None

    def clear_caches(self):
        self._kv_caches = None

    def __call__(self, x, t, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros_like(x)


def run_combo(f2, f3):
    net = MockNet(model_sink_cache=f2, skip_clean_cache=f3)
    os.environ["LOOKAHEAD_DEBUG_TRACE"] = "1"

    stream = io.StringIO()
    sink_id = logging_utils.logger.add(stream, format="{message}", level="INFO")
    try:
        x = torch.zeros(1, 16, 9, 28, 56)
        condition = {"first_frame_cond": torch.zeros_like(x)}
        t_list = torch.tensor([0.999, 0.955, 0.875, 0.700, 0.0])
        CausVidModel._student_sample_loop(
            net=net, x=x, t_list=t_list, condition=condition,
            student_sample_type="sde",
        )
    finally:
        logging_utils.logger.remove(sink_id)
        os.environ.pop("LOOKAHEAD_DEBUG_TRACE", None)
    return stream.getvalue(), net.calls


def main():
    print("=" * 70)
    print("LOOKAHEAD + F2 + F3 PIPELINE VERIFICATION")
    print("=" * 70)

    all_ok = True

    for f2, f3 in [(False, False), (False, True), (True, False), (True, True)]:
        print(f"\n--- F2={f2}, F3={f3} ---")
        log, calls = run_combo(f2, f3)
        # Expected call counts
        # F2=off,F3=off: 3 chunks x 5 = 15
        # F2=off,F3=on:  3 chunks x 4 = 12
        # F2=on, F3=off: 3 chunks x 5 = 15
        # F2=on, F3=on:  chunk 0: 5, chunks 1,2: 4 each = 13
        expected = {(False, False): 15, (False, True): 12, (True, False): 15, (True, True): 13}
        exp = expected[(f2, f3)]
        ok = len(calls) == exp
        status = "OK" if ok else f"FAIL: expected {exp} calls, got {len(calls)}"
        print(f"  Forward calls: {len(calls)} ({status})")
        all_ok = all_ok and ok

        # Show a sample of the log
        lines = [l for l in log.splitlines() if "[sample_loop]" in l]
        print(f"  [sample_loop] lines: {len(lines)}")
        for l in lines[:3]:
            print(f"    {l}")
        if len(lines) > 3:
            print(f"    ... ({len(lines) - 3} more)")

    # Verify the multi-chunk lookahead trigger
    print("\n--- Multi-chunk lookahead trigger check ---")
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    frame_seqlen = 4
    sink_frames = 1
    chunk_frames = 3
    sink_tokens = sink_frames * frame_seqlen
    fire_status = []

    for chunk_idx in range(4):
        if chunk_idx == 0:
            cached_frames = 0
            query_offset = 0
        else:
            cached_frames = chunk_idx * chunk_frames
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
        fired = len(calls) == 3
        fire_status.append(fired)
        print(f"  Chunk {chunk_idx}: lookahead {'FIRED' if fired else 'did NOT fire'} "
              f"(query_offset_in_win={query_offset})")

    expected_fire = [False, True, True, True]
    if fire_status == expected_fire:
        print("  Trigger pattern OK")
    else:
        print(f"  FAIL: expected {expected_fire}, got {fire_status}")
        all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("ALL VERIFICATION CHECKS PASSED")
        sys.exit(0)
    else:
        print("VERIFICATION FAILED -- see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
