"""Test F2+F3 toggles in scripts/inference/inference_causal.py::run_inference.

Uses MockNet pattern — no 14B model loading. Verifies the inline AR loop
follows the same F2+F3 matrix as CausVidModel._student_sample_loop.
"""
import importlib.util
import os
import sys
from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class MockNoiseScheduler:
    def forward_process(self, x, eps, t):
        return x


class MockNet:
    def __init__(self, total_frames=9, chunk_size=3,
                 model_sink_cache=False, skip_clean_cache=False):
        self.chunk_size = chunk_size
        self.total_num_frames = total_frames
        self.noise_scheduler = MockNoiseScheduler()
        self.calls = []
        self._model_sink_cache = model_sink_cache
        self._skip_clean_cache_pass = skip_clean_cache

    def clear_caches(self):
        pass

    def __call__(self, x, t, **kwargs):
        self.calls.append({
            "t": float(t[0].item()) if t.numel() else 0.0,
            "store_kv": kwargs.get("store_kv", False),
            "apply_anchor": kwargs.get("apply_anchor", True),
            "cur_start_frame": kwargs.get("cur_start_frame", 0),
        })
        return torch.zeros_like(x)


def _load_run_inference():
    spec = importlib.util.spec_from_file_location(
        "inference_causal", str(ROOT / "scripts" / "inference" / "inference_causal.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Disable argparse execution by not calling spec.loader.exec_module on __main__
    # The script uses argparse only in __main__ block, so importing is safe.
    spec.loader.exec_module(mod)
    return mod.run_inference


def _run(model_sink_cache, skip_clean_cache):
    run_inference = _load_run_inference()
    net = MockNet(model_sink_cache=model_sink_cache, skip_clean_cache=skip_clean_cache)
    condition = {"first_frame_cond": torch.zeros(1, 16, 9, 28, 56)}
    run_inference(
        model=net, condition=condition, num_latent_frames=9, chunk_size=3,
        context_noise=0.0, seed=0, device=torch.device("cpu"), dtype=torch.float32,
        anchor_first_frame=True,
    )
    return net.calls


def test_both_off_baseline():
    calls = _run(False, False)
    # 3 chunks × 5 calls (4 denoise + 1 cache pass) = 15
    assert len(calls) == 15
    for chunk_idx in range(3):
        chunk_calls = calls[chunk_idx * 5:(chunk_idx + 1) * 5]
        for c in chunk_calls[:4]:
            assert c["store_kv"] is False
            assert c["apply_anchor"] is True
        assert chunk_calls[4]["store_kv"] is True
        assert chunk_calls[4]["apply_anchor"] is True


def test_f3_on_skips_cache_pass():
    calls = _run(False, True)
    # 3 chunks × 4 calls = 12
    assert len(calls) == 12
    for chunk_idx in range(3):
        chunk_calls = calls[chunk_idx * 4:(chunk_idx + 1) * 4]
        for c in chunk_calls[:3]:
            assert c["store_kv"] is False
        assert chunk_calls[3]["store_kv"] is True
        for c in chunk_calls:
            assert c["apply_anchor"] is True


def test_f2_on_sink_chunk_suppresses_anchor():
    calls = _run(True, False)
    assert len(calls) == 15
    chunk0 = calls[0:5]
    # Chunk 0 sink: last denoise + cache pass have apply_anchor=False
    assert chunk0[3]["apply_anchor"] is False
    assert chunk0[3]["store_kv"] is False
    assert chunk0[4]["apply_anchor"] is False
    assert chunk0[4]["store_kv"] is True
    # Other chunks unaffected
    for chunk_idx in [1, 2]:
        for c in calls[chunk_idx * 5:(chunk_idx + 1) * 5]:
            assert c["apply_anchor"] is True


def test_f2_overrides_f3_for_sink_chunk():
    calls = _run(True, True)
    # Chunk 0: 5 calls (F2 forces cache pass). Chunks 1, 2: 4 calls. Total 13.
    assert len(calls) == 13
    chunk0 = calls[0:5]
    assert chunk0[3]["apply_anchor"] is False
    assert chunk0[3]["store_kv"] is False
    assert chunk0[4]["apply_anchor"] is False
    assert chunk0[4]["store_kv"] is True
    # Chunk 1: F3 active
    chunk1 = calls[5:9]
    assert len(chunk1) == 4
    assert chunk1[3]["store_kv"] is True
    for c in chunk1:
        assert c["apply_anchor"] is True


def test_inference_causal_propagates_apply_input_anchor_f2():
    """F2 on sink chunk's last denoise + cache pass: both anchors off together."""
    from scripts.inference import inference_causal

    class _InfSpy:
        def __init__(self):
            self.chunk_size = 3
            self.total_num_frames = 3
            self._model_sink_cache = True   # F2 active
            self._skip_clean_cache_pass = False
            self._kv_caches = None
            self.calls = []
            from types import SimpleNamespace
            self.noise_scheduler = SimpleNamespace(
                forward_process=lambda x, eps, t: x,
            )

        def __call__(self, x, t, **kwargs):
            self.calls.append(dict(kwargs, **{"x_shape": tuple(x.shape)}))
            return x.clone()

        def clear_caches(self): pass

    spy = _InfSpy()
    B, C, T, H, W = 1, 16, 3, 8, 8
    import torch
    condition = {
        "text_embeds": torch.zeros(B, 512, 4096),
        "clip_features": torch.zeros(B, 257, 1280),
        "audio_emb": torch.zeros(B, 93, 5, 12, 768),
        "first_frame_cond": torch.full((B, C, 21, H, W), 5.0),
    }
    _ = inference_causal.run_inference(
        spy, condition, num_latent_frames=T, chunk_size=3,
        context_noise=0.0, seed=42,
        device="cpu", dtype=torch.float32,
        anchor_first_frame=True,
    )

    denoise_calls = [c for c in spy.calls if not c.get("store_kv")]
    cache_calls = [c for c in spy.calls if c.get("store_kv")]

    assert denoise_calls, "no denoise forwards recorded"
    # Last denoise on chunk 0 with F2: both anchors off
    last = denoise_calls[-1]
    assert last.get("apply_anchor") is False, "F2 last denoise: apply_anchor should be False"
    assert last.get("apply_input_anchor") is False, \
        "F2 last denoise: apply_input_anchor should also be False"

    # F2 cache pass: both off
    assert cache_calls, "expected a cache-store forward"
    assert cache_calls[-1].get("apply_anchor") is False
    assert cache_calls[-1].get("apply_input_anchor") is False
