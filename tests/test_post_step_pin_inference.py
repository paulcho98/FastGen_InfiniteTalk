"""Tests for post-scheduler-step frame-0 pinning in validation/inference paths."""
import torch
from types import SimpleNamespace
import pytest


class _SpyNet:
    """Minimal mock causal network that records forward kwargs and returns identity x0."""
    def __init__(self, chunk_size=3, total_num_frames=21):
        self.chunk_size = chunk_size
        self.total_num_frames = total_num_frames
        self._enable_first_frame_anchor = True
        self._anchor_eval_only = False
        self._model_sink_cache = False
        self._skip_clean_cache_pass = False
        self.training = False
        self.forward_calls = []
        self.noise_scheduler = SimpleNamespace(
            forward_process=lambda x, eps, t: x * (1 - t.view(-1, 1, 1, 1, 1)) + eps * t.view(-1, 1, 1, 1, 1),
            x0_to_eps=lambda xt, x0, t: torch.randn_like(xt),
            max_sigma=1.0,
        )
        self._kv_caches = None

    def __call__(self, x, t, **kwargs):
        self.forward_calls.append({
            "x": x.detach().clone(),
            "t": t.detach().clone(),
            "apply_anchor": kwargs.get("apply_anchor", True),
            "apply_input_anchor": kwargs.get("apply_input_anchor", True),
            "store_kv": kwargs.get("store_kv", False),
            "cur_start_frame": kwargs.get("cur_start_frame", 0),
        })
        # Return identity x0 (whatever x came in is treated as x0); this makes
        # it easy to trace what gets passed where
        return x.clone()

    def clear_caches(self):
        pass


def test_sample_loop_post_step_pins_frame_0_on_chunk0():
    """After forward_process, frame 0 of the next-step input must be pinned to ffc."""
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    net = _SpyNet(chunk_size=3, total_num_frames=6)
    B, C, T, H, W = 1, 16, 6, 8, 8
    x = torch.randn(B, C, T, H, W)
    ffc = torch.full((B, C, 21, H, W), 7.0)
    condition = {
        "text_embeds": torch.zeros(B, 512, 4096),
        "clip_features": torch.zeros(B, 257, 1280),
        "audio_emb": torch.zeros(B, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t_list = torch.tensor([0.9, 0.5, 0.0])  # 2 denoise steps + 0

    out = CausVidModel._student_sample_loop(
        net=net, x=x, t_list=t_list, condition=condition,
        student_sample_type="sde", context_noise=0.0,
    )

    # With post-step pinning: forward calls on chunk 0 at steps 1+ (after the
    # first forward_process) should see x[:, :, 0] == 7.0.
    chunk0_forwards = [c for c in net.forward_calls if c["cur_start_frame"] == 0 and not c["store_kv"]]
    assert len(chunk0_forwards) >= 2, "Expected at least 2 denoise forwards on chunk 0"
    # Post-step pin affects step 1's input (the second forward)
    assert chunk0_forwards[1]["x"][:, :, 0].eq(7.0).all(), \
        "After post-step pin, step 1's input frame 0 should equal first_frame_cond"


def test_sample_loop_no_post_step_pin_on_nonzero_chunk():
    """Non-zero chunks (cur_start_frame != 0) should NOT get post-step pinning."""
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    net = _SpyNet(chunk_size=3, total_num_frames=6)
    B, C, T, H, W = 1, 16, 6, 8, 8
    x = torch.full((B, C, T, H, W), 1.0)   # nonzero start values for all frames
    ffc = torch.full((B, C, 21, H, W), 7.0)
    condition = {
        "text_embeds": torch.zeros(B, 512, 4096),
        "clip_features": torch.zeros(B, 257, 1280),
        "audio_emb": torch.zeros(B, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t_list = torch.tensor([0.9, 0.5, 0.0])

    _ = CausVidModel._student_sample_loop(
        net=net, x=x, t_list=t_list, condition=condition,
        student_sample_type="sde", context_noise=0.0,
    )

    # Chunk 1 forwards: cur_start_frame=3. Frame 0 of local chunk is NOT the
    # global reference frame, so it should NOT be pinned to ffc[:, :, 0].
    chunk1_forwards = [c for c in net.forward_calls if c["cur_start_frame"] == 3 and not c["store_kv"]]
    if len(chunk1_forwards) >= 2:
        # Second forward on chunk 1 — frame 0 should NOT be 7.0 (the ffc value)
        assert not chunk1_forwards[1]["x"][:, :, 0].eq(7.0).all(), \
            "Non-zero chunk should NOT be post-step pinned to ffc[:, :, 0]"


def test_sample_loop_f2_propagates_both_anchors():
    """With F2 active, both apply_anchor and apply_input_anchor must be False
    on chunk 0's last denoise step and cache-store pass."""
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    net = _SpyNet(chunk_size=3, total_num_frames=3)
    net._model_sink_cache = True   # F2 enabled
    B, C, T, H, W = 1, 16, 3, 8, 8
    x = torch.randn(B, C, T, H, W)
    ffc = torch.full((B, C, 21, H, W), 7.0)
    condition = {
        "text_embeds": torch.zeros(B, 512, 4096),
        "clip_features": torch.zeros(B, 257, 1280),
        "audio_emb": torch.zeros(B, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t_list = torch.tensor([0.9, 0.5, 0.0])  # 2 denoise steps

    _ = CausVidModel._student_sample_loop(
        net=net, x=x, t_list=t_list, condition=condition,
        student_sample_type="sde", context_noise=0.0,
    )

    chunk0_denoise = [c for c in net.forward_calls if c["cur_start_frame"] == 0 and not c["store_kv"]]
    chunk0_cache = [c for c in net.forward_calls if c["cur_start_frame"] == 0 and c["store_kv"]]

    # Last denoise forward on chunk 0 under F2: both anchors off
    last_denoise = chunk0_denoise[-1]
    assert last_denoise["apply_anchor"] is False, "F2 last denoise: apply_anchor should be False"
    assert last_denoise["apply_input_anchor"] is False, \
        "F2 last denoise: apply_input_anchor should also be False"

    # Non-last denoise: both True
    assert chunk0_denoise[0]["apply_anchor"] is True
    assert chunk0_denoise[0]["apply_input_anchor"] is True

    # Cache-store pass under F2: both off
    assert len(chunk0_cache) >= 1
    assert chunk0_cache[0]["apply_anchor"] is False
    assert chunk0_cache[0]["apply_input_anchor"] is False


def test_sample_loop_non_f2_keeps_both_anchors_on():
    """Without F2, all forwards have both anchors on."""
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    net = _SpyNet(chunk_size=3, total_num_frames=3)
    # F2 off (default)
    B, C, T, H, W = 1, 16, 3, 8, 8
    x = torch.randn(B, C, T, H, W)
    ffc = torch.full((B, C, 21, H, W), 7.0)
    condition = {
        "text_embeds": torch.zeros(B, 512, 4096),
        "clip_features": torch.zeros(B, 257, 1280),
        "audio_emb": torch.zeros(B, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t_list = torch.tensor([0.9, 0.5, 0.0])

    _ = CausVidModel._student_sample_loop(
        net=net, x=x, t_list=t_list, condition=condition,
        student_sample_type="sde", context_noise=0.0,
    )

    for call in net.forward_calls:
        assert call["apply_anchor"] is True, f"Non-F2 call should have apply_anchor=True: {call}"
        assert call["apply_input_anchor"] is True, f"Non-F2 call should have apply_input_anchor=True: {call}"
