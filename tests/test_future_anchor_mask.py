"""Tests for future-anchor token support in _build_block_mask.

Verifies that anchor tokens appended after the main video sequence are
globally visible (attend to everything, attended by everything) while
preserving main-to-main causal semantics.

Requires CUDA (FlexAttention compiles CUDA kernels for mask evaluation).
"""

import pytest
import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not FLEX_AVAILABLE, reason="FlexAttention not available"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_net(chunk_size: int = 3):
    """Create a bare CausalInfiniteTalkWan without loading 14B weights."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    net = CausalInfiniteTalkWan.__new__(CausalInfiniteTalkWan)
    net.chunk_size = chunk_size
    return net


def _capture_mask_fn(net, **build_kwargs):
    """Call _build_block_mask while capturing the raw mask function.

    Returns (block_mask, mask_fn) where mask_fn has signature
    ``(b, h, q_idx, kv_idx) -> bool``.
    """
    import fastgen.networks.InfiniteTalk.network_causal as mod

    captured = {}
    original_create = mod.create_block_mask

    def spy_create(mask_fn, **kwargs):
        captured["fn"] = mask_fn
        return original_create(mask_fn, **kwargs)

    mod.create_block_mask = spy_create
    try:
        block_mask = net._build_block_mask(**build_kwargs)
    finally:
        mod.create_block_mask = original_create

    return block_mask, captured["fn"]


def _eval_mask(fn, q_idx: int, kv_idx: int) -> bool:
    """Evaluate the captured mask function at scalar indices."""
    result = fn(0, 0, torch.tensor(q_idx, device="cuda"), torch.tensor(kv_idx, device="cuda"))
    return bool(result)


# ---------------------------------------------------------------------------
# Test dimensions: 6 frames, 4 tokens/frame, chunk_size=3
#   -> chunk 0: frames 0-2, tokens 0-11
#   -> chunk 1: frames 3-5, tokens 12-23
#   total_main = 24 tokens
# ---------------------------------------------------------------------------

NUM_FRAMES = 6
FRAME_SEQLEN = 4
CHUNK_SIZE = 3
TOTAL_MAIN = NUM_FRAMES * FRAME_SEQLEN  # 24


class TestNoAnchorUnchanged:
    """With num_anchor_tokens=0, mask is identical to original behaviour."""

    def test_mask_no_anchor_builds_successfully(self):
        net = _make_net(CHUNK_SIZE)
        block_mask, fn = _capture_mask_fn(
            net,
            device=torch.device("cuda"),
            num_frames=NUM_FRAMES,
            frame_seqlen=FRAME_SEQLEN,
            chunk_size=CHUNK_SIZE,
            num_anchor_tokens=0,
        )
        assert block_mask is not None

    def test_chunk0_cannot_see_chunk1(self):
        """Standard causal: chunk 0 tokens cannot attend to chunk 1 tokens."""
        net = _make_net(CHUNK_SIZE)
        _, fn = _capture_mask_fn(
            net,
            device=torch.device("cuda"),
            num_frames=NUM_FRAMES,
            frame_seqlen=FRAME_SEQLEN,
            chunk_size=CHUNK_SIZE,
            num_anchor_tokens=0,
        )
        # Token 0 (chunk 0) should NOT see token 12 (chunk 1)
        assert not _eval_mask(fn, q_idx=0, kv_idx=12)
        # Token 5 (chunk 0) should NOT see token 15 (chunk 1)
        assert not _eval_mask(fn, q_idx=5, kv_idx=15)

    def test_chunk1_can_see_chunk0(self):
        """Standard causal: chunk 1 tokens CAN attend to chunk 0 tokens."""
        net = _make_net(CHUNK_SIZE)
        _, fn = _capture_mask_fn(
            net,
            device=torch.device("cuda"),
            num_frames=NUM_FRAMES,
            frame_seqlen=FRAME_SEQLEN,
            chunk_size=CHUNK_SIZE,
            num_anchor_tokens=0,
        )
        # Token 12 (chunk 1) should see token 0 (chunk 0)
        assert _eval_mask(fn, q_idx=12, kv_idx=0)
        # Token 20 (chunk 1) should see token 5 (chunk 0)
        assert _eval_mask(fn, q_idx=20, kv_idx=5)

    def test_within_chunk_bidirectional(self):
        """Tokens within the same chunk attend bidirectionally."""
        net = _make_net(CHUNK_SIZE)
        _, fn = _capture_mask_fn(
            net,
            device=torch.device("cuda"),
            num_frames=NUM_FRAMES,
            frame_seqlen=FRAME_SEQLEN,
            chunk_size=CHUNK_SIZE,
            num_anchor_tokens=0,
        )
        # Within chunk 0: token 0 sees token 11, token 11 sees token 0
        assert _eval_mask(fn, q_idx=0, kv_idx=11)
        assert _eval_mask(fn, q_idx=11, kv_idx=0)


class TestAnchorGloballyVisible:
    """With num_anchor_tokens > 0, anchor tokens are globally visible."""

    NUM_ANCHOR = 4  # 1 anchor frame worth of tokens
    ANCHOR_START = TOTAL_MAIN  # 24
    ANCHOR_END = TOTAL_MAIN + NUM_ANCHOR  # 28

    def _build(self):
        net = _make_net(CHUNK_SIZE)
        block_mask, fn = _capture_mask_fn(
            net,
            device=torch.device("cuda"),
            num_frames=NUM_FRAMES,
            frame_seqlen=FRAME_SEQLEN,
            chunk_size=CHUNK_SIZE,
            num_anchor_tokens=self.NUM_ANCHOR,
        )
        return block_mask, fn

    def test_main_query_sees_anchor_kv(self):
        """Every main-sequence query can attend to every anchor KV token."""
        _, fn = self._build()
        for q in [0, 5, 11, 12, 20, 23]:  # tokens from both chunks
            for kv in range(self.ANCHOR_START, self.ANCHOR_END):
                assert _eval_mask(fn, q_idx=q, kv_idx=kv), (
                    f"Main query {q} should see anchor kv {kv}"
                )

    def test_anchor_query_sees_main_kv(self):
        """Every anchor query can attend to every main-sequence KV token."""
        _, fn = self._build()
        for q in range(self.ANCHOR_START, self.ANCHOR_END):
            for kv in [0, 5, 11, 12, 20, 23]:  # tokens from both chunks
                assert _eval_mask(fn, q_idx=q, kv_idx=kv), (
                    f"Anchor query {q} should see main kv {kv}"
                )

    def test_anchor_query_sees_other_anchor(self):
        """Anchor queries attend to other anchor tokens."""
        _, fn = self._build()
        for q in range(self.ANCHOR_START, self.ANCHOR_END):
            for kv in range(self.ANCHOR_START, self.ANCHOR_END):
                assert _eval_mask(fn, q_idx=q, kv_idx=kv), (
                    f"Anchor query {q} should see anchor kv {kv}"
                )


class TestMainToMainStillCausal:
    """With anchor tokens present, main-to-main attention is still causal."""

    NUM_ANCHOR = 4

    def _build(self):
        net = _make_net(CHUNK_SIZE)
        _, fn = _capture_mask_fn(
            net,
            device=torch.device("cuda"),
            num_frames=NUM_FRAMES,
            frame_seqlen=FRAME_SEQLEN,
            chunk_size=CHUNK_SIZE,
            num_anchor_tokens=self.NUM_ANCHOR,
        )
        return fn

    def test_chunk0_cannot_see_chunk1_with_anchors(self):
        """Chunk 0 main tokens still cannot see chunk 1 main tokens."""
        fn = self._build()
        # Token 0 (chunk 0) should NOT see token 12 (chunk 1)
        assert not _eval_mask(fn, q_idx=0, kv_idx=12)
        # Token 5 (chunk 0) should NOT see token 15 (chunk 1)
        assert not _eval_mask(fn, q_idx=5, kv_idx=15)
        # Token 11 (last in chunk 0) should NOT see token 12 (first in chunk 1)
        assert not _eval_mask(fn, q_idx=11, kv_idx=12)

    def test_chunk1_can_see_chunk0_with_anchors(self):
        """Chunk 1 main tokens can still see chunk 0 main tokens."""
        fn = self._build()
        assert _eval_mask(fn, q_idx=12, kv_idx=0)
        assert _eval_mask(fn, q_idx=23, kv_idx=11)

    def test_within_chunk_bidirectional_with_anchors(self):
        """Within-chunk bidirectional attention still works."""
        fn = self._build()
        assert _eval_mask(fn, q_idx=0, kv_idx=11)
        assert _eval_mask(fn, q_idx=11, kv_idx=0)
        assert _eval_mask(fn, q_idx=12, kv_idx=23)
        assert _eval_mask(fn, q_idx=23, kv_idx=12)
