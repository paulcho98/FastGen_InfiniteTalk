# Input-Side Anchor and Full InfiniteTalk Inference Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add input-side frame-0 pinning (`x_t[:, :, 0:1] = first_frame_cond[:, :, 0:1]` before every model forward) so teacher/fake/student all see clean frame 0 as input — matching InfiniteTalk's training distribution. Add explicit post-step pinning to inference paths (validation + production) so the full InfiniteTalk inference convention is replicated bit-for-bit.

**Architecture:** Introduce a separate `apply_input_anchor` kwarg that mirrors the existing `apply_anchor` (output-side). Default `True`; F2 code paths set it to `False` alongside `apply_anchor=False` to preserve F2's "cache model's raw prediction" semantics. The input pin is applied at the top of `forward()` in both `CausalInfiniteTalkWan` and `InfiniteTalkWan`, gated on `cur_start_frame == 0` (causal) or unconditionally (bidirectional). Inference paths additionally apply a post-scheduler-step pin so the displayed latent literally matches InfiniteTalk's `multitalk.py:711, 773` behavior.

**Tech Stack:** Python 3.12, PyTorch 2.8, FSDP2, bf16. Test framework: pytest (no timeout plugin — mock-based tests complete in <30s).

---

## Interaction Review (context, not tasks)

### Why output-side anchor alone is insufficient

The current code (`network.py:620-631`, `network_causal.py:1025-1057`) pins the model's **output** at frame 0 to `first_frame_cond[:, :, 0:1]`. The model's **input** at frame 0 is whatever the caller passed — typically noisy (from `forward_process` or pure initial noise).

InfiniteTalk's teacher was trained with clean `x[:, :, 0]` at every timestep (verified: `batch_inference.py --disable_i2v_overwrite` produces utter garbage at frame 0, which is the Option A training signature). Feeding the teacher a noisy frame 0 is out-of-distribution: the output override masks the frame-0 position but the model's temporal attention at frames 1+ routed through the OOD context and produces subtly corrupted downstream frames.

Adding an input-side pin fixes this: every forward pass sees in-distribution input at frame 0.

### Interaction with F2 (`_model_sink_cache`)

F2 is designed to cache the model's **raw** (unanchored) frame-0 prediction for use as a sink token by subsequent chunks. Its implementation sets `apply_anchor=False` at specific forwards (chunk 0's last denoise step and chunk 0's cache-store pass).

The K/V stored at position 0 of the cache is computed from the forward pass's **input**, not its output. Current F2 relies on the input naturally being OOD (noisy frame 0 drifted through scheduler) because input pinning doesn't exist — so the K/V at position 0 reflects the model's behavior on "realistic" input.

With input-side pinning added unconditionally, F2 would cache K/V computed from clean reference input — nullifying F2's effect (the K/V would be close to what non-F2 caches anyway).

**Resolution:** Introduce `apply_input_anchor: bool = True` as a separate toggle. F2 paths set it to `False` alongside `apply_anchor=False`. Default training/inference paths set both to `True`. F2 semantics are preserved; standard paths get the corrected input distribution.

### Interaction with F3 (`_skip_clean_cache_pass`)

F3 skips the separate cache-store forward and stores KV during the final denoise step instead. With input pinning on that final step (default `True`), the KV cache at position 0 comes from clean reference input — same as non-F3 cache-store behavior. No additional change needed; F3 semantics are unchanged.

### Interaction with `context_noise` (noisy KV feature)

`context_noise` in `rollout_with_gradient` (`self_forcing.py:213-220`) and `_student_sample_loop` (`causvid.py:200-207`) adds noise to the denoised chunk output before the cache-store forward, so frames 1+ in the cache reflect a partially-noisy context (mirroring real self-forcing distribution where subsequent chunks see imperfectly denoised prior outputs).

With input pinning on the cache-store forward (default `True`), frame 0's K/V is computed from clean_ref regardless of `context_noise`, while frames 1+ receive the `context_noise` perturbation. This is **correct and desired**:
- Frame 0 is the fixed reference anchor — it never drifts at inference.
- Frames 1+ are generated; their cached K/V should reflect self-forcing distribution.
- Training distribution (frame 0 clean, rest noised) matches inference distribution — no train/deploy shift.

When F2 is active (cache pass has `apply_input_anchor=False`), `context_noise` affects frame 0 too. This is F2's intent — cache the model's behavior on noised input.

No changes to `context_noise` itself are required.

### Interaction with rollout gradient tracking

`rollout_with_gradient` exits at a stochastic step per block; gradients only flow from the exit-step forward onward. Input pinning assigns a constant (`first_frame_cond[:, :, 0:1]`), cutting gradient flow at position 0. This is correct — frame 0's position 0 should NOT receive student gradient (its content is externally specified). Position-1+ gradients are preserved.

The existing output-side anchor also cuts gradient at position 0 (same constant assignment). No net change to gradient topology.

### Decision: keep output-side anchor too

We keep `apply_anchor` (output-side) as a guardrail alongside new `apply_input_anchor` (input-side). They're cheap, redundant-but-safe when both are True, and the F2 mechanism already uses `apply_anchor=False` in specific places. Removing output-side anchor would require rewriting F2/F3 logic; adding input-side alongside is minimally invasive.

### What changes, concretely

| Call site | Before | After |
|-----------|--------|-------|
| SF `rollout_with_gradient` denoising steps | `store_kv=False`, relies on output anchor | Add `apply_input_anchor=True` (default) — in-distribution input |
| SF `rollout_with_gradient` cache-store | `store_kv=True`, output anchor applied | Add `apply_input_anchor=True` (default) — frame 0 pinned on input |
| CausVid `_student_sample_loop` denoise steps | `apply_anchor=apply_anchor_here` | Add `apply_input_anchor=apply_anchor_here` (F2-aware) + post-step pin |
| CausVid `_student_sample_loop` cache-store | `apply_anchor=apply_anchor_cache` | Add `apply_input_anchor=apply_anchor_cache` (F2-aware) |
| `inference_causal.py::run_inference` denoise steps | `apply_anchor=apply_anchor_here` | Add `apply_input_anchor=apply_anchor_here` + post-step pin |
| `inference_causal.py::run_inference` cache-store | `apply_anchor=apply_anchor_cache` | Add `apply_input_anchor=apply_anchor_cache` |
| DMD2 teacher/fake forwards | Pass through | No change — default `apply_input_anchor=True` propagates |
| InfiniteTalkSF 3-call CFG teacher forwards | Pass through | No change — default propagates |

---

## File Structure

**Files modified:**
- `fastgen/networks/InfiniteTalk/network_causal.py` — add `_maybe_apply_input_anchor` helper + `apply_input_anchor` kwarg on forward
- `fastgen/networks/InfiniteTalk/network.py` — add same helper + kwarg on bidirectional forward
- `fastgen/methods/distribution_matching/causvid.py` — propagate `apply_input_anchor` in F2-aware call sites; add post-step pin for full inference convention
- `scripts/inference/inference_causal.py` — same changes as causvid for production AR loop

**Files unchanged (rely on default `apply_input_anchor=True`):**
- `fastgen/methods/distribution_matching/self_forcing.py::rollout_with_gradient`
- `fastgen/methods/distribution_matching/dmd2.py`
- `fastgen/methods/infinitetalk_self_forcing.py` (inherits rollout + teacher/fake call pattern)

**Tests created:**
- `tests/test_input_anchor_plumbing.py` — unit tests for helper + kwarg on both networks
- `tests/test_f2_input_anchor_interaction.py` — F2 correctly disables both anchors in sync
- `tests/test_post_step_pin_inference.py` — validation/inference paths apply post-step pin
- Extend existing `tests/test_apply_anchor_kwarg.py` with input-anchor equivalents

---

## Task 1: Helper function `_maybe_apply_input_anchor` in network_causal.py

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py` — add helper next to existing `_maybe_apply_first_frame_anchor`
- Test: `tests/test_input_anchor_plumbing.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_input_anchor_plumbing.py`:

```python
"""Unit tests for input-side frame-0 anchor helper and plumbing."""
import torch
import pytest
from types import SimpleNamespace


# The helper under test (we import it in step 3 after implementation).
# For now, reference it by module path; the test fails on ImportError.
def test_input_anchor_helper_pins_frame_0_when_active():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(2, 16, 21, 28, 56)
    ffc = torch.ones(2, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}

    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True,
        _anchor_eval_only=False,
        training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=condition, apply_input_anchor=True,
    )
    # Frame 0 pinned
    assert out[:, :, 0].eq(7.0).all(), "frame 0 not pinned to first_frame_cond"
    # Other frames untouched
    assert out[:, :, 1:].eq(0.0).all(), "frames 1+ should be unchanged"
    # Original x_t is NOT mutated (function must clone)
    assert x_t[:, :, 0].eq(0.0).all(), "input x_t was mutated in place"


def test_input_anchor_helper_noop_when_apply_input_anchor_false():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 21, 28, 56)
    ffc = torch.ones(1, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True, _anchor_eval_only=False, training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=condition, apply_input_anchor=False,
    )
    assert out is x_t, "apply_input_anchor=False should return input unchanged"


def test_input_anchor_helper_noop_when_cur_start_nonzero():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 3, 28, 56)
    ffc = torch.ones(1, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True, _anchor_eval_only=False, training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=3, condition=condition, apply_input_anchor=True,
    )
    assert out is x_t, "non-zero cur_start_frame should skip pinning"


def test_input_anchor_helper_respects_hard_disable():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 21, 28, 56)
    ffc = torch.ones(1, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=False,  # hard-disabled (e.g., teacher_anchor_disabled)
        _anchor_eval_only=False,
        training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=condition, apply_input_anchor=True,
    )
    assert out is x_t, "_enable_first_frame_anchor=False should skip pinning"


def test_input_anchor_helper_eval_only_during_training():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 21, 28, 56)
    ffc = torch.ones(1, 16, 21, 28, 56) * 7.0
    condition = {"first_frame_cond": ffc}
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True,
        _anchor_eval_only=True,   # eval-only
        training=True,            # but we are training
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=condition, apply_input_anchor=True,
    )
    assert out is x_t, "_anchor_eval_only + training=True should skip pinning"


def test_input_anchor_helper_no_condition():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor

    x_t = torch.zeros(1, 16, 21, 28, 56)
    net_module = SimpleNamespace(
        _enable_first_frame_anchor=True, _anchor_eval_only=False, training=False,
    )

    out = _maybe_apply_input_anchor(
        x_t, net_module, cur_start_frame=0, condition=None, apply_input_anchor=True,
    )
    assert out is x_t, "missing condition should skip pinning"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
python -m pytest tests/test_input_anchor_plumbing.py -v
```

Expected: all 6 tests FAIL with `ImportError: cannot import name '_maybe_apply_input_anchor'`.

- [ ] **Step 3: Implement the helper in network_causal.py**

In `fastgen/networks/InfiniteTalk/network_causal.py`, immediately after the existing `_maybe_apply_first_frame_anchor` function (around line 1058), add:

```python
def _maybe_apply_input_anchor(
    x_t: torch.Tensor,
    net_module,
    cur_start_frame: int,
    condition,
    apply_input_anchor: bool = True,
) -> torch.Tensor:
    """Pin x_t[:, :, 0:1] to the clean reference frame when enabled.

    Mirrors _maybe_apply_first_frame_anchor but operates on the forward-pass
    INPUT (x_t) rather than the output. Applied at the top of forward() so
    the model always sees clean frame 0 as input — matching InfiniteTalk's
    training distribution where x_t[:, :, 0] was held clean at every timestep.

    Modes (instance attributes on net_module), identical semantics to output
    anchor:
      _enable_first_frame_anchor = True (default): anchor is active
      _enable_first_frame_anchor = False: anchor fully disabled
      _anchor_eval_only = True: anchor only in eval mode (not during training)

    The explicit ``apply_input_anchor=False`` argument overrides all of the
    above — used by F2 (model-sink-cache) paths to let the model see its
    own drifted frame 0 as input so the cached K/V reflects that behavior.

    Returns the (possibly pinned) tensor. Does not mutate x_t in place.
    """
    if not apply_input_anchor:
        return x_t
    if cur_start_frame != 0:
        return x_t
    if not isinstance(condition, dict) or "first_frame_cond" not in condition:
        return x_t
    anchor_active = getattr(net_module, "_enable_first_frame_anchor", True)
    if anchor_active and getattr(net_module, "_anchor_eval_only", False):
        anchor_active = not net_module.training
    if not anchor_active:
        return x_t
    first_frame_cond = condition["first_frame_cond"]
    x_t = x_t.clone()
    x_t[:, :, 0:1] = first_frame_cond[:, :, 0:1]
    return x_t
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_input_anchor_plumbing.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_input_anchor_plumbing.py
git commit -m "$(cat <<'EOF'
feat: add _maybe_apply_input_anchor helper for input-side frame-0 pinning

Mirrors _maybe_apply_first_frame_anchor but operates on the forward
INPUT instead of output. Pins x_t[:, :, 0:1] to first_frame_cond so the
model sees clean frame 0 as input — matching InfiniteTalk's training
distribution (verified via batch_inference.py --disable_i2v_overwrite
producing garbage at frame 0 without the overwrite).

Helper respects the same _enable_first_frame_anchor / _anchor_eval_only
attribute semantics as the output-side anchor. Explicit
apply_input_anchor=False kwarg overrides all attrs (for F2 paths).

No callers yet — follow-up commits wire this into forward() and into
the training/inference loops.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Plumb `apply_input_anchor` into `CausalInfiniteTalkWan.forward`

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:1975-2160` (forward signature + body)
- Test: `tests/test_input_anchor_plumbing.py` (extend)

- [ ] **Step 1: Write the failing test (extend the existing test file)**

Append to `tests/test_input_anchor_plumbing.py`:

```python
def test_apply_input_anchor_kwarg_pins_forward_input(monkeypatch):
    """Verify CausalInfiniteTalkWan.forward applies input anchor before dispatch."""
    from fastgen.networks.InfiniteTalk import network_causal

    captured_x = {}

    def fake_forward_ar(self, x, timestep, context, clip_fea, y, audio,
                       current_start, store_kv, use_gradient_checkpointing):
        # Capture the x that gets passed to the AR forward
        captured_x["x"] = x.detach().clone()
        # Return a dummy tensor matching expected output shape
        B, C, T, H, W = x.shape
        return torch.zeros(B, C, T, H, W, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(
        network_causal.CausalInfiniteTalkWan,
        "_forward_ar",
        fake_forward_ar,
    )

    # Build a minimal-mocked instance
    net = SimpleNamespace(
        _enable_first_frame_anchor=True,
        _anchor_eval_only=False,
        training=False,
        _use_gradient_checkpointing=False,
        _audio_window=5,
        _patch_size=(1, 2, 2),
        _lookahead_sink_enabled=False,
        _lookahead_distance_min=0,
        _lookahead_distance_max=0,
        net_pred_type="flow",
        noise_scheduler=SimpleNamespace(
            rescale_t=lambda t: t * 1000,
            convert_model_output=lambda x_t, model_out, t, src_pred_type, target_pred_type: model_out,
        ),
        _build_y=lambda cond, T, start_frame: torch.zeros(cond["first_frame_cond"].shape[0], 20, T, cond["first_frame_cond"].shape[3], cond["first_frame_cond"].shape[4], dtype=cond["first_frame_cond"].dtype),
    )

    # x_t has noisy frame 0
    x_t = torch.randn(1, 16, 3, 28, 56)  # chunk of 3 frames
    ffc = torch.full((1, 16, 21, 28, 56), 5.0)  # clean ref = 5.0 everywhere
    condition = {
        "text_embeds": torch.zeros(1, 512, 4096),
        "clip_features": torch.zeros(1, 257, 1280),
        "audio_emb": torch.zeros(1, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t = torch.tensor([0.5])

    _ = network_causal.CausalInfiniteTalkWan.forward(
        net, x_t, t, condition=condition,
        cur_start_frame=0, is_ar=True,
        apply_input_anchor=True, apply_anchor=True,
    )

    # Verify that _forward_ar received x with frame 0 pinned to 5.0
    assert "x" in captured_x, "_forward_ar was not called"
    assert captured_x["x"][:, :, 0].eq(5.0).all(), \
        "Input anchor did NOT pin frame 0 before forward"
    # Non-frame-0 positions are unchanged (from original x_t, which was random)
    # We can't check exact values (random), but we check they weren't all 5.0
    assert not captured_x["x"][:, :, 1:].eq(5.0).all(), \
        "Non-frame-0 positions were unexpectedly pinned"


def test_apply_input_anchor_false_skips_pinning(monkeypatch):
    """Verify apply_input_anchor=False leaves x_t untouched."""
    from fastgen.networks.InfiniteTalk import network_causal

    captured_x = {}

    def fake_forward_ar(self, x, timestep, context, clip_fea, y, audio,
                       current_start, store_kv, use_gradient_checkpointing):
        captured_x["x"] = x.detach().clone()
        B, C, T, H, W = x.shape
        return torch.zeros(B, C, T, H, W, dtype=x.dtype, device=x.device)

    monkeypatch.setattr(
        network_causal.CausalInfiniteTalkWan,
        "_forward_ar",
        fake_forward_ar,
    )

    net = SimpleNamespace(
        _enable_first_frame_anchor=True, _anchor_eval_only=False, training=False,
        _use_gradient_checkpointing=False, _audio_window=5, _patch_size=(1, 2, 2),
        _lookahead_sink_enabled=False, _lookahead_distance_min=0, _lookahead_distance_max=0,
        net_pred_type="flow",
        noise_scheduler=SimpleNamespace(
            rescale_t=lambda t: t * 1000,
            convert_model_output=lambda x_t, model_out, t, src_pred_type, target_pred_type: model_out,
        ),
        _build_y=lambda cond, T, start_frame: torch.zeros(cond["first_frame_cond"].shape[0], 20, T, cond["first_frame_cond"].shape[3], cond["first_frame_cond"].shape[4], dtype=cond["first_frame_cond"].dtype),
    )

    original_x = torch.randn(1, 16, 3, 28, 56)
    x_t = original_x.clone()
    ffc = torch.full((1, 16, 21, 28, 56), 5.0)
    condition = {
        "text_embeds": torch.zeros(1, 512, 4096),
        "clip_features": torch.zeros(1, 257, 1280),
        "audio_emb": torch.zeros(1, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t = torch.tensor([0.5])

    _ = network_causal.CausalInfiniteTalkWan.forward(
        net, x_t, t, condition=condition,
        cur_start_frame=0, is_ar=True,
        apply_input_anchor=False,  # disabled
        apply_anchor=True,
    )

    assert torch.equal(captured_x["x"], original_x), \
        "apply_input_anchor=False should leave x_t unchanged"
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_input_anchor_plumbing.py::test_apply_input_anchor_kwarg_pins_forward_input \
    tests/test_input_anchor_plumbing.py::test_apply_input_anchor_false_skips_pinning -v
```

Expected: both FAIL with `TypeError: forward() got an unexpected keyword argument 'apply_input_anchor'`.

- [ ] **Step 3: Add the kwarg to forward signature + apply at top of function body**

In `fastgen/networks/InfiniteTalk/network_causal.py`:

Find the forward method signature around line 1975-1992. Add `apply_input_anchor: bool = True` alongside the existing `apply_anchor: bool = True`:

```python
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        cur_start_frame: int = 0,
        store_kv: bool = False,
        is_ar: bool = True,
        use_gradient_checkpointing: Optional[bool] = None,
        apply_anchor: bool = True,
        apply_input_anchor: bool = True,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
```

Update the docstring (in the "Args:" block of that same method, around line 2014):

```python
            apply_anchor: If True (default), apply the frame-0 anchor overwrite
                on the model OUTPUT when cur_start_frame==0 and network anchor
                mode allows. If False, bypass the output anchor (used by F2).
            apply_input_anchor: If True (default), pin the model INPUT
                x_t[:, :, 0:1] = first_frame_cond[:, :, 0:1] at the top of
                forward() when cur_start_frame==0 and network anchor mode
                allows. Matches InfiniteTalk's training distribution. F2 paths
                set this to False to let the model see its own drifted frame 0
                at the cache-store forward.
```

Now, in the body of forward, find the line that unpacks `B, C, T, H, W = x_t.shape` (around line 2091). **Immediately after** that line, add the input anchor call:

```python
        B, C, T, H, W = x_t.shape

        # Input-side frame-0 anchor: pin x_t[:, :, 0:1] to clean reference
        # BEFORE any downstream computation. Matches InfiniteTalk training
        # distribution (clean x[:, :, 0] at every timestep). See
        # _maybe_apply_input_anchor for modes / gating semantics.
        x_t = _maybe_apply_input_anchor(
            x_t, self, cur_start_frame, condition,
            apply_input_anchor=apply_input_anchor,
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_input_anchor_plumbing.py -v
```

Expected: all tests pass (8 total — 6 helper tests + 2 plumbing tests).

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_input_anchor_plumbing.py
git commit -m "$(cat <<'EOF'
feat: add apply_input_anchor kwarg to CausalInfiniteTalkWan.forward

Pins x_t[:, :, 0:1] to first_frame_cond[:, :, 0:1] at the top of
forward() when apply_input_anchor=True (default) and network anchor
mode permits. Ensures the model sees clean frame 0 as INPUT — matching
InfiniteTalk's training distribution.

The explicit apply_input_anchor=False kwarg is reserved for F2
(model-sink-cache) paths that need the model to see its own drifted
frame 0 so the cached K/V reflects natural behavior.

Default behavior (apply_input_anchor=True) pins frame 0 at every
forward, which resolves the "teacher produces garbage at frame 0
without overwrite" failure mode (see
docs/sf-validation-garbage-output-analysis.md).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Plumb `apply_input_anchor` into `InfiniteTalkWan.forward` (bidirectional teacher/fake)

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network.py:473-660` (forward signature + body, and add the helper alongside existing anchor code)
- Test: `tests/test_input_anchor_plumbing.py` (extend)

Bidirectional network has no `cur_start_frame` concept (it always operates on the full-length video with frame 0 at position 0), so the helper unconditionally pins `x_t[:, :, 0:1]` when anchor mode allows.

- [ ] **Step 1: Write the failing test (extend)**

Append to `tests/test_input_anchor_plumbing.py`:

```python
def test_bidirectional_input_anchor_kwarg_pins_forward_input(monkeypatch):
    """Verify InfiniteTalkWan (bidirectional) forward applies input anchor."""
    from fastgen.networks.InfiniteTalk import network as itw_network

    captured = {}

    def fake_model_forward(x, t, context, seq_len, clip_fea, y, audio,
                           use_gradient_checkpointing, feature_indices, return_features_early):
        captured["x_list"] = [xi.detach().clone() for xi in x]
        # Return dummy tensor shaped like stacked unpatchify output [B, C, T, H, W]
        B = len(x)
        C, T, H, W = x[0].shape
        return torch.zeros(B, C, T, H, W, dtype=x[0].dtype, device=x[0].device)

    net = SimpleNamespace(
        _enable_first_frame_anchor=True,
        _anchor_eval_only=False,
        training=False,
        _use_gradient_checkpointing=False,
        net_pred_type="flow",
        noise_scheduler=SimpleNamespace(
            rescale_t=lambda t: t * 1000,
            convert_model_output=lambda x_t, mo, t, src_pred_type, target_pred_type: mo,
        ),
        model=SimpleNamespace(
            patch_size=(1, 2, 2),
            __call__=fake_model_forward,
        ),
        _build_y=lambda cond, T: torch.zeros(cond["first_frame_cond"].shape[0], 20, T, cond["first_frame_cond"].shape[3], cond["first_frame_cond"].shape[4], dtype=cond["first_frame_cond"].dtype),
    )
    # Make model callable by patching its __call__
    net.model = type("M", (), {"patch_size": (1, 2, 2), "__call__": lambda self, **kw: fake_model_forward(**kw)})()

    x_t = torch.randn(1, 16, 21, 28, 56)
    ffc = torch.full((1, 16, 21, 28, 56), 9.0)
    condition = {
        "text_embeds": torch.zeros(1, 512, 4096),
        "clip_features": torch.zeros(1, 257, 1280),
        "audio_emb": torch.zeros(1, 93, 5, 12, 768),
        "first_frame_cond": ffc,
    }
    t = torch.tensor([0.5])

    _ = itw_network.InfiniteTalkWan.forward(
        net, x_t, t, condition=condition,
        apply_input_anchor=True,
    )

    # x is a list of per-sample tensors; check frame 0 pinned on all
    assert len(captured["x_list"]) == 1
    assert captured["x_list"][0][:, 0].eq(9.0).all(), \
        "Bidirectional input anchor did NOT pin frame 0"
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_input_anchor_plumbing.py::test_bidirectional_input_anchor_kwarg_pins_forward_input -v
```

Expected: FAIL.

- [ ] **Step 3: Add helper + kwarg to network.py**

In `fastgen/networks/InfiniteTalk/network.py`, near the top of the file (next to other module-level helpers), add:

```python
def _maybe_apply_input_anchor_bidir(
    x_t: torch.Tensor,
    net_module,
    condition,
    apply_input_anchor: bool = True,
) -> torch.Tensor:
    """Pin x_t[:, :, 0:1] to clean reference for bidirectional teacher/fake.

    Simpler than the causal variant: no cur_start_frame gating — the
    bidirectional network always operates on a full-length video with
    frame 0 as the reference.

    See network_causal._maybe_apply_input_anchor for mode semantics.
    """
    if not apply_input_anchor:
        return x_t
    if not isinstance(condition, dict) or "first_frame_cond" not in condition:
        return x_t
    anchor_active = getattr(net_module, "_enable_first_frame_anchor", True)
    if anchor_active and getattr(net_module, "_anchor_eval_only", False):
        anchor_active = not net_module.training
    if not anchor_active:
        return x_t
    first_frame_cond = condition["first_frame_cond"]
    x_t = x_t.clone()
    x_t[:, :, 0:1] = first_frame_cond[:, :, 0:1]
    return x_t
```

Now update the `forward` method signature (around line 473). Add `apply_input_anchor: bool = True` to the signature:

```python
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Any = None,
        r: Optional[torch.Tensor] = None,
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        apply_input_anchor: bool = True,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
```

In the body of forward, find the line `B, C, T, H, W = x_t.shape` (around line 548). **Immediately after** that line, add:

```python
        B, C, T, H, W = x_t.shape

        # Input-side frame-0 anchor (bidirectional): pin x_t[:, :, 0:1] to
        # clean reference before building y and running the model. Matches
        # InfiniteTalk training distribution for teacher/fake.
        x_t = _maybe_apply_input_anchor_bidir(
            x_t, self, condition, apply_input_anchor=apply_input_anchor,
        )
```

Also update the docstring for forward (in its Args: block) to include the new kwarg, mirroring the text from Task 2 but without the `cur_start_frame` mention.

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_input_anchor_plumbing.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network.py tests/test_input_anchor_plumbing.py
git commit -m "$(cat <<'EOF'
feat: add apply_input_anchor kwarg to InfiniteTalkWan.forward (bidirectional)

Mirrors the causal variant's kwarg. Bidirectional teacher/fake
networks always operate on a full-length video with frame 0 at
position 0, so no cur_start_frame gating is needed — the helper
unconditionally pins x_t[:, :, 0:1] when anchor mode allows.

Teacher and fake_score calls in DMD2 and InfiniteTalkSelfForcingModel
inherit the default apply_input_anchor=True and start seeing clean
frame 0 on input immediately after this commit.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: F2-aware propagation in `_student_sample_loop` + post-step pin

**Files:**
- Modify: `fastgen/methods/distribution_matching/causvid.py:87-234` (the `_student_sample_loop` method)
- Test: `tests/test_sample_loop_toggles.py` (extend with F2 + input anchor coupling test; create `tests/test_post_step_pin_inference.py` for new behavior)

The existing F2 logic sets `apply_anchor_here` and `apply_anchor_cache` booleans. We add the same wiring for `apply_input_anchor_here` and `apply_input_anchor_cache` — they take **the same value** as their output-side counterparts. This preserves F2 semantics: when F2 wants the model's raw frame-0 output cached, it also wants the model to see its own frame 0 as input.

We also add a post-step pin after the `forward_process` call so the full InfiniteTalk convention (both pre- and post-step pinning) is replicated on the validation/inference path.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_post_step_pin_inference.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_post_step_pin_inference.py -v
```

Expected: all 4 tests FAIL. Specifically:
- `test_sample_loop_post_step_pins_frame_0_on_chunk0`: FAIL — no post-step pin exists yet.
- `test_sample_loop_f2_propagates_both_anchors`: FAIL — `apply_input_anchor` key is absent from recorded calls (spy never sees it because sample_loop doesn't pass it).

- [ ] **Step 3: Modify `_student_sample_loop` to propagate `apply_input_anchor` and add post-step pin**

In `fastgen/methods/distribution_matching/causvid.py`, edit `_student_sample_loop`.

Find the denoising-step forward call (around lines 155-168). Change:

```python
            t_cur = t_list[step].expand(batch_size)
            x_cur = x_next
            x_next = net(
                x_cur,
                t_cur,
                condition=condition,
                fwd_pred_type="x0",
                cache_tag="pos",
                cur_start_frame=start,
                store_kv=store_kv_here,
                is_ar=True,
                apply_anchor=apply_anchor_here,
                **kwargs,
            )
```

to:

```python
            t_cur = t_list[step].expand(batch_size)
            x_cur = x_next
            x_next = net(
                x_cur,
                t_cur,
                condition=condition,
                fwd_pred_type="x0",
                cache_tag="pos",
                cur_start_frame=start,
                store_kv=store_kv_here,
                is_ar=True,
                apply_anchor=apply_anchor_here,
                apply_input_anchor=apply_anchor_here,
                **kwargs,
            )
```

Find the forward_process block (around lines 170-185). Change:

```python
                t_next = t_list[step + 1]
                if t_next > 0:
                    t_chunk_next = t_next.expand(batch_size)
                    if student_sample_type == "sde":
                        eps_infer = torch.randn_like(x_next)
                    elif student_sample_type == "ode":
                        eps_infer = net.noise_scheduler.x0_to_eps(
                            xt=x_cur, x0=x_next, t=t_cur
                        )
                    else:
                        raise NotImplementedError(
                            f"student_sample_type must be one of 'sde', 'ode' but got {student_sample_type}"
                        )
                    x_next = net.noise_scheduler.forward_process(
                        x_next, eps_infer, t_chunk_next
                    )
```

to:

```python
                t_next = t_list[step + 1]
                if t_next > 0:
                    t_chunk_next = t_next.expand(batch_size)
                    if student_sample_type == "sde":
                        eps_infer = torch.randn_like(x_next)
                    elif student_sample_type == "ode":
                        eps_infer = net.noise_scheduler.x0_to_eps(
                            xt=x_cur, x0=x_next, t=t_cur
                        )
                    else:
                        raise NotImplementedError(
                            f"student_sample_type must be one of 'sde', 'ode' but got {student_sample_type}"
                        )
                    x_next = net.noise_scheduler.forward_process(
                        x_next, eps_infer, t_chunk_next
                    )

                    # Post-scheduler-step pin — matches InfiniteTalk's
                    # multitalk.py:773 convention. Redundant with the next
                    # step's input anchor but explicitly replicates
                    # InfiniteTalk's full inference convention for the
                    # validation/inference path. Only pin on chunk 0 where
                    # local frame 0 is the global reference frame.
                    if (
                        start == 0
                        and isinstance(condition, dict)
                        and "first_frame_cond" in condition
                    ):
                        ffc = condition["first_frame_cond"]
                        x_next = x_next.clone()
                        x_next[:, :, 0:1] = ffc[:, :, 0:1]
```

Find the cache-store forward (around lines 219-230). Change:

```python
                _ = net(
                    x_cache,
                    t_cache,
                    condition=condition,
                    fwd_pred_type="x0",
                    cache_tag="pos",
                    cur_start_frame=start,
                    store_kv=True,
                    is_ar=True,
                    apply_anchor=apply_anchor_cache,
                    **kwargs,
                )
```

to:

```python
                _ = net(
                    x_cache,
                    t_cache,
                    condition=condition,
                    fwd_pred_type="x0",
                    cache_tag="pos",
                    cur_start_frame=start,
                    store_kv=True,
                    is_ar=True,
                    apply_anchor=apply_anchor_cache,
                    apply_input_anchor=apply_anchor_cache,
                    **kwargs,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_post_step_pin_inference.py tests/test_sample_loop_toggles.py -v
```

Expected: all new tests pass, existing `tests/test_sample_loop_toggles.py` still passes (no regressions).

- [ ] **Step 5: Commit**

```bash
git add fastgen/methods/distribution_matching/causvid.py tests/test_post_step_pin_inference.py
git commit -m "$(cat <<'EOF'
feat: full InfiniteTalk inference convention in _student_sample_loop

Three changes to CausVidModel._student_sample_loop:

1. Propagate apply_input_anchor=apply_anchor_here on denoising forwards.
   Same value as apply_anchor — so F2 disables both in sync.

2. Propagate apply_input_anchor=apply_anchor_cache on cache-store pass.
   Preserves F2's "cache model's raw behavior" semantics.

3. Add post-scheduler-step pin: after forward_process, pin
   x_next[:, :, 0:1] = first_frame_cond[:, :, 0:1] (chunk 0 only).
   Matches InfiniteTalk's multitalk.py:711, 773 inference convention.
   Redundant with the next step's input anchor for intermediate
   denoising steps, but explicitly replicates the full convention.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: F2-aware propagation + post-step pin in `inference_causal.py::run_inference`

**Files:**
- Modify: `scripts/inference/inference_causal.py:705-832` (the `run_inference` function)
- Test: `tests/test_inference_causal_toggles.py` (extend if it exists; otherwise create minimal test)

This is the production AR inference path. Same changes as Task 4, applied to the inline AR loop in `run_inference`.

- [ ] **Step 1: Check if inference_causal tests exist**

```bash
ls tests/test_inference_causal_toggles.py 2>/dev/null && echo FOUND || echo MISSING
```

Expected: FOUND (file exists per earlier grep at line count 20).

- [ ] **Step 2: Write the failing test**

Append to `tests/test_inference_causal_toggles.py`:

```python
def test_inference_causal_propagates_apply_input_anchor_f2(monkeypatch):
    """F2 on sink chunk's last denoise + cache pass: both anchors off together."""
    import sys
    # Avoid heavy imports from scripts/inference/inference_causal.py by
    # isolating run_inference via a minimal model mock.
    # NOTE: run_inference imports torch etc at module load; we reuse the
    # existing test harness pattern from test_inference_causal_toggles.py.
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
    condition = {
        "text_embeds": __import__("torch").zeros(B, 512, 4096),
        "clip_features": __import__("torch").zeros(B, 257, 1280),
        "audio_emb": __import__("torch").zeros(B, 93, 5, 12, 768),
        "first_frame_cond": __import__("torch").full((B, C, 21, H, W), 5.0),
    }
    import torch
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
python -m pytest tests/test_inference_causal_toggles.py::test_inference_causal_propagates_apply_input_anchor_f2 -v
```

Expected: FAIL (key `apply_input_anchor` missing from recorded calls).

- [ ] **Step 4: Modify `run_inference` in `scripts/inference/inference_causal.py`**

Find the denoise forward call (around lines 766-776). Change:

```python
            x0_pred = model(
                noisy_input,
                t_cur.expand(B),
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=store_kv_here,
                is_ar=True,
                fwd_pred_type="x0",
                apply_anchor=apply_anchor_here,
                use_gradient_checkpointing=False,
            )
```

to:

```python
            x0_pred = model(
                noisy_input,
                t_cur.expand(B),
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=store_kv_here,
                is_ar=True,
                fwd_pred_type="x0",
                apply_anchor=apply_anchor_here,
                apply_input_anchor=apply_anchor_here,
                use_gradient_checkpointing=False,
            )
```

Find the forward_process block (around lines 784-790). Change:

```python
            if t_next > 0:
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.expand(B),
                )
            else:
                noisy_input = x0_pred
```

to:

```python
            if t_next > 0:
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.expand(B),
                )
                # Post-scheduler-step pin — matches InfiniteTalk's
                # multitalk.py:773 convention on the production AR path.
                if cur_start_frame == 0 and isinstance(condition, dict) and "first_frame_cond" in condition:
                    noisy_input = noisy_input.clone()
                    noisy_input[:, :, 0:1] = first_frame_latent
            else:
                noisy_input = x0_pred
```

Find the cache-store forward (around lines 816-826). Change:

```python
            model(
                cache_input,
                t_cache,
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=True,
                is_ar=True,
                fwd_pred_type="x0",
                apply_anchor=apply_anchor_cache,
                use_gradient_checkpointing=False,
            )
```

to:

```python
            model(
                cache_input,
                t_cache,
                condition=condition,
                cur_start_frame=cur_start_frame,
                store_kv=True,
                is_ar=True,
                fwd_pred_type="x0",
                apply_anchor=apply_anchor_cache,
                apply_input_anchor=apply_anchor_cache,
                use_gradient_checkpointing=False,
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_inference_causal_toggles.py -v
```

Expected: new test passes; existing tests in that file still pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/inference/inference_causal.py tests/test_inference_causal_toggles.py
git commit -m "$(cat <<'EOF'
feat: full InfiniteTalk inference convention in inference_causal.py

Mirrors the _student_sample_loop change for the production AR path:

1. apply_input_anchor propagated alongside apply_anchor on denoise
   forwards (F2-aware).
2. apply_input_anchor propagated on cache-store pass (F2-aware).
3. Post-scheduler-step pin on chunk 0 after forward_process — matches
   multitalk.py:773 behavior.

Production inference now uses the same frame-0 conditioning convention
that InfiniteTalk's own generate_infinitetalk uses.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Verify existing test suite + 6-step gradient-sanity test

**Files:**
- Test: `tests/test_rollout_gradient_sanity.py` (create)
- No other code changes.

The goal is to verify that (a) nothing regresses in the existing suite, and (b) the new plumbing survives a gradient roundtrip through the student rollout. We use a MockNet-based test with a small number of synthetic iterations — consistent with the repo's existing "no 14B model required" testing convention (see `CLAUDE.md::Testing patterns`).

- [ ] **Step 1: Run the full test suite to check for regressions**

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
python -m pytest tests/ -v
```

Expected: all tests pass. Check specifically:
- `tests/test_apply_anchor_kwarg.py` — existing output-anchor tests unchanged
- `tests/test_apply_attention_config.py` — `_apply_anchor_config` stamping unchanged
- `tests/test_sample_loop_toggles.py` — F2/F3 behavior preserved
- `tests/test_inference_causal_toggles.py` — production path F2/F3 preserved
- New `tests/test_input_anchor_plumbing.py` — 9 tests pass
- New `tests/test_post_step_pin_inference.py` — 4 tests pass

If any existing test fails, investigate: the change should be purely additive.

- [ ] **Step 2: Write a 6-step rollout gradient-sanity test**

Create `tests/test_rollout_gradient_sanity.py`:

```python
"""Verify that the student rollout + new input-anchor plumbing survives
gradient backprop across multiple chunks.

Uses a MockNet that implements enough of the CausalInfiniteTalkWan interface
to exercise _student_sample_loop end-to-end with gradients. The backprop
path is: rollout → concat → simple MSE against a target → loss.backward().

This is a fast (<5s), CPU-only test that catches bugs like:
  - new apply_input_anchor kwarg blocking autograd
  - shape errors in the post-step pin
  - F2 path not propagating both anchor kwargs
"""
import torch
import torch.nn as nn
from types import SimpleNamespace
import pytest


class _MockCausalNet(nn.Module):
    """Minimal nn.Module that exposes everything _student_sample_loop needs
    and produces gradients on a single learnable parameter.
    """
    def __init__(self, chunk_size=3, total_num_frames=6):
        super().__init__()
        # Learnable parameter that gradients should flow through
        self.scale = nn.Parameter(torch.tensor(1.0))

        self.chunk_size = chunk_size
        self.total_num_frames = total_num_frames
        self._enable_first_frame_anchor = True
        self._anchor_eval_only = False
        self._model_sink_cache = False
        self._skip_clean_cache_pass = False
        self._kv_caches = None

        # Minimal noise_scheduler surface used by the sample loop
        self.noise_scheduler = SimpleNamespace(
            forward_process=lambda x, eps, t: x * (1 - t.view(-1, 1, 1, 1, 1)) + eps * t.view(-1, 1, 1, 1, 1),
            x0_to_eps=lambda xt, x0, t: (xt - x0 * (1 - t.view(-1, 1, 1, 1, 1))) / t.view(-1, 1, 1, 1, 1).clamp(min=1e-3),
            max_sigma=1.0,
        )
        # Record the last-seen apply_anchor / apply_input_anchor for assertions
        self.last_seen = []

    def clear_caches(self):
        pass

    def forward(self, x, t, **kwargs):
        # Record call metadata for test assertions
        self.last_seen.append({
            "apply_anchor": kwargs.get("apply_anchor", True),
            "apply_input_anchor": kwargs.get("apply_input_anchor", True),
            "cur_start_frame": kwargs.get("cur_start_frame", 0),
            "store_kv": kwargs.get("store_kv", False),
            # Capture the frame-0 value of the input — tells us whether input
            # anchor actually fired in the real network (test will NOT see it
            # because MockNet doesn't dispatch to _maybe_apply_input_anchor,
            # but useful for debugging in richer tests later).
            "input_frame0_mean": x[:, :, 0].mean().item(),
        })
        # Simple differentiable transform: multiply by learnable scale
        return x * self.scale


def _build_condition(B=1, C=16, T=21, H=8, W=8, device="cpu", dtype=torch.float32):
    return {
        "text_embeds": torch.zeros(B, 512, 4096, device=device, dtype=dtype),
        "clip_features": torch.zeros(B, 257, 1280, device=device, dtype=dtype),
        "audio_emb": torch.zeros(B, 93, 5, 12, 768, device=device, dtype=dtype),
        "first_frame_cond": torch.full((B, C, T, H, W), 3.0, device=device, dtype=dtype),
    }


def test_rollout_gradient_flows_with_input_anchor():
    """6-iteration rollout gradient-sanity test.

    Runs the student sample loop 6 times on a toy problem, backprops loss
    each iteration, and verifies grad flows onto the mock network's
    learnable parameter. Shows the new apply_input_anchor plumbing doesn't
    block autograd.
    """
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    torch.manual_seed(0)
    net = _MockCausalNet(chunk_size=3, total_num_frames=6)
    optim = torch.optim.SGD(net.parameters(), lr=0.01)

    B, C, T, H, W = 1, 16, 6, 8, 8
    condition = _build_condition(B=B, C=C, T=T, H=H, W=W)
    t_list = torch.tensor([0.9, 0.5, 0.0])  # 2 denoise steps
    target = torch.randn(B, C, T, H, W)
    initial_scale = net.scale.item()

    losses = []
    for step_idx in range(6):
        noise = torch.randn(B, C, T, H, W)
        out = CausVidModel._student_sample_loop(
            net=net, x=noise, t_list=t_list, condition=condition,
            student_sample_type="sde", context_noise=0.0,
        )
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    # Gradient sanity assertions
    assert all(
        torch.isfinite(torch.tensor(L)) for L in losses
    ), f"Non-finite losses across 6 iterations: {losses}"
    assert net.scale.grad is None or torch.isfinite(net.scale.grad), \
        "Gradient on scale is non-finite"
    assert abs(net.scale.item() - initial_scale) > 1e-6, (
        "Scale parameter did not update — gradients aren't flowing through "
        "the rollout. Check that the new apply_input_anchor plumbing isn't "
        "blocking autograd."
    )


def test_rollout_gradient_with_f2_enabled():
    """Same as above but with F2 on — verifies that propagating
    apply_input_anchor=False alongside apply_anchor=False still allows
    gradient flow.
    """
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    torch.manual_seed(0)
    net = _MockCausalNet(chunk_size=3, total_num_frames=6)
    net._model_sink_cache = True  # F2 ON
    optim = torch.optim.SGD(net.parameters(), lr=0.01)

    B, C, T, H, W = 1, 16, 6, 8, 8
    condition = _build_condition(B=B, C=C, T=T, H=H, W=W)
    t_list = torch.tensor([0.9, 0.5, 0.0])
    target = torch.randn(B, C, T, H, W)
    initial_scale = net.scale.item()

    losses = []
    for step_idx in range(6):
        noise = torch.randn(B, C, T, H, W)
        out = CausVidModel._student_sample_loop(
            net=net, x=noise, t_list=t_list, condition=condition,
            student_sample_type="sde", context_noise=0.0,
        )
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(L)) for L in losses), \
        f"Non-finite losses with F2 on: {losses}"
    assert abs(net.scale.item() - initial_scale) > 1e-6, \
        "Gradients not flowing with F2 on — F2 anchor propagation broke autograd"


def test_rollout_gradient_with_context_noise():
    """Verify gradient flow when context_noise>0 is combined with input anchor.
    This exercises the subtle cache-pass frame-0 pinning (see plan's
    interaction review — non-F2 + context_noise)."""
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    torch.manual_seed(0)
    net = _MockCausalNet(chunk_size=3, total_num_frames=6)
    optim = torch.optim.SGD(net.parameters(), lr=0.01)

    B, C, T, H, W = 1, 16, 6, 8, 8
    condition = _build_condition(B=B, C=C, T=T, H=H, W=W)
    t_list = torch.tensor([0.9, 0.5, 0.0])
    target = torch.randn(B, C, T, H, W)
    initial_scale = net.scale.item()

    for _ in range(6):
        noise = torch.randn(B, C, T, H, W)
        out = CausVidModel._student_sample_loop(
            net=net, x=noise, t_list=t_list, condition=condition,
            student_sample_type="sde", context_noise=0.1,   # <-- noisy KV
        )
        loss = ((out - target) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert abs(net.scale.item() - initial_scale) > 1e-6, \
        "Gradients not flowing with context_noise>0"
```

- [ ] **Step 3: Run the gradient-sanity test**

```bash
python -m pytest tests/test_rollout_gradient_sanity.py -v
```

Expected: 3 tests pass. Runtime should be <5s on CPU.

If any test fails:
- Non-finite losses → probably a shape mismatch in the new post-step pin. Check that the pin only fires on chunk 0 and `x_next.shape[2] >= 1`.
- Scale parameter didn't update → autograd was blocked somewhere. Trace through by setting `torch.autograd.set_detect_anomaly(True)` at the top of the test.
- Test for F2 fails but base passes → the F2 `apply_input_anchor=False` wiring is broken. Double-check Task 4's sample_loop changes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_rollout_gradient_sanity.py
git commit -m "$(cat <<'EOF'
test: 6-iter rollout gradient-sanity test for input-anchor plumbing

Three mock-based tests that run the student sample loop 6 times with
a learnable MockNet parameter, backprop each iteration, and verify:

  1. Baseline (no F2, no context_noise) — gradients flow
  2. F2 enabled — apply_input_anchor=False propagation doesn't break autograd
  3. context_noise>0 — non-F2 cache-pass input anchor compatible with noisy KV

Fast (<5s, CPU-only) and matches the repo's mock-based testing
convention. Does NOT require 14B model loading.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update CLAUDE.md with input-anchor notes

**Files:**
- Modify: `CLAUDE.md` (top-level) — add section near the "Known footguns" or "Anchor" area

- [ ] **Step 1: Read current CLAUDE.md**

```bash
wc -l CLAUDE.md
```

- [ ] **Step 2: Append new section**

Use the Edit tool to add after the "Known footguns" section:

```markdown
## Anchor plumbing (input-side + output-side)

Two independent anchor toggles on CausalInfiniteTalkWan.forward /
InfiniteTalkWan.forward, both default `True`:

- **`apply_anchor`** (output-side, older): pins `out[:, :, 0:1]` after
  the model forward. See `_maybe_apply_first_frame_anchor`.
- **`apply_input_anchor`** (input-side, newer): pins `x_t[:, :, 0:1]`
  at the top of forward() — so the model sees clean frame 0 as INPUT.
  Matches InfiniteTalk's training distribution (verified: teacher
  produces garbage at frame 0 without this; see
  `docs/sf-validation-garbage-output-analysis.md`).

Both respect `_enable_first_frame_anchor` / `_anchor_eval_only` stamped
by `_apply_anchor_config`. Causal forward additionally gates on
`cur_start_frame == 0`.

**F2 interaction**: when F2 (`_model_sink_cache`) wants the cache to
capture the model's raw prediction, callers set both `apply_anchor` and
`apply_input_anchor` to `False` on the same forward. They must move in
lockstep — disabling output anchor alone leaves the cache's K/V coming
from a clean-pinned input, which partially defeats F2.

**Validation/inference path**: `_student_sample_loop` and
`run_inference` additionally apply a post-`forward_process` pin on
chunk 0 (matches `multitalk.py:773`). Redundant with the next step's
input anchor on intermediate denoise steps, but explicitly replicates
InfiniteTalk's full convention.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md — document input-anchor + F2 interaction"
```

---

## Self-Review Summary

Spec coverage:
- ✅ Input-side pinning on teacher/fake (Task 3 plumbs it into bidirectional network; DMD2 and SF_3call_CFG call through default)
- ✅ Input-side pinning on student rollout (Task 2 plumbs it; `rollout_with_gradient` uses default `True` via the kwarg default — no code change needed there)
- ✅ Full InfiniteTalk convention (pre + post step pin) for validation (Task 4) and production inference (Task 5)
- ✅ F2 interaction handled (Tasks 4, 5: `apply_input_anchor` mirrors `apply_anchor` in F2 code paths)
- ✅ F3 interaction reviewed (no change needed — F3 just relocates cache storage to the last denoise step which already has `apply_anchor=True`)
- ✅ `context_noise` interaction reviewed (interaction review section; no code change needed)
- ✅ Integration smoke test (Task 6)
- ✅ Docs (Task 7)

Placeholder scan: no TBD/TODO entries; every step has concrete code or command.

Type consistency: `apply_input_anchor: bool = True` is the only new kwarg, consistent across both network modules and all call sites.
