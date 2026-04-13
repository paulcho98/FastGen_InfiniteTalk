# Lookahead Sink + Model-Generated Sink Cache + Skip-Clean-Cache-Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three independently-toggleable features to the InfiniteTalk causal student for SF training + inference: (F1) lookahead attention sink placed at a configurable future temporal RoPE position; (F2) cache model-generated frame-0 K/V (not the anchor-overwritten reference) for the sink chunk; (F3) skip the separate clean-cache forward pass by letting the last denoise step write the K/V.

**Architecture:** Four orthogonal code surfaces:
1. `CausalSelfAttention.forward` dynamic-RoPE branch — splits `k_win` into sink/rest and rotates each with a different `start_frame` when lookahead enabled. Trigger condition fires in BOTH the contiguous and non-contiguous branches (so training and inference are consistent).
2. `CausalInfiniteTalkWan.forward` — gains an `apply_anchor: bool = True` kwarg that gates the existing anchor overwrite.
3. `CausVidModel._student_sample_loop` — interleaves F2+F3 logic (used in **validation only** via `InfiniteTalkSelfForcingModel.validation_step`). F2 forces `apply_anchor=False` on the last denoise step of chunk 0 and manually anchors the display tensor; F3 sets `store_kv=True` on the last denoise step and skips the separate cache pass. F2 overrides F3 for chunk 0 (keeps the separate cache pass alive) so the model-generated sink K/V is actually produced.
4. **`scripts/inference/inference_causal.py::run_inference`** — the ACTUAL production inference path for InfiniteTalk. Constructs `CausalInfiniteTalkWan` directly (bypassing `_apply_anchor_config`), has its own inline AR loop, dynamically sets `model.total_num_frames` from audio duration, relies on rolling-window cache for arbitrary lengths. Needs CLI flags + parallel F2/F3 logic.

Config fields live on `InfiniteTalkSFModelConfig` (for SF training + validation). `_apply_anchor_config` stamps `_lookahead_sink_enabled`, `_lookahead_distance`, `_model_sink_cache`, `_skip_clean_cache_pass` onto `self.net` so `_student_sample_loop` can read them via `getattr(net, ..., default)`. For production inference (`inference_causal.py`), the same attributes are set directly on the `CausalInfiniteTalkWan` instance after construction, via new CLI flags.

`CausVidModel.generator_fn_extrapolation` is a GENERIC multi-segment utility used only by `scripts/inference/video_model_inference.py` (NOT by InfiniteTalk inference). Modifying it is low-priority; included for consistency but production impact is zero.

**Tech Stack:** PyTorch 2.8, FSDP2, flash_attn 2.8.3, FlexAttention (for non-AR training path), existing causal-network and causvid machinery.

---

## Key Interaction Tables (reference for all tasks)

### Feature 2 + Feature 3 matrix — what the sample loop must do

For each chunk during inference, the `_student_sample_loop` runs `len(t_list) - 1` denoise steps followed by an optional cache-prefill pass. "Last denoise step" = `step == len(t_list) - 2` (the loop index).

`is_sink_chunk = (cur_start_frame == 0)` — only the first chunk is affected by the anchor.

| F2 | F3 | is_sink_chunk | Last denoise `store_kv` | Last denoise `apply_anchor` | Separate cache pass? | Cache pass `apply_anchor` |
|---|---|---|---|---|---|---|
| off | off | any | False | True | Yes | True |
| off | on | any | True | True | No | — |
| on | off | True | False | **False** | Yes | **False** |
| on | off | False | False | True | Yes | True |
| on | on | True | False | **False** | Yes (F2 overrides F3 for chunk 0) | **False** |
| on | on | False | True | True | No | — |

Reading rules:
- `F2=on & is_sink_chunk` ⇒ `apply_anchor=False` on last denoise step AND cache pass; display tensor is anchored manually AFTER last denoise step.
- `F3=on & not (F2 & is_sink_chunk)` ⇒ last denoise `store_kv=True`, no cache pass.
- In all "Yes, separate cache pass" rows, the pass uses `x_cache = x_next` as input.

### Feature 1 scope (revised — applies in both training and inference)

Lookahead applies whenever `k_win` contains a recognizable sink slab. Concretely, in the dynamic-RoPE branch, the trigger is:

```python
use_lookahead = (
    lookahead_sink_enabled
    and sink_tokens > 0
    and k_win.shape[1] > sink_tokens   # sink is NOT the whole window
)
```

This fires in:
- **Contiguous branch (line 543 of current code):** when chunks past chunk 0 read cache, `k_win` is `[sink | rolling | current]` with sink at the start. Covers SF training (full cache) and early-chunk inference.
- **Non-contiguous branch (line 526):** when eviction forces rolling past sink. Covers late-chunk inference only.

This trigger condition is false for chunk 0 itself — when chunk 0 is being generated, `new_local_start == 0` so `k_win` consists only of the current chunk's K (`k_win.shape[1] == current_chunk_tokens == sink_tokens + other_current_tokens`, but since there's no cache content, `k_win` doesn't have a separate "sink slab" to shift). Lookahead is a no-op for chunk 0 — matching the intent that chunk 0 IS the sink and "lookahead to yourself" is meaningless.

### RoPE position formula (revised — preserves Q and rolling positions)

When `use_lookahead == True`:
- **Sink K**: rotated at temporal position `F_window - 1 + lookahead_distance` (one past the last frame in the window, plus the lookahead).
- **Rest K (rolling + current)**: rotated at positions `[sink_frames_n, sink_frames_n + 1, ..., F_window - 1]`. This is identical to what they'd get if lookahead were disabled — sink's old slot at position 0 is simply left empty.
- **Q**: rotated at its natural position `query_offset_in_win // frame_seqlen`. Same in both lookahead and no-lookahead modes.

**Key property:** only the sink's absolute RoPE position changes when the flag flips. Rolling frames, current-chunk Q, and all K-K/Q-K relative distances among non-sink tokens are IDENTICAL in both modes. This minimizes drift from pretrained-DF-checkpoint weights when fine-tuning with lookahead.

Example with `F_window=12, chunk=3, L=4, sink_frames=1`:
- **No lookahead:** sink at 0, rolling at [1..8], Q at [9, 10, 11]. Q-sink distance = 9, 10, 11.
- **With lookahead:** sink at 15, rolling at [1..8], Q at [9, 10, 11]. Q-sink distance = -6, -5, -4 (sink is 4–6 frames ahead of Q).

### Lookahead validation rules

Enforced at `CausalInfiniteTalkWan.__init__` when `lookahead_sink_enabled=True`:
1. `use_dynamic_rope=True` required (static RoPE pre-rotates K before caching).
2. `lookahead_distance >= 1` required (distance=0 would collide sink with the last current frame).
3. `freqs[0].shape[0] >= total_num_frames + lookahead_distance` (freqs table must index the sink's shifted position).

All three raise `ValueError` with clear messages.

---

## File Plan

Files to create or modify, with responsibilities:

**Modified files:**
- `fastgen/configs/methods/config_infinitetalk_sf.py` — add 4 new config fields
- `fastgen/networks/InfiniteTalk/network_causal.py` — add params to `CausalInfiniteTalkWan.__init__`, propagate to `CausalSelfAttention` instance attrs; modify dynamic-RoPE branch in `CausalSelfAttention.forward`; add `apply_anchor` kwarg to `CausalInfiniteTalkWan.forward`
- `fastgen/methods/distribution_matching/causvid.py` — modify `_student_sample_loop` (production validation path) AND `generator_fn_extrapolation` (low-priority, generic utility) per F2+F3 matrix
- `fastgen/methods/infinitetalk_self_forcing.py` — extend `_apply_anchor_config` to also stamp attention/cache toggles onto `self.net`
- **`scripts/inference/inference_causal.py`** — add CLI flags (`--lookahead_sink`, `--lookahead_distance`, `--model_sink_cache`, `--skip_clean_cache_pass`, `--use_dynamic_rope`); pass to `CausalInfiniteTalkWan` constructor / set as post-construction attrs; mirror F2+F3 logic in the inline `run_inference` AR loop

**Created files:**
- `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead.py` — training config with F1 enabled (and dynamic RoPE on), F2 and F3 off by default
- `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_valtest.py` — valtest variant
- `scripts/run_sf_w9s1_lookahead.sh` — training launcher
- `scripts/run_sf_w9s1_lookahead_valtest.sh` — valtest launcher
- `tests/test_lookahead_sink_attention.py` — unit tests for F1 attention math
- `tests/test_apply_anchor_kwarg.py` — unit test for F2's apply_anchor plumbing
- `tests/test_sample_loop_toggles.py` — unit tests for F2+F3 sample-loop logic using a mock network
- `tests/test_inference_causal_toggles.py` — unit tests for the inline AR-loop toggles in `inference_causal.py` using the same mock-network pattern

---

## Task 1: Add the four config fields to `InfiniteTalkSFModelConfig`

**Files:**
- Modify: `fastgen/configs/methods/config_infinitetalk_sf.py`
- Test: `tests/test_config_fields.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_fields.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
python -m pytest tests/test_config_fields.py -v
```
Expected: `AttributeError: 'InfiniteTalkSFModelConfig' object has no attribute 'lookahead_sink_enabled'`

- [ ] **Step 3: Add the fields**

In `fastgen/configs/methods/config_infinitetalk_sf.py`, immediately after the `teacher_anchor_disabled` field (~line 74), add:

```python
    # Lookahead attention sink (Feature 1):
    # When True, the sink K/V (stored at buffer positions [0, sink_size) in
    # frames) is rotated at attention read-time with a RoPE temporal position
    # equal to (last_current_frame_pos + lookahead_distance), placing it
    # "in the future" relative to the current generating block.
    # Requires use_dynamic_rope=True on the student network (the static-RoPE
    # path cannot retroactively shift cached-key positions).
    # Only takes effect in chunks past the sink (cur_start_frame > 0).
    lookahead_sink_enabled: bool = False

    # Distance in frames for lookahead sink. Only meaningful when
    # lookahead_sink_enabled=True. Typical values 0..8.
    lookahead_distance: int = 0

    # Model-generated sink cache (Feature 2):
    # When True, the last denoise step of the sink chunk (cur_start_frame=0)
    # runs with apply_anchor=False, and its output is used as the input to a
    # subsequent cache-prefill forward pass so the cached sink K/V is computed
    # from the student's OWN frame-0 prediction (not the reference-image
    # overwrite). The displayed video still has the reference image at frame 0
    # (anchor applied manually outside forward). Inference-time only; no effect
    # during training because anchoring is gated on self.training.
    model_sink_cache_enabled: bool = False

    # Skip clean-cache pass (Feature 3):
    # When True, the separate cache-prefill forward pass after each chunk's
    # denoise loop is skipped. Instead, the last denoise step runs with
    # store_kv=True, so the K/V cached for that chunk comes from the (slightly
    # noisy) input to the last denoise step (at t=t_list[-2]).
    # Saves 1/(sample_steps+1) of inference forwards per chunk.
    # For the sink chunk, model_sink_cache_enabled=True overrides this setting
    # and keeps the separate cache pass alive (otherwise F2 cannot function).
    skip_clean_cache_pass: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config_fields.py -v
```
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fastgen/configs/methods/config_infinitetalk_sf.py tests/test_config_fields.py
git commit -m "feat: add lookahead-sink + model-sink-cache + skip-cache-pass config fields

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add `apply_anchor` kwarg to `CausalInfiniteTalkWan.forward`

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:1805-1974` (the top-level `forward`)
- Test: `tests/test_apply_anchor_kwarg.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_apply_anchor_kwarg.py`:

```python
"""Unit test that apply_anchor=False suppresses the frame-0 overwrite.

Uses a lightweight stand-in: we subclass CausalInfiniteTalkWan minimally
and monkey-patch _forward_ar / _forward_full_sequence to return a known
tensor. The test asserts that with apply_anchor=False, the returned tensor
is NOT overwritten with first_frame_cond, and with apply_anchor=True it IS.
"""
import torch


def test_apply_anchor_false_skips_overwrite():
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    # Stub model output: known garbage values at frame 0
    stub_output = torch.full((1, 16, 3, 28, 56), 7.0)
    first_frame_cond = torch.full((1, 16, 3, 28, 56), -3.0)

    class StubNet:
        """Minimal stub exposing only what the anchor block on forward touches."""
        training = False
        _enable_first_frame_anchor = True
        _anchor_eval_only = False
        # The noise_scheduler is called for rescale_t + convert_model_output.
        # For this test we bypass forward() almost entirely by monkey-patching.

    # Instead of instantiating the real network (expensive), we directly exercise
    # the anchor logic as a standalone function call. Extract the anchor block
    # into a helper in the plan.
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_first_frame_anchor

    out_anchored = _maybe_apply_first_frame_anchor(
        stub_output.clone(), StubNet(), cur_start_frame=0,
        condition={"first_frame_cond": first_frame_cond},
        apply_anchor=True,
    )
    out_not_anchored = _maybe_apply_first_frame_anchor(
        stub_output.clone(), StubNet(), cur_start_frame=0,
        condition={"first_frame_cond": first_frame_cond},
        apply_anchor=False,
    )

    # apply_anchor=True: frame 0 is overwritten with -3.0
    assert out_anchored[:, :, 0:1].eq(-3.0).all()
    # apply_anchor=False: frame 0 keeps its original 7.0
    assert out_not_anchored[:, :, 0:1].eq(7.0).all()


def test_apply_anchor_nonzero_start_frame_is_noop():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_first_frame_anchor

    stub_output = torch.full((1, 16, 3, 28, 56), 7.0)
    first_frame_cond = torch.full((1, 16, 3, 28, 56), -3.0)

    class StubNet:
        training = False
        _enable_first_frame_anchor = True
        _anchor_eval_only = False

    # cur_start_frame > 0 — anchor never applies regardless of apply_anchor flag
    out = _maybe_apply_first_frame_anchor(
        stub_output.clone(), StubNet(), cur_start_frame=3,
        condition={"first_frame_cond": first_frame_cond},
        apply_anchor=True,
    )
    assert out[:, :, 0:1].eq(7.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_apply_anchor_kwarg.py -v
```
Expected: `ImportError: cannot import name '_maybe_apply_first_frame_anchor'`

- [ ] **Step 3: Extract the anchor block into a module-level helper and add the kwarg**

In `fastgen/networks/InfiniteTalk/network_causal.py`, add the helper just before the `CausalInfiniteTalkWan` class definition (~line 930):

```python
def _maybe_apply_first_frame_anchor(
    out: torch.Tensor,
    net_module,
    cur_start_frame: int,
    condition,
    apply_anchor: bool = True,
) -> torch.Tensor:
    """Hard-anchor frame 0 to the clean reference when enabled.

    Modes (instance attributes on net_module):
      _enable_first_frame_anchor = True (default): anchor is active
      _enable_first_frame_anchor = False: anchor fully disabled
      _anchor_eval_only = True: anchor only in eval mode (not during training)

    The explicit ``apply_anchor=False`` argument overrides all of the above —
    used by the sampling loop's model-generated-sink-cache path.
    """
    if not apply_anchor:
        return out
    if cur_start_frame != 0:
        return out
    if not isinstance(condition, dict) or "first_frame_cond" not in condition:
        return out
    anchor_active = getattr(net_module, "_enable_first_frame_anchor", True)
    if anchor_active and getattr(net_module, "_anchor_eval_only", False):
        anchor_active = not net_module.training
    if not anchor_active:
        return out
    first_frame_cond = condition["first_frame_cond"]
    out = out.clone()
    out[:, :, 0:1] = first_frame_cond[:, :, 0:1]
    return out
```

Then replace the inline anchor block in `CausalInfiniteTalkWan.forward` (currently at lines 1947-1960):

```python
        # Hard-anchor frame 0 to clean reference when processing the first chunk.
        # See _maybe_apply_first_frame_anchor for mode semantics.
        out = _maybe_apply_first_frame_anchor(
            out, self, cur_start_frame, condition, apply_anchor=apply_anchor,
        )
```

And add `apply_anchor: bool = True` to the `forward` signature (currently ends at line 1819 with `**fwd_kwargs`). Insert BEFORE `**fwd_kwargs`:

```python
        apply_anchor: bool = True,
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_apply_anchor_kwarg.py -v
```
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_apply_anchor_kwarg.py
git commit -m "refactor: extract anchor overwrite to helper, add apply_anchor forward kwarg

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Add lookahead-sink params to `CausalInfiniteTalkWan` + propagate to attention

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py`
  - `CausalSelfAttention.__init__` (line 347) — add default instance attrs
  - `CausalInfiniteTalkWan.__init__` (line 951) — accept params, propagate
- Test: `tests/test_lookahead_sink_attention.py` (create, only default-values test at this stage)

- [ ] **Step 1: Write the failing test**

Create `tests/test_lookahead_sink_attention.py`:

```python
"""Unit tests for lookahead sink attention-math behavior."""
import math
import torch


def test_attention_module_default_lookahead_fields():
    """CausalSelfAttention has lookahead fields defaulting to disabled."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalSelfAttention

    attn = CausalSelfAttention(
        dim=64, num_heads=4, local_attn_size=10, sink_size=1,
        use_dynamic_rope=True,
    )
    assert attn.lookahead_sink_enabled is False
    assert attn.lookahead_distance == 0


def test_causal_wan_propagates_lookahead_to_attention():
    """CausalInfiniteTalkWan constructor forwards lookahead_* to each block's
    self-attention module."""
    # Build a minimal 2-block network (tiny hidden dims so init is fast).
    # Skip if model weight load fails — this test only exercises constructor plumbing.
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan

    # Build with no LoRA and no base weights — pure constructor path.
    net = CausalInfiniteTalkWan(
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
        lookahead_distance=4,
        # Force a minimal config to avoid loading the 14B checkpoint:
        skip_weight_load=True,
    )
    for block in net.blocks:
        assert block.self_attn.lookahead_sink_enabled is True
        assert block.self_attn.lookahead_distance == 4
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_lookahead_sink_attention.py::test_attention_module_default_lookahead_fields -v
```
Expected: `AttributeError: 'CausalSelfAttention' object has no attribute 'lookahead_sink_enabled'`

- [ ] **Step 3: Add default instance attrs to `CausalSelfAttention.__init__`**

In `fastgen/networks/InfiniteTalk/network_causal.py`, `CausalSelfAttention.__init__` (ends at line 376 with `self.norm_k`), add right after `self.norm_k`:

```python
        # Lookahead sink (Feature 1) — toggleable, defaults off.
        # Set externally by CausalInfiniteTalkWan.__init__ when propagating.
        self.lookahead_sink_enabled: bool = False
        self.lookahead_distance: int = 0
```

- [ ] **Step 4: Run first test to verify it passes**

```bash
python -m pytest tests/test_lookahead_sink_attention.py::test_attention_module_default_lookahead_fields -v
```
Expected: PASS.

- [ ] **Step 5: Add constructor params + propagation to `CausalInfiniteTalkWan`**

In `fastgen/networks/InfiniteTalk/network_causal.py`, find the `CausalInfiniteTalkWan.__init__` signature (starts line 951). Examine existing params `local_attn_size` and `sink_size` and add next to them:

```python
        lookahead_sink_enabled: bool = False,
        lookahead_distance: int = 0,
```

Then in the body, after `self.local_attn_size = local_attn_size` etc. assignments (search for `self.local_attn_size`), add:

```python
        self._lookahead_sink_enabled = lookahead_sink_enabled
        self._lookahead_distance = lookahead_distance

        # Validate lookahead config at construction time
        if lookahead_sink_enabled:
            if not use_dynamic_rope:
                raise ValueError(
                    "lookahead_sink_enabled=True requires use_dynamic_rope=True "
                    "on the causal student. The static-RoPE path caches keys with "
                    "absolute positions already applied and cannot retroactively "
                    "shift the sink's position."
                )
            if lookahead_distance < 1:
                raise ValueError(
                    f"lookahead_sink_enabled=True requires lookahead_distance >= 1, "
                    f"got {lookahead_distance}. distance=0 would collide sink with "
                    f"the last current frame."
                )
            # Freqs table bound: sink position = (total_num_frames - 1) + lookahead_distance
            # at worst. freqs[0] must cover it.
            max_pos_needed = total_num_frames + lookahead_distance
            if self.freqs[0].shape[0] < max_pos_needed:
                raise ValueError(
                    f"freqs[0] has {self.freqs[0].shape[0]} temporal positions but "
                    f"lookahead needs at least {max_pos_needed} "
                    f"(total_num_frames={total_num_frames} + lookahead_distance="
                    f"{lookahead_distance}). Enlarge freq table at model construction "
                    f"or reduce lookahead_distance."
                )
```

Then, after the `self.blocks = nn.ModuleList([...])` construction, add propagation:

```python
        # Propagate lookahead config to every block's self-attention module
        for block in self.blocks:
            block.self_attn.lookahead_sink_enabled = lookahead_sink_enabled
            block.self_attn.lookahead_distance = lookahead_distance
```

- [ ] **Step 6: Run second test to verify it passes**

```bash
python -m pytest tests/test_lookahead_sink_attention.py::test_causal_wan_propagates_lookahead_to_attention -v
```
Expected: PASS. (Note: if the test errors on `skip_weight_load` not being a real kwarg, you'll need to handle that flag — if it's simpler, use an existing test fixture or make the test an integration test via the real configs. As a fallback, mark `test_causal_wan_propagates_lookahead_to_attention` with `@pytest.mark.integration` and skip in CI; verify manually via `python -c "import ..."` later.)

- [ ] **Step 7: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_lookahead_sink_attention.py
git commit -m "feat: add lookahead_sink_enabled/lookahead_distance params to causal network

Constructor params propagate to each block's CausalSelfAttention module.
Lookahead requires use_dynamic_rope=True; enforced at construction time.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Implement lookahead RoPE split in `CausalSelfAttention.forward`

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py:556-574` (dynamic-RoPE branch)
- Test: `tests/test_lookahead_sink_attention.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lookahead_sink_attention.py`:

```python
def test_lookahead_rope_math_on_sink_portion():
    """Verify that when lookahead_sink_enabled=True and sink_tokens > 0, the
    dynamic-RoPE branch rotates the sink portion at a different temporal
    position than the rest of the window."""
    import math
    from fastgen.networks.InfiniteTalk.network_causal import (
        CausalSelfAttention, causal_rope_apply,
    )

    # Tiny config: 1 frame_seqlen=4 tokens, 1 sink frame, 4 rolling frames, 1 current frame.
    # Window has 6 frames × 4 tokens = 24 tokens.
    B, n_heads, d = 1, 2, 16
    dim = n_heads * d
    frame_seqlen = 4
    sink_frames = 1
    rest_frames = 5  # 4 rolling + 1 current
    F_window = sink_frames + rest_frames

    torch.manual_seed(0)

    attn = CausalSelfAttention(
        dim=dim, num_heads=n_heads, local_attn_size=F_window, sink_size=sink_frames,
        use_dynamic_rope=True,
    )
    attn.lookahead_sink_enabled = True
    attn.lookahead_distance = 4

    # Build synthetic K of shape [B, F_window*frame_seqlen, n_heads, d] with
    # easily-distinguishable sink vs rest values.
    k_win_sink = torch.full((B, sink_frames * frame_seqlen, n_heads, d), 1.0)
    k_win_rest = torch.full((B, rest_frames * frame_seqlen, n_heads, d), 2.0)
    k_win = torch.cat([k_win_sink, k_win_rest], dim=1)

    # Mimic the RoPE application from the new dynamic+lookahead branch.
    # Sink at position F_window - 1 + lookahead_distance = 5 + 4 = 9.
    # Rest at positions [1..5] (natural positions — sink's slot 0 left empty).
    grid_sizes = torch.tensor([[sink_frames, 2, 2]], dtype=torch.long)
    freqs = attn.__class__.__dict__  # stub to make test self-contained: use real freqs
    # Actually, we need freqs. Easiest: build them from a real CausalInfiniteTalkWan
    # or use a helper. For this test, use a simplified freqs that matches head_dim.
    # Rationale: we only care that the position indices differ between sink and rest.

    # Build minimal freqs (f/h/w) sized for a head_dim of 16: per-axis size = 16//6 rounded.
    # To keep scope small, mark this test as checking POSITION indices via logging —
    # we monkey-patch causal_rope_apply to record start_frame and caller args.

    calls = []
    original_rope = causal_rope_apply

    def spy_rope(x, grid, freqs, start_frame=0):
        calls.append({
            "x_shape": tuple(x.shape),
            "start_frame": start_frame,
            "f": int(grid[0, 0]),
        })
        return x  # return unchanged; we only care about the recording

    import fastgen.networks.InfiniteTalk.network_causal as ncm
    ncm.causal_rope_apply = spy_rope
    try:
        # Drive the relevant code path. Easiest: build a tiny q and call forward with a
        # minimal kv_cache. But that's brittle. Alternative: call the subroutine we
        # factor out in Step 3 — see the helper extraction there.
        from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

        q = torch.randn(B, frame_seqlen, n_heads, d)  # 1 current frame worth of queries
        query_offset_in_win = sink_frames * frame_seqlen + 4 * frame_seqlen  # q at last frame
        roped_q, roped_k = _apply_window_rope(
            q=q, k_win=k_win, grid_sizes=grid_sizes,
            freqs=(torch.zeros(64, 4, dtype=torch.complex64),) * 3,  # stub
            frame_seqlen=frame_seqlen, sink_tokens=sink_frames * frame_seqlen,
            query_offset_in_win=query_offset_in_win,
            lookahead_sink_enabled=True, lookahead_distance=4, use_dynamic_rope=True,
        )
        # Three calls expected: sink K, rest K, Q
        assert len(calls) == 3
        sink_call, rest_call, q_call = calls
        # Sink: 1 frame, start_frame = F_window - 1 + lookahead_distance = 5 + 4 = 9
        assert sink_call["f"] == sink_frames
        assert sink_call["start_frame"] == 9
        # Rest: 5 frames, start_frame = sink_frames_n = 1
        # (rest keeps its natural positions [1, 2, 3, 4, 5] — sink's position 0 is empty)
        assert rest_call["f"] == rest_frames
        assert rest_call["start_frame"] == 1
        # Q: 1 frame, start_frame = query_offset_in_win // frame_seqlen = 5
        # (natural position, same as no-lookahead case)
        assert q_call["f"] == 1
        assert q_call["start_frame"] == 5
    finally:
        ncm.causal_rope_apply = original_rope


def test_lookahead_disabled_matches_original_rope_positions():
    """When lookahead_sink_enabled=False, the dynamic-RoPE path uses contiguous
    positions [0..F_window-1] for the whole k_win and start_frame for Q based
    on query_offset_in_win // frame_seqlen (existing behavior)."""
    import fastgen.networks.InfiniteTalk.network_causal as ncm
    from fastgen.networks.InfiniteTalk.network_causal import _apply_window_rope

    calls = []

    def spy_rope(x, grid, freqs, start_frame=0):
        calls.append({"start_frame": start_frame, "f": int(grid[0, 0])})
        return x

    original_rope = ncm.causal_rope_apply
    ncm.causal_rope_apply = spy_rope
    try:
        B, frame_seqlen = 1, 4
        k_win = torch.zeros(B, 6 * frame_seqlen, 2, 16)
        q = torch.zeros(B, frame_seqlen, 2, 16)
        grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)

        _apply_window_rope(
            q=q, k_win=k_win, grid_sizes=grid_sizes,
            freqs=(torch.zeros(64, 4, dtype=torch.complex64),) * 3,
            frame_seqlen=frame_seqlen, sink_tokens=frame_seqlen,  # sink=1 frame
            query_offset_in_win=5 * frame_seqlen,  # q at last frame (pos 5)
            lookahead_sink_enabled=False, lookahead_distance=0, use_dynamic_rope=True,
        )
        # Two calls: K (whole window, start_frame=0), Q (last frame, start_frame=5)
        assert len(calls) == 2
        k_call, q_call = calls
        assert k_call["f"] == 6 and k_call["start_frame"] == 0
        assert q_call["f"] == 1 and q_call["start_frame"] == 5
        # Also verify that toggling lookahead ON with same inputs produces
        # the SAME Q start_frame (property: Q position unchanged by flag).
        calls.clear()
        _apply_window_rope(
            q=q, k_win=k_win, grid_sizes=grid_sizes,
            freqs=(torch.zeros(64, 4, dtype=torch.complex64),) * 3,
            frame_seqlen=frame_seqlen, sink_tokens=frame_seqlen,
            query_offset_in_win=5 * frame_seqlen,
            lookahead_sink_enabled=True, lookahead_distance=4, use_dynamic_rope=True,
        )
        # Three calls: sink K, rest K, Q. Q start_frame should be 5 (unchanged).
        assert len(calls) == 3
        q_call_after = calls[2]
        assert q_call_after["f"] == 1 and q_call_after["start_frame"] == 5
    finally:
        ncm.causal_rope_apply = original_rope
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_lookahead_sink_attention.py::test_lookahead_rope_math_on_sink_portion -v
```
Expected: `ImportError: cannot import name '_apply_window_rope'`

- [ ] **Step 3: Extract the dynamic-RoPE branch into a helper, add lookahead logic**

In `fastgen/networks/InfiniteTalk/network_causal.py`, add a module-level helper right after `causal_rope_apply` (line ~240):

```python
def _apply_window_rope(
    q: torch.Tensor,
    k_win: torch.Tensor,
    grid_sizes: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    frame_seqlen: int,
    sink_tokens: int,
    query_offset_in_win: int,
    *,
    lookahead_sink_enabled: bool,
    lookahead_distance: int,
    use_dynamic_rope: bool,
    # Fallback (static-RoPE) tensors for when use_dynamic_rope=False.
    # Callers pass the already-rotated q and key window.
    static_roped_q: Optional[torch.Tensor] = None,
    static_k_win: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE for the AR attention window.

    Dynamic-RoPE mode rotates Q and the (un-rotated) k_win at read time. When
    lookahead is active (lookahead_sink_enabled=True AND sink_tokens > 0 AND
    k_win contains a distinct sink slab at its front):
      - Sink K rotated at position F_window - 1 + lookahead_distance
      - Rest K (rolling + current) rotated at positions
        [sink_frames_n, ..., F_window - 1] (sink's old slot at position 0 is
        left empty; rest keeps the positions it would have without lookahead)
      - Q rotated at its natural position query_offset_in_win // frame_seqlen
        (same in both lookahead and no-lookahead modes)

    This design keeps Q and all non-sink K at identical positions regardless of
    the lookahead flag. ONLY the sink's position changes — minimizes drift
    from pretrained weights when enabling the feature.

    Static-RoPE mode returns the pre-rotated tensors unchanged.

    Returns:
        (roped_q, roped_k) ready for flash_attn.
    """
    if not use_dynamic_rope:
        return static_roped_q, static_k_win

    F_window = k_win.shape[1] // frame_seqlen
    # Lookahead applies whenever sink is a separate slab in k_win — this fires
    # in both the contiguous-branch and non-contiguous-branch callsites, so
    # training (full cache, contiguous) and inference (small cache, possible
    # non-contiguous) are consistent.
    use_lookahead = (
        lookahead_sink_enabled
        and sink_tokens > 0
        and sink_tokens < k_win.shape[1]
    )

    if use_lookahead:
        sink_frames_n = sink_tokens // frame_seqlen
        rest_frames_n = F_window - sink_frames_n

        # Sink shifted: placed lookahead_distance frames past the last window frame
        sink_pos = F_window - 1 + lookahead_distance

        sink_grid = grid_sizes.clone()
        sink_grid[:, 0] = sink_frames_n
        k_sink_roped = causal_rope_apply(
            k_win[:, :sink_tokens], sink_grid, freqs, start_frame=sink_pos,
        ).type_as(q)

        # Rest keeps its "natural" positions [sink_frames_n, ..., F_window-1]
        # — exactly what it would have without the flag, minus sink's position 0.
        rest_grid = grid_sizes.clone()
        rest_grid[:, 0] = rest_frames_n
        k_rest_roped = causal_rope_apply(
            k_win[:, sink_tokens:], rest_grid, freqs, start_frame=sink_frames_n,
        ).type_as(q)

        roped_key = torch.cat([k_sink_roped, k_rest_roped], dim=1)
    else:
        k_grid = grid_sizes.clone()
        k_grid[:, 0] = F_window
        roped_key = causal_rope_apply(
            k_win, k_grid, freqs, start_frame=0,
        ).type_as(q)

    # Q position: natural, same in both branches.
    q_frame_start = query_offset_in_win // frame_seqlen
    roped_query = causal_rope_apply(
        q, grid_sizes, freqs, start_frame=q_frame_start,
    ).type_as(q)
    return roped_query, roped_key
```

Then in `CausalSelfAttention.forward` (line ~556), replace the dynamic-RoPE branch (lines 556-574) with:

```python
            # -- 6. Apply RoPE (mode-dependent, sink-aware) --
            roped_query, roped_key = _apply_window_rope(
                q=q,
                k_win=k_win,
                grid_sizes=grid_sizes,
                freqs=freqs,
                frame_seqlen=frame_seqlen,
                sink_tokens=sink_tokens,
                query_offset_in_win=query_offset_in_win,
                lookahead_sink_enabled=self.lookahead_sink_enabled,
                lookahead_distance=self.lookahead_distance,
                use_dynamic_rope=self.use_dynamic_rope,
                static_roped_q=(roped_q if not self.use_dynamic_rope else None),
                static_k_win=(k_win if not self.use_dynamic_rope else None),
            )
```

(This replaces the entire `if not self.use_dynamic_rope: ... else: ...` block currently on lines 556-574.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_lookahead_sink_attention.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_lookahead_sink_attention.py
git commit -m "feat: lookahead sink attention — sink RoPE at future position

Extracts window-RoPE into _apply_window_rope helper. When lookahead enabled,
rotates sink portion of k_win at temporal position rest_frames-1+distance
while rest uses window-local positions [0..rest_frames-1].

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Modify `_student_sample_loop` for F2+F3 logic

**Files:**
- Modify: `fastgen/methods/distribution_matching/causvid.py:87-185`
- Test: `tests/test_sample_loop_toggles.py` (create)

- [ ] **Step 1: Write the failing test using a mock network**

Create `tests/test_sample_loop_toggles.py`:

```python
"""Test the F2 (model-sink-cache) and F3 (skip-clean-cache-pass) interplay
in CausVidModel._student_sample_loop.

We mock the net with a class that records every forward call (store_kv,
apply_anchor, cur_start_frame, t value) and returns a known tensor. We then
verify the recorded call sequence against the expected matrix.
"""
import torch


class MockNoiseScheduler:
    def x0_to_eps(self, xt, x0, t):
        return xt - x0

    def forward_process(self, x, eps, t):
        return x  # identity for test

    def get_t_list(self, steps, device=None, dtype=None):
        return torch.tensor([0.999, 0.955, 0.875, 0.700, 0.0])


class MockNet:
    """Records every forward call and returns a fixed shape."""

    def __init__(self, chunk_size=3, total_frames=9, model_sink_cache=False,
                 skip_clean_cache=False):
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
            "t": float(t[0]) if t.numel() else 0.0,
            "store_kv": kwargs.get("store_kv", False),
            "apply_anchor": kwargs.get("apply_anchor", True),
            "cur_start_frame": kwargs.get("cur_start_frame", 0),
        })
        return torch.zeros_like(x)


def _run_loop(mock_net, **kwargs):
    from fastgen.methods.distribution_matching.causvid import CausVidModel
    x = torch.zeros(1, 16, mock_net.total_num_frames, 28, 56)
    t_list = torch.tensor([0.999, 0.955, 0.875, 0.700, 0.0])
    return CausVidModel._student_sample_loop(
        net=mock_net, x=x, t_list=t_list,
        condition={"first_frame_cond": torch.zeros_like(x)},
        student_sample_type="sde",
        **kwargs,
    )


def test_both_off_preserves_current_behavior():
    """F2 off, F3 off: 4 denoise steps with store_kv=False, apply_anchor=True
    + 1 cache pass with store_kv=True, apply_anchor=True, per chunk."""
    net = MockNet(model_sink_cache=False, skip_clean_cache=False)
    _run_loop(net)
    # 3 chunks (9 frames / 3 chunk_size), 5 calls each (4 denoise + 1 cache) = 15
    assert len(net.calls) == 15
    # Every denoise step has store_kv=False; cache pass has store_kv=True
    for chunk_idx in range(3):
        chunk_calls = net.calls[chunk_idx * 5:(chunk_idx + 1) * 5]
        for i, c in enumerate(chunk_calls[:4]):  # 4 denoise steps
            assert c["store_kv"] is False
            assert c["apply_anchor"] is True
        # Cache pass (call index 4)
        assert chunk_calls[4]["store_kv"] is True
        assert chunk_calls[4]["apply_anchor"] is True


def test_f3_on_skips_cache_pass():
    """F3 on, F2 off: 4 denoise steps; last one has store_kv=True; no cache pass."""
    net = MockNet(model_sink_cache=False, skip_clean_cache=True)
    _run_loop(net)
    # 3 chunks × 4 calls each = 12
    assert len(net.calls) == 12
    for chunk_idx in range(3):
        chunk_calls = net.calls[chunk_idx * 4:(chunk_idx + 1) * 4]
        assert chunk_calls[0]["store_kv"] is False
        assert chunk_calls[1]["store_kv"] is False
        assert chunk_calls[2]["store_kv"] is False
        assert chunk_calls[3]["store_kv"] is True  # last denoise step writes cache
        for c in chunk_calls:
            assert c["apply_anchor"] is True


def test_f2_on_suppresses_anchor_on_sink_last_step_only():
    """F2 on, F3 off:
    - Chunk 0 last denoise step: apply_anchor=False
    - Chunk 0 cache pass: apply_anchor=False
    - All other calls: apply_anchor=True
    - Still 5 calls per chunk (cache pass preserved for all chunks)."""
    net = MockNet(model_sink_cache=True, skip_clean_cache=False)
    _run_loop(net)
    assert len(net.calls) == 15

    chunk0 = net.calls[0:5]
    # Steps 0-2: apply_anchor=True
    for i in range(3):
        assert chunk0[i]["apply_anchor"] is True
    # Step 3 (last denoise): apply_anchor=False (F2 activates)
    assert chunk0[3]["apply_anchor"] is False
    assert chunk0[3]["store_kv"] is False  # F3 off, still no store here
    # Cache pass: apply_anchor=False (F2: cache from model output, not anchored)
    assert chunk0[4]["apply_anchor"] is False
    assert chunk0[4]["store_kv"] is True

    # Chunks 1, 2: unchanged (anchor never fires for cur_start_frame > 0)
    for chunk_idx in [1, 2]:
        for c in net.calls[chunk_idx * 5:(chunk_idx + 1) * 5]:
            assert c["apply_anchor"] is True


def test_f2_and_f3_on_f2_overrides_for_sink_chunk():
    """F2 on, F3 on:
    - Chunk 0 keeps separate cache pass (F2 overrides F3 for sink chunk).
    - Chunks > 0 skip cache pass (F3 active).
    - Chunk 0 last denoise: apply_anchor=False, store_kv=False.
    - Chunk 0 cache pass: apply_anchor=False, store_kv=True.
    - Other chunks last denoise: store_kv=True, apply_anchor=True.
    """
    net = MockNet(model_sink_cache=True, skip_clean_cache=True)
    _run_loop(net)
    # Chunk 0: 5 calls (F3 overridden). Chunks 1, 2: 4 calls each. Total 13.
    assert len(net.calls) == 13

    chunk0 = net.calls[0:5]
    assert chunk0[3]["apply_anchor"] is False
    assert chunk0[3]["store_kv"] is False
    assert chunk0[4]["apply_anchor"] is False
    assert chunk0[4]["store_kv"] is True

    chunk1 = net.calls[5:9]
    assert len(chunk1) == 4
    for i in range(3):
        assert chunk1[i]["store_kv"] is False
    assert chunk1[3]["store_kv"] is True  # F3 active
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_sample_loop_toggles.py -v
```
Expected: all four FAIL — one will pass accidentally (test_both_off), but `test_f3_on_skips_cache_pass` will fail because F3 logic doesn't exist yet; `test_f2_on_suppresses_anchor` will fail because apply_anchor isn't passed through.

- [ ] **Step 3: Modify `_student_sample_loop`**

In `fastgen/methods/distribution_matching/causvid.py`, replace the body of `_student_sample_loop` (lines 113-184) with the toggle-aware version:

```python
        logger.debug("Using generator_fn in CausVidModel")

        # Read toggles from the net module (set by _apply_anchor_config).
        model_sink_cache = getattr(net, "_model_sink_cache", False)
        skip_clean_cache = getattr(net, "_skip_clean_cache_pass", False)

        # cleanup caches before sampling
        net.clear_caches()

        batch_size, num_frames = x.shape[0], x.shape[2]
        chunk_size = net.chunk_size
        num_chunks = num_frames // chunk_size
        remaining_size = num_frames % chunk_size

        num_denoise_steps = len(t_list) - 1  # e.g. 4 for t_list of length 5
        last_step_idx = num_denoise_steps - 1

        for i in range(max(1, num_chunks)):
            if num_chunks == 0:
                start, end = 0, remaining_size
            else:
                start = 0 if i == 0 else chunk_size * i + remaining_size
                end = chunk_size * (i + 1) + remaining_size

            is_sink_chunk = (start == 0)
            # F2: suppress anchor on sink-chunk's last denoise step AND cache pass
            f2_active = model_sink_cache and is_sink_chunk
            # F3: skip separate cache pass, unless F2 is active for this chunk
            # (F2 requires a separate pass to cache the model-generated frame 0)
            do_cache_pass = (not skip_clean_cache) or f2_active

            x_next = x[:, :, start:end, ...]
            for step in range(num_denoise_steps):
                is_last = (step == last_step_idx)
                # store_kv only on last step if F3 on AND we're NOT overriding
                # with F2's separate-cache-pass requirement
                store_kv_here = skip_clean_cache and is_last and not f2_active
                # apply_anchor only matters on sink chunk's last denoise step
                apply_anchor_here = not (f2_active and is_last)

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

                # Forward process to next timestep
                t_next = t_list[step + 1]
                if t_next > 0:
                    t_chunk_next = t_next.expand(batch_size)
                    if student_sample_type == "sde":
                        eps_infer = torch.randn_like(x_next)
                    elif student_sample_type == "ode":
                        eps_infer = net.noise_scheduler.x0_to_eps(xt=x_cur, x0=x_next, t=t_cur)
                    else:
                        raise NotImplementedError(
                            f"student_sample_type must be one of 'sde', 'ode' but got {student_sample_type}"
                        )
                    x_next = net.noise_scheduler.forward_process(x_next, eps_infer, t_chunk_next)

            # Store displayed output. If F2 active, manually anchor frame 0 for display.
            if f2_active and isinstance(condition, dict) and "first_frame_cond" in condition:
                x_display = x_next.clone()
                x_display[:, :, 0:1] = condition["first_frame_cond"][:, :, 0:1]
                x[:, :, start:end, ...] = x_display
            else:
                x[:, :, start:end, ...] = x_next

            # Separate cache pass (skipped by F3 unless F2 forces it for sink chunk)
            if do_cache_pass:
                x_cache = x_next  # NON-anchored when f2_active, anchored otherwise
                t_cache = t_list[-1].expand(batch_size)
                if context_noise > 0:
                    t_cache = torch.full(
                        (batch_size,), context_noise, device=x.device, dtype=x.dtype
                    )
                    x_cache = net.noise_scheduler.forward_process(
                        x_next, torch.randn_like(x_next), t_cache
                    )
                # apply_anchor on cache pass: False when F2 active (cache model's K/V,
                # not ref-image's K/V). Also harmless for non-sink chunks since anchor
                # gates on cur_start_frame==0 anyway.
                apply_anchor_cache = not f2_active
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

        # cleanup caches after full sampling
        net.clear_caches()
        return x
```

- [ ] **Step 4: Run tests to verify all four pass**

```bash
python -m pytest tests/test_sample_loop_toggles.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fastgen/methods/distribution_matching/causvid.py tests/test_sample_loop_toggles.py
git commit -m "feat: F2 (model-sink-cache) + F3 (skip-clean-cache-pass) toggles in student_sample_loop

F2 causes chunk 0's last denoise step and cache pass to run with apply_anchor=False;
display tensor is manually anchored afterward. F3 skips the separate cache pass
and sets store_kv=True on the last denoise step. F2 overrides F3 for chunk 0
(separate cache pass is required to cache model-generated K/V).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6 (LOW PRIORITY): Mirror F2+F3 logic in `generator_fn_extrapolation`

> **Note:** `generator_fn_extrapolation` is used ONLY by `scripts/inference/video_model_inference.py` (generic multi-segment utility). InfiniteTalk production inference does NOT use it (uses `inference_causal.py::run_inference` instead — see Task 6b). Keeping this task for consistency with the generic path. Skip if short on time.



**Files:**
- Modify: `fastgen/methods/distribution_matching/causvid.py:187-428` (the `generator_fn_extrapolation` classmethod)
- Test: extend `tests/test_sample_loop_toggles.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sample_loop_toggles.py`:

```python
def test_extrapolation_honors_f2_f3_toggles():
    """Same matrix as _student_sample_loop, but through generator_fn_extrapolation
    with num_segments=1 (so it reduces to a single _run_segment)."""
    from fastgen.methods.distribution_matching.causvid import CausVidModel

    net = MockNet(model_sink_cache=True, skip_clean_cache=True, total_frames=9)
    noise = torch.zeros(1, 16, 9, 28, 56)
    t_list = torch.tensor([0.999, 0.955, 0.875, 0.700, 0.0])

    CausVidModel.generator_fn_extrapolation(
        net=net, noise=noise,
        condition={"first_frame_cond": torch.zeros_like(noise)},
        num_segments=1, overlap_frames=0,
        student_sample_steps=4, student_sample_type="sde",
        t_list=t_list, precision_amp=None, context_noise=0,
    )
    # Same counts as single-loop (num_segments=1): 13 calls total.
    assert len(net.calls) == 13
    # Chunk 0 has 5 calls; last denoise + cache pass both apply_anchor=False
    chunk0 = net.calls[0:5]
    assert chunk0[3]["apply_anchor"] is False
    assert chunk0[4]["apply_anchor"] is False
    assert chunk0[4]["store_kv"] is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_sample_loop_toggles.py::test_extrapolation_honors_f2_f3_toggles -v
```
Expected: FAIL — `generator_fn_extrapolation` hasn't been updated yet.

- [ ] **Step 3: Refactor `_run_segment` inside `generator_fn_extrapolation` with the same toggle logic**

In `fastgen/methods/distribution_matching/causvid.py`, locate `_run_segment` (the inner function inside `generator_fn_extrapolation`, currently at lines 282-375). Replace the main denoise loop + cache-pass block (lines 313-375) with:

```python
                # Read toggles from the net module
                model_sink_cache = getattr(net, "_model_sink_cache", False)
                skip_clean_cache = getattr(net, "_skip_clean_cache_pass", False)

                num_denoise_steps = len(t_list) - 1
                last_step_idx = num_denoise_steps - 1

                start_frame = prefill_frames
                while start_frame < segment_frames:
                    end_frame = min(start_frame + chunk_size, segment_frames)
                    x_next = x[:, :, start_frame:end_frame, ...]

                    is_sink_chunk = (start_frame == 0 and frame_offset == 0)
                    f2_active = model_sink_cache and is_sink_chunk
                    do_cache_pass = (not skip_clean_cache) or f2_active

                    for step in range(num_denoise_steps):
                        is_last = (step == last_step_idx)
                        store_kv_here = skip_clean_cache and is_last and not f2_active
                        apply_anchor_here = not (f2_active and is_last)

                        t_cur = t_list[step].expand(batch_size)
                        x_cur = x_next
                        x_next = net(
                            x_cur,
                            t_cur,
                            condition=condition,
                            fwd_pred_type="x0",
                            cache_tag="pos",
                            cur_start_frame=start_frame,
                            frame_offset=frame_offset,
                            store_kv=store_kv_here,
                            is_ar=True,
                            apply_anchor=apply_anchor_here,
                            **kwargs,
                        )

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

                    # Write the generated slice back. Manual anchor for F2.
                    if f2_active and isinstance(condition, dict) and "first_frame_cond" in condition:
                        x_display = x_next.clone()
                        x_display[:, :, 0:1] = condition["first_frame_cond"][:, :, 0:1]
                        x[:, :, start_frame:end_frame, ...] = x_display
                    else:
                        x[:, :, start_frame:end_frame, ...] = x_next

                    if do_cache_pass:
                        x_cache = x_next
                        t_cache = t_list[-1].expand(batch_size)
                        if context_noise and context_noise > 0:
                            t_cache = torch.full(
                                (batch_size,), context_noise, device=device, dtype=dtype,
                            )
                            x_cache = net.noise_scheduler.forward_process(
                                x_next, torch.randn_like(x_next), t_cache
                            )
                        apply_anchor_cache = not f2_active
                        _ = net(
                            x_cache,
                            t_cache,
                            condition=condition,
                            fwd_pred_type="x0",
                            cache_tag="pos",
                            cur_start_frame=start_frame,
                            frame_offset=frame_offset,
                            store_kv=True,
                            is_ar=True,
                            apply_anchor=apply_anchor_cache,
                            **kwargs,
                        )

                    start_frame = end_frame
```

Leave the `_prefill_caches`, segment composition, VAE bridge, etc. untouched.

- [ ] **Step 4: Run all sample-loop tests to verify pass**

```bash
python -m pytest tests/test_sample_loop_toggles.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add fastgen/methods/distribution_matching/causvid.py tests/test_sample_loop_toggles.py
git commit -m "feat: mirror F2+F3 toggles in generator_fn_extrapolation

Same chunk-0-aware logic as _student_sample_loop; applies to each segment's
first chunk (where anchoring fires via cur_start_frame=0).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6b (HIGH PRIORITY): Plumb F1/F2/F3 through `scripts/inference/inference_causal.py`

**Context:** This is THE production inference path for InfiniteTalk. It constructs `CausalInfiniteTalkWan` directly (bypassing `InfiniteTalkSelfForcingModel._apply_anchor_config`), has its own inline AR loop (lines 644-741), and dynamically sets `model.total_num_frames = num_latent` based on audio duration. Without this task, F1/F2/F3 apply during validation but NOT during production inference.

**Files:**
- Modify: `scripts/inference/inference_causal.py`
  - Argparse (line ~125-165) — add 5 CLI flags
  - `load_diffusion_model` (line ~573-637) — pass lookahead params to constructor, optionally flip `use_dynamic_rope`, set F2/F3 attrs post-construction
  - `run_inference` (line ~644-741) — mirror the F2+F3 matrix from Task 5 inside the inline AR loop
- Test: `tests/test_inference_causal_toggles.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_inference_causal_toggles.py`:

```python
"""Test the F2+F3 toggles in scripts/inference/inference_causal.py::run_inference.

Uses the same MockNet pattern as test_sample_loop_toggles.py — we're testing
the same control-flow logic, but in the inline AR loop instead of
CausVidModel._student_sample_loop.
"""
import sys
import torch


class MockNoiseScheduler:
    def forward_process(self, x, eps, t):
        return x  # identity for test


class MockNet:
    """Records every forward call."""
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
            "t": float(t[0]) if t.numel() else 0.0,
            "store_kv": kwargs.get("store_kv", False),
            "apply_anchor": kwargs.get("apply_anchor", True),
            "cur_start_frame": kwargs.get("cur_start_frame", 0),
        })
        return torch.zeros_like(x)


def _run(model_sink_cache, skip_clean_cache):
    # Import run_inference from the script module. The script has
    # top-level argparse but run_inference is a plain function.
    sys.path.insert(0, "scripts/inference")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "inference_causal", "scripts/inference/inference_causal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    net = MockNet(model_sink_cache=model_sink_cache, skip_clean_cache=skip_clean_cache)
    condition = {"first_frame_cond": torch.zeros(1, 16, 9, 28, 56)}
    mod.run_inference(
        model=net, condition=condition, num_latent_frames=9, chunk_size=3,
        context_noise=0.0, seed=0, device=torch.device("cpu"), dtype=torch.float32,
        anchor_first_frame=True,
    )
    return net.calls


def test_both_off_unchanged_behavior():
    calls = _run(False, False)
    # 3 chunks × 5 calls (4 denoise + 1 cache) = 15
    assert len(calls) == 15


def test_f3_on_skips_cache_pass():
    calls = _run(False, True)
    assert len(calls) == 12
    for chunk_idx in range(3):
        chunk_calls = calls[chunk_idx * 4:(chunk_idx + 1) * 4]
        assert chunk_calls[3]["store_kv"] is True


def test_f2_on_sink_chunk_only():
    calls = _run(True, False)
    chunk0 = calls[0:5]
    assert chunk0[3]["apply_anchor"] is False   # last denoise suppresses anchor
    assert chunk0[4]["apply_anchor"] is False   # cache pass suppresses anchor
    assert chunk0[4]["store_kv"] is True


def test_f2_overrides_f3_for_sink():
    calls = _run(True, True)
    # Chunk 0: 5 calls (F2 overrides F3). Chunks 1, 2: 4 calls. Total 13.
    assert len(calls) == 13
    chunk0 = calls[0:5]
    assert chunk0[4]["apply_anchor"] is False
    assert chunk0[4]["store_kv"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_inference_causal_toggles.py -v
```
Expected: FAIL — `run_inference` doesn't accept the new `apply_anchor`/`store_kv` matrix yet (its loop currently hardcodes `store_kv=False` for denoise, `store_kv=True` for cache).

- [ ] **Step 3: Add CLI flags**

In `scripts/inference/inference_causal.py` argparse block (around line 138-165, near the existing `--local_attn_size` and `--sink_size` flags), add:

```python
    p.add_argument("--use_dynamic_rope", action="store_true",
                   help="Enable dynamic RoPE on the student (required by --lookahead_sink)")
    p.add_argument("--lookahead_sink", action="store_true",
                   help="Place the attention sink at a future RoPE position "
                        "(F1). Requires --use_dynamic_rope.")
    p.add_argument("--lookahead_distance", type=int, default=0,
                   help="Distance in frames for lookahead sink (>= 1 when "
                        "--lookahead_sink is on).")
    p.add_argument("--model_sink_cache", action="store_true",
                   help="Cache model-generated sink K/V (F2). Only the first "
                        "chunk is affected; display video still anchors frame 0.")
    p.add_argument("--skip_clean_cache_pass", action="store_true",
                   help="Skip the separate clean-cache forward pass (F3). "
                        "Last denoise step caches slightly-noisy K/V instead.")
```

- [ ] **Step 4: Wire flags into model construction**

In `load_diffusion_model` (around line 583-597), replace the `CausalInfiniteTalkWan(...)` constructor call with:

```python
    model = CausalInfiniteTalkWan(
        base_model_paths=args.base_model_paths,
        infinitetalk_ckpt_path=args.infinitetalk_ckpt,
        lora_ckpt_path="",
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        chunk_size=args.chunk_size,
        total_num_frames=num_latent,
        net_pred_type="flow",
        schedule_type="rf",
        shift=7.0,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        use_dynamic_rope=args.use_dynamic_rope,
        lookahead_sink_enabled=args.lookahead_sink,
        lookahead_distance=args.lookahead_distance,
    )

    # F2/F3 toggles: set as post-construction attrs (read by run_inference via getattr)
    model._model_sink_cache = args.model_sink_cache
    model._skip_clean_cache_pass = args.skip_clean_cache_pass
    print(f"  Lookahead sink: {'ON, distance=' + str(args.lookahead_distance) if args.lookahead_sink else 'OFF'}")
    print(f"  Model-generated sink cache (F2): {'ON' if args.model_sink_cache else 'OFF'}")
    print(f"  Skip clean cache pass (F3): {'ON' if args.skip_clean_cache_pass else 'OFF'}")
```

- [ ] **Step 5: Mirror F2+F3 logic in `run_inference`**

In `scripts/inference/inference_causal.py`, replace the body of `run_inference`'s main AR loop (lines ~677-735) with:

```python
    # Read F2/F3 toggles from the model (set in load_diffusion_model)
    model_sink_cache = getattr(model, "_model_sink_cache", False)
    skip_clean_cache = getattr(model, "_skip_clean_cache_pass", False)
    num_denoise_steps = len(t_list_t) - 1
    last_step_idx = num_denoise_steps - 1

    for block_idx in range(num_blocks):
        cur_start_frame = block_idx * chunk_size
        noisy_input = noise[:, :, cur_start_frame:cur_start_frame + chunk_size]

        is_sink_chunk = (cur_start_frame == 0)
        f2_active = model_sink_cache and is_sink_chunk
        do_cache_pass = (not skip_clean_cache) or f2_active

        for step_idx in range(num_denoise_steps):
            is_last = (step_idx == last_step_idx)
            store_kv_here = skip_clean_cache and is_last and not f2_active
            apply_anchor_here = not (f2_active and is_last)

            t_cur = t_list_t[step_idx]
            t_next = t_list_t[step_idx + 1]

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

            # Explicit safety anchor override (redundant with forward's builtin
            # anchor, but kept to match original code). Only applies when
            # apply_anchor_here=True and we're in chunk 0 — otherwise skip.
            if (anchor_first_frame and cur_start_frame == 0
                    and apply_anchor_here):
                x0_pred = x0_pred.clone()
                x0_pred[:, :, 0:1] = first_frame_latent

            if t_next > 0:
                eps = torch.randn_like(x0_pred)
                noisy_input = model.noise_scheduler.forward_process(
                    x0_pred, eps, t_next.expand(B),
                )
            else:
                noisy_input = x0_pred

        # Write displayed output. For F2 on sink chunk, manually anchor for display.
        if f2_active and isinstance(condition, dict) and "first_frame_cond" in condition:
            x_display = x0_pred.clone()
            x_display[:, :, 0:1] = first_frame_latent
            output[:, :, cur_start_frame:cur_start_frame + chunk_size] = x_display
        else:
            output[:, :, cur_start_frame:cur_start_frame + chunk_size] = x0_pred

        if do_cache_pass:
            cache_input = x0_pred  # non-anchored when f2_active, anchored otherwise
            t_cache = torch.full((B,), context_noise, device=device, dtype=dtype)
            if context_noise > 0:
                cache_eps = torch.randn_like(x0_pred)
                cache_input = model.noise_scheduler.forward_process(
                    x0_pred, cache_eps,
                    torch.tensor(context_noise, device=device, dtype=torch.float64).expand(B),
                )
            apply_anchor_cache = not f2_active
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

        print(f"    Block {block_idx + 1}/{num_blocks} done "
              f"(frames {cur_start_frame}-{cur_start_frame + chunk_size - 1})")
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_inference_causal_toggles.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/inference/inference_causal.py tests/test_inference_causal_toggles.py
git commit -m "feat: plumb F1/F2/F3 through inference_causal.py production AR loop

Adds CLI flags --use_dynamic_rope, --lookahead_sink, --lookahead_distance,
--model_sink_cache, --skip_clean_cache_pass. Mirrors the F2+F3 toggle matrix
from _student_sample_loop in the inline run_inference AR loop so production
inference has feature parity with validation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Wire config fields onto `self.net` in `_apply_anchor_config`

**Files:**
- Modify: `fastgen/methods/infinitetalk_self_forcing.py:170-227` (the `_apply_anchor_config` method)
- Test: `tests/test_apply_attention_config.py` (create — uses mock objects)

- [ ] **Step 1: Write the failing test**

Create `tests/test_apply_attention_config.py`:

```python
"""Verify that the new F1/F2/F3 config fields get stamped onto self.net
when _apply_anchor_config runs."""


class _Net:
    training = False
    # Minimal attrs the anchor-config function checks:
    _enable_first_frame_anchor = True
    _anchor_eval_only = False


class _Model:
    """Stand-in for InfiniteTalkSelfForcingModel, exposing just .config, .net,
    and the _apply_anchor_config method bound from the real class."""

    def __init__(self, cfg):
        self.config = cfg
        self.net = _Net()
        self.teacher = None
        self.fake_score = None

    # Bind the method:
    from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel
    _apply_anchor_config = InfiniteTalkSelfForcingModel._apply_anchor_config


class _Cfg:
    def __init__(self, **kw):
        self.student_anchor_eval_only = kw.get("student_anchor_eval_only", False)
        self.fake_score_anchor_eval_only = kw.get("fake_score_anchor_eval_only", False)
        self.teacher_anchor_disabled = kw.get("teacher_anchor_disabled", False)
        self.lookahead_sink_enabled = kw.get("lookahead_sink_enabled", False)
        self.lookahead_distance = kw.get("lookahead_distance", 0)
        self.model_sink_cache_enabled = kw.get("model_sink_cache_enabled", False)
        self.skip_clean_cache_pass = kw.get("skip_clean_cache_pass", False)


def test_new_fields_stamped_onto_net():
    m = _Model(_Cfg(
        lookahead_sink_enabled=True,
        lookahead_distance=4,
        model_sink_cache_enabled=True,
        skip_clean_cache_pass=True,
    ))
    m._apply_anchor_config()
    assert m.net._lookahead_sink_enabled is True
    assert m.net._lookahead_distance == 4
    assert m.net._model_sink_cache is True
    assert m.net._skip_clean_cache_pass is True


def test_new_fields_default_off():
    m = _Model(_Cfg())
    m._apply_anchor_config()
    assert m.net._lookahead_sink_enabled is False
    assert m.net._lookahead_distance == 0
    assert m.net._model_sink_cache is False
    assert m.net._skip_clean_cache_pass is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_apply_attention_config.py -v
```
Expected: `AssertionError: net has no _lookahead_sink_enabled`.

- [ ] **Step 3: Extend `_apply_anchor_config`**

In `fastgen/methods/infinitetalk_self_forcing.py`, at the end of `_apply_anchor_config` (after the teacher-anchor block), add:

```python
        # --- F1/F2/F3 toggles stamped onto self.net for the sample loops to read ---
        lookahead_enabled = getattr(self.config, "lookahead_sink_enabled", False)
        lookahead_distance = getattr(self.config, "lookahead_distance", 0)
        self.net._lookahead_sink_enabled = lookahead_enabled
        self.net._lookahead_distance = lookahead_distance

        # Also sync down onto every block's self-attention (these may have been
        # set by the network constructor; re-stamp for runtime overrides).
        if hasattr(self.net, "blocks"):
            for block in self.net.blocks:
                if hasattr(block, "self_attn"):
                    block.self_attn.lookahead_sink_enabled = lookahead_enabled
                    block.self_attn.lookahead_distance = lookahead_distance

        if lookahead_enabled:
            logger.info(
                f"[attn] Lookahead sink ENABLED, distance={lookahead_distance} frames"
            )
        else:
            logger.info("[attn] Lookahead sink: disabled (standard sink)")

        self.net._model_sink_cache = getattr(self.config, "model_sink_cache_enabled", False)
        if self.net._model_sink_cache:
            logger.info("[attn] Model-generated sink cache: ENABLED (F2)")
        else:
            logger.info("[attn] Model-generated sink cache: disabled")

        self.net._skip_clean_cache_pass = getattr(
            self.config, "skip_clean_cache_pass", False
        )
        if self.net._skip_clean_cache_pass:
            logger.info("[attn] Skip clean cache pass: ENABLED (F3)")
        else:
            logger.info("[attn] Skip clean cache pass: disabled")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_apply_attention_config.py -v
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add fastgen/methods/infinitetalk_self_forcing.py tests/test_apply_attention_config.py
git commit -m "feat: stamp F1/F2/F3 config fields onto self.net in _apply_anchor_config

Also propagates lookahead_sink_enabled/distance to every block.self_attn so
runtime overrides work even after network construction.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Create experiment config + valtest + run scripts

**Files:**
- Create: `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead.py`
- Create: `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_valtest.py`
- Create: `scripts/run_sf_w9s1_lookahead.sh`
- Create: `scripts/run_sf_w9s1_lookahead_valtest.sh`

- [ ] **Step 1: Create `config_sf_w9s1_lookahead.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk Self-Forcing — w9s1 + Lookahead Sink + optional F2/F3

Inherits config_sf_w9s1 and flips:
  - use_dynamic_rope = True        (required by lookahead sink)
  - lookahead_sink_enabled = True
  - lookahead_distance = 4
  - model_sink_cache_enabled  (via env MODEL_SINK_CACHE=1, default off)
  - skip_clean_cache_pass     (via env SKIP_CLEAN_CACHE=1, default off)

The lookahead distance is tunable via env LOOKAHEAD_DISTANCE.

Usage:
    bash scripts/run_sf_w9s1_lookahead.sh
    # Or with toggles:
    LOOKAHEAD_DISTANCE=6 bash scripts/run_sf_w9s1_lookahead.sh
    MODEL_SINK_CACHE=1 SKIP_CLEAN_CACHE=1 bash scripts/run_sf_w9s1_lookahead.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1 import (
    create_config as create_w9s1_config,
)


def create_config():
    config = create_w9s1_config()

    # ---- Student network: enable dynamic RoPE (required for lookahead) ----
    config.model.net.use_dynamic_rope = True

    # ---- Feature 1: lookahead sink ----
    config.model.lookahead_sink_enabled = True
    config.model.lookahead_distance = int(os.environ.get("LOOKAHEAD_DISTANCE", "4"))

    # ---- Feature 2: model-generated sink cache ----
    config.model.model_sink_cache_enabled = bool(os.environ.get("MODEL_SINK_CACHE", ""))

    # ---- Feature 3: skip clean cache pass ----
    config.model.skip_clean_cache_pass = bool(os.environ.get("SKIP_CLEAN_CACHE", ""))

    # ---- Logging ----
    f2_tag = "_f2" if config.model.model_sink_cache_enabled else ""
    f3_tag = "_f3" if config.model.skip_clean_cache_pass else ""
    config.log_config.group = f"infinitetalk_sf_w9s1_lookahead{f2_tag}{f3_tag}"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        L = config.model.lookahead_distance
        run_name = f"sf_w9s1_la{L}{f2_tag}{f3_tag}_freq5_lr1e5_{timestamp}"
    config.log_config.name = run_name

    return config
```

- [ ] **Step 2: Create `config_sf_w9s1_lookahead_valtest.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""
Validation-only variant of config_sf_w9s1_lookahead.py.

Iter 0 + iter 1 validation, no training.

Expected startup logs:
    [attn] Lookahead sink ENABLED, distance=4 frames
    [attn] Model-generated sink cache: <disabled|ENABLED>
    [attn] Skip clean cache pass: <disabled|ENABLED>

Because validation has cur_start_frame > 0 only on chunks past the sink, the
lookahead effect is only visible in chunks 1+. F2/F3 observable in wandb as
(a) frame-0 consistency with reference, (b) per-chunk forward count in logs.

Usage:
    bash scripts/run_sf_w9s1_lookahead_valtest.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_lookahead import (
    create_config as create_lookahead_config,
)


def create_config():
    config = create_lookahead_config()

    config.trainer.max_iter = 1
    config.trainer.skip_iter0_validation = False
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 2}]
    config.trainer.save_ckpt_iter = 99999

    val_list = os.environ.get(
        "INFINITETALK_VAL_LIST",
        "data/precomputed_talkvid/val_quarter_2.txt",
    )
    if hasattr(config, "dataloader_val") and config.dataloader_val is not None:
        config.dataloader_val.data_list_path = val_list

    config.log_config.group = "infinitetalk_sf_w9s1_lookahead_valtest"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        L = config.model.lookahead_distance
        run_name = f"valtest_w9s1_la{L}_{timestamp}"
    config.log_config.name = run_name
    config.log_config.wandb_mode = "online"

    return config
```

- [ ] **Step 3: Create `scripts/run_sf_w9s1_lookahead.sh`**

```bash
#!/bin/bash
# InfiniteTalk SF Training — w9s1 + Lookahead Sink (+ optional F2/F3)
#
# Environment overrides:
#   LOOKAHEAD_DISTANCE=N   (default 4)    lookahead distance in frames
#   MODEL_SINK_CACHE=1     (default off)  F2: cache model-generated sink K/V
#   SKIP_CLEAN_CACHE=1     (default off)  F3: skip separate clean cache pass
#   STUDENT_ANCHOR_EVAL_ONLY=1, FAKE_SCORE_ANCHOR_EVAL_ONLY=1 — anchor modes
#
# Usage:
#   bash scripts/run_sf_w9s1_lookahead.sh
#   LOOKAHEAD_DISTANCE=6 MODEL_SINK_CACHE=1 bash scripts/run_sf_w9s1_lookahead.sh

set -e

export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

export INFINITETALK_TEACHER_LORA_CKPT="${INFINITETALK_TEACHER_LORA_CKPT:-}"
export INFINITETALK_STUDENT_LORA_CKPT="${INFINITETALK_STUDENT_LORA_CKPT:-}"

DF_CKPT_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter/quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints"
if [ -z "${INFINITETALK_DF_CKPT:-}" ]; then
    INFINITETALK_DF_CKPT=$(ls -1 "$DF_CKPT_DIR"/*.pth 2>/dev/null | sort -V | tail -1)
    if [ -z "$INFINITETALK_DF_CKPT" ]; then
        echo "ERROR: No .pth files found in $DF_CKPT_DIR"
        exit 1
    fi
fi
export INFINITETALK_DF_CKPT

export INFINITETALK_TRAIN_LIST="${INFINITETALK_TRAIN_LIST:-data/precomputed_talkvid/train_excl_val30.txt}"
export INFINITETALK_VAL_LIST="${INFINITETALK_VAL_LIST:-data/precomputed_talkvid/val_quarter_30.txt}"
export INFINITETALK_NEG_TEXT_EMB="${INFINITETALK_NEG_TEXT_EMB:-data/precomputed_talkvid/neg_text_embeds.pt}"
export INFINITETALK_AUDIO_ROOT="${INFINITETALK_AUDIO_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data}"
export INFINITETALK_CSV_PATH="${INFINITETALK_CSV_PATH:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list_cleaned.csv}"
export INFINITETALK_RAW_DATA_ROOT="${INFINITETALK_RAW_DATA_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/}"
export INFINITETALK_WAV2VEC_DIR="${INFINITETALK_WAV2VEC_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base}"

export WANDB_ENTITY="paulhcho"
export WANDB_API_KEY="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"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export NUM_GPUS

echo "============================================================"
echo "InfiniteTalk SF Training — w9s1 + Lookahead Sink"
echo "============================================================"
echo "GPUs:               $NUM_GPUS"
echo "Student attn:       local_attn_size=10, sink_size=1"
echo "Dynamic RoPE:       ON (required by lookahead)"
echo "Lookahead distance: ${LOOKAHEAD_DISTANCE:-4} frames"
echo "F2 (model-sink):    ${MODEL_SINK_CACHE:-off}"
echo "F3 (skip cache):    ${SKIP_CLEAN_CACHE:-off}"
echo "DF checkpoint:      $INFINITETALK_DF_CKPT"
echo "============================================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead.py
```

- [ ] **Step 4: Create `scripts/run_sf_w9s1_lookahead_valtest.sh`**

```bash
#!/bin/bash
# InfiniteTalk SF Val-Only Test — w9s1 + Lookahead (+ optional F2/F3)
#
# Usage:
#   bash scripts/run_sf_w9s1_lookahead_valtest.sh
#   MODEL_SINK_CACHE=1 SKIP_CLEAN_CACHE=1 bash scripts/run_sf_w9s1_lookahead_valtest.sh

set -e

export INFINITETALK_WEIGHTS_DIR="${INFINITETALK_WEIGHTS_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P}"
export INFINITETALK_CKPT="${INFINITETALK_CKPT:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors}"
export INFINITETALK_VAE_PATH="${INFINITETALK_VAE_PATH:-${INFINITETALK_WEIGHTS_DIR}/Wan2.1_VAE.pth}"

export INFINITETALK_TEACHER_LORA_CKPT="${INFINITETALK_TEACHER_LORA_CKPT:-}"
export INFINITETALK_STUDENT_LORA_CKPT="${INFINITETALK_STUDENT_LORA_CKPT:-}"

DF_CKPT_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter/quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints"
if [ -z "${INFINITETALK_DF_CKPT:-}" ]; then
    INFINITETALK_DF_CKPT=$(ls -1 "$DF_CKPT_DIR"/*.pth 2>/dev/null | sort -V | tail -1)
    if [ -z "$INFINITETALK_DF_CKPT" ]; then
        echo "ERROR: No .pth files found in $DF_CKPT_DIR"
        exit 1
    fi
fi
export INFINITETALK_DF_CKPT

export INFINITETALK_TRAIN_LIST="${INFINITETALK_TRAIN_LIST:-data/precomputed_talkvid/train_excl_val30.txt}"
export INFINITETALK_VAL_LIST="${INFINITETALK_VAL_LIST:-data/precomputed_talkvid/val_quarter_2.txt}"
export INFINITETALK_NEG_TEXT_EMB="${INFINITETALK_NEG_TEXT_EMB:-data/precomputed_talkvid/neg_text_embeds.pt}"
export INFINITETALK_AUDIO_ROOT="${INFINITETALK_AUDIO_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data}"
export INFINITETALK_CSV_PATH="${INFINITETALK_CSV_PATH:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list_cleaned.csv}"
export INFINITETALK_RAW_DATA_ROOT="${INFINITETALK_RAW_DATA_ROOT:-/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/}"
export INFINITETALK_WAV2VEC_DIR="${INFINITETALK_WAV2VEC_DIR:-/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base}"

export WANDB_ENTITY="paulhcho"
export WANDB_API_KEY="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"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export NUM_GPUS

echo "============================================================"
echo "InfiniteTalk SF Val-Only Test — w9s1 + Lookahead"
echo "============================================================"
echo "GPUs:               $NUM_GPUS"
echo "Dynamic RoPE:       ON"
echo "Lookahead distance: ${LOOKAHEAD_DISTANCE:-4} frames"
echo "F2 (model-sink):    ${MODEL_SINK_CACHE:-off}"
echo "F3 (skip cache):    ${SKIP_CLEAN_CACHE:-off}"
echo "Max iter:           1  (val at iter 0 + iter 1, no training)"
echo "Val list:           $INFINITETALK_VAL_LIST"
echo "DF checkpoint:      $INFINITETALK_DF_CKPT"
echo "============================================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    train.py \
    --config fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_valtest.py
```

- [ ] **Step 5: Make scripts executable + syntax-check**

```bash
chmod +x scripts/run_sf_w9s1_lookahead.sh scripts/run_sf_w9s1_lookahead_valtest.sh
bash -n scripts/run_sf_w9s1_lookahead.sh
bash -n scripts/run_sf_w9s1_lookahead_valtest.sh
python -c "
import ast
for p in ['fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead.py',
          'fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_valtest.py']:
    ast.parse(open(p).read())
    print(f'OK {p}')
"
```
Expected: `bash OK`, both files `OK`.

- [ ] **Step 6: Runtime-verify configs resolve with env-var toggles**

```bash
python -c "
import os
os.environ.setdefault('INFINITETALK_WEIGHTS_DIR', '/tmp/fake')
os.environ.setdefault('INFINITETALK_CKPT', '/tmp/fake.st')
os.environ.setdefault('INFINITETALK_DF_CKPT', '/tmp/fake.pth')
os.environ.setdefault('INFINITETALK_NEG_TEXT_EMB', '/tmp/fake.pt')
os.environ['LOOKAHEAD_DISTANCE'] = '6'
os.environ['MODEL_SINK_CACHE'] = '1'
os.environ['SKIP_CLEAN_CACHE'] = '1'
from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_lookahead import create_config
c = create_config()
assert c.model.net.use_dynamic_rope == True
assert c.model.lookahead_sink_enabled == True
assert c.model.lookahead_distance == 6
assert c.model.model_sink_cache_enabled == True
assert c.model.skip_clean_cache_pass == True
print('All config knobs plumb correctly')
"
```
Expected: `All config knobs plumb correctly`.

- [ ] **Step 7: Commit**

```bash
git add fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead.py \
        fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_lookahead_valtest.py \
        scripts/run_sf_w9s1_lookahead.sh \
        scripts/run_sf_w9s1_lookahead_valtest.sh
git commit -m "feat: add lookahead + F2 + F3 experiment configs and run scripts

Training config enables dynamic RoPE (required by lookahead), sets distance=4
by default. F2 and F3 are env-var toggleable (MODEL_SINK_CACHE, SKIP_CLEAN_CACHE).
Valtest variant runs iter-0 validation on a 2-sample val list.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: End-to-end smoke test via valtest

**Files:**
- (None created; this is an integration smoke test)

- [ ] **Step 1: Run the valtest with defaults (F1 only)**

```bash
cd /data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk
bash scripts/run_sf_w9s1_lookahead_valtest.sh 2>&1 | tee /tmp/valtest_f1.log
```
Expected completion: non-zero exit only on model-load errors. Scan log for:
- `[attn] Lookahead sink ENABLED, distance=4 frames`
- `[attn] Model-generated sink cache: disabled`
- `[attn] Skip clean cache pass: disabled`
- `[val] pixel_video stats: shape=[1, 3, 81, 224, 448], ...`
- `[val_end] Logging N val videos at iteration X...`

- [ ] **Step 2: Run the valtest with F2 on**

```bash
MODEL_SINK_CACHE=1 bash scripts/run_sf_w9s1_lookahead_valtest.sh 2>&1 | tee /tmp/valtest_f1_f2.log
```
Expected: new log line `[attn] Model-generated sink cache: ENABLED (F2)`. Validation still produces coherent videos — look at wandb `val0/generated`.

- [ ] **Step 3: Run the valtest with F3 on**

```bash
SKIP_CLEAN_CACHE=1 bash scripts/run_sf_w9s1_lookahead_valtest.sh 2>&1 | tee /tmp/valtest_f1_f3.log
```
Expected: new log line `[attn] Skip clean cache pass: ENABLED (F3)`. Validation produces videos. Forward-count savings not directly logged but wall-clock-time per-validation should be slightly lower than the defaults run.

- [ ] **Step 4: Run the valtest with F2 + F3 both on**

```bash
MODEL_SINK_CACHE=1 SKIP_CLEAN_CACHE=1 bash scripts/run_sf_w9s1_lookahead_valtest.sh 2>&1 | tee /tmp/valtest_f1_f2_f3.log
```
Expected: both `[attn] Model-generated sink cache: ENABLED (F2)` AND `[attn] Skip clean cache pass: ENABLED (F3)` log lines. Validation still coherent.

- [ ] **Step 5: Runtime-verify the failure mode for lookahead without dynamic RoPE**

Create a temp config to exercise the error path:

```bash
python -c "
import os
os.environ.setdefault('INFINITETALK_WEIGHTS_DIR', '/tmp/fake')
os.environ.setdefault('INFINITETALK_CKPT', '/tmp/fake.st')
os.environ.setdefault('INFINITETALK_DF_CKPT', '/tmp/fake.pth')
os.environ.setdefault('INFINITETALK_NEG_TEXT_EMB', '/tmp/fake.pt')
# Build w/ lookahead + static RoPE — should raise ValueError at net construction
from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_lookahead import create_config
c = create_config()
c.model.net.use_dynamic_rope = False  # force bad combination
from fastgen.utils import instantiate
try:
    instantiate(c.model.net)
    raise AssertionError('expected ValueError')
except ValueError as e:
    assert 'lookahead_sink_enabled=True requires use_dynamic_rope=True' in str(e)
    print('Error correctly raised')
"
```
Expected: `Error correctly raised`.

- [ ] **Step 6: Smoke-test the production inference path (`inference_causal.py`)**

This is the path actually used in `run_inference_causal.sh` / `run_eval_*`. After Task 6b, it should accept the new CLI flags.

```bash
# Pick one precomputed sample to generate, CLI-only (no wandb):
python scripts/inference/inference_causal.py \
    --precomputed_dir data/precomputed_talkvid/data_-_uuPMrdok8_-_uuPMrdok8_835701_840743 \
    --output_path /tmp/inference_f1_test.mp4 \
    --ckpt_path $INFINITETALK_DF_CKPT \
    --base_model_paths "$INFINITETALK_WEIGHTS_DIR/diffusion_pytorch_model-00001-of-00007.safetensors,..." \
    --infinitetalk_ckpt $INFINITETALK_CKPT \
    --chunk_size 3 --lora_rank 128 --lora_alpha 64 \
    --local_attn_size 10 --sink_size 1 \
    --use_dynamic_rope --lookahead_sink --lookahead_distance 4 \
    --quarter_res --target_h 224 --target_w 448
```
Expected: console prints
```
  Lookahead sink: ON, distance=4
  Model-generated sink cache (F2): OFF
  Skip clean cache pass (F3): OFF
  AR loop: N blocks x 4 denoising steps = ... forward passes
```
`/tmp/inference_f1_test.mp4` generated successfully. Re-run with `--model_sink_cache --skip_clean_cache_pass` appended to test the combined path.

- [ ] **Step 7: Commit a brief notes/README documenting the matrix**

Create `docs/superpowers/specs/lookahead-sink-notes.md`:

```markdown
# Lookahead Sink + Model-Sink-Cache + Skip-Cache-Pass — Field Notes

## Flag matrix

| MODEL_SINK_CACHE | SKIP_CLEAN_CACHE | Sink chunk behavior | Other chunks behavior |
|---|---|---|---|
| off | off | 4 denoise + 1 cache pass; anchor on both | same |
| off | on  | 4 denoise (last writes cache); no cache pass | same |
| on  | off | 4 denoise + 1 cache pass; anchor OFF on last denoise + cache pass; manual display anchor | same as off/off |
| on  | on  | 4 denoise + 1 cache pass (F2 overrides F3 for sink); anchor OFF on last denoise + cache pass | 4 denoise (last writes cache); no cache pass |

## Inspecting correctness from wandb

- `val_gt/videos` — reference images (should always match)
- `val0/generated` — frame 0 should MATCH reference regardless of F2 state
  (display path always anchors; F2 only affects cache K/V, not display)
- Flicker/discontinuity between chunks 0 and 1 — potential sign of
  lookahead RoPE mismatch between sink's "future" pos and rest's window-local
  pos. Tune `LOOKAHEAD_DISTANCE` if present.

## Current defaults

Lookahead distance = 4 frames (experimental). Adjust via `LOOKAHEAD_DISTANCE` env.
```

Then commit:

```bash
git add docs/superpowers/specs/lookahead-sink-notes.md
git commit -m "docs: field notes for lookahead + sink-cache + skip-cache-pass toggles

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Checklist (run before handoff)

1. **Spec coverage:**
   - F1 (lookahead sink): Tasks 3, 4, 6b, 7, 8 ✓
   - F2 (model sink cache): Tasks 2, 5, 6, 6b, 7, 8 ✓
   - F3 (skip cache pass): Tasks 5, 6, 6b, 7, 8 ✓
   - Toggles: Task 1 ✓ + all experiment configs + CLI flags in Task 6b ✓
   - Training (SF rollout — lookahead via contiguous-branch trigger): Tasks 3, 4 ✓
   - Validation (`_student_sample_loop`): Task 5 ✓
   - **Production inference (`scripts/inference/inference_causal.py`): Task 6b ✓**
   - Generic multi-segment inference (`generator_fn_extrapolation`, low priority): Task 6 ✓
   - Existing anchor+sink+window honored: preserved by design (all config set-points untouched) ✓

2. **Placeholder scan:** no "TBD" / "implement later" / "appropriate error handling" left. All code blocks contain the actual code.

3. **Type consistency:**
   - Method name `_apply_anchor_config` is kept (Task 7) — callsites don't need updating ✓
   - `apply_anchor: bool = True` kwarg name used consistently across `forward`, `_student_sample_loop`, `generator_fn_extrapolation`, and `inference_causal.py::run_inference` ✓
   - Net-module attributes `_lookahead_sink_enabled`, `_lookahead_distance`, `_model_sink_cache`, `_skip_clean_cache_pass` used consistently ✓
   - Config field names: `lookahead_sink_enabled`, `lookahead_distance`, `model_sink_cache_enabled`, `skip_clean_cache_pass` used consistently in Task 1 + all consumers ✓
   - Helper `_apply_window_rope` signature matches its callsite in `CausalSelfAttention.forward` (Task 4) ✓
   - Helper `_maybe_apply_first_frame_anchor` signature matches its callsite in `CausalInfiniteTalkWan.forward` (Task 2) ✓

---

## Resolved design questions (captured here for history)

- **A1 (lookahead in training):** RESOLVED → apply in both contiguous and non-contiguous attention branches. Trigger: `sink_tokens > 0 AND sink_tokens < k_win.shape[1]`. Fires during SF training (from chunk 1 onward) and inference (early chunks via contiguous branch; late chunks via non-contiguous after eviction).
- **A2 (chunk 0 skip):** RESOLVED → naturally falls out of the trigger condition. When chunk 0 runs, `k_win` is just the current chunk's K (no cache content); `sink_tokens == k_win.shape[1]` is FALSE but `sink_tokens < k_win.shape[1]` requires k_win to have content beyond sink, which it doesn't in chunk 0. Lookahead = no-op on chunk 0.
- **A3 (F2+F3 interaction):** RESOLVED → F2 overrides F3 for the sink chunk (chunk 0). F3 applies to chunks > 0 normally. Matrix documented in "Feature 2 + Feature 3 matrix" section.
- **A5 (multi-segment inference):** RESOLVED → production InfiniteTalk inference (`inference_causal.py`) uses a **single-pass rolling-window approach** — dynamically sets `model.total_num_frames = num_latent` from audio duration, then runs one AR pass. Never calls `generator_fn_extrapolation`. The multi-segment concern is purely theoretical for InfiniteTalk.
- **A6 (lookahead_distance=0):** RESOLVED → `lookahead_distance < 1` raises `ValueError` at `CausalInfiniteTalkWan.__init__` time.
- **A7/A8 (Q position):** RESOLVED → Q keeps its natural position `query_offset_in_win // frame_seqlen` regardless of lookahead flag. Only sink's position shifts. Rolling's positions `[sink_frames_n, ..., F_window - 1]` = what they'd be without the flag. Identical train/eval.

---

## Remaining open questions (tunable during execution)

1. **Sink RoPE position formula — alternate tunings (if `F_window - 1 + lookahead_distance` produces bad outputs):**
   - `F_window + lookahead_distance` (one more position of future; sink becomes clearly beyond the window)
   - `F_window - chunk_frames + lookahead_distance` (distance measured from start of current block instead of end)

   Not a blocker; first tuning knob to try if the default underperforms.

2. **Handling of `cache_local_end_override` interaction.** The existing `cache_local_end_override` kwarg in `CausalSelfAttention.forward` is used for gradient-checkpointing determinism. The new `_apply_window_rope` helper receives `query_offset_in_win` which is computed from cache state, so the override-vs-no-override paths need to produce the same `query_offset_in_win` value for lookahead position math to be deterministic across forward and recomputation. Verify in Task 4 Step 4 — existing tests should catch this if broken.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-13-lookahead-sink-and-cache-toggles.md`. 10 tasks total (1–9 plus 6b).

**Critical task dependencies:**
- Task 1 (config fields) → Task 7 (plumbing into `_apply_anchor_config`) → Tasks 8, 9 (experiment configs)
- Task 2 (`apply_anchor` kwarg) → Task 5 (`_student_sample_loop`) → Task 6 (`generator_fn_extrapolation`, low-priority)
- Task 2 → Task 6b (`inference_causal.py`, high-priority production path)
- Tasks 3, 4 (attention + lookahead RoPE) → validates F1 math for training + validation + inference
- Task 6b is independent of Task 7 (uses post-construction attr-setting instead of `_apply_anchor_config`)

**Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task with `superpowers:subagent-driven-development`, review between tasks. Tasks 1, 2, 3, 4, 5, 6b, 7 are mostly independent (each adds a capability). Task 8 depends on Tasks 1–7 being in place. Task 9 is an integration smoke test.

**2. Inline Execution** — run tasks here with `superpowers:executing-plans`, batch execution with checkpoints for review.

Which approach?
