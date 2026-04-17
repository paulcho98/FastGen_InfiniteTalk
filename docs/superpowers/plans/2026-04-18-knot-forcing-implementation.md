# Knot Forcing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Knot Forcing (Xiao et al. 2025, arXiv:2512.21734v2) on our InfiniteTalk SF stack — temporal knot, running-ahead dynamic RoPE, last-frame reference — and train a new SF run from scratch.

**Architecture:** Hybrid approach (not paper-exact): keep our frame-0 sink for global reference at iter 0; overload it with the knot at iter > 0. Temporal knot denoises `c+k=4` frames per chunk, commits first c, saves last k. Fusion (Eq. 5) averages the boundary frame across adjacent chunks at exit step. Running-ahead maintains an absolute RoPE position for the sink, advanced when rollout catches up. All KF components gated behind config flags; KF-off must be bit-exact to existing SF.

**Tech Stack:** torch 2.8, FSDP2, bf16, existing FastGen + InfiniteTalk DiT codebase. No new deps.

**Worktree:** `/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk_knot/` (branch `feat/knot-forcing` from `main`).

**Understanding doc:** `docs/analysis/2026-04-17-knot-forcing-algorithm-understanding.md` (read this first).

---

## File Structure

### New files
- `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_knot_runahead.py` — training config
- `scripts/run_sf_w9s1_knot_runahead.sh` — training launcher
- `scripts/inference/run_inference_knot_runahead.sh` — inference launcher
- `scripts/inference/monitor_sf_w9s1_knot.sh` — monitor/eval loop for new run
- `tests/test_knot_forcing.py` — unit tests for KF components

### Modified files
- `fastgen/configs/methods/config_infinitetalk_sf.py` — add KF config fields
- `fastgen/networks/InfiniteTalk/network_causal.py` — generalize anchor helpers, pin-source override, running-ahead RoPE
- `fastgen/methods/distribution_matching/self_forcing.py` — KF rollout in `rollout_with_gradient`
- `fastgen/methods/distribution_matching/causvid.py` — KF in `_student_sample_loop` (validation path)
- `fastgen/datasets/infinitetalk_dataloader.py` — last-frame reference + audio +k slicing
- `scripts/inference/inference_causal.py` — KF in `run_inference` (CLI)

### Unchanged (verify only)
- `fastgen/networks/InfiniteTalk/network.py` — bidirectional teacher; processes full rolled-out output, no KF awareness needed.
- `fastgen/methods/distribution_matching/dmd2.py` — DMD loss; unchanged.

---

## Phase 1 — Config flags + regression scaffold

### Task 1: Add KF config fields to `config_infinitetalk_sf.py`

**Files:**
- Modify: `fastgen/configs/methods/config_infinitetalk_sf.py`

- [ ] **Step 1: Write the failing test**

Create file `tests/test_knot_forcing.py`:

```python
"""Knot Forcing unit tests — config flags, anchor pin-source, knot injection,
fusion, running-ahead, dataloader last-frame reference."""
import pytest
import torch


def test_kf_config_flags_default_off():
    """KF flags must default to off — KF-off must be bit-exact to existing SF."""
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    assert cfg.use_temporal_knot is False
    assert cfg.knot_size == 1
    assert cfg.use_running_ahead is False
    assert cfg.running_ahead_step == 4
    assert cfg.running_ahead_init_n == 8
    assert cfg.use_last_frame_reference is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/.../FastGen_InfiniteTalk_knot && python -m pytest tests/test_knot_forcing.py::test_kf_config_flags_default_off -v`
Expected: FAIL with `AttributeError: 'InfiniteTalkSFModelConfig' object has no attribute 'use_temporal_knot'`

- [ ] **Step 3: Add fields to config class**

Note: `InfiniteTalkSFModelConfig` uses `@attrs.define(slots=False)` (not `@dataclass`). Fields are declared as `name: type = default`.

In `fastgen/configs/methods/config_infinitetalk_sf.py`, inside the `InfiniteTalkSFModelConfig` class (near the existing `student_anchor_eval_only` field), add:

```python
    # ── Knot Forcing (KF) flags — all default off, KF-off = bit-exact existing SF ──
    # Temporal knot: denoise c+k frames per chunk, commit c, hold k as knot,
    # inject previous knot via I2V mask on frame 0 of next chunk, fuse boundaries.
    use_temporal_knot: bool = False
    knot_size: int = 1
    
    # Running-ahead: reference KV at dynamic absolute RoPE position; advances by
    # running_ahead_step when the rollout cursor catches up. Initial position
    # given by running_ahead_init_n. Must be mutually exclusive with
    # lookahead_sink (F1) — validated at runtime.
    use_running_ahead: bool = False
    running_ahead_step: int = 4
    running_ahead_init_n: int = 8
    
    # Dataloader: use last frame of clip as reference (first_frame_cond source)
    # instead of the first frame. Encodes via vae_latents[:, :, -1:] broadcast.
    use_last_frame_reference: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_knot_forcing.py::test_kf_config_flags_default_off -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/configs/methods/config_infinitetalk_sf.py tests/test_knot_forcing.py
git commit -m "feat(kf): add KF config flags (all default off)"
```

---

### Task 2: Config validation — running-ahead and lookahead-sink mutually exclusive

**Files:**
- Modify: `fastgen/configs/methods/config_infinitetalk_sf.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_knot_forcing.py`:

```python
def test_kf_running_ahead_incompatible_with_lookahead_sink():
    """Running-ahead and F1 lookahead_sink both modify sink RoPE — must not coexist."""
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    cfg.use_running_ahead = True
    cfg.lookahead_sink_enabled = True
    with pytest.raises(ValueError, match="running_ahead.*lookahead_sink"):
        cfg.validate_kf_flags()


def test_kf_validation_passes_when_only_one_is_on():
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    cfg = InfiniteTalkSFModelConfig()
    cfg.use_running_ahead = True
    cfg.lookahead_sink_enabled = False
    cfg.validate_kf_flags()  # should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_knot_forcing.py -k "kf_running_ahead or kf_validation_passes" -v`
Expected: FAIL (method `validate_kf_flags` not defined)

- [ ] **Step 3: Add validator method**

Add to `InfiniteTalkSFModelConfig` class body:

```python
    def validate_kf_flags(self):
        """Validate KF flag combinations. Call after config is fully populated."""
        if self.use_running_ahead and getattr(self, "lookahead_sink_enabled", False):
            raise ValueError(
                "use_running_ahead=True is incompatible with lookahead_sink_enabled=True: "
                "both mechanisms modify the sink's RoPE position and cannot coexist. "
                "Pick one."
            )
        if self.use_temporal_knot and self.knot_size < 1:
            raise ValueError(f"knot_size must be >= 1 when use_temporal_knot=True, got {self.knot_size}")
        if self.use_running_ahead and self.running_ahead_step < 1:
            raise ValueError(f"running_ahead_step must be >= 1, got {self.running_ahead_step}")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_knot_forcing.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add fastgen/configs/methods/config_infinitetalk_sf.py tests/test_knot_forcing.py
git commit -m "feat(kf): config validator for incompatible KF/F1 flag combos"
```

---

## Phase 2 — Network pin-source override (anchor helpers)

### Task 3: Generalize `_maybe_apply_input_anchor` to accept arbitrary pin source

Currently the helper reads `condition["first_frame_cond"]` as the pin source. Under KF, we need to pin to the knot (`x̄`) at iter > 0.

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_knot_forcing.py`:

```python
def test_input_anchor_uses_custom_pin_source():
    """When condition has 'anchor_pin_source', use it instead of 'first_frame_cond'."""
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_input_anchor
    
    # Stub net_module
    class M:
        training = False
    m = M()
    m._enable_first_frame_anchor = True
    m._anchor_eval_only = False
    
    B, C, T, H, W = 1, 16, 4, 8, 8
    x_t = torch.randn(B, C, T, H, W)
    first_frame_cond = torch.ones(B, C, T, H, W) * 7.0
    custom_pin = torch.ones(B, C, T, H, W) * -3.0
    
    # Without custom pin source → uses first_frame_cond
    out1 = _maybe_apply_input_anchor(
        x_t, m, cur_start_frame=0,
        condition={"first_frame_cond": first_frame_cond},
    )
    assert torch.allclose(out1[:, :, 0:1], first_frame_cond[:, :, 0:1])
    
    # With custom pin source → uses custom
    out2 = _maybe_apply_input_anchor(
        x_t, m, cur_start_frame=0,
        condition={"first_frame_cond": first_frame_cond, "anchor_pin_source": custom_pin},
    )
    assert torch.allclose(out2[:, :, 0:1], custom_pin[:, :, 0:1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_knot_forcing.py::test_input_anchor_uses_custom_pin_source -v`
Expected: FAIL (pin source override not yet implemented)

- [ ] **Step 3: Modify `_maybe_apply_input_anchor`**

In `fastgen/networks/InfiniteTalk/network_causal.py`, change the body of `_maybe_apply_input_anchor` — locate the line `first_frame_cond = condition["first_frame_cond"]` and replace with:

```python
    # Pin source override: KF injects the knot x̄ via condition["anchor_pin_source"]
    # to reuse the anchor machinery without duplicating plumbing. Default source
    # remains first_frame_cond (standard I2V reference).
    pin_source = condition.get("anchor_pin_source")
    if pin_source is None:
        pin_source = condition["first_frame_cond"]
    x_t = x_t.clone()
    x_t[:, :, 0:1] = pin_source[:, :, 0:1]
    return x_t
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_input_anchor_uses_custom_pin_source -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_knot_forcing.py
git commit -m "feat(kf): input anchor accepts custom pin source via condition['anchor_pin_source']"
```

---

### Task 4: Generalize `_maybe_apply_first_frame_anchor` (output anchor) similarly + add disable-at-position-0 path

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_output_anchor_uses_custom_pin_source():
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_first_frame_anchor
    class M:
        training = False
    m = M()
    m._enable_first_frame_anchor = True
    m._anchor_eval_only = False
    
    B, C, T, H, W = 1, 16, 4, 8, 8
    out = torch.randn(B, C, T, H, W)
    first_frame_cond = torch.ones(B, C, T, H, W) * 7.0
    custom_pin = torch.ones(B, C, T, H, W) * -3.0
    
    o1 = _maybe_apply_first_frame_anchor(out, m, 0, {"first_frame_cond": first_frame_cond})
    assert torch.allclose(o1[:, :, 0:1], first_frame_cond[:, :, 0:1])
    
    o2 = _maybe_apply_first_frame_anchor(
        out, m, 0, {"first_frame_cond": first_frame_cond, "anchor_pin_source": custom_pin}
    )
    assert torch.allclose(o2[:, :, 0:1], custom_pin[:, :, 0:1])


def test_output_anchor_disabled_for_fusion():
    """When condition has 'output_anchor_disabled_for_fusion': True, skip output pin
    so the model's raw prediction flows through for Eq. 5 averaging."""
    from fastgen.networks.InfiniteTalk.network_causal import _maybe_apply_first_frame_anchor
    class M:
        training = False
    m = M()
    m._enable_first_frame_anchor = True
    m._anchor_eval_only = False
    
    B, C, T, H, W = 1, 16, 4, 8, 8
    out = torch.randn(B, C, T, H, W)
    first_frame_cond = torch.zeros(B, C, T, H, W)
    
    out_unchanged = _maybe_apply_first_frame_anchor(
        out, m, 0,
        {"first_frame_cond": first_frame_cond, "output_anchor_disabled_for_fusion": True},
    )
    assert torch.allclose(out_unchanged, out)  # untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_knot_forcing.py -k "output_anchor" -v`
Expected: FAIL both

- [ ] **Step 3: Modify `_maybe_apply_first_frame_anchor`**

Locate the function body. After the existing gating checks (`anchor_active`, `cur_start_frame != 0`), and BEFORE the final pin assignment, add:

```python
    # KF: explicit disable flag — lets the model's raw prediction flow through
    # so the caller can apply Eq. 5 fusion afterwards.
    if isinstance(condition, dict) and condition.get("output_anchor_disabled_for_fusion", False):
        return out
    
    # Pin source override: KF injects x̄ via condition["anchor_pin_source"].
    pin_source = condition.get("anchor_pin_source") if isinstance(condition, dict) else None
    if pin_source is None:
        pin_source = condition["first_frame_cond"]
    out = out.clone()
    out[:, :, 0:1] = pin_source[:, :, 0:1]
    return out
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_knot_forcing.py -k "output_anchor" -v`
Expected: PASS both

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_knot_forcing.py
git commit -m "feat(kf): output anchor accepts pin source override + disable-for-fusion flag"
```

---

## Phase 3 — Y-tensor knot injection

### Task 5: Build-y helper that injects VAE(knot) at position 0

The `_build_y()` method constructs the 20-channel y-tensor (4 mask + 16 VAE). Under KF iter > 0, positions 0's VAE channels should carry VAE(knot) instead of VAE(reference).

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_y_tensor_knot_injection():
    """When condition['y_position_0_override'] is set, position 0 of y's VAE
    channels = override, not first_frame_cond."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    # We need a build_y method; use the existing one with knot override.
    # This test only checks the math; no network instantiation needed.
    # Use a small standalone helper added to the module:
    from fastgen.networks.InfiniteTalk.network_causal import _build_y_with_knot
    
    B, C, T, H, W = 1, 16, 4, 8, 8
    first_frame_cond = torch.ones(B, C, T, H, W) * 7.0
    knot = torch.ones(B, C, 1, H, W) * -3.0
    
    # No override: position 0 VAE chans = first_frame_cond[:, :, 0]
    y_normal = _build_y_with_knot(first_frame_cond, knot_override=None, T=T)
    assert y_normal.shape == (B, 20, T, H, W)
    assert torch.allclose(y_normal[:, 4:, 0:1], first_frame_cond[:, :, 0:1])
    assert torch.allclose(y_normal[:, 0:4, 0:1], torch.ones(B, 4, 1, H, W))  # mask=1
    
    # With override: position 0 VAE chans = knot
    y_knot = _build_y_with_knot(first_frame_cond, knot_override=knot, T=T)
    assert torch.allclose(y_knot[:, 4:, 0:1], knot)
    assert torch.allclose(y_knot[:, 0:4, 0:1], torch.ones(B, 4, 1, H, W))
    # Positions 1..T-1 unchanged
    assert torch.allclose(y_knot[:, 4:, 1:], first_frame_cond[:, :, 1:])
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_y_tensor_knot_injection -v`
Expected: FAIL — `_build_y_with_knot` not defined.

- [ ] **Step 3: Add helper function**

At module top-level in `fastgen/networks/InfiniteTalk/network_causal.py` (near other helpers around the anchor functions):

```python
def _build_y_with_knot(first_frame_cond: torch.Tensor,
                       knot_override: torch.Tensor = None,
                       T: int = None) -> torch.Tensor:
    """Build the 20-channel y-tensor (4 mask + 16 VAE) with optional knot override.
    
    The y-tensor carries the I2V inpainting conditioning:
      - channels [0:4] = binary mask, set to 1 at position 0 (ref/knot slot), 0 elsewhere
      - channels [4:20] = VAE latents. Position 0 = first_frame_cond or knot_override.
    
    Args:
        first_frame_cond: [B, 16, T, H, W] — standard reference image latent.
        knot_override: Optional [B, 16, 1, H, W] — the knot x̄ from previous chunk.
            If given, overrides first_frame_cond at position 0 only.
        T: Number of temporal positions (defaults to first_frame_cond.shape[2]).
    
    Returns:
        y: [B, 20, T, H, W] — concatenated mask + VAE conditioning.
    """
    B, C, T_ff, H, W = first_frame_cond.shape
    if T is None:
        T = T_ff
    assert C == 16, f"Expected 16-channel VAE latent, got {C}"
    
    mask = torch.zeros(B, 4, T, H, W, dtype=first_frame_cond.dtype, device=first_frame_cond.device)
    mask[:, :, 0:1] = 1.0
    
    vae_chans = first_frame_cond.clone()
    if knot_override is not None:
        assert knot_override.shape == (B, 16, 1, H, W), (
            f"knot_override expected shape ({B}, 16, 1, {H}, {W}), got {knot_override.shape}"
        )
        vae_chans[:, :, 0:1] = knot_override
    
    return torch.cat([mask, vae_chans], dim=1)
```

**Note**: this helper duplicates the logic currently inline in `_build_y()` / `generate_infinitetalk_nativeres()`. For now, we keep both codepaths; refactoring to use this helper globally is a follow-up.

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_y_tensor_knot_injection -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_knot_forcing.py
git commit -m "feat(kf): _build_y_with_knot helper — y-tensor with optional knot override at position 0"
```

---

## Phase 4 — Running-ahead: dynamic absolute-RoPE + re-cache

### Task 6: Add `_running_ahead_n` attribute + RoPE position accessor on net

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_running_ahead_initial_and_advance():
    """Net tracks absolute RoPE position for the sink via _running_ahead_n."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    # We're only testing the attribute plumbing, not full forward.
    # Minimal stub — manually set required attrs.
    class NetStub:
        _running_ahead_enabled = True
        _running_ahead_n = 8
        _running_ahead_step = 4
    
    # Import the advance function (to be added).
    from fastgen.networks.InfiniteTalk.network_causal import advance_running_ahead
    
    net = NetStub()
    # cursor at i=0, c=3, k=1 → i+c+k = 4, not > 8, no advance
    advanced, new_n = advance_running_ahead(net, i=0, c=3, k=1)
    assert advanced is False
    assert new_n == 8
    # cursor at i=6 → i+c+k = 10 > 8, advance by s=4 → new n=12
    advanced, new_n = advance_running_ahead(net, i=6, c=3, k=1)
    assert advanced is True
    assert new_n == 12
    assert net._running_ahead_n == 12
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_running_ahead_initial_and_advance -v`
Expected: FAIL (`advance_running_ahead` not defined)

- [ ] **Step 3: Add advancement helper**

Add to `fastgen/networks/InfiniteTalk/network_causal.py` at module level:

```python
def advance_running_ahead(net_module, i: int, c: int, k: int) -> tuple[bool, int]:
    """Advance the net's running-ahead RoPE position if the rollout caught up.
    
    Returns (advanced, new_n). When advanced=True, caller should re-cache the
    sink KV at the new RoPE position.
    
    Fires when i + c + k > n. Advances n by running_ahead_step. No-op when
    running_ahead is disabled on the net.
    """
    if not getattr(net_module, "_running_ahead_enabled", False):
        return False, getattr(net_module, "_running_ahead_n", 0)
    n = net_module._running_ahead_n
    if i + c + k > n:
        new_n = n + net_module._running_ahead_step
        net_module._running_ahead_n = new_n
        return True, new_n
    return False, n
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_running_ahead_initial_and_advance -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_knot_forcing.py
git commit -m "feat(kf): advance_running_ahead helper — trigger + state update for dynamic RoPE"
```

---

### Task 7: Stamp `_running_ahead_*` attrs on the net + hook into sink RoPE calculation

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_running_ahead_overrides_sink_rope_position():
    """When running_ahead is on, sink K RoPE position = _running_ahead_n (absolute),
    NOT the F1 lookahead_distance relative offset."""
    from fastgen.networks.InfiniteTalk.network_causal import compute_sink_rope_position
    
    # F1 baseline: fixed relative offset
    class NetF1:
        _running_ahead_enabled = False
        lookahead_sink_enabled = True
        lookahead_distance = 4
    pos_f1 = compute_sink_rope_position(NetF1(), F_window=10)
    assert pos_f1 == 10 - 1 + 4  # F_window - 1 + lookahead_distance = 13
    
    # KF running-ahead: absolute position from _running_ahead_n
    class NetRA:
        _running_ahead_enabled = True
        _running_ahead_n = 8
        lookahead_sink_enabled = False
    pos_ra = compute_sink_rope_position(NetRA(), F_window=10)
    assert pos_ra == 8
    
    # Neither: natural position 0
    class NetNone:
        _running_ahead_enabled = False
        lookahead_sink_enabled = False
    pos_none = compute_sink_rope_position(NetNone(), F_window=10)
    assert pos_none == 0
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_running_ahead_overrides_sink_rope_position -v`
Expected: FAIL

- [ ] **Step 3: Add `compute_sink_rope_position` helper + refactor existing F1 logic**

Add at module level:

```python
def compute_sink_rope_position(net_module, F_window: int) -> int:
    """Compute the RoPE position to use for the sink's K rotation.
    
    Precedence (enforced by config validation, both can't be True):
      - running_ahead enabled → absolute position from _running_ahead_n
      - lookahead_sink (F1) enabled → F_window - 1 + lookahead_distance
      - neither → natural position 0
    """
    if getattr(net_module, "_running_ahead_enabled", False):
        return int(net_module._running_ahead_n)
    if getattr(net_module, "lookahead_sink_enabled", False):
        return int(F_window - 1 + net_module.lookahead_distance)
    return 0
```

Then, in the existing attention code where F1's sink RoPE position is currently computed (look for `lookahead_distance` usage in `CausalSelfAttention.forward`), replace the inline computation with a call to `compute_sink_rope_position(self, F_window)`. Preserve existing gating (`sink_tokens > 0`, `query_offset_in_win > 0`).

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_running_ahead_overrides_sink_rope_position -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_knot_forcing.py
git commit -m "feat(kf): compute_sink_rope_position — unified F1/running-ahead precedence"
```

---

### Task 8: `_apply_running_ahead_config` helper + config propagation in SF model builder

Mirrors the pattern in `_apply_anchor_config` from `infinitetalk_self_forcing.py`.

**Files:**
- Modify: `fastgen/methods/infinitetalk_self_forcing.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_apply_running_ahead_config():
    """After build_model, net + fake_score + teacher all have _running_ahead_* attrs
    stamped if use_running_ahead is True in config."""
    # Minimal integration test — no actual model build. Just validate the method exists
    # and sets attrs on stubs.
    from fastgen.methods.infinitetalk_self_forcing import InfiniteTalkSelfForcingModel
    
    class Stub:
        pass
    model = Stub()
    model.net = Stub()
    model.fake_score = Stub()
    model.teacher = Stub()
    model.config = Stub()
    model.config.use_running_ahead = True
    model.config.running_ahead_step = 4
    model.config.running_ahead_init_n = 8
    
    # Bind the method from the class
    InfiniteTalkSelfForcingModel._apply_running_ahead_config(model)
    
    assert model.net._running_ahead_enabled is True
    assert model.net._running_ahead_step == 4
    assert model.net._running_ahead_n == 8
    # Teacher shouldn't have running_ahead (bidirectional, processes full sequence)
    assert getattr(model.teacher, "_running_ahead_enabled", False) is False
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_apply_running_ahead_config -v`
Expected: FAIL (`_apply_running_ahead_config` not defined)

- [ ] **Step 3: Add method to `InfiniteTalkSelfForcingModel`**

In `fastgen/methods/infinitetalk_self_forcing.py`, add method alongside `_apply_anchor_config`:

```python
    def _apply_running_ahead_config(self):
        """Stamp _running_ahead_* attributes on the student net (and fake_score)
        based on config flags. Teacher is bidirectional and does NOT receive
        running-ahead — it sees the full rolled-out sequence in DMD.
        """
        enabled = getattr(self.config, "use_running_ahead", False)
        step = getattr(self.config, "running_ahead_step", 4)
        init_n = getattr(self.config, "running_ahead_init_n", 8)
        
        for role, net in [("student", getattr(self, "net", None)),
                          ("fake_score", getattr(self, "fake_score", None))]:
            if net is None:
                continue
            net._running_ahead_enabled = enabled
            net._running_ahead_step = step
            net._running_ahead_n = init_n
            if enabled:
                logger.info(f"[running_ahead] {role}: enabled, step={step}, init_n={init_n}")
            else:
                logger.info(f"[running_ahead] {role}: disabled")
```

And call it at the end of `build_model` (alongside the existing `self._apply_anchor_config()` call):

```python
        self._apply_anchor_config()
        self._apply_running_ahead_config()
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_apply_running_ahead_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/methods/infinitetalk_self_forcing.py tests/test_knot_forcing.py
git commit -m "feat(kf): _apply_running_ahead_config — stamps attrs on student + fake_score"
```

---

### Task 9: Sink KV re-cache helper — re-encode reference at new RoPE position

When running-ahead fires and advances `n`, the sink KV needs to be re-computed at the new absolute RoPE position. The sink holds the reference image's KV, so we re-run the model's "cache-store" forward on `first_frame_cond` but with the new RoPE position.

**Files:**
- Modify: `fastgen/networks/InfiniteTalk/network_causal.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_sink_kv_recache_stub():
    """Sink-KV re-cache entry point exists and can be invoked. Actual re-encoding
    validation is an integration test — here we only verify plumbing."""
    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    # Verify method exists on class
    assert hasattr(CausalInfiniteTalkWan, "recache_sink_at_rope_position"), (
        "CausalInfiniteTalkWan must expose recache_sink_at_rope_position"
    )
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_sink_kv_recache_stub -v`
Expected: FAIL (method not present)

- [ ] **Step 3: Add method to `CausalInfiniteTalkWan`**

Add to the class body (near existing cache management methods):

```python
    @torch.no_grad()
    def recache_sink_at_rope_position(self, first_frame_cond: torch.Tensor,
                                       condition: dict,
                                       new_rope_pos: int) -> None:
        """Re-encode the sink chunk (frame 0 = reference image) at a new RoPE
        position and overwrite the sink slot in the KV cache.
        
        Used by running-ahead: when n advances, the sink's K rotation changes,
        so its cached values become stale. This method recomputes them.
        
        Args:
            first_frame_cond: [B, 16, T, H, W] — reference image latent.
            condition: full condition dict (text_embeds, audio_emb, clip_features, etc.).
            new_rope_pos: new absolute RoPE position for the sink.
        """
        self._running_ahead_n = new_rope_pos
        # Run a cache-store forward on frame 0 only.
        # The sink (frame 0) is part of the first c frames in the normal cache pass;
        # re-running that pass with running_ahead enabled will write new sink K/V
        # at the updated RoPE position.
        B = first_frame_cond.shape[0]
        # Single-frame cache write at t=0
        t_zero = torch.zeros(B, device=first_frame_cond.device, dtype=first_frame_cond.dtype)
        _ = self(
            first_frame_cond[:, :, 0:1],
            t_zero,
            condition=condition,
            cache_tag="pos",
            store_kv=True,
            cur_start_frame=0,
            fwd_pred_type="x0",
            is_ar=True,
        )
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_sink_kv_recache_stub -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/networks/InfiniteTalk/network_causal.py tests/test_knot_forcing.py
git commit -m "feat(kf): recache_sink_at_rope_position — re-encode sink KV at new RoPE"
```

---

## Phase 5 — Dataloader: last-frame reference + audio pre-check

### Task 10: Dataloader last-frame reference option

**Files:**
- Modify: `fastgen/datasets/infinitetalk_dataloader.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write failing test**

Append:

```python
def test_dataloader_last_frame_reference_option():
    """When use_last_frame_reference=True, first_frame_cond's position 0
    comes from vae_latents[:, :, -1:] instead of the precomputed file."""
    # This test uses a stub sample dir with fake tensors. If full dataloader
    # setup is too heavy, test the swap logic directly:
    from fastgen.datasets.infinitetalk_dataloader import _maybe_swap_last_frame_ref
    
    C, T, H, W = 16, 5, 4, 4
    vae_latents = torch.arange(C * T * H * W, dtype=torch.float32).reshape(C, T, H, W)
    first_frame_cond = torch.zeros(C, T, H, W)
    first_frame_cond[:, 0] = 1.0  # only position 0 carries info normally
    
    # Default behavior (use_last_frame=False) — unchanged
    ffc_default = _maybe_swap_last_frame_ref(
        first_frame_cond.clone(), vae_latents, use_last_frame=False
    )
    assert torch.allclose(ffc_default, first_frame_cond)
    
    # use_last_frame=True — position 0 replaced by vae_latents[:, -1]
    ffc_last = _maybe_swap_last_frame_ref(
        first_frame_cond.clone(), vae_latents, use_last_frame=True
    )
    assert torch.allclose(ffc_last[:, 0], vae_latents[:, -1])
    # Positions 1..T-1 still zero
    assert torch.allclose(ffc_last[:, 1:], torch.zeros(C, T-1, H, W))
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_dataloader_last_frame_reference_option -v`
Expected: FAIL

- [ ] **Step 3: Add helper in dataloader module**

Add to `fastgen/datasets/infinitetalk_dataloader.py` near the top (with other helpers):

```python
def _maybe_swap_last_frame_ref(first_frame_cond: torch.Tensor,
                                vae_latents: torch.Tensor,
                                use_last_frame: bool) -> torch.Tensor:
    """If use_last_frame=True, replace first_frame_cond[:, 0] with the last
    latent frame from vae_latents. Otherwise return first_frame_cond unchanged.
    
    Semantics: under KF, the 'reference image' is the clip's last frame
    (future anchor / pseudo-last-frame), not the first frame. vae_latents[:, -1]
    is the last latent frame from the full-clip VAE encoding — not strictly
    equivalent to VAE-encoding just the last video frame (VAE has temporal
    convs), but captures the clip's endpoint semantics.
    
    Args:
        first_frame_cond: [C=16, T, H, W] — original (first-frame-based) ref conditioning.
        vae_latents: [C=16, T, H, W] — full-clip VAE latent sequence.
        use_last_frame: if True, swap; if False, return unchanged.
    
    Returns:
        first_frame_cond_maybe_swapped: [C=16, T, H, W]
    """
    if not use_last_frame:
        return first_frame_cond
    out = first_frame_cond.clone()
    out[:, 0] = vae_latents[:, -1]
    return out
```

Then, inside `__getitem__`, after the block that loads `first_frame_cond` and `vae_latents`, call the helper:

```python
        first_frame_cond = _maybe_swap_last_frame_ref(
            first_frame_cond, vae_latents,
            use_last_frame=getattr(self, "use_last_frame_reference", False),
        )
```

And in the dataset `__init__`, accept the option:

```python
        self.use_last_frame_reference = kwargs.get("use_last_frame_reference", False)
```

Also update the config `config_infinitetalk_sf.py` `dataloader_train` section to pass this through from the SF config when `use_last_frame_reference=True`.

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_dataloader_last_frame_reference_option -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/datasets/infinitetalk_dataloader.py fastgen/configs/methods/config_infinitetalk_sf.py tests/test_knot_forcing.py
git commit -m "feat(kf): dataloader use_last_frame_reference option (from vae_latents[-1])"
```

---

## Phase 6 — Rollout-with-gradient: c+k denoising + knot + fusion + commit-c

### Task 11: Extend `rollout_with_gradient` to `c+k` denoise window (feature-flagged)

This is the largest change. Split into sub-steps: first restructure, then add knot injection, then add fusion.

**Files:**
- Modify: `fastgen/methods/distribution_matching/self_forcing.py`
- Test: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
def test_rollout_cplusk_allocates_correctly():
    """When use_temporal_knot=True, each chunk's noisy_input slice is c+k frames,
    not c. Committed output remains num_chunks * c."""
    # Integration test using a MockNet pattern from existing tests.
    # Reference tests/test_sample_loop_toggles.py for MockNet template.
    pytest.skip("Integration test — implement after MockNet template is reusable")


def test_rollout_knot_fusion_at_exit_step():
    """At exit step for iter > 0, output[:, :, 0:k] = (x̄ + x̂) / 2."""
    pytest.skip("Integration test — see above")


def test_rollout_commits_only_c_frames():
    """rollout_with_gradient returns [B, 16, num_chunks*c, H, W] regardless of k."""
    pytest.skip("Integration test — see above")
```

These skipped tests are TODO markers — to be fully implemented in Task 14 once MockNet is reusable.

- [ ] **Step 2: Refactor `rollout_with_gradient` with KF branch**

In `fastgen/methods/distribution_matching/self_forcing.py`, modify the `rollout_with_gradient` method. Locate the main loop (currently starts at `for block_idx in range(num_blocks):`). Replace it with:

```python
        # KF flags (all default off → behavior unchanged)
        use_knot = getattr(self.config, "use_temporal_knot", False)
        k = int(getattr(self.config, "knot_size", 1)) if use_knot else 0
        denoise_len = chunk_size + k  # c+k frames per denoise forward under KF
        
        # KF requires num_frames = num_chunks * c + k (extra k for last chunk's knot)
        # If use_knot is off, falls back to existing c-per-chunk logic.
        if use_knot:
            assert noise.shape[2] >= num_blocks * chunk_size + k, (
                f"For KF: noise must have at least num_chunks*c+k = "
                f"{num_blocks * chunk_size + k} frames, got {noise.shape[2]}"
            )
        
        knot_latent = None  # x̄; None at iter 0
        
        denoised_blocks = []
        for block_idx in range(num_blocks):
            if num_blocks == 0:
                cur_start_frame, cur_end_frame = 0, remaining_size
            else:
                cur_start_frame = 0 if block_idx == 0 else chunk_size * block_idx + remaining_size
                cur_end_frame = cur_start_frame + denoise_len
            
            noisy_input = noise[:, :, cur_start_frame:cur_end_frame]
            
            # Build per-chunk condition with knot injection for iter > 0.
            # Strategy: overwrite first_frame_cond[:, :, 0:1] with knot_latent so
            # all downstream readers (input anchor, output anchor, _build_y,
            # post-step pin) automatically pick up the knot without extra keys.
            # Only 'output_anchor_disabled_for_fusion' is a new flag — disables
            # the output pin at position 0 so fusion can happen.
            if use_knot and block_idx > 0 and knot_latent is not None:
                modified_ffc = condition["first_frame_cond"].clone()
                modified_ffc[:, :, 0:1] = knot_latent
                chunk_condition = {
                    **condition,
                    "first_frame_cond": modified_ffc,
                    "output_anchor_disabled_for_fusion": True,
                }
            else:
                chunk_condition = condition
            
            # Denoising loop — unchanged structure, just uses chunk_condition
            for step, t_cur in enumerate(t_list):
                if self.config.same_step_across_blocks:
                    exit_flag = step == denoising_end_steps[0]
                else:
                    exit_flag = step == denoising_end_steps[block_idx]
                
                t_chunk_cur = t_cur.expand(batch_size)
                
                if not exit_flag:
                    with torch.no_grad():
                        x0_pred_chunk = self.net(
                            noisy_input, t_chunk_cur,
                            condition=chunk_condition,
                            cache_tag="pos", store_kv=False,
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0", is_ar=True,
                        )
                    t_next = t_list[step + 1]
                    t_chunk_next = t_next.expand(batch_size)
                    if self.config.student_sample_type == "sde":
                        eps_infer = torch.randn_like(x0_pred_chunk)
                    elif self.config.student_sample_type == "ode":
                        eps_infer = self.net.noise_scheduler.x0_to_eps(
                            xt=noisy_input, x0=x0_pred_chunk, t=t_chunk_cur
                        )
                    noisy_input = self.net.noise_scheduler.forward_process(
                        x0_pred_chunk, eps_infer, t_chunk_next
                    )
                else:
                    enable_grad = (
                        enable_gradient and torch.is_grad_enabled()
                        and (cur_start_frame >= start_gradient_frame)
                    )
                    with torch.set_grad_enabled(enable_grad):
                        x0_pred_chunk = self.net(
                            noisy_input, t_chunk_cur,
                            condition=chunk_condition,
                            cache_tag="pos",
                            store_kv=getattr(self.net, "_skip_clean_cache_pass", False) and not use_knot,
                            # Under KF: F3 OFF — we'll use a separate cache pass for committed c frames only.
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0", is_ar=True,
                        )
                    
                    # KF: fusion at exit step
                    if use_knot and block_idx > 0 and knot_latent is not None:
                        x0_pred_chunk = x0_pred_chunk.clone()
                        x0_pred_chunk[:, :, 0:k] = (knot_latent + x0_pred_chunk[:, :, 0:k]) / 2.0
                    
                    # KF: save new knot (last k frames of c+k prediction)
                    if use_knot:
                        knot_latent = x0_pred_chunk[:, :, chunk_size:chunk_size + k]
                    
                    break
            
            # KF: commit only first c frames to output
            if use_knot:
                committed = x0_pred_chunk[:, :, 0:chunk_size]
            else:
                committed = x0_pred_chunk
            denoised_blocks.append(committed)
            
            # KV cache update: committed c frames only under KF (separate pass).
            # When use_knot=False, existing logic unchanged.
            if use_knot:
                # KF always uses the separate cache pass, never F3.
                with torch.no_grad():
                    if self.config.context_noise > 0:
                        t_cache = torch.full((batch_size,), self.config.context_noise,
                                             device=self.device, dtype=dtype)
                        x0_pred_cache = self.net.noise_scheduler.forward_process(
                            committed, torch.randn_like(committed), t_cache,
                        )
                    else:
                        x0_pred_cache = committed
                        t_cache = torch.zeros(batch_size, device=self.device, dtype=dtype)
                    _ = self.net(
                        x0_pred_cache, t_cache,
                        condition=condition,  # no knot in cache pass
                        cache_tag="pos", store_kv=True,
                        cur_start_frame=cur_start_frame,
                        fwd_pred_type="x0", is_ar=True,
                    )
            else:
                # Existing non-KF cache logic (F3-aware)
                skip_clean_cache = getattr(self.net, "_skip_clean_cache_pass", False)
                if not skip_clean_cache:
                    with torch.no_grad():
                        # ... existing code ...
                        if self.config.context_noise > 0:
                            t_cache = torch.full((batch_size,), self.config.context_noise,
                                                 device=self.device, dtype=dtype)
                            x0_pred_cache = self.net.noise_scheduler.forward_process(
                                committed, torch.randn_like(committed), t_cache,
                            )
                        else:
                            x0_pred_cache = committed
                            t_cache = torch.zeros(batch_size, device=self.device, dtype=dtype)
                        _ = self.net(
                            x0_pred_cache, t_cache, condition=condition,
                            cache_tag="pos", store_kv=True,
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0", is_ar=True,
                        )
        
        output = torch.cat(denoised_blocks, dim=2) if denoised_blocks else torch.empty_like(noise)
        self.net.clear_caches()
        return output
```

- [ ] **Step 3: Run existing regression tests**

Run: `python -m pytest tests/test_rollout_gradient_sanity.py tests/test_rollout_f3_toggle.py -v`
Expected: PASS (use_knot=False path unchanged → existing tests still pass)

- [ ] **Step 4: Commit**

```bash
git add fastgen/methods/distribution_matching/self_forcing.py tests/test_knot_forcing.py
git commit -m "feat(kf): rollout_with_gradient — c+k window, knot injection, fusion, commit-c"
```

---

### Task 12: Running-ahead trigger in rollout outer loop

**Files:**
- Modify: `fastgen/methods/distribution_matching/self_forcing.py`

- [ ] **Step 1: Insert running-ahead trigger at top of rollout loop**

Before the denoising loop inside `for block_idx in range(num_blocks):`, add:

```python
            # Running-ahead: advance sink RoPE + re-cache if rollout caught up
            if getattr(self.net, "_running_ahead_enabled", False):
                from fastgen.networks.InfiniteTalk.network_causal import advance_running_ahead
                advanced, new_n = advance_running_ahead(
                    self.net, i=cur_start_frame, c=chunk_size, k=k
                )
                if advanced:
                    # Re-cache the sink KV at new RoPE position
                    ffc = condition.get("first_frame_cond")
                    if ffc is not None:
                        self.net.recache_sink_at_rope_position(ffc, condition, new_n)
```

- [ ] **Step 2: Sanity-check the existing test suite**

Run: `python -m pytest tests/test_rollout_gradient_sanity.py -v`
Expected: PASS (no-op when running_ahead disabled)

- [ ] **Step 3: Commit**

```bash
git add fastgen/methods/distribution_matching/self_forcing.py
git commit -m "feat(kf): running-ahead trigger + sink re-cache in rollout_with_gradient"
```

---

## Phase 7 — Validation path (`_student_sample_loop`)

### Task 13: Mirror KF changes to validation loop

The validation path at `fastgen/methods/distribution_matching/causvid.py::_student_sample_loop` must match training rollout.

**Files:**
- Modify: `fastgen/methods/distribution_matching/causvid.py`

- [ ] **Step 1: Port the KF branch**

Follow the exact same pattern as Task 11 Step 2, applied to `_student_sample_loop`. Replace the main chunk loop with the KF-flagged version that:
1. Denoises `c+k` frames per chunk when `use_knot` is True.
2. Injects `knot_latent` via `anchor_pin_source` / `output_anchor_disabled_for_fusion`.
3. Applies fusion at the final step.
4. Saves `knot_latent ← x̂_{i+c:i+c+k}` and commits `x̂_{0:c}`.
5. Cache update uses committed frames only under KF.
6. Running-ahead advancement at top of chunk loop (same pattern as rollout).

**Key difference from training rollout**: `_student_sample_loop` uses `no_grad` throughout and has an in-place writeback pattern `x[:, :, start:end] = x_next` (pre-existing, gradient-unsafe — not a concern here since we're not backpropping through it).

- [ ] **Step 2: Sanity-check existing tests**

Run: `python -m pytest tests/test_sample_loop_toggles.py tests/test_post_step_pin_inference.py -v`
Expected: PASS (unchanged when use_knot=False)

- [ ] **Step 3: Commit**

```bash
git add fastgen/methods/distribution_matching/causvid.py
git commit -m "feat(kf): _student_sample_loop — KF path mirror (validation)"
```

---

## Phase 8 — Inference CLI (`run_inference`)

### Task 14: Mirror KF changes to production inference

**Files:**
- Modify: `scripts/inference/inference_causal.py`

- [ ] **Step 1: Port KF branch into `run_inference`**

Apply the identical KF logic from Tasks 11-12 to `run_inference()` in `scripts/inference/inference_causal.py`. Add CLI flags:

```python
    p.add_argument("--use_temporal_knot", action="store_true",
                   help="Enable Knot Forcing temporal knot (c+k denoise window, fusion, commit-c).")
    p.add_argument("--knot_size", type=int, default=1, help="Knot length k (paper uses 1).")
    p.add_argument("--use_running_ahead", action="store_true",
                   help="Enable running-ahead dynamic RoPE + sink re-cache.")
    p.add_argument("--running_ahead_step", type=int, default=4, help="Running-ahead step s.")
    p.add_argument("--running_ahead_init_n", type=int, default=8, help="Initial reference RoPE position.")
    p.add_argument("--use_last_frame_reference", action="store_true",
                   help="Source the reference image from the last video frame latent (KF training convention).")
```

Wire these flags through to the model's attributes via `_apply_running_ahead_config`-equivalent code at model-load time.

- [ ] **Step 2: Add integration smoke — CLI parses flags correctly**

Run: `python scripts/inference/inference_causal.py --help 2>&1 | grep -E "use_temporal_knot|running_ahead|last_frame_reference"`
Expected: All 6 new flags listed.

- [ ] **Step 3: Commit**

```bash
git add scripts/inference/inference_causal.py
git commit -m "feat(kf): inference_causal.py — KF CLI flags + run_inference KF path"
```

---

## Phase 9 — Audio slicing extension

### Task 15: Ensure audio_emb can handle c+k slices per chunk

Audio is precomputed for the full clip. The dataloader already loads the full tensor and asserts `audio_emb.shape[0] >= _train_pixel_frames`, slicing to exactly `_train_pixel_frames`.

**Key check**: the stored audio may already have more slices than `_train_pixel_frames` (if precompute used a larger `num_video_frames`). If so, we can just slice more for KF — no padding needed. If stored audio is exactly `_train_pixel_frames`, we right-pad by k extra latent-frame-equivalent pixel slices (repeat last slice). Task 15 covers both cases.

**Files:**
- Modify: `fastgen/datasets/infinitetalk_dataloader.py`

- [ ] **Step 1: Write test**

Append:

```python
def test_audio_emb_padded_for_kf():
    from fastgen.datasets.infinitetalk_dataloader import _maybe_pad_audio_for_knot
    audio = torch.arange(21 * 12 * 768, dtype=torch.float32).reshape(21, 12, 768)
    # Non-KF: unchanged
    out = _maybe_pad_audio_for_knot(audio, extra=0)
    assert out.shape == audio.shape
    # KF with extra=4 pixel frames: last slice repeated 4x
    out_k = _maybe_pad_audio_for_knot(audio, extra=4)
    assert out_k.shape == (25, 12, 768)
    assert torch.allclose(out_k[:21], audio)
    assert torch.allclose(out_k[21], audio[20])  # last slice repeated
    assert torch.allclose(out_k[24], audio[20])
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_audio_emb_padded_for_kf -v`
Expected: FAIL

- [ ] **Step 3: Modify audio slicing in `__getitem__`**

In `fastgen/datasets/infinitetalk_dataloader.py::__getitem__`, locate the block:

```python
        audio_emb = audio_emb[:self._train_pixel_frames]  # [train_pixel_frames, 12, 768]
```

Replace with:

```python
        # KF: extend slice by extra_k_pixel frames if available on disk; else pad.
        extra_k_pixel = getattr(self, "knot_size_extra_audio_pixel", 0)
        target_len = self._train_pixel_frames + extra_k_pixel
        if audio_emb.shape[0] >= target_len:
            # Enough stored on disk — just slice more.
            audio_emb = audio_emb[:target_len]
        else:
            # Slice to what's available, then repeat-last-pad to target_len.
            audio_emb = audio_emb[:self._train_pixel_frames]
            audio_emb = _maybe_pad_audio_for_knot(audio_emb, target_len - audio_emb.shape[0])
```

Add helper at module level:

```python
def _maybe_pad_audio_for_knot(audio_emb: torch.Tensor, extra: int) -> torch.Tensor:
    """Right-pad audio_emb by `extra` slices (repeat last). No-op when extra <= 0."""
    if extra <= 0:
        return audio_emb
    last = audio_emb[-1:]
    pad = last.repeat(extra, *([1] * (audio_emb.dim() - 1)))
    return torch.cat([audio_emb, pad], dim=0)
```

Add to `__init__`:

```python
        self.knot_size_extra_audio_pixel = kwargs.get("knot_size_extra_audio_pixel", 0)
```

**Choosing `knot_size_extra_audio_pixel`**: for our quarter-res setup with num_latent_frames=21 and _train_pixel_frames=81, one latent frame ≈ 4 pixel frames. So for k=1, set `knot_size_extra_audio_pixel = 4`. The config propagates this from `config.model.knot_size * vae_temporal_stride` (approximately 4).

And in `__init__`:

```python
        self.knot_size_extra_audio = kwargs.get("knot_size_extra_audio", 0)
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_knot_forcing.py::test_audio_emb_padded_for_kf -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add fastgen/datasets/infinitetalk_dataloader.py tests/test_knot_forcing.py
git commit -m "feat(kf): audio_emb padding for c+k last chunk"
```

---

## Phase 10 — Training config + launch scripts

### Task 16: New training config `config_sf_w9s1_knot_runahead.py`

**Files:**
- Create: `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_knot_runahead.py`

- [ ] **Step 1: Create the config**

```python
# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk Self-Forcing — w9s1 + Knot Forcing + Running-Ahead.

Baked feature configuration:
  - w9s1 attention   (sink_size=1, rolling=9, local_attn_size=10)
  - F1 lookahead     OFF (replaced by running-ahead)
  - F2 sink cache    OFF
  - F3 skip clean    OFF (KF uses separate cache pass for committed c frames only)
  - Temporal Knot    ON, k=1
  - Running-Ahead    ON, s=4, init_n=8
  - Last-frame ref   ON (dataloader uses vae_latents[:, :, -1])
  - Anchors          ON (input + output, student + fake_score + teacher)

Derived from config_sf_w9s1.py; do NOT inherit from config_sf_w9s1_lookahead_f3.py
(lookahead-sink is incompatible with running-ahead).

Usage:
    bash scripts/run_sf_w9s1_knot_runahead.sh
"""
import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1 import (
    create_config as create_w9s1_config,
)


def create_config():
    config = create_w9s1_config()
    
    # ---- Disable F1/F2/F3 ----
    config.model.net.use_dynamic_rope = True            # still needed by RA's RoPE override
    config.model.lookahead_sink_enabled = False
    config.model.lookahead_distance = 0
    config.model.model_sink_cache_enabled = False
    config.model.skip_clean_cache_pass = False
    
    # ---- Temporal Knot ----
    config.model.use_temporal_knot = True
    config.model.knot_size = 1
    
    # ---- Running-Ahead ----
    config.model.use_running_ahead = True
    config.model.running_ahead_step = 4
    config.model.running_ahead_init_n = 8
    
    # ---- Last-frame reference ----
    config.model.use_last_frame_reference = True
    # Propagate to dataloader
    config.dataloader_train.use_last_frame_reference = True
    # k=1 latent frame ≈ 4 video pixel frames (VAE temporal stride ~3.86)
    config.dataloader_train.knot_size_extra_audio_pixel = 4
    
    # ---- Validate ----
    config.model.validate_kf_flags()
    
    # ---- Propagate to net constructor ----
    config.model.net.lookahead_sink_enabled = False
    config.model.net.lookahead_distance = 0
    
    # ---- Logging / run name ----
    config.log_config.group = "infinitetalk_sf"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"sf_w9s1_knot_k1_ra_s4_n8_lastref_freq5_lr1e5_{timestamp}"
    config.log_config.name = run_name
    
    return config
```

- [ ] **Step 2: Smoke-test the config loads**

Run: `python -c "from fastgen.configs.experiments.InfiniteTalk.config_sf_w9s1_knot_runahead import create_config; c = create_config(); print('OK:', c.log_config.name)"`
Expected: prints `OK: sf_w9s1_knot_k1_ra_s4_n8_lastref_freq5_lr1e5_...`

- [ ] **Step 3: Commit**

```bash
git add fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_knot_runahead.py
git commit -m "feat(kf): config_sf_w9s1_knot_runahead — training config with KF + RA + last-ref"
```

---

### Task 17: Training launcher `run_sf_w9s1_knot_runahead.sh`

**Files:**
- Create: `scripts/run_sf_w9s1_knot_runahead.sh`

- [ ] **Step 1: Create the launcher**

Model this on `scripts/run_sf_w9s1_lookahead_f3.sh`. Copy that file to `scripts/run_sf_w9s1_knot_runahead.sh` and modify:
- Banner messages: "w9s1 + Knot Forcing + Running-Ahead" instead of "lookahead + F3"
- Config path: `config_sf_w9s1_knot_runahead.py`
- DF init: same anchor-ON stochastic DF checkpoint (`quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints/` — latest)
- Remove F1/F2/F3 overrides (not applicable)
- Add KF-specific echo lines:
  ```
  echo "Temporal Knot: ON (k=1, commit=3, denoise=4)"
  echo "Running Ahead: ON (s=4, init_n=8)"
  echo "Last-frame ref: ON (dataloader uses vae_latents[:, -1])"
  ```

Make executable: `chmod +x scripts/run_sf_w9s1_knot_runahead.sh`

- [ ] **Step 2: Dry-run parse the script (no torchrun)**

Run: `bash -n scripts/run_sf_w9s1_knot_runahead.sh && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_sf_w9s1_knot_runahead.sh
git commit -m "feat(kf): run_sf_w9s1_knot_runahead.sh — training launcher"
```

---

### Task 18: Inference launcher + monitor for KF run

**Files:**
- Create: `scripts/inference/run_inference_knot_runahead.sh`
- Create: `scripts/inference/monitor_sf_w9s1_knot.sh`

- [ ] **Step 1: Copy + adapt the existing F3 inference launcher**

Copy `scripts/inference/run_inference_w9s1_lookahead_f3.sh` to `scripts/inference/run_inference_knot_runahead.sh`. Modify flags:

Remove: `--use_dynamic_rope --lookahead_sink --lookahead_distance 4 --skip_clean_cache_pass`
Add: `--use_temporal_knot --knot_size 1 --use_running_ahead --running_ahead_step 4 --running_ahead_init_n 8 --use_last_frame_reference`

- [ ] **Step 2: Copy + adapt the monitor**

Copy `InfiniteTalk/monitor_sf_w9s1_la4_f3.sh` (our existing monitor) to `scripts/inference/monitor_sf_w9s1_knot.sh`. Modify:
- `RUN_NAME`: change to match new run (pattern: `sf_w9s1_knot_k1_ra_s4_n8_lastref_freq5_lr1e5_*`)
- Inference call: replace the F3 flags with KF flags (as in Step 1)
- `OUTPUT_ROOT` / `EVAL_ROOT`: update to match the new RUN_NAME

Make executable: `chmod +x scripts/inference/run_inference_knot_runahead.sh scripts/inference/monitor_sf_w9s1_knot.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/inference/run_inference_knot_runahead.sh scripts/inference/monitor_sf_w9s1_knot.sh
git commit -m "feat(kf): inference launcher + monitor for KF training run"
```

---

## Phase 11 — Regression + integration validation

### Task 19: Regression test — KF-off bit-exact

**Files:**
- Modify: `tests/test_knot_forcing.py`

- [ ] **Step 1: Write regression test**

Append:

```python
def test_kf_off_is_bit_exact_to_sf():
    """With all KF flags off, rollout output must be bit-exact to pre-KF code.
    
    This is a smoke test using MockNet — full model bit-exactness would need
    a deterministic fixture checkpoint which is overkill here. Instead we
    verify the code path branching is clean.
    """
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig
    
    cfg = InfiniteTalkSFModelConfig()
    # Verify all KF flags default off
    assert not cfg.use_temporal_knot
    assert not cfg.use_running_ahead
    assert not cfg.use_last_frame_reference
    
    # Validator passes when all off
    cfg.validate_kf_flags()
    
    # With use_knot=False, rollout code should take the pre-existing branch
    # (verified visually in self_forcing.py; full end-to-end validation happens
    # via the existing test suite not regressing).
```

- [ ] **Step 2: Run full existing test suite as the real regression**

Run: `cd /data/.../FastGen_InfiniteTalk_knot && python -m pytest tests/test_rollout_gradient_sanity.py tests/test_rollout_f3_toggle.py tests/test_sample_loop_toggles.py tests/test_post_step_pin_inference.py tests/test_input_anchor_plumbing.py tests/test_apply_anchor_kwarg.py tests/test_apply_attention_config.py tests/test_inference_causal_toggles.py tests/test_lookahead_pipeline_integration.py tests/test_stochastic_lookahead_distance.py tests/test_rolling_attention.py tests/test_knot_forcing.py -v`
Expected: ALL PASS (our changes preserve existing behavior when KF flags are off).

- [ ] **Step 3: Commit**

```bash
git add tests/test_knot_forcing.py
git commit -m "test(kf): regression harness — KF-off bit-exact to existing SF"
```

---

### Task 20: Smoke test — valtest config with KF on

**Files:**
- Create: `fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_knot_valtest.py`
- Create: `scripts/run_sf_w9s1_knot_valtest.sh`

- [ ] **Step 1: Create valtest config**

Mirror `config_sf_w9s1_lookahead_valtest.py` pattern. Inherit from `config_sf_w9s1_knot_runahead`, override to:
- Tiny train list (`val_quarter_2.txt`)
- `dataloader_train.raw_data_root = None` (skip VAE+CLIP+wav2vec load per CLAUDE.md fast-path)
- `max_iter = 10`

- [ ] **Step 2: Create valtest launcher**

Model on `scripts/run_sf_w9s1_lookahead_valtest.sh`.

- [ ] **Step 3: Run valtest (end-to-end smoke)**

Run: `bash scripts/run_sf_w9s1_knot_valtest.sh 2>&1 | tail -40`
Expected: 10 training iterations complete without crash. Loss values finite. Gradients don't explode.

- [ ] **Step 4: Commit**

```bash
git add fastgen/configs/experiments/InfiniteTalk/config_sf_w9s1_knot_valtest.py scripts/run_sf_w9s1_knot_valtest.sh
git commit -m "test(kf): valtest config + launcher for end-to-end smoke"
```

---

### Task 21: Document the run — new analysis / spec doc

**Files:**
- Create: `docs/2026-04-18-knot-forcing-implementation-session.md`

- [ ] **Step 1: Write session record**

Modeled on `docs/2026-04-16-input-anchor-f3-training-session.md`. Should cover:
- Summary of KF mechanism (link to understanding doc)
- Files changed
- Config choices (k=1, s=4, n_init=8, last-ref ON)
- DF init checkpoint used
- Known differences from paper (hybrid vs pathway-separated design)
- Open questions / follow-ups (training-time running-ahead firing frequency, fusion gradient compute cost observed, etc.)

- [ ] **Step 2: Commit**

```bash
git add docs/2026-04-18-knot-forcing-implementation-session.md
git commit -m "docs(kf): session record for Knot Forcing implementation"
```

---

## Phase 12 — Launch training (deferred; not a task for the implementer)

Not part of this plan's automated execution. After all implementation tasks pass + valtest smoke succeeds, the user manually:

1. Verifies 8-GPU cluster availability.
2. Runs `bash scripts/run_sf_w9s1_knot_runahead.sh`.
3. Starts monitor in parallel: `bash scripts/inference/monitor_sf_w9s1_knot.sh`.
4. Watches wandb group `infinitetalk_sf`, run name `sf_w9s1_knot_k1_ra_s4_n8_lastref_freq5_lr1e5_*`.

---

## Notes for the implementer

- **TDD discipline**: write the failing test, watch it fail with the expected error, then implement. Do not write implementation before test.
- **KF-off regression is sacred**: every commit should keep the existing test suite green. When KF flags are off, the code path must be bit-exact to pre-KF behavior.
- **Skipped integration tests** (marked `pytest.skip`) in Task 11 are TODOs — implementing them requires setting up a MockNet harness. Defer unless time permits.
- **Gradient flow through fusion**: at iter > 0 exit step, `(x̄ + x̂) / 2` has two autograd paths (`x̄` from prev iter, `x̂` from current). This is correct but expensive. If memory/compute is prohibitive during training, fall back to `x̄.detach()` (flagged as Task 22 follow-up, not in this plan).
- **F3 under KF**: turned off for simplicity (separate cache-store pass used instead). Can be re-enabled later with a masked cache-write if speed matters.
- **No new precomputed data required**: KF uses existing `vae_latents.pt`, `audio_emb.pt`, etc. Only dataloader indexing changes.
