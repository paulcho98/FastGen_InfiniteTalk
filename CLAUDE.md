# FastGen_InfiniteTalk — Codebase Notes

## Environment
- 8× A100-SXM4-80GB, torch 2.8, flash_attn 2.8.3, FSDP2, bf16. loguru via `fastgen.utils.logging_utils`.
- `pytest --timeout=N` is NOT supported (no pytest-timeout installed). Use `timeout N python -m pytest …` or accept longer tests.

## Config hierarchy (SF training)
`config_infinitetalk_sf.py` (base) → `config_sf.py` → `config_sf_w9s1.py` → `config_sf_w9s1_lookahead.py` → `config_sf_w9s1_lookahead_noanchor.py`

## Run script / env var interaction (gotcha)
Run scripts `export VAR="${VAR:-default}"` BEFORE torchrun. Defaults set in the script OVERRIDE any config-level default for the same env var. To change behavior, either edit the script or pass `VAR=value bash scripts/…` at the call site.

## Known footguns
- `bool(os.environ.get("VAR", ""))` is truthy for `"0"` — see `config_sf.py:257`. Used throughout; be careful setting flags to `0` to disable.
- `wandb.Video` expects numpy `[T, C, H, W]`, NOT `[T, H, W, C]`. Passing the wrong layout silently produces 5–15 KB garbled MP4s instead of 150–800 KB valid videos.
- `sink_tokens` in `CausalSelfAttention.forward` is a **constant** (`self.sink_size * frame_seqlen`) — does NOT reflect whether `k_win` actually contains a cached sink slab. Any lookahead/stochastic logic that uses it must additionally gate on `query_offset_in_win > 0` to exclude chunk 0.
- `_apply_anchor_config` runs AFTER FSDP wrapping. Setting `self.net.attr = X` and `block.self_attn.attr = X` both propagate through FSDP transparently.

## Dataloader init is slow (NFS)
`InfiniteTalkDataLoader.__init__` does `torch.load(vae_latents.pt)` for every sample as a shape check. For `train_excl_val30.txt` (~147k samples) this is a full-dataset NFS scan, 20+ min.
- Validation-only / smoke tests: set train list to `val_quarter_2.txt` (2 samples) AND `config.dataloader_train.raw_data_root = None` (disables the VAE+CLIP+wav2vec load that lazy caching triggers).
- See `config_sf_w9s1_lookahead_valtest.py` for the pattern.

## Testing patterns
- Mock-network tests: `MockNet` class with `__call__` recording kwargs, `clear_caches()`, `_kv_caches=None`, `chunk_size`, `total_num_frames`, `noise_scheduler` stub. No 14B model load. See `tests/test_sample_loop_toggles.py` for the canonical template.
- Spy on `causal_rope_apply` by monkey-patching the module attribute, recording `start_frame` and grid shape. See `tests/test_apply_window_rope.py`.
- Capture loguru output: `logger.add(io.StringIO(), format="{message}", level="INFO")` + `logger.remove(sink_id)` in finally. `pytest capsys` does NOT capture loguru.
- Test count as of 2026-04-13: 46 tests across 9 files in `tests/test_*.py`, all mock-based (no 14B model required), complete in <30s.

## Wandb group convention
All SF experiments → group `infinitetalk_sf`. Run name encodes the variant: `sf_w9s1_la<dist>_<anchor_tag>_<f2/f3_tags>_freq5_lr1e5_<timestamp>`.

## DF checkpoint sources (for SF init)
- Stochastic attention (`quarter_stoch_r128_bs4_accum1_8gpu_0409_2248/checkpoints/`): 9500 steps, trained with 5 sampled attention configs incl. `local_attn_size=10, sink=1` at 20%. Best init for w9s1 SF.
- Non-stochastic (`quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints/`): 4300 steps, single fixed attention config. Worse init for w9s1.

## Lookahead sink semantics (current experiment)
- Sink K rotated at RoPE position `F_window - 1 + lookahead_distance`; rest and Q keep natural positions (only sink moves).
- Trigger: `use_lookahead = lookahead_sink_enabled AND sink_tokens > 0 AND sink_tokens < k_win.shape[1] AND query_offset_in_win > 0`. The last conjunct excludes chunk 0 (where k_win is the current chunk only, no cached sink).
- Stochastic distance sampling: `self.training` gated. Eval/inference uses fixed distance for reproducibility. Env: `LOOKAHEAD_DISTANCE_MIN` / `LOOKAHEAD_DISTANCE_MAX`.
- F2 (model-sink-cache) overrides F3 (skip-clean-cache-pass) for chunk 0: F2 requires the separate cache pass to exist because the cached K/V comes from a forward call's INPUT, and F2 wants model-predicted frame 0 as that input.
- Debug traces gated by `LOOKAHEAD_DEBUG_TRACE=1`: `[lookahead]` (RoPE positions, permanent), `[sample_loop]` (per-chunk F2/F3 decisions, ephemeral — slated for removal once F2/F3 verified in production).

## Quick diagnostics
- Config override not taking effect? Check the shell script's `export VAR="${VAR:-...}"` — it overrides the config's `os.environ.get(..., default)`.
- SF init hanging after `[attn]` log lines? Likely the train dataloader is doing its NFS shape-check scan. See "Dataloader init is slow".
- Valtest hung at `Uploading GT validation videos`? Unknown root cause — rank-0 VAE decode + wandb upload can block for >15 min. Use the inference CLI (`inference_causal.py`) with `LOOKAHEAD_DEBUG_TRACE=1` for fast end-to-end logic verification instead.
