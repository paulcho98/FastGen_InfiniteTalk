# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk Self-Forcing — w9s1 + Knot Forcing + Running-Ahead + Last-frame ref.

Paper: Xiao et al., "Knot Forcing: Taming Autoregressive Video Diffusion Models
for Real-time Infinite Interactive Portrait Animation", arXiv:2512.21734v2.

Baked feature configuration:
  - w9s1 attention   (sink_size=1, rolling=9, local_attn_size=10)
  - F1 lookahead     OFF (replaced by running-ahead — mutually exclusive)
  - F2 sink cache    OFF
  - F3 skip clean    OFF (KF uses separate cache pass for committed c frames only)
  - Temporal Knot    ON, k=1 (paper default)
  - Running-Ahead    ON, s=4, init_n=8
  - Last-frame ref   ON (dataloader uses vae_latents[:, :, -1] as reference)
  - Anchors          ON (student + fake_score + teacher, input + output)

Derived from config_sf_w9s1.py. Use dynamic_rope=True is required by running-ahead.

Optional env overrides:
  KNOT_SIZE=N                        knot length (default 1, paper uses 1)
  RUNNING_AHEAD_STEP=N               s value (default 4)
  RUNNING_AHEAD_INIT_N=N             initial n (default 8)

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

    # ---- Dynamic RoPE required for running-ahead ----
    config.model.net.use_dynamic_rope = True

    # ---- Disable F1/F2/F3 (incompatible with KF running-ahead) ----
    config.model.lookahead_sink_enabled = False
    config.model.lookahead_distance = 0
    config.model.lookahead_distance_min = 0
    config.model.lookahead_distance_max = 0
    config.model.model_sink_cache_enabled = False
    config.model.skip_clean_cache_pass = False

    # Propagate to net constructor too
    config.model.net.lookahead_sink_enabled = False
    config.model.net.lookahead_distance = 0
    config.model.net.lookahead_distance_min = 0
    config.model.net.lookahead_distance_max = 0

    # ---- Temporal Knot (paper Section 3.2) ----
    config.model.use_temporal_knot = True
    config.model.knot_size = int(os.environ.get("KNOT_SIZE", "1"))

    # ---- Running-Ahead (paper Section 3.3, Algorithm 1 lines 6-9) ----
    config.model.use_running_ahead = True
    config.model.running_ahead_step = int(os.environ.get("RUNNING_AHEAD_STEP", "4"))
    config.model.running_ahead_init_n = int(os.environ.get("RUNNING_AHEAD_INIT_N", "8"))

    # ---- Last-frame reference (paper Section 3.3) ----
    config.model.use_last_frame_reference = True
    config.dataloader_train.use_last_frame_reference = True
    # k=1 latent frame ≈ 4 pixel frames (VAE temporal stride ~3.86)
    config.dataloader_train.knot_size_extra_audio_pixel = 4 * config.model.knot_size

    # ---- Validate KF flag combinations ----
    config.model.validate_kf_flags()

    # ---- Logging: distinct group/name ----
    config.log_config.group = "infinitetalk_sf"
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        k = config.model.knot_size
        s = config.model.running_ahead_step
        n = config.model.running_ahead_init_n
        run_name = (
            f"sf_w9s1_knot_k{k}_ra_s{s}_n{n}_lastref_freq5_lr1e5_{timestamp}"
        )
    config.log_config.name = run_name

    return config
