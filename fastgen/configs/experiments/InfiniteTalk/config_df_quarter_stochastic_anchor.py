# SPDX-License-Identifier: Apache-2.0
"""
Quarter-resolution DF config with stochastic attention + future anchor.

Extends config_df_quarter_stochastic with future_anchor=True on 50% of
attention configs. During training, configs with future_anchor inject a clean
GT latent from a random future distance (1-5 latent frames past the sequence)
as globally-visible anchor tokens in attention, teaching the model
distance-aware identity conditioning.

Requires: future_anchor_latents[_quarter].pt precomputed in each sample dir.
    python scripts/precompute_future_anchors.py --help

Usage:
    bash scripts/run_df_training_quarter_stochastic_anchor.sh
"""

import os
import time

from fastgen.configs.experiments.InfiniteTalk.config_df_quarter_stochastic import (
    create_config as create_stochastic_config,
)


def create_config():
    config = create_stochastic_config()

    # Override stochastic configs: 50% with future anchor, 50% without
    config.model.net.stochastic_attn_configs = [
        # Without future anchor (baseline configs)
        {"local_attn_size": 7,  "sink_size": 1, "weight": 0.1},
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.1},
        {"local_attn_size": 13, "sink_size": 1, "weight": 0.1},
        {"local_attn_size": 9,  "sink_size": 3, "weight": 0.1},
        {"local_attn_size": 12, "sink_size": 3, "weight": 0.1},
        # With future anchor (distance-aware conditioning)
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
        {"local_attn_size": 13, "sink_size": 1, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
        {"local_attn_size": 9,  "sink_size": 3, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
        {"local_attn_size": 12, "sink_size": 3, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 5]},
        {"local_attn_size": 10, "sink_size": 1, "weight": 0.1,
         "future_anchor": True, "future_anchor_distance_range": [1, 3]},
    ]

    # Logging
    config.log_config.group = "infinitetalk_df_quarter_stochastic_anchor"
    num_gpus = int(os.environ.get("NUM_GPUS", "8"))
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"quarter_stoch_anchor_r128_bs4_{num_gpus}gpu_{timestamp}"
    config.log_config.name = run_name

    return config
