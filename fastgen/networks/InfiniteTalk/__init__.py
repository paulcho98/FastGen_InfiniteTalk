# SPDX-License-Identifier: Apache-2.0
# InfiniteTalk network components for FastGen Self-Forcing adaptation.

from .audio_modules import AudioProjModel, SingleStreamAttention
from .wan_model import WanModel
from .network import InfiniteTalkWan

# Causal wrapper import — deferred to avoid ImportError when network_causal.py
# is not yet created.  The class is registered here once available.
try:
    from .network_causal import CausalInfiniteTalkWan
except ImportError:
    CausalInfiniteTalkWan = None

__all__ = [
    "AudioProjModel",
    "SingleStreamAttention",
    "WanModel",
    "InfiniteTalkWan",
    "CausalInfiniteTalkWan",
]
