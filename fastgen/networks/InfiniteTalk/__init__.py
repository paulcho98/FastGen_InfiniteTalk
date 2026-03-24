# SPDX-License-Identifier: Apache-2.0
# InfiniteTalk network components for FastGen Self-Forcing adaptation.

from .audio_modules import AudioProjModel, SingleStreamAttention
from .wan_model import WanModel
from .network import InfiniteTalkWan

__all__ = [
    "AudioProjModel",
    "SingleStreamAttention",
    "WanModel",
    "InfiniteTalkWan",
]
