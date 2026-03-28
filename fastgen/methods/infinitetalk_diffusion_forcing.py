# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InfiniteTalk Diffusion Forcing model for Stage 1 initialization.

Alternative to ODE-based KD (InfiniteTalkKDModel). Instead of pre-computing
ODE trajectories from the teacher, this adds Gaussian noise to real data at
inhomogeneous block-wise timesteps and trains the student to denoise with L2 loss.
No teacher model or ODE generation needed.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, TYPE_CHECKING, Callable
from functools import partial

import torch
import torch.nn.functional as F

from fastgen.methods.knowledge_distillation.KD import KDModel
from fastgen.methods.distribution_matching.causvid import CausVidModel
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.config import BaseModelConfig as ModelConfig


class InfiniteTalkDiffusionForcingModel(KDModel):
    """Diffusion Forcing on real data -- alternative to ODE KD for Stage 1.

    Adds noise to real data at inhomogeneous block-wise timesteps.
    Student denoises -> L2 loss vs clean data. No teacher ODE needed.

    Inheritance: InfiniteTalkDiffusionForcingModel -> KDModel -> FastGenModel
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def build_model(self):
        """Build model and re-apply LoRA freeze.

        VAE is NOT loaded here — loading it manipulates sys.modules (xformers mock)
        which poisons torch.compile/inductor for FlexAttention. Instead, VAE is
        loaded lazily on the first _get_outputs call that needs it (after torch.compile
        is done).
        """
        super().build_model()
        from fastgen.networks.InfiniteTalk.lora import freeze_base
        freeze_base(self.net)
        self._vae_load_attempted = False
        logger.info(f"[build_model] config type={type(self.config).__name__}, "
                    f"has vae_path={hasattr(self.config, 'vae_path')}, "
                    f"vae_path={getattr(self.config, 'vae_path', 'MISSING')}")

    def _load_vae(self, vae_path: str):
        """Load WanVAE for decoding generated samples in wandb visual logging."""
        import types
        import importlib.machinery

        # Mock xformers (wan/__init__.py pulls it in transitively)
        class _MockModule(types.ModuleType):
            def __getattr__(self, name):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                return lambda *a, **k: None

        for p in ["xformers", "xformers.ops", "xformers.ops.fmha",
                   "xformers.ops.fmha.attn_bias"]:
            parts = p.split(".")
            for i in range(len(parts)):
                partial_name = ".".join(parts[:i + 1])
                if partial_name not in sys.modules:
                    m = _MockModule(partial_name)
                    if i < len(parts) - 1:
                        m.__path__ = []
                    m.__spec__ = importlib.machinery.ModuleSpec(partial_name, None)
                    sys.modules[partial_name] = m

        from flash_attn import flash_attn_func
        sys.modules["xformers"].ops = sys.modules["xformers.ops"]
        sys.modules["xformers.ops"].memory_efficient_attention = (
            lambda q, k, v, attn_bias=None, op=None: flash_attn_func(q, k, v)
        )

        # Mock other deps pulled by wan/__init__.py
        import inspect
        if not hasattr(inspect, 'ArgSpec'):
            inspect.ArgSpec = inspect.FullArgSpec
        for p in ["decord", "src.vram_management"]:
            if p not in sys.modules:
                m = _MockModule(p)
                m.__spec__ = importlib.machinery.ModuleSpec(p, None)
                sys.modules[p] = m
        sys.modules["decord"].cpu = lambda n=0: None
        sys.modules["decord"].VideoReader = None

        # Add InfiniteTalk root for wan.modules.vae
        it_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../InfiniteTalk"))
        if it_root not in sys.path:
            sys.path.insert(0, it_root)

        from wan.modules.vae import WanVAE
        device_str = f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
        self.net.vae = WanVAE(vae_pth=vae_path, device=device_str)
        logger.info(f"Loaded WanVAE from {vae_path} for visual logging")

    # Use CausVidModel's AR sample loop for visualization (chunk-by-chunk with KV cache).
    # Without this, FastGenModel._student_sample_loop processes the entire video as one
    # bidirectional pass, which doesn't reflect actual AR inference behavior.
    _student_sample_loop = CausVidModel._student_sample_loop

    def _build_condition(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build InfiniteTalk condition dict from data batch.

        Expected shapes (after collation, with batch dim):
            text_embeds:      [B, 1, 512, 4096] or [B, 512, 4096]
            first_frame_cond: [B, 16, T, H, W]
            clip_features:    [B, 1, 257, 1280] or [B, 257, 1280]
            audio_emb:        [B, 81, 12, 768]
        """
        for key in ("text_embeds", "first_frame_cond", "clip_features", "audio_emb"):
            assert key in data, f"Missing required key '{key}' in data batch"
        assert data["first_frame_cond"].shape[1] == 16, (
            f"first_frame_cond should have 16 channels (VAE only, mask added at runtime), "
            f"got {data['first_frame_cond'].shape[1]}"
        )
        assert data["audio_emb"].shape[-2:] == (12, 768), (
            f"audio_emb should have shape [..., 12, 768], got {list(data['audio_emb'].shape)}"
        )
        return {
            "text_embeds": data["text_embeds"],
            "first_frame_cond": data["first_frame_cond"],
            "clip_features": data["clip_features"],
            "audio_emb": data["audio_emb"],
        }

    def _ensure_vae_loaded(self):
        """Lazily load VAE on first call. Must happen AFTER torch.compile finishes.

        VAE path comes from INFINITETALK_VAE_PATH env var (not config.model.vae_path,
        which gets lost during attrs serialization in DDP).
        """
        if self._vae_load_attempted:
            return hasattr(self.net, 'vae')
        self._vae_load_attempted = True
        # Try env var first (reliable across DDP), then config attr (fallback)
        vae_path = os.environ.get("INFINITETALK_VAE_PATH") or getattr(self.config, "vae_path", None)
        if vae_path and os.path.exists(vae_path):
            try:
                self._load_vae(vae_path)
                return True
            except Exception as e:
                logger.warning(f"Failed to load VAE from {vae_path}: {e}")
                import traceback
                traceback.print_exc()
                return False
        logger.info(f"No VAE path found (env={os.environ.get('INFINITETALK_VAE_PATH')}, "
                    f"config={getattr(self.config, 'vae_path', None)}) — visual logging disabled")
        return False

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        # Lazily load VAE (deferred from build_model to avoid poisoning torch.compile)
        has_vae = self._ensure_vae_loaded()
        if has_vae and condition is not None:
            noise = torch.randn_like(gen_data, dtype=self.precision)

            # Wrap the AR generation + VAE decode in a single callable.
            # The base WandbCallback only decodes if model.net has init_preprocessors
            # (diffusers-specific), which our standalone port doesn't have.
            # So we decode here and return pixel-space video directly.
            def _generate_and_decode():
                with torch.no_grad():
                    latent = self.generator_fn(
                        net=self.net_inference,
                        noise=noise,
                        condition=condition,
                        student_sample_steps=self.config.student_sample_steps,
                        t_list=self.config.sample_t_cfg.t_list,
                        precision_amp=self.precision_amp_infer,
                    )
                    # Decode latent → pixel video
                    video = self.net.vae.decode(latent[:1].float())
                    # WanVAE.decode may return a list; ensure tensor [B, C, T, H, W]
                    if isinstance(video, (list, tuple)):
                        video = torch.stack(video) if len(video) > 1 else video[0].unsqueeze(0)
                    return video

            return {"gen_rand": _generate_and_decode, "input_rand": noise, "gen_rand_train": gen_data}
        return {"gen_rand_train": gen_data}

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | Callable]]:
        """Single training step using diffusion forcing on real data.

        Instead of gathering from pre-computed ODE trajectories (as in CausalKDModel),
        this adds Gaussian noise to real data at inhomogeneous block-wise timesteps.
        """
        real_data = data["real"]  # [B, 16, 21, H, W]
        condition = self._build_condition(data)

        batch_size, num_frames = real_data.shape[0], real_data.shape[2]
        chunk_size = self.net.chunk_size

        # Sample inhomogeneous block-wise timesteps
        t_inhom, _ = self.net.noise_scheduler.sample_t_inhom(
            batch_size,
            num_frames,
            chunk_size,
            sample_steps=self.config.student_sample_steps,
            t_list=self.config.sample_t_cfg.t_list,
            device=self.device,
            dtype=real_data.dtype,
        )  # [B, T]

        # Diffusion forcing: add noise to real data at sampled timesteps
        eps = torch.randn_like(real_data)
        t_inhom_expanded = t_inhom[:, None, :, None, None]  # [B, 1, T, 1, 1]
        noisy_data = self.net.noise_scheduler.forward_process(real_data, eps, t_inhom_expanded)

        # Student denoise
        gen_data = self.gen_data_from_net(noisy_data, t_inhom, condition=condition)

        # L2 loss — real_data is the target (no grad needed)
        loss = 0.5 * F.mse_loss(gen_data, real_data, reduction="mean")

        # Outputs for logging (detached to avoid holding autograd references)
        outputs = self._get_outputs(gen_data.detach(), condition=condition)

        loss_map = {"total_loss": loss, "recon_loss": loss.detach()}
        return loss_map, outputs
