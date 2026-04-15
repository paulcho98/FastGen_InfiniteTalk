# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bidirectional InfiniteTalk wrapper for FastGen.

Used as both the teacher (frozen 14B) and fake_score (trainable 14B with LoRA)
networks in the Self-Forcing distillation pipeline.  Wraps the standalone WanModel
DiT from ``fastgen.networks.InfiniteTalk.wan_model`` behind the ``FastGenNetwork``
abstract interface so that FastGen's training loop can call it uniformly.

Unlike OmniAvatar (14B teacher -> 1.3B student), InfiniteTalk uses the same 14B
architecture for all three roles.  LoRA adapters make training feasible:
  - Teacher:    base Wan I2V + InfiniteTalk ckpt merged, fully frozen
  - Student:    causal variant with LoRA adapters (separate file)
  - Fake score: bidirectional with LoRA adapters (this file, apply_lora_adapters=True)

Weight loading pipeline:
    1. Base Wan 2.1 I2V-14B safetensor shards (7 files) -> WanModel
       Keys already match our WanModel naming (blocks.N.self_attn.q.weight etc.)
    2. InfiniteTalk checkpoint on top (audio modules + any base weight overrides)
    3. (optional) Merge external LoRA from file into base weights
    4. (optional) Apply runtime LoRA adapters for trainable fake_score

I2V conditioning:
    InfiniteTalk uses a 20-channel y-tensor: 4 mask channels + 16 VAE-encoded
    reference frame channels.  The mask marks frame 0 as the reference (=1) and
    all subsequent frames as generation targets (=0), with VAE temporal stride
    handling via repeat_interleave.
"""

import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.networks.InfiniteTalk.wan_model import WanModel
from fastgen.networks.InfiniteTalk.lora import (
    apply_lora,
    freeze_base,
    merge_lora_from_file,
)
import fastgen.utils.logging_utils as logger


# ---------------------------------------------------------------------------
# 14B architecture defaults (Wan 2.1 I2V-14B-480P)
# ---------------------------------------------------------------------------
_14B_DEFAULTS = dict(
    dim=5120,
    ffn_dim=13824,
    freq_dim=256,
    num_heads=40,
    num_layers=40,
)

_COMMON_CFG = dict(
    model_type="i2v",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=36,   # 16 noise + 4 mask + 16 VAE ref (I2V concat happens before patch_embedding)
    out_dim=16,
    text_dim=4096,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
)

# Audio module defaults (InfiniteTalk single-speaker)
_AUDIO_DEFAULTS = dict(
    audio_window=5,
    intermediate_dim=512,
    output_dim=768,
    context_tokens=32,
    vae_scale=4,
    norm_input_visual=True,
    norm_output_audio=True,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# InfiniteTalkWan: FastGenNetwork wrapper
# ---------------------------------------------------------------------------

class InfiniteTalkWan(FastGenNetwork):
    """Bidirectional InfiniteTalk wrapper for teacher and fake_score.

    Wraps the standalone ``WanModel`` DiT behind FastGen's ``FastGenNetwork``
    interface.  Handles:
      - Model construction with 14B architecture defaults
      - Multi-stage weight loading (base Wan I2V + InfiniteTalk ckpt + LoRA)
      - I2V conditioning tensor assembly (20ch: 4 mask + 16 VAE ref)
      - Prediction-type conversion (flow -> x0)
      - Optional runtime LoRA adapters for fake_score training
    """

    def __init__(
        self,
        base_model_paths: str = "",
        infinitetalk_ckpt_path: str = "",
        lora_ckpt_path: str = "",
        lora_rank: int = 32,
        lora_alpha: int = 32,
        apply_lora_adapters: bool = False,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        shift: float = 7.0,
        disable_grad_ckpt: bool = False,
        # WanModel architecture params (defaults for 14B):
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        num_heads: int = 40,
        num_layers: int = 40,
        audio_window: int = 5,
        vae_scale: int = 4,
        **kwargs,
    ):
        """
        Args:
            base_model_paths: Comma-separated safetensor shard paths for the base
                Wan 2.1 I2V-14B weights (7 shards).  Keys in these files already
                match our WanModel naming convention (no conversion needed).
            infinitetalk_ckpt_path: Path to ``infinitetalk.safetensors`` containing
                audio modules and any base weight overrides from InfiniteTalk
                fine-tuning.  Loaded with ``strict=False`` on top of base weights.
            lora_ckpt_path: Path to an external LoRA checkpoint to merge into the
                base model weights (InfiniteTalk ``diffusion_model.*`` key format).
                Used for the teacher to absorb LoRA into frozen base.
            lora_rank: LoRA rank for both merge and runtime adapters.
            lora_alpha: LoRA alpha scaling for both merge and runtime adapters.
            apply_lora_adapters: If True, inject runtime ``LoRALinear`` adapters
                into the model and freeze base weights.  Used for the trainable
                fake_score network.
            net_pred_type: Network prediction type (``"flow"`` for rectified flow).
            schedule_type: Noise schedule type (``"rf"`` for rectified flow).
            shift: Timestep shift for the RF noise schedule.  InfiniteTalk uses 7.0
                for 480p generation.
            disable_grad_ckpt: If True, disable gradient checkpointing.
            dim: Hidden dimension of the transformer (5120 for 14B).
            ffn_dim: FFN intermediate dimension (13824 for 14B).
            freq_dim: Sinusoidal embedding dimension (256).
            num_heads: Number of attention heads (40 for 14B).
            num_layers: Number of transformer blocks (40 for 14B).
            audio_window: Audio context window size (5).
            vae_scale: VAE temporal compression ratio (4).
            **kwargs: Additional kwargs passed to ``FastGenNetwork.__init__``
                (forwarded to noise schedule constructor).
        """
        super().__init__(
            net_pred_type=net_pred_type,
            schedule_type=schedule_type,
            shift=shift,
            **kwargs,
        )

        self.base_model_paths = base_model_paths
        self.infinitetalk_ckpt_path = infinitetalk_ckpt_path
        self.lora_ckpt_path = lora_ckpt_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.apply_lora_adapters = apply_lora_adapters
        self.shift = shift

        self._use_gradient_checkpointing = not disable_grad_ckpt

        # Build the WanModel DiT
        self.model = WanModel(
            model_type=_COMMON_CFG["model_type"],
            patch_size=_COMMON_CFG["patch_size"],
            text_len=_COMMON_CFG["text_len"],
            in_dim=_COMMON_CFG["in_dim"],
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=_COMMON_CFG["text_dim"],
            out_dim=_COMMON_CFG["out_dim"],
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=_COMMON_CFG["window_size"],
            qk_norm=_COMMON_CFG["qk_norm"],
            cross_attn_norm=_COMMON_CFG["cross_attn_norm"],
            eps=_COMMON_CFG["eps"],
            audio_window=audio_window,
            intermediate_dim=_AUDIO_DEFAULTS["intermediate_dim"],
            output_dim=_AUDIO_DEFAULTS["output_dim"],
            context_tokens=_AUDIO_DEFAULTS["context_tokens"],
            vae_scale=vae_scale,
            norm_input_visual=_AUDIO_DEFAULTS["norm_input_visual"],
            norm_output_audio=_AUDIO_DEFAULTS["norm_output_audio"],
            use_gradient_checkpointing=self._use_gradient_checkpointing,
            weight_init=True,
        )

        # Load weights (unless we are in meta device context for FSDP)
        if not self._is_in_meta_context():
            self._load_weights()
            self.model.to(torch.bfloat16)
        else:
            # Meta-init (non-rank-0): skip weight loading but still apply LoRA
            # adapters so model structure matches rank 0 for FSDP sync.
            if self.apply_lora_adapters:
                apply_lora(self.model, rank=self.lora_rank, alpha=self.lora_alpha)
                freeze_base(self.model)

    # ------------------------------------------------------------------
    # FSDP2 sharding
    # ------------------------------------------------------------------

    def fully_shard(self, **kwargs):
        """Apply FSDP2 sharding to the internal WanModel.

        Shards each transformer block individually, then wraps self.model as
        the FSDP root. The root wrapper is critical: without it, child FSDP
        units all-gather params during forward but never reshard (no parent
        post-forward hook to trigger it), causing ~40 GB memory leak per model.
        """
        from torch.distributed._composable.fsdp import fully_shard

        # Shard each transformer block independently
        for block in self.model.blocks:
            fully_shard(block, **kwargs)

        # Shard root (self.model) — this triggers resharding of all children
        # after forward completes. self.model is a plain nn.Module (WanModel),
        # so __class__ assignment works fine.
        fully_shard(self.model, **kwargs)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        """Multi-stage weight loading pipeline.

        Stage 1: Load base Wan I2V-14B safetensor shards.
                 The shard keys (blocks.N.self_attn.q.weight, etc.) match our
                 WanModel directly -- no key conversion needed.

        Stage 2: Load InfiniteTalk checkpoint (audio modules + weight overrides).
                 Uses ``strict=False`` so audio-only keys don't fail.

        Stage 3: Merge external LoRA weights into base model (for teacher).
                 Uses ``merge_lora_from_file()`` which handles InfiniteTalk's
                 ``diffusion_model.`` prefix and lora_down/lora_up naming.

        Stage 4: Apply runtime LoRA adapters (for trainable fake_score).
                 Injects ``LoRALinear`` wrappers and freezes base weights.
        """
        # --- Stage 1: Base Wan I2V safetensor shards ---
        if self.base_model_paths:
            paths = [p.strip() for p in self.base_model_paths.split(",") if p.strip()]
            logger.info(
                f"[InfiniteTalkWan] Loading base Wan I2V from {len(paths)} shard(s)"
            )

            # Load and merge all shards into a single state dict
            base_sd: Dict[str, torch.Tensor] = {}
            for p in paths:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Base model shard not found: {p}")
                shard = safetensors_load_file(p, device="cpu")
                base_sd.update(shard)
                logger.info(
                    f"[InfiniteTalkWan]   Loaded shard: {os.path.basename(p)} "
                    f"({len(shard)} tensors)"
                )

            logger.info(
                f"[InfiniteTalkWan] Total base tensors: {len(base_sd)}"
            )

            # Load into model -- strict=False because audio modules are not
            # present in the base I2V checkpoint
            missing, unexpected = self.model.load_state_dict(base_sd, strict=False)
            loaded = len(base_sd) - len(unexpected)
            logger.info(
                f"[InfiniteTalkWan] Base weights: {loaded} loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

            # Categorize missing keys for diagnostic clarity
            if missing:
                audio_missing = [k for k in missing if "audio" in k.lower()]
                other_missing = [k for k in missing if "audio" not in k.lower()]
                if audio_missing:
                    logger.info(
                        f"[InfiniteTalkWan] Expected missing (audio): "
                        f"{len(audio_missing)} keys"
                    )
                if other_missing:
                    logger.warning(
                        f"[InfiniteTalkWan] Unexpected missing keys: "
                        f"{other_missing[:10]}{'...' if len(other_missing) > 10 else ''}"
                    )
            if unexpected:
                logger.warning(
                    f"[InfiniteTalkWan] Unexpected keys in base shards: "
                    f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
                )

            del base_sd  # free memory
        else:
            logger.info(
                "[InfiniteTalkWan] No base_model_paths provided, using random init"
            )

        # --- Stage 2: InfiniteTalk checkpoint ---
        if self.infinitetalk_ckpt_path:
            if not os.path.isfile(self.infinitetalk_ckpt_path):
                raise FileNotFoundError(
                    f"InfiniteTalk checkpoint not found: {self.infinitetalk_ckpt_path}"
                )

            logger.info(
                f"[InfiniteTalkWan] Loading InfiniteTalk checkpoint: "
                f"{self.infinitetalk_ckpt_path}"
            )
            it_sd = safetensors_load_file(self.infinitetalk_ckpt_path, device="cpu")
            logger.info(
                f"[InfiniteTalkWan] InfiniteTalk checkpoint: {len(it_sd)} tensors"
            )

            # Load on top of base weights -- overrides audio modules and any
            # fine-tuned base parameters
            missing, unexpected = self.model.load_state_dict(it_sd, strict=False)
            loaded = len(it_sd) - len(unexpected)
            logger.info(
                f"[InfiniteTalkWan] InfiniteTalk weights: {loaded} loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )

            del it_sd  # free memory
        else:
            logger.info(
                "[InfiniteTalkWan] No infinitetalk_ckpt_path provided, "
                "audio modules use random init"
            )

        # --- Stage 3: Merge external LoRA into base weights ---
        if self.lora_ckpt_path:
            if not os.path.isfile(self.lora_ckpt_path):
                raise FileNotFoundError(
                    f"LoRA checkpoint not found: {self.lora_ckpt_path}"
                )

            logger.info(
                f"[InfiniteTalkWan] Merging LoRA from: {self.lora_ckpt_path}"
            )
            lora_sd = safetensors_load_file(self.lora_ckpt_path, device="cpu")
            applied = merge_lora_from_file(
                self.model,
                lora_sd,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                prefix="diffusion_model.",
            )
            logger.info(
                f"[InfiniteTalkWan] LoRA merge complete: {applied} updates applied"
            )
            del lora_sd

        # --- Stage 4: Runtime LoRA adapters (for fake_score) ---
        if self.apply_lora_adapters:
            logger.info(
                f"[InfiniteTalkWan] Applying runtime LoRA adapters "
                f"(rank={self.lora_rank}, alpha={self.lora_alpha})"
            )
            apply_lora(self.model, rank=self.lora_rank, alpha=self.lora_alpha)
            freeze_base(self.model)

        # Reinitialize RoPE frequencies (not stored in checkpoints)
        self.model.init_freqs()

    # ------------------------------------------------------------------
    # I2V conditioning
    # ------------------------------------------------------------------

    @staticmethod
    def _construct_i2v_mask(
        frame_num: int,
        lat_h: int,
        lat_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the 4-channel temporal mask for I2V conditioning.

        The mask marks frame 0 as the reference (value=1) and all subsequent
        frames as generation targets (value=0).  The first pixel frame is
        replicated 4 times to account for the VAE's temporal stride of 4
        (``repeat_interleave``), then the tensor is reshaped into latent-space
        dimensions ``[1, 4, T_lat, lat_h, lat_w]``.

        This exactly reproduces InfiniteTalk's mask construction from
        ``wan/multitalk.py`` lines 551-559.

        Args:
            frame_num: Number of pixel-space frames (e.g. 81).
            lat_h: Latent spatial height.
            lat_w: Latent spatial width.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Mask tensor of shape ``[1, 4, T_lat, lat_h, lat_w]``.
        """
        # Build per-frame spatial mask: first frame = 1, rest = 0
        msk = torch.ones(1, frame_num, lat_h, lat_w, device=device)
        msk[:, 1:] = 0

        # VAE temporal stride handling: replicate frame 0 four times,
        # then concat with remaining frames
        msk = torch.cat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
            msk[:, 1:]
        ], dim=1)

        # Reshape: [1, T_pixel+3, lat_h, lat_w] -> [1, 4, T_lat, lat_h, lat_w]
        # T_pixel+3 = frame_num + 3 = (T_lat - 1)*4 + 1 + 3 = T_lat * 4
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2).to(dtype)  # [1, 4, T_lat, lat_h, lat_w]

        return msk

    def _build_y(
        self,
        condition: Dict[str, torch.Tensor],
        T: int,
    ) -> torch.Tensor:
        """Build the 20-channel I2V conditioning tensor.

        Concatenates the 4-channel temporal mask with the 16-channel
        VAE-encoded reference frame to produce ``y = [msk, first_frame_cond]``
        of shape ``[B, 20, T, H, W]``.

        Args:
            condition: Dict containing ``"first_frame_cond"`` with shape
                ``[B, 16, T, H, W]`` (VAE-encoded reference frame, zero-padded
                after frame 0).
            T: Number of latent time steps (e.g. 21).

        Returns:
            y tensor of shape ``[B, 20, T, H, W]``.
        """
        first_frame_cond = condition["first_frame_cond"]  # [B, 16, T, H, W]
        B, _, _, H, W = first_frame_cond.shape

        # Pixel frame count from latent frames: T_lat=21 -> 81 pixel frames
        frame_num = (T - 1) * 4 + 1

        msk = self._construct_i2v_mask(
            frame_num, H, W,
            device=first_frame_cond.device,
            dtype=first_frame_cond.dtype,
        )
        msk = msk.expand(B, -1, -1, -1, -1)  # [B, 4, T, H, W]

        # Concatenate: [B, 4, T, H, W] + [B, 16, T, H, W] -> [B, 20, T, H, W]
        y = torch.cat([msk, first_frame_cond], dim=1)
        return y

    # ------------------------------------------------------------------
    # Forward pass (FastGenNetwork interface)
    # ------------------------------------------------------------------

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
        """Forward pass of the InfiniteTalk diffusion score model.

        Converts from FastGen's batched tensor interface to WanModel's
        list-of-unbatched-tensors interface, runs the DiT, and converts the
        output prediction type as needed.

        Args:
            x_t: Noisy latent ``[B, 16, T, H, W]``.
            t: Timestep in ``[0, 1)`` range, shape ``[B]``.
            condition: Dict with keys:
                - ``text_embeds``:      ``[B, 1, 512, 4096]`` or ``[B, 512, 4096]``
                - ``first_frame_cond``: ``[B, 16, T, H, W]`` (VAE-encoded ref frame)
                - ``clip_features``:    ``[B, 257, 1280]``
                - ``audio_emb``:        ``[B, num_audio_frames, audio_window, 12, 768]``
            r: Reserved for mean-flow models (unused here).
            return_features_early: If True, return intermediate features and exit.
            feature_indices: Set of block indices to extract features from.
            return_logvar: If True, return a dummy logvar alongside the output.
            fwd_pred_type: Override prediction type for output conversion.
            apply_input_anchor: If True (default), pin the model INPUT
                x_t[:, :, 0:1] = first_frame_cond[:, :, 0:1] at the top of
                forward() when network anchor mode allows. Matches
                InfiniteTalk's training distribution (clean frame 0 at every
                timestep).
            **fwd_kwargs: Additional kwargs (``use_gradient_checkpointing``, etc.).

        Returns:
            Model output tensor, or features list, depending on flags.
        """
        if feature_indices is None:
            feature_indices = set()

        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            if fwd_pred_type not in NET_PRED_TYPES:
                raise ValueError(
                    f"Unsupported fwd_pred_type '{fwd_pred_type}'. "
                    f"Supported: {NET_PRED_TYPES}"
                )

        # --- Unpack condition dict ---
        assert isinstance(condition, dict), (
            f"condition must be a dict, got {type(condition)}"
        )

        text_embeds = condition["text_embeds"]          # [B, 1, 512, 4096] or [B, 512, 4096]
        clip_features = condition["clip_features"]      # [B, 257, 1280] or [B, 1, 257, 1280]
        audio_emb = condition["audio_emb"]              # [B, num_frames, 12, 768] or [B, num_frames, 5, 12, 768]

        # Handle extra dims from dataset collation
        if text_embeds.dim() == 4:
            text_embeds = text_embeds.squeeze(1)
        if clip_features.dim() == 4:
            clip_features = clip_features.squeeze(1)

        # Apply 5-frame sliding window to audio if not already windowed
        if audio_emb is not None and audio_emb.dim() == 4:
            num_frames = audio_emb.shape[1]
            half_win = 5 // 2  # audio_window=5
            indices = torch.arange(5, device=audio_emb.device) - half_win
            center_indices = torch.arange(num_frames, device=audio_emb.device).unsqueeze(1) + indices.unsqueeze(0)
            center_indices = center_indices.clamp(0, num_frames - 1)
            audio_emb = audio_emb[:, center_indices]  # [B, num_frames, 5, 12, 768]

        B, C, T, H, W = x_t.shape

        # Input-side frame-0 anchor (bidirectional): pin x_t[:, :, 0:1] to
        # clean reference before building y and running the model. Matches
        # InfiniteTalk training distribution for teacher/fake.
        x_t = _maybe_apply_input_anchor_bidir(
            x_t, self, condition, apply_input_anchor=apply_input_anchor,
        )

        # --- Build I2V conditioning y-tensor ---
        y = self._build_y(condition, T)  # [B, 20, T, H, W]

        # --- Rescale timestep ---
        # FastGen uses t in [0, 1), WanModel expects t * 1000
        t_rescaled = self.noise_scheduler.rescale_t(t)

        # --- Prepare inputs for WanModel ---
        # WanModel expects lists of unbatched tensors for x, y, and context

        # x: list of [C_in, T, H, W] per sample
        x_list = [x_t[i] for i in range(B)]

        # y: list of [C_cond, T, H, W] per sample
        y_list = [y[i] for i in range(B)]

        # context (text embeddings): list of [seq_len, text_dim] per sample
        # Handle both [B, 1, 512, 4096] and [B, 512, 4096] shapes
        if text_embeds.dim() == 4:
            # [B, 1, 512, 4096] -> squeeze the extra dim
            text_embeds = text_embeds.squeeze(1)  # [B, 512, 4096]
        context_list = [text_embeds[i] for i in range(B)]  # list of [512, 4096]

        # Compute max sequence length for padding
        # seq_len = T_patches * H_patches * W_patches
        patch_t, patch_h, patch_w = self.model.patch_size
        # Input to patch_embedding is in_dim + cond_channels = 16 + 20 = 36 channels
        # but seq_len is determined by spatial/temporal dims after patching
        n_t = T // patch_t
        n_h = H // patch_h
        n_w = W // patch_w
        seq_len = n_t * n_h * n_w

        # --- Forward through WanModel ---
        use_gradient_checkpointing = fwd_kwargs.get(
            "use_gradient_checkpointing", self._use_gradient_checkpointing
        )
        has_features = feature_indices is not None and len(feature_indices) > 0

        model_output = self.model(
            x=x_list,
            t=t_rescaled,
            context=context_list,
            seq_len=seq_len,
            clip_fea=clip_features,
            y=y_list,
            audio=audio_emb,
            use_gradient_checkpointing=use_gradient_checkpointing,
            feature_indices=feature_indices if has_features else None,
            return_features_early=return_features_early,
        )

        # --- Handle early exit for feature extraction ---
        if return_features_early and has_features:
            return model_output  # List of [B, dim, T, H, W] feature tensors

        # --- Unpack if model returned (output, features) tuple ---
        features = None
        if has_features and isinstance(model_output, tuple):
            model_output, features = model_output

        # model_output is a stacked tensor [B, C_out, T, H, W] from unpatchify

        # --- Convert prediction type ---
        out = self.noise_scheduler.convert_model_output(
            x_t, model_output, t,
            src_pred_type=self.net_pred_type,
            target_pred_type=fwd_pred_type,
        )

        # Hard-anchor frame 0 to clean reference.
        # By default, always anchors (teacher behavior).
        # When _anchor_eval_only=True, only anchors in eval mode — used for
        # fake_score in soft-anchor training so it tracks the student distribution
        # (which also doesn't anchor during training).
        anchor_active = getattr(self, "_enable_first_frame_anchor", True)
        if anchor_active and getattr(self, "_anchor_eval_only", False):
            anchor_active = not self.training
        if anchor_active and isinstance(condition, dict) and "first_frame_cond" in condition:
            first_frame_cond = condition["first_frame_cond"]
            out = out.clone()
            out[:, :, 0:1] = first_frame_cond[:, :, 0:1]

        # --- Return format depends on what was requested ---
        if features is not None:
            if return_logvar:
                logvar = torch.zeros(
                    out.shape[0], 1, device=out.device, dtype=out.dtype
                )
                return [out, features], logvar
            return [out, features]

        if return_logvar:
            logvar = torch.zeros(
                out.shape[0], 1, device=out.device, dtype=out.dtype
            )
            return out, logvar

        return out
