# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
InfiniteTalk Self-Forcing model for audio-driven talking-face distillation.

Overrides _prepare_training_data to build InfiniteTalk-specific condition dicts
with text, first-frame conditioning, CLIP features, and audio embeddings.

Also overrides _apply_classifier_free_guidance to implement 3-call CFG with
separate text and audio guidance scales, matching InfiniteTalk's original
inference pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

from fastgen.methods.distribution_matching.self_forcing import SelfForcingModel
from fastgen.methods.distribution_matching.causvid import CausVidModel
from fastgen.utils import instantiate
from fastgen.utils.distributed import synchronize, is_rank0
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.configs.methods.config_infinitetalk_sf import InfiniteTalkSFModelConfig as ModelConfig


class InfiniteTalkSelfForcingModel(SelfForcingModel):
    """Self-Forcing distillation for InfiniteTalk audio-driven talking-face generation.

    Inherits the full Self-Forcing training loop (rollout_with_gradient, VSD loss,
    fake_score/discriminator updates). Only overrides:
      - _prepare_training_data: maps dataset output to (real_data, condition, neg_condition)
      - _apply_classifier_free_guidance: 3-call CFG with separate text and audio scales
      - _setup_grad_requirements: LoRA-safe parameter toggling for fake_score
      - _get_outputs: lazy VAE decode for visual logging
      - build_model: re-freeze LoRA base after parent initialization

    3-call CFG formula (matching InfiniteTalk's original inference):
        output = uncond + text_scale * (cond - drop_text) + audio_scale * (drop_text - uncond)
    where:
        cond = teacher(x_t, t, condition)                    # full conditioning
        drop_text = teacher(x_t, t, neg_text_condition)      # drop text, keep audio
        uncond = teacher(x_t, t, neg_condition)              # drop both text and audio
    """

    # Use CausVid's AR sample loop for visualization (chunk-by-chunk with KV cache)
    # instead of base FastGenModel's bidirectional loop.
    # Both DF and KD have this same override.
    _student_sample_loop = CausVidModel._student_sample_loop

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._vae_load_attempted = False

    def build_model(self):
        """Build model with sequential FSDP sharding to avoid CPU RAM OOM.

        3x 14B models × 8 ranks = ~900 GB CPU RAM if built simultaneously.
        Instead, build each network → FSDP-shard to GPU → free CPU → build next.
        Peak CPU RAM: ~1 model × 8 ranks ≈ 310 GB (fits on any multi-GPU node).

        This flag is controlled by config.model.fsdp_sequential_init. When False,
        falls back to the standard build-all-then-shard pattern.
        """
        import gc
        from fastgen.methods.model import FastGenModel
        from fastgen.networks.InfiniteTalk.lora import freeze_base, has_lora

        sequential = getattr(self.config, "fsdp_sequential_init", False)
        logger.info(f"[build_model] fsdp_sequential_init={sequential}, config type={type(self.config).__name__}")

        if not sequential:
            # Standard path: build all, let trainer FSDP-wrap later
            super(InfiniteTalkSelfForcingModel, self).build_model()
            # (this calls DMD2Model.build_model → FastGenModel.build_model + build_teacher + fake_score)
            freeze_base(self.net)
            if hasattr(self, 'fake_score') and self.fake_score is not None:
                if has_lora(self.fake_score):
                    freeze_base(self.fake_score)
            self._apply_anchor_config()
            self._apply_running_ahead_config()  # KF: stamp running-ahead attrs on student
            return

        # === Sequential build + FSDP shard ===
        from fastgen.utils.distributed.fsdp import fsdp_shard_single_network

        fsdp_kwargs = dict(
            precision=self.precision,
            precision_fsdp=self.precision_fsdp,
            apply_cpu_offload=False,
            sharding_group_size=None,
        )

        # Step 1: Build student
        FastGenModel.build_model(self)
        freeze_base(self.net)

        # Load pretrained DF checkpoint into student BEFORE FSDP sharding
        # (loading into FSDP DTensors requires different API)
        pretrained_path = self.config.pretrained_ckpt_path if hasattr(self.config, "pretrained_ckpt_path") else ""
        # The trainer normally handles this, but with sequential init we do it here
        # The path comes from trainer config, not model config — check env var
        import os
        pretrained_path = os.environ.get("INFINITETALK_DF_CKPT", "")
        if pretrained_path and os.path.exists(pretrained_path):
            logger.info(f"[sequential_init] Loading pretrained DF checkpoint into student: {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            if "model" in ckpt and "net" in ckpt["model"]:
                load_info = self.net.load_state_dict(ckpt["model"]["net"], strict=False)
                logger.info(f"[sequential_init] Loaded DF checkpoint: {load_info}")
                self._pretrained_iter = ckpt.get("iteration", 0)
            del ckpt
            gc.collect()

        gc.collect()
        logger.info(f"[sequential_init] Student built. CPU RAM free: {self._get_free_ram_gb():.1f} GB")

        # FSDP-shard student to GPU, freeing CPU copy
        fsdp_shard_single_network(self.net, "net", **fsdp_kwargs)
        logger.info(f"[sequential_init] Student sharded to GPU. CPU RAM free: {self._get_free_ram_gb():.1f} GB")

        # Step 2: Build teacher
        self.build_teacher()
        gc.collect()
        logger.info(f"[sequential_init] Teacher built. CPU RAM free: {self._get_free_ram_gb():.1f} GB")

        # FSDP-shard teacher to GPU
        fsdp_shard_single_network(self.teacher, "teacher", **fsdp_kwargs)
        logger.info(f"[sequential_init] Teacher sharded to GPU. CPU RAM free: {self._get_free_ram_gb():.1f} GB")

        # Step 3: Load student weights + EMA (student already on GPU as FSDP DTensors)
        # Skip load_student_weights_and_ema — pretrained ckpt is loaded by the trainer
        # after build_model. EMA is not used in SF training.
        self.load_student_weights_and_ema()

        # Step 4: Build fake_score
        logger.info("Instantiating the fake_score")
        fake_score_cfg = getattr(self.config, "fake_score_net", None)
        if fake_score_cfg is not None:
            logger.info("Instantiating fake_score from config.fake_score_net")
            self.fake_score = instantiate(fake_score_cfg)
        else:
            self.fake_score = instantiate(self.teacher_config)
            model_path = self.config.pretrained_model_path
            if model_path is not None and len(model_path) > 0:
                self.fake_score.load_state_dict(self.teacher.state_dict())
        synchronize()
        if has_lora(self.fake_score):
            freeze_base(self.fake_score)
        gc.collect()
        logger.info(f"[sequential_init] Fake score built. CPU RAM free: {self._get_free_ram_gb():.1f} GB")

        # FSDP-shard fake_score to GPU
        fsdp_shard_single_network(self.fake_score, "fake_score", **fsdp_kwargs)
        logger.info(f"[sequential_init] Fake score sharded to GPU. CPU RAM free: {self._get_free_ram_gb():.1f} GB")

        if self.config.gan_loss_weight_gen > 0:
            self.discriminator = instantiate(self.config.discriminator)
            synchronize()
        torch.cuda.empty_cache()

        # Mark that FSDP wrapping is already done (trainer should skip its own)
        self._fsdp_already_wrapped = True
        self._apply_anchor_config()
        self._apply_running_ahead_config()  # KF: stamp running-ahead attrs on student

    def _apply_anchor_config(self):
        """Apply first-frame anchor configuration to student, teacher, and fake_score.

        Three independent flags (each defaults to the I2V-always behavior):

        config.student_anchor_eval_only (default False):
          - False: student anchors frame 0 always (current I2V behavior)
          - True: student anchors only in eval mode (inference/validation),
                  no anchor during training → student learns frame 0 via VSD gradient.

        config.fake_score_anchor_eval_only (default False):
          - False: fake_score always anchors (original behavior)
          - True: fake_score anchors only in eval mode — matches student's soft
                  conditioning so it tracks the student's actual distribution.

        config.teacher_anchor_disabled (default False):
          - False: teacher always anchors (original I2V target distribution)
          - True: teacher's _enable_first_frame_anchor is permanently set to False.
                  Teacher never anchors — used to make the VSD target distribution
                  consistent with an anchor-free training rollout.
                  (Teacher is always in eval mode, so _anchor_eval_only would
                  paradoxically leave it anchoring; we use hard-disable instead.)

        Typical "train-time anchor-free, student-anchored eval" mode:
            student_anchor_eval_only = True
            fake_score_anchor_eval_only = True
            teacher_anchor_disabled = True
        """
        # Student anchor mode
        student_eval_only = getattr(self.config, "student_anchor_eval_only", False)
        if student_eval_only:
            self.net._anchor_eval_only = True
            logger.info("[anchor] Student: eval-only (no anchor during training rollout)")
        else:
            self.net._anchor_eval_only = False
            logger.info("[anchor] Student: always (anchor during training + inference)")

        # Fake_score anchor mode
        fake_eval_only = getattr(self.config, "fake_score_anchor_eval_only", False)
        if fake_eval_only and hasattr(self, "fake_score") and self.fake_score is not None:
            self.fake_score._anchor_eval_only = True
            logger.info("[anchor] Fake score: eval-only (tracks student distribution)")
        else:
            logger.info("[anchor] Fake score: always (anchors during training)")

        # Teacher anchor mode — hard-disable path (eval_only would leave it anchoring
        # because teacher.training is False)
        teacher_disabled = getattr(self.config, "teacher_anchor_disabled", False)
        if teacher_disabled and hasattr(self, "teacher") and self.teacher is not None:
            self.teacher._enable_first_frame_anchor = False
            logger.info("[anchor] Teacher: DISABLED (anchor-free target distribution)")
        else:
            logger.info("[anchor] Teacher: always (anchors during training rollout)")

        # --- F1/F2/F3 toggles stamped onto self.net for the sample loops to read ---
        lookahead_enabled = getattr(self.config, "lookahead_sink_enabled", False)
        lookahead_distance = getattr(self.config, "lookahead_distance", 0)
        lookahead_distance_min = getattr(self.config, "lookahead_distance_min", 0)
        lookahead_distance_max = getattr(self.config, "lookahead_distance_max", 0)
        self.net._lookahead_sink_enabled = lookahead_enabled
        self.net._lookahead_distance = lookahead_distance
        self.net._lookahead_distance_min = lookahead_distance_min
        self.net._lookahead_distance_max = lookahead_distance_max

        # Also sync down onto every block's self-attention (runtime override in
        # case the network was constructed with a different lookahead config).
        if hasattr(self.net, "blocks"):
            for block in self.net.blocks:
                if hasattr(block, "self_attn"):
                    block.self_attn.lookahead_sink_enabled = lookahead_enabled
                    block.self_attn.lookahead_distance = lookahead_distance

        if lookahead_enabled:
            if lookahead_distance_min > 0 and lookahead_distance_max > 0:
                logger.info(
                    f"[attn] Lookahead sink ENABLED, stochastic distance in "
                    f"[{lookahead_distance_min}, {lookahead_distance_max}] "
                    f"(eval uses fixed={lookahead_distance})"
                )
            else:
                logger.info(
                    f"[attn] Lookahead sink ENABLED, distance={lookahead_distance} frames"
                )
        else:
            logger.info("[attn] Lookahead sink: disabled (standard sink)")

        self.net._model_sink_cache = getattr(
            self.config, "model_sink_cache_enabled", False
        )
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

    def _apply_running_ahead_config(self):
        """Apply Knot Forcing running-ahead configuration to the student net.

        Paper Section 3.3 — the reference image's RoPE position is maintained
        "ahead" of the current generation chunk. Advancement logic lives in
        `network_causal.advance_running_ahead`; this method stamps the initial
        state on the module attributes that `_apply_window_rope` reads.

        Only the causal student (`self.net`) needs these attrs. The teacher and
        fake_score use `InfiniteTalkWan` (bidirectional, full attention, no
        sliding-window sink) — running-ahead does not apply.
        """
        enabled = getattr(self.config, "use_running_ahead", False)
        step = getattr(self.config, "running_ahead_step", 4)
        init_n = getattr(self.config, "running_ahead_init_n", 8)

        self.net._running_ahead_enabled = enabled
        self.net._running_ahead_step = step
        self.net._running_ahead_n = init_n

        if enabled:
            logger.info(
                f"[running_ahead] Student: enabled "
                f"(step={step}, init_n={init_n})"
            )
        else:
            logger.info("[running_ahead] Student: disabled")

    @staticmethod
    def _get_free_ram_gb() -> float:
        """Get free system RAM in GB."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024 * 1024)
        except Exception:
            return -1.0

    def single_train_step(self, data: Dict[str, Any], iteration: int):
        """Combined fake_score + student update on student steps (true 1:N ratio).

        Matches the original Self-Forcing training loop where the critic updates
        EVERY step, including on the generator (student) step. This gives a true
        1:N ratio (N critic updates per student update in an N-step cycle).

        On non-student steps (iter % freq != 0): delegate to base class (fake_score only).
        On student steps (iter % freq == 0): run fake_score backward manually
        (freeing its graph to save memory), then return student loss for the
        trainer's backward. Both sets of gradients accumulate across grad_accum
        rounds; the trainer steps both optimizers at the end.

        CRITICAL: Do NOT toggle requires_grad_ on self.net — this breaks FSDP2
        gradient checkpointing recomputation. Use no_grad() context instead.
        """
        if iteration % self.config.student_update_freq != 0:
            # Critic-only step — unchanged from base class
            return super().single_train_step(data, iteration)

        # === Combined step: fake_score + student ===
        from fastgen.networks.InfiniteTalk.lora import freeze_base

        real_data, condition, neg_condition = self._prepare_training_data(data)
        grad_accum_rounds = getattr(self.config, "grad_accum_rounds", None) or 1

        # --- Step 1: Fake score forward + manual backward (frees graph) ---
        # Enable fake_score LoRA params for training, keep base frozen.
        # Do NOT touch self.net.requires_grad_ — it stays as-is.
        self.fake_score.train()
        freeze_base(self.fake_score)

        input_fs, t_student_fs, t_fs, eps_fs = self._generate_noise_and_time(real_data)

        fake_loss_map, _ = self._fake_score_discriminator_update_step(
            input_fs, t_student_fs, t_fs, eps_fs, real_data, condition=condition,
        )

        # Manual backward with same scaling the trainer uses for the student loss.
        # This ensures fake_score grads accumulate correctly across grad_accum rounds.
        # The graph is freed here so it doesn't overlap with the student forward.
        (fake_loss_map["total_loss"] / grad_accum_rounds).backward()

        # --- Step 2: Student forward (returned for trainer's backward) ---
        # Freeze fake_score for student update phase
        self.fake_score.eval()
        for p in self.fake_score.parameters():
            p.requires_grad = False

        self.net.clear_caches()

        input_student, t_student, t, eps = self._generate_noise_and_time(real_data)

        student_loss_map, student_outputs = self._student_update_step(
            input_student, t_student, t, eps, data,
            condition=condition, neg_condition=neg_condition,
        )

        # Attach fake_score loss for logging (detached — no gradient)
        student_loss_map["fake_score_loss"] = fake_loss_map["total_loss"].detach()

        return student_loss_map, student_outputs

    def get_optimizers(self, iteration: int) -> list:
        """On student steps, return BOTH optimizers (trainer steps both)."""
        if iteration % self.config.student_update_freq == 0:
            return [self.net_optimizer, self.fake_score_optimizer]
        else:
            if self.config.gan_loss_weight_gen > 0:
                return [self.fake_score_optimizer, self.discriminator_optimizer]
            else:
                return [self.fake_score_optimizer]

    def get_lr_schedulers(self, iteration: int) -> list:
        """On student steps, return BOTH schedulers (trainer steps both)."""
        if iteration % self.config.student_update_freq == 0:
            return [self.net_lr_scheduler, self.fake_score_lr_scheduler]
        else:
            if self.config.gan_loss_weight_gen > 0:
                return [self.fake_score_lr_scheduler, self.discriminator_lr_scheduler]
            else:
                return [self.fake_score_lr_scheduler]

    def _setup_grad_requirements(self, iteration: int) -> None:
        """Override to respect LoRA freeze when toggling fake_score trainability.

        The base DMD2Model calls requires_grad_(True) on the entire fake_score
        module during its update step, which unfreezes all 14B base parameters.
        We instead toggle only the LoRA and audio_proj parameters.

        NOTE: On student steps (iter % freq == 0), single_train_step handles
        grad setup directly — this method is only called on fake_score-only steps.
        """
        from fastgen.networks.InfiniteTalk.lora import freeze_base

        if iteration % self.config.student_update_freq == 0:
            # Student update: freeze fake_score entirely
            self.fake_score.eval()
            for p in self.fake_score.parameters():
                p.requires_grad = False
            if self.config.gan_loss_weight_gen > 0 and hasattr(self, 'discriminator'):
                self.discriminator.eval()
                for p in self.discriminator.parameters():
                    p.requires_grad = False
        else:
            # Fake_score update: enable only LoRA + audio_proj params
            self.fake_score.train()
            freeze_base(self.fake_score)
            if self.config.gan_loss_weight_gen > 0 and hasattr(self, 'discriminator'):
                self.discriminator.train()
                for p in self.discriminator.parameters():
                    p.requires_grad = True

    def _prepare_training_data(self, data: Dict[str, Any]) -> tuple[torch.Tensor, Any, Any]:
        """Build InfiniteTalk condition and neg_condition dicts from dataset output.

        The InfiniteTalk dataset returns:
            real: [B, 16, 21, H, W] -- clean video latents
            first_frame_cond: [B, 16, 21, H, W] -- VAE-encoded reference frame
            clip_features: [B, 1, 257, 1280] -- CLIP features
            audio_emb: [B, 81, 12, 768] -- wav2vec2 audio embeddings
            text_embeds: [B, 1, 512, 4096] -- T5 text embedding
            neg_text_embeds: [B, 1, 512, 4096] -- negative text embedding

        Returns:
            real_data: [B, 16, 21, H, W]
            condition: dict with all InfiniteTalk conditioning
            neg_condition: dict with null audio + negative text (for 2-call CFG base path)
        """
        real_data = data["real"]

        # Positive condition
        condition = {
            "text_embeds": data["text_embeds"],
            "first_frame_cond": data["first_frame_cond"],
            "clip_features": data["clip_features"],
            "audio_emb": data["audio_emb"],
            # Stash neg_text_embeds inside condition for 3-call CFG override
            "neg_text_embeds": data["neg_text_embeds"],
        }

        # Negative condition: drop BOTH text and audio (fully unconditional)
        neg_condition = {
            "text_embeds": data["neg_text_embeds"],
            "first_frame_cond": data["first_frame_cond"],
            "clip_features": data["clip_features"],
            "audio_emb": torch.zeros_like(data["audio_emb"]),
        }

        return real_data, condition, neg_condition

    def _apply_classifier_free_guidance(
        self,
        perturbed_data: torch.Tensor,
        t: torch.Tensor,
        teacher_x0: torch.Tensor,
        neg_condition: Optional[Any] = None,
    ) -> torch.Tensor:
        """Apply 3-call classifier-free guidance to teacher predictions.

        InfiniteTalk uses separate text and audio guidance scales:
            output = uncond + text_scale * (cond - drop_text) + audio_scale * (drop_text - uncond)

        The base class _apply_classifier_free_guidance does standard 2-call CFG.
        We override to implement the 3-call variant.

        Args:
            perturbed_data: Noisy data tensor [B, C, T, H, W]
            t: Timestep tensor [B]
            teacher_x0: Teacher x0 prediction with FULL conditioning (cond call already done)
            neg_condition: Fully unconditional condition (neg text + zero audio)

        Returns:
            CFG-adjusted teacher_x0
        """
        text_guide_scale = getattr(self.config, "text_guide_scale", 5.0)
        audio_guide_scale = getattr(self.config, "audio_guide_scale", 4.0)

        with torch.no_grad():
            # teacher_x0 is already the FULL conditioned prediction (text + audio).
            # We need two more calls via the stashed _current_condition.

            if not hasattr(self, "_current_condition") or self._current_condition is None:
                raise RuntimeError(
                    "InfiniteTalkSelfForcingModel._apply_classifier_free_guidance: "
                    "_current_condition not set. The 3-call CFG requires condition "
                    "stashing via _student_update_step. Check that single_train_step "
                    "flows through _student_update_step before reaching CFG."
                )

            # Build neg_text_condition: negative text + positive audio
            neg_text_condition = {
                **self._current_condition,
                "text_embeds": self._current_condition["neg_text_embeds"],
            }
            # Remove the stashed neg_text_embeds from the passed condition
            neg_text_condition = {
                k: v for k, v in neg_text_condition.items() if k != "neg_text_embeds"
            }

            # Call 2: drop text only
            kwargs = {"condition": neg_text_condition, "fwd_pred_type": "x0"}
            if self.config.skip_layers is not None:
                kwargs["skip_layers"] = self.config.skip_layers
            teacher_drop_text = self.teacher(perturbed_data, t, **kwargs)

            # Call 3: drop both (fully unconditional) -- this is neg_condition
            kwargs_uncond = {"condition": neg_condition, "fwd_pred_type": "x0"}
            if self.config.skip_layers is not None:
                kwargs_uncond["skip_layers"] = self.config.skip_layers
            teacher_uncond = self.teacher(perturbed_data, t, **kwargs_uncond)

            # 3-call CFG formula
            teacher_x0 = (
                teacher_uncond
                + text_guide_scale * (teacher_x0 - teacher_drop_text)
                + audio_guide_scale * (teacher_drop_text - teacher_uncond)
            )

        return teacher_x0

    def _student_update_step(self, input_student, t_student, t, eps, data, condition=None, neg_condition=None):
        """Override to stash condition for 3-call CFG access in _apply_classifier_free_guidance."""
        # Stash the full condition so _apply_classifier_free_guidance can access it
        self._current_condition = condition
        try:
            result = super()._student_update_step(
                input_student, t_student, t, eps, data, condition=condition, neg_condition=neg_condition
            )
        finally:
            self._current_condition = None
        return result

    # ------------------------------------------------------------------
    # Validation: causal AR inference (no teacher, no fake_score)
    # ------------------------------------------------------------------

    def validation_step(self, data: Dict[str, Any], iteration: int) -> tuple[dict, dict]:
        """Validation using CausVid's causal AR inference (chunk-by-chunk with KV cache).

        Uses CausVidModel._student_sample_loop which does proper AR inference:
        chunk-by-chunk denoising with KV cache updates, matching inference behavior.
        No teacher, no fake_score — just the student generating video.
        """
        real_data, condition, neg_condition = self._prepare_training_data(data)
        B, C, T, H, W = real_data.shape

        logger.info(f"[val] Starting CausVid AR inference (B={B}, T={T}, steps={self.config.student_sample_steps})")

        noise = torch.randn_like(real_data)
        context_noise = getattr(self.config, "context_noise", 0)

        with torch.no_grad():
            gen_data = CausVidModel.generator_fn(
                net=self.net,
                noise=noise,
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                context_noise=context_noise,
                precision_amp=self.precision_amp_infer,
            )

            # VAE decode latents → pixel video for visual logging
            self._ensure_vae_loaded()
            has_vae = hasattr(self.net, "vae")

            # Diagnostic: log latent stats to distinguish bad model output vs bad decode
            logger.info(
                f"[val] gen_data stats: shape={list(gen_data.shape)}, "
                f"min={gen_data.min().item():.3f}, max={gen_data.max().item():.3f}, "
                f"mean={gen_data.mean().item():.3f}, std={gen_data.std().item():.3f}"
            )

            if has_vae:
                pixel_video = self.net.vae.decode(gen_data[:1].float())
                if isinstance(pixel_video, (list, tuple)):
                    pixel_video = torch.stack(pixel_video) if len(pixel_video) > 1 else pixel_video[0].unsqueeze(0)
                logger.info(
                    f"[val] pixel_video stats: shape={list(pixel_video.shape)}, "
                    f"min={pixel_video.min().item():.3f}, max={pixel_video.max().item():.3f}, "
                    f"mean={pixel_video.mean().item():.3f}"
                )
                gen_output = pixel_video
            else:
                gen_output = gen_data

        loss_map = {"total_loss": torch.tensor(0.0, device=self.device)}
        outputs = {"gen_rand": gen_output}

        return loss_map, outputs

    # ------------------------------------------------------------------
    # Visual logging: lazy VAE loading + _get_outputs with decode
    # ------------------------------------------------------------------

    def _ensure_vae_loaded(self) -> bool:
        """Lazily load VAE for visual logging (avoids torch.compile poisoning).

        Reuses the dataloader's _add_infinitetalk_to_path() which handles all
        mock module setup (xformers, decord, etc.) and is proven to work during
        training.  Stores the VAE on self.net.vae.
        """
        if self._vae_load_attempted:
            return hasattr(self.net, "vae")
        self._vae_load_attempted = True

        vae_path = os.environ.get("INFINITETALK_VAE_PATH") or getattr(self.config, "vae_path", None)
        if not vae_path or not os.path.exists(vae_path):
            logger.warning(f"VAE not found at {vae_path!r}, visual logging disabled")
            return False

        try:
            from fastgen.datasets.infinitetalk_dataloader import _add_infinitetalk_to_path
            _add_infinitetalk_to_path()

            from wan.modules.vae import WanVAE
            device_str = f"cuda:{self.device}" if isinstance(self.device, int) else str(self.device)
            self.net.vae = WanVAE(vae_pth=vae_path, device=device_str)
            logger.info(f"VAE loaded from {vae_path} on {device_str}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load VAE: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_outputs(self, gen_data, input_student=None, condition=None):
        """Return callable that generates AR sample + decodes to pixel video."""
        has_vae = self._ensure_vae_loaded()

        if has_vae and condition is not None:
            noise = torch.randn_like(gen_data, dtype=self.precision)

            def _generate_and_decode():
                with torch.no_grad():
                    latent = self.generator_fn(
                        net=self.net_inference,
                        noise=noise,
                        condition=condition,
                        student_sample_steps=self.config.student_sample_steps,
                        student_sample_type=self.config.student_sample_type,
                        t_list=self.config.sample_t_cfg.t_list,
                        precision_amp=self.precision_amp_infer,
                        context_noise=getattr(self.config, "context_noise", 0),
                    )
                    # Decode latent → pixel video (VAE already on correct device)
                    video = self.net.vae.decode(latent[:1].float())
                    if isinstance(video, (list, tuple)):
                        video = torch.stack(video) if len(video) > 1 else video[0].unsqueeze(0)
                    return video

            return {
                "gen_rand": _generate_and_decode,
                "input_rand": noise,
                "gen_rand_train": gen_data,
            }

        # Fallback: return latent-space callable (no VAE decode)
        from functools import partial
        noise = torch.randn_like(gen_data, dtype=self.precision)
        gen_rand_func = partial(
            self.generator_fn,
            net=self.net_inference,
            noise=noise,
            condition=condition,
            student_sample_steps=self.config.student_sample_steps,
            student_sample_type=self.config.student_sample_type,
            t_list=self.config.sample_t_cfg.t_list,
            precision_amp=self.precision_amp_infer,
            context_noise=getattr(self.config, "context_noise", 0),
        )
        return {"gen_rand": gen_rand_func, "input_rand": noise, "gen_rand_train": gen_data}
