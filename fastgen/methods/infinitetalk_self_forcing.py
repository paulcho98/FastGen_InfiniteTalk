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

from typing import Any, Dict, Optional, TYPE_CHECKING

import torch

from fastgen.methods.distribution_matching.self_forcing import SelfForcingModel
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
      - build_model: supports separate fake_score architecture via config.fake_score_net

    3-call CFG formula (matching InfiniteTalk's original inference):
        output = uncond + text_scale * (cond - drop_text) + audio_scale * (drop_text - uncond)
    where:
        cond = teacher(x_t, t, condition)                    # full conditioning
        drop_text = teacher(x_t, t, neg_text_condition)      # drop text, keep audio
        uncond = teacher(x_t, t, neg_condition)              # drop both text and audio
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def build_model(self):
        """Override to instantiate fake_score from config.fake_score_net if provided.

        The base DMD2Model.build_model() now natively checks config.fake_score_net,
        so this override is only needed if additional logic is required. Kept for
        backward compatibility and explicitness.
        """
        super().build_model()

        # The base dmd2.py now handles fake_score_net natively.
        # If additional per-model-class logic is needed later, add it here.

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
            # We need two more calls:

            # 1. Drop text only (neg_text + keep audio) -- construct on the fly
            # The positive condition is stored in the training step context.
            # We access it through the stashed neg_text_embeds in condition.
            # The base class passes neg_condition which has BOTH dropped.
            # We need to build an intermediate: neg_text + positive audio.

            # Build neg_text_condition from neg_condition + original audio
            # neg_condition has: neg_text_embeds as text, zero audio
            # We need: neg_text_embeds as text, original audio
            # The original audio was stashed... but the base class doesn't pass condition here.

            # Strategy: Use the stashed neg_text_embeds from the condition dict.
            # The _student_update_step passes condition to _compute_teacher_prediction_gan_loss
            # which calls teacher with full condition. neg_condition is passed here.
            # We reconstruct neg_text_condition from neg_condition + the original condition's audio.

            # Since we can't access the original condition here, we use a different approach:
            # Store the condition on self during single_train_step execution.
            if hasattr(self, "_current_condition") and self._current_condition is not None:
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
            else:
                # Fallback to standard 2-call CFG if condition not stashed
                logger.warning(
                    "InfiniteTalkSelfForcingModel: _current_condition not set, "
                    "falling back to standard 2-call CFG"
                )
                kwargs = {"condition": neg_condition, "fwd_pred_type": "x0"}
                if self.config.skip_layers is not None:
                    kwargs["skip_layers"] = self.config.skip_layers
                teacher_x0_neg = self.teacher(perturbed_data, t, **kwargs)
                guidance_scale = getattr(self.config, "guidance_scale", None)
                if guidance_scale is None:
                    guidance_scale = text_guide_scale  # reasonable fallback
                teacher_x0 = teacher_x0 + (guidance_scale - 1) * (teacher_x0 - teacher_x0_neg)

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
