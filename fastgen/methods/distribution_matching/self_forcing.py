# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist

from fastgen.methods import CausVidModel

import fastgen.utils.logging_utils as logger
from fastgen.networks.network import CausalFastGenNetwork
from fastgen.utils.basic_utils import convert_cfg_to_dict
from fastgen.utils.distributed import is_rank0, world_size

if TYPE_CHECKING:
    from fastgen.configs.methods.config_self_forcing import ModelConfig


class SelfForcingModel(CausVidModel):
    """Self-Forcing model for distribution matching distillation
    Inheritance hierarchy:
    SelfForcingModel -> CausVidModel -> DMD2Model -> FastGenModel

    The major difference between SelfForcingModel and DMD2Model is how we get
    the gen_data in the single_train_step() function.  In SelfForcingModel, we
    use self.rollout_with_gradient() to get the gen_data, which
    does the rollout with gradient tracking at the last denoising step.  The
    number of denoising steps is stochastic, and is sampled from the
    denoising_step_list.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

    def _generate_noise_and_time(
        self, real_data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random noises and time step

        Args:
            batch_size: Batch size
            real_data: Real data tensor for dtype/device reference

        Returns:
            input_student: Random noise used by the student
            t_max: Time step used by the student
            t: Time step for distribution matching
            eps: Random noise used by a forward process
        """
        batch_size = real_data.shape[0]

        eps_student = torch.randn(batch_size, *self.input_shape, device=self.device, dtype=real_data.dtype)
        t_student = torch.full(
            (batch_size,),
            self.net.noise_scheduler.max_t,
            device=self.device,
            dtype=self.net.noise_scheduler.t_precision,
        )
        input_student = self.net.noise_scheduler.latents(noise=eps_student)

        t = self.net.noise_scheduler.sample_t(
            batch_size, **convert_cfg_to_dict(self.config.sample_t_cfg), device=self.device
        )

        eps = torch.randn_like(real_data, device=self.device, dtype=real_data.dtype)

        return input_student, t_student, t, eps

    def _sample_denoising_end_steps(self, num_blocks: int) -> List[int]:
        """Sample a list of denoising end indices for each block"""
        sample_steps = self.config.student_sample_steps

        if is_rank0():
            if self.config.last_step_only:
                indices = torch.full((num_blocks,), sample_steps - 1, dtype=torch.long, device=self.device)
            else:
                indices = torch.randint(low=0, high=sample_steps, size=(num_blocks,), device=self.device)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=self.device)

        # Broadcast the random indices to all ranks
        if world_size() > 1:
            dist.broadcast(indices, src=0)

        return indices.tolist()

    def rollout_with_gradient(
        self,
        noise: torch.Tensor,
        condition: Optional[Any] = None,
        enable_gradient: bool = True,
        start_gradient_frame: int = 0,
    ) -> torch.Tensor:
        """
        Perform self-forcing rollout with gradient tracking at the last step of each block.

        No external KV cache is used. Instead, we update the model's internal caches
        once per completed block using `store_kv=True` under no_grad.

        Knot Forcing (paper arXiv:2512.21734v2): when `use_temporal_knot=True` in config:
          - Each chunk denoises c+k frames (was: c).
          - For block_idx > 0: inject previous chunk's tail prediction as I2V
            conditioning at position 0 of the current chunk via
            `condition["knot_latent_for_chunk_start"]`; pin `noisy_input[:, :, 0:k]`
            to the knot so the model sees it as clean input.
          - At exit step: fuse `x0_pred[:, :, 0:k]` with the saved knot via Eq. 5
            averaging; then save the new knot from `x0_pred[:, :, c:c+k]`.
          - Commit only first c frames to the output tensor. The knot is held
            for the next iteration's fusion; it is NOT written to the KV cache.
          - F3 is forced OFF under KF (separate cache pass used).

        Running-ahead: when `_running_ahead_enabled` on self.net, the sink's RoPE
        position is advanced by advance_running_ahead when the rollout cursor
        catches up. Re-caching is implicit under dynamic RoPE.

        Args:
            noise: Initial noise tensor [B, C, T, H, W]. Under KF, internally
                extended by k extra frames for the last chunk's knot slot.
            condition: Conditioning dict.
            enable_gradient: Whether to enable gradients at the exit step
            start_gradient_frame: Frame index to start gradient tracking

        Returns:
            generated_frames: [B, C, num_blocks * c, H, W] (KF commits c frames/block).
                When KF is off, same as input noise shape (legacy behavior).
        """
        assert isinstance(self.net, CausalFastGenNetwork), f"{self.net} must be a CausalFastGenNetwork"
        self.net.clear_caches()

        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=self.device)

        batch_size, C, num_frames, H, W = noise.shape
        chunk_size = self.net.chunk_size
        num_blocks = num_frames // chunk_size
        remaining_size = num_frames % chunk_size
        sample_steps = self.config.student_sample_steps
        dtype = noise.dtype

        # ── KF flags ──
        use_knot = getattr(self.config, "use_temporal_knot", False)
        k = int(getattr(self.config, "knot_size", 1)) if use_knot else 0
        denoise_len = chunk_size + k  # c+k frames per chunk under KF, c otherwise

        # Under KF, the last chunk needs k extra noise frames beyond num_blocks*c.
        # Allocate fresh noise and append. Total: num_blocks * c + k frames.
        if use_knot:
            extra = torch.randn(
                batch_size, C, k, H, W,
                device=noise.device, dtype=noise.dtype,
            )
            noise = torch.cat([noise, extra], dim=2)

        # F3 forced OFF under KF — always use the separate cache pass so the KV
        # cache is populated with committed c frames only (not c+k).
        skip_clean_cache = (
            getattr(self.net, "_skip_clean_cache_pass", False) and not use_knot
        )

        denoising_end_steps = self._sample_denoising_end_steps(num_blocks)
        logger.debug(f"denoising_end_steps: {denoising_end_steps}")

        t_list = self.config.sample_t_cfg.t_list
        if t_list is None:
            t_list = self.net.noise_scheduler.get_t_list(sample_steps, device=self.device)
        else:
            assert (
                len(t_list) - 1 == sample_steps
            ), f"t_list length (excluding zero) != student_sample_steps: {len(t_list) - 1} != {sample_steps}"
            t_list = torch.tensor(t_list, device=self.device, dtype=self.net.noise_scheduler.t_precision)

        # ── KF state ──
        knot_latent = None  # x̄ buffer — populated at iter 0 exit step, used at iter 1+

        # Lazy import to avoid circular dep at module load.
        from fastgen.networks.InfiniteTalk.network_causal import advance_running_ahead

        denoised_blocks = []
        for block_idx in range(num_blocks):
            if num_blocks == 0:
                cur_start_frame, cur_end_frame = 0, remaining_size
            else:
                cur_start_frame = 0 if block_idx == 0 else chunk_size * block_idx + remaining_size
                cur_end_frame = cur_start_frame + denoise_len

            # Running-ahead: advance absolute RoPE n when rollout catches up.
            # Re-cache is implicit under dynamic RoPE (K stored un-rotated,
            # rotation applied at each attention read).
            if getattr(self.net, "_running_ahead_enabled", False):
                advance_running_ahead(self.net, i=cur_start_frame, c=chunk_size, k=k)

            noisy_input = noise[:, :, cur_start_frame:cur_end_frame]

            # KF iter > 0: build condition with knot key + pin noisy_input at pos 0
            if use_knot and block_idx > 0 and knot_latent is not None:
                chunk_condition = dict(condition)
                chunk_condition["knot_latent_for_chunk_start"] = knot_latent
                # Pin position 0 of noisy_input to the clean knot (model sees clean input)
                noisy_input = noisy_input.clone()
                noisy_input[:, :, 0:k] = knot_latent
            else:
                chunk_condition = condition

            for step, t_cur in enumerate(t_list):
                if self.config.same_step_across_blocks:
                    exit_flag = step == denoising_end_steps[0]
                else:
                    exit_flag = step == denoising_end_steps[block_idx]

                t_chunk_cur = t_cur.expand(batch_size)

                if not exit_flag:
                    with torch.no_grad():
                        x0_pred_chunk = self.net(
                            noisy_input,
                            t_chunk_cur,
                            condition=chunk_condition,
                            cache_tag="pos",
                            store_kv=False,
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0",
                            is_ar=True,
                        )

                    t_next = t_list[step + 1]
                    t_chunk_next = t_next.expand(batch_size)
                    if self.config.student_sample_type == "sde":
                        eps_infer = torch.randn_like(x0_pred_chunk)
                    elif self.config.student_sample_type == "ode":
                        eps_infer = self.net.noise_scheduler.x0_to_eps(
                            xt=noisy_input, x0=x0_pred_chunk, t=t_chunk_cur
                        )
                    else:
                        raise NotImplementedError(
                            f"student_sample_type must be one of 'sde', 'ode' but got {self.config.student_sample_type}"
                        )
                    noisy_input = self.net.noise_scheduler.forward_process(
                        x0_pred_chunk, eps_infer, t_chunk_next
                    )

                    # KF iter > 0: re-pin the knot after forward_process (so next step
                    # still sees clean knot at position 0 of noisy_input)
                    if use_knot and block_idx > 0 and knot_latent is not None:
                        noisy_input = noisy_input.clone()
                        noisy_input[:, :, 0:k] = knot_latent
                else:
                    enable_grad = (
                        enable_gradient
                        and torch.is_grad_enabled()
                        and (cur_start_frame >= start_gradient_frame)
                    )
                    with torch.set_grad_enabled(enable_grad):
                        x0_pred_chunk = self.net(
                            noisy_input,
                            t_chunk_cur,
                            condition=chunk_condition,
                            cache_tag="pos",
                            store_kv=skip_clean_cache,
                            cur_start_frame=cur_start_frame,
                            fwd_pred_type="x0",
                            is_ar=True,
                        )

                    # KF: fusion at exit step + save new knot
                    if use_knot:
                        if block_idx > 0 and knot_latent is not None:
                            # Eq. 5: average previous chunk's knot prediction with
                            # current chunk's prediction at the shared boundary.
                            x0_pred_chunk = x0_pred_chunk.clone()
                            x0_pred_chunk[:, :, 0:k] = (
                                knot_latent + x0_pred_chunk[:, :, 0:k]
                            ) / 2.0
                        # Save new knot from the last k frames of the c+k prediction.
                        # Kept gradient-enabled so future-chunk fusion can backprop.
                        knot_latent = x0_pred_chunk[:, :, chunk_size:chunk_size + k]

                    break

            # Commit: under KF, only first c frames go to output; knot is held
            if use_knot:
                committed = x0_pred_chunk[:, :, 0:chunk_size]
            else:
                committed = x0_pred_chunk
            denoised_blocks.append(committed)

            # KV cache update: always separate pass under KF (F3 forced off for KF).
            # Caches only committed c frames, not the knot.
            if not skip_clean_cache:
                with torch.no_grad():
                    if self.config.context_noise > 0:
                        t_cache = torch.full(
                            (batch_size,), self.config.context_noise,
                            device=self.device, dtype=dtype,
                        )
                        x0_pred_cache = self.net.noise_scheduler.forward_process(
                            committed,
                            torch.randn_like(committed),
                            t_cache,
                        )
                    else:
                        x0_pred_cache = committed
                        t_cache = torch.zeros(batch_size, device=self.device, dtype=dtype)

                    # Cache pass uses the BASE condition (no knot key) — we're
                    # caching the committed clean frames, not the knot.
                    _ = self.net(
                        x0_pred_cache,
                        t_cache,
                        condition=condition,
                        cache_tag="pos",
                        store_kv=True,
                        cur_start_frame=cur_start_frame,
                        fwd_pred_type="x0",
                        is_ar=True,
                    )

        output = torch.cat(denoised_blocks, dim=2) if len(denoised_blocks) > 0 else torch.empty_like(noise)

        self.net.clear_caches()
        return output

    def gen_data_from_net(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        del t_student
        gen_data = self.rollout_with_gradient(
            noise=input_student,
            condition=condition,
            enable_gradient=self.config.enable_gradient_in_rollout,
            start_gradient_frame=self.config.start_gradient_frame,
        )
        return gen_data
