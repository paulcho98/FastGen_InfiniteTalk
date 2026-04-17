#!/usr/bin/env python
"""
Benchmark GPU memory for InfiniteTalk Self-Forcing at different batch sizes.

Loads models once (sequential FSDP init), then tests the combined student step
(iteration % freq == 0, the memory-heaviest step) at BS=1,2,4.

Uses real data with lazy encoding active to capture true peak memory.

Usage:
    torchrun --nproc_per_node=8 scripts/benchmark_sf_memory.py \
        --config fastgen/configs/experiments/InfiniteTalk/config_sf.py
"""

import gc
import argparse
import warnings

import torch
import torch.distributed as dist

from fastgen.configs.config import BaseConfig
from fastgen.utils import instantiate
from fastgen.utils.distributed import synchronize, clean_up, is_rank0
import fastgen.utils.logging_utils as logger
from fastgen.utils.scripts import parse_args, setup

warnings.filterwarnings("ignore", "Grad strides do not match bucket view strides")


def get_peak_memory_gb(device: int = 0) -> float:
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


def get_current_memory_gb(device: int = 0) -> float:
    return torch.cuda.memory_allocated(device) / (1024 ** 3)


def reset_memory_stats(device: int = 0):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def make_batch(sample: dict, batch_size: int) -> dict:
    """Stack a single sample into a batch of the desired size."""
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v[:1].expand(batch_size, *v.shape[1:]).contiguous()
        else:
            batch[k] = v
    return batch


def main(config: BaseConfig):
    device = torch.cuda.current_device()

    # Build model (same as train.py)
    config.model_class.config = config.model
    model = instantiate(config.model_class)
    config.model_class.config = None
    synchronize()

    logger.info("Building model...")
    model.build_model()
    synchronize()

    # Set _is_fsdp flag (normally done by trainer's on_train_begin)
    model._is_fsdp = True

    # Init optimizers (after FSDP wrapping, same as trainer)
    model.init_optimizers()
    synchronize()

    if is_rank0():
        logger.info(f"Model built. Baseline GPU: {get_current_memory_gb(device):.1f} GB")
        logger.info(f"student_update_freq: {config.model.student_update_freq}")

    # Load one real sample via the dataloader (with lazy encoding active)
    logger.info("Loading real data sample (with lazy encoding)...")
    dataloader = instantiate(config.dataloader_train)
    data_iter = iter(dataloader)
    sample = next(data_iter)

    # Move to device and model precision
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v.to(device=device, dtype=model.precision)

    if is_rank0():
        logger.info(f"Sample loaded. real shape: {sample['real'].shape}")
        logger.info(f"GPU after data load: {get_current_memory_gb(device):.1f} GB")

    # The combined student step is the heaviest — iteration % freq == 0
    student_iter = config.model.student_update_freq  # e.g. 5

    batch_sizes = [1, 2, 4]
    results = []

    for bs in batch_sizes:
        if is_rank0():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing BS={bs} — combined student step (iteration={student_iter})")
            logger.info(f"{'='*60}")

        batch = make_batch(sample, bs)

        # Clear all model state
        if hasattr(model.net, 'clear_caches'):
            model.net.clear_caches()

        # Zero all optimizer gradients (prevents stale state from previous BS test)
        model.optimizers_zero_grad(student_iter)

        # Reset peak memory tracking
        reset_memory_stats(device)
        baseline = get_current_memory_gb(device)
        synchronize()

        try:
            # Run combined student step (fake_score backward + student forward)
            with model.autocast():
                loss_map, outputs = model.single_train_step(batch, student_iter)

            # Student backward (same as trainer — fake_score backward already done inside)
            grad_accum = getattr(config.model, "grad_accum_rounds", 1) or 1
            model.grad_scaler.scale(loss_map["total_loss"] / grad_accum).backward()

            synchronize()
            peak = get_peak_memory_gb(device)
            current = get_current_memory_gb(device)

            if is_rank0():
                losses = {k: f"{v.item():.4f}" for k, v in loss_map.items() if isinstance(v, torch.Tensor)}
                logger.info(f"BS={bs} PASSED — Peak: {peak:.1f} GB, Post-step: {current:.1f} GB, Baseline: {baseline:.1f} GB")
                logger.info(f"  Losses: {losses}")
                results.append((bs, peak, current, "PASS"))

            # Clean up gradients fully
            model.optimizers_zero_grad(student_iter)

        except torch.cuda.OutOfMemoryError:
            synchronize()
            peak = get_peak_memory_gb(device)
            if is_rank0():
                logger.info(f"BS={bs} OOM — Peak before crash: {peak:.1f} GB")
                results.append((bs, peak, -1, "OOM"))
            gc.collect()
            torch.cuda.empty_cache()
            if is_rank0():
                logger.info("Stopping — higher batch sizes will also OOM.")
            break

        except Exception as e:
            if is_rank0():
                import traceback
                logger.info(f"BS={bs} ERROR: {e}")
                traceback.print_exc()
                results.append((bs, -1, -1, f"ERROR: {type(e).__name__}"))
            break

        # Clean up between tests
        del loss_map, outputs, batch
        gc.collect()
        torch.cuda.empty_cache()
        synchronize()

    # Summary
    if is_rank0():
        gpu_total = torch.cuda.get_device_properties(device).total_mem / 1e9
        ws = dist.get_world_size()

        logger.info(f"\n{'='*60}")
        logger.info("MEMORY BENCHMARK RESULTS (per GPU)")
        logger.info(f"{'='*60}")
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}, {gpu_total:.0f} GB")
        logger.info(f"Models: 3x 14B (student + teacher + fake_score), LoRA rank=128")
        logger.info(f"Resolution: quarter-res {config.model.input_shape}")
        logger.info(f"FSDP: world_size={ws}, precision_fsdp={config.model.precision_fsdp}")
        logger.info(f"student_update_freq: {config.model.student_update_freq}")
        logger.info(f"{'─'*60}")
        logger.info(f"{'BS':>4} | {'Peak GB':>8} | {'Post-step GB':>12} | {'Status':>8}")
        logger.info(f"{'─'*60}")
        for bs, peak, current, status in results:
            cur_str = f"{current:.1f}" if current >= 0 else "N/A"
            peak_str = f"{peak:.1f}" if peak >= 0 else "N/A"
            logger.info(f"{bs:>4} | {peak_str:>8} | {cur_str:>12} | {status:>8}")
        logger.info(f"{'─'*60}")

        max_bs = max((bs for bs, _, _, s in results if s == "PASS"), default=0)
        if max_bs > 0:
            best_peak = next(p for b, p, _, s in results if b == max_bs and s == "PASS")
            headroom = gpu_total - best_peak
            logger.info(f"\nMax passing BS: {max_bs} (peak {best_peak:.1f} GB, headroom {headroom:.1f} GB)")

            target_effective = 32
            grad_accum = max(1, target_effective // (max_bs * ws))
            actual_effective = max_bs * ws * grad_accum
            logger.info(f"Suggested: BS={max_bs}, grad_accum={grad_accum}, "
                        f"world_size={ws} → effective BS={actual_effective}")

    synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SF Memory Benchmark")
    args = parse_args(parser)
    config = setup(args)

    main(config)

    clean_up()
    logger.info("Benchmark complete.")
