#!/usr/bin/env python3
"""Precompute T5 text embeddings only — lightweight, fast, parallelizable.

This is the first step before lazy caching of other modalities (VAE, CLIP,
audio) during training. T5 is the memory bottleneck (~20GB) that prevents
on-the-fly encoding during training, so we precompute it separately.

Automatically distributes across all visible GPUs using multiprocessing.

Usage:
    # Use all available GPUs automatically:
    python scripts/precompute_t5_only.py \
        --csv_path /path/to/video_list.csv \
        --output_dir /path/to/precomputed/ \
        --weights_dir /path/to/Wan2.1-I2V-14B-480P

    # Use specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/precompute_t5_only.py ...

Timing: ~0.5s/sample. 172K samples on 8 GPUs ≈ 3 hours.
"""

import argparse
import csv
import inspect
import logging
import os
import sys
import time

# Python 3.12: inspect.ArgSpec was removed
if not hasattr(inspect, 'ArgSpec'):
    inspect.ArgSpec = inspect.FullArgSpec
import torch
import torch.multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_NEG_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)


def _add_infinitetalk_to_path():
    """Add InfiniteTalk to sys.path and install necessary mocks."""
    it_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../InfiniteTalk"))
    if it_root not in sys.path:
        sys.path.insert(0, it_root)

    import types
    import importlib.machinery

    class _MockModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return lambda *a, **k: None

    for mock_pkg in [
        "xformers", "xformers.ops", "xformers.ops.fmha",
        "xformers.ops.fmha.attn_bias",
        "xfuser", "xfuser.core", "xfuser.core.distributed",
        "sageattn", "decord",
    ]:
        parts = mock_pkg.split(".")
        for i in range(len(parts)):
            partial = ".".join(parts[:i + 1])
            if partial not in sys.modules:
                mod = _MockModule(partial)
                if i < len(parts) - 1:
                    mod.__path__ = []
                mod.__spec__ = importlib.machinery.ModuleSpec(partial, None)
                sys.modules[partial] = mod

    sys.modules["xformers"].ops = sys.modules["xformers.ops"]
    sys.modules["xformers.ops"].memory_efficient_attention = lambda *a, **k: None

    vm = _MockModule("src.vram_management")
    vm.__spec__ = importlib.machinery.ModuleSpec("src.vram_management", None)
    vm.AutoWrappedQLinear = None
    vm.AutoWrappedLinear = None
    vm.AutoWrappedModule = None
    vm.enable_vram_management = lambda *a, **k: None
    sys.modules["src.vram_management"] = vm

    return it_root


def worker(gpu_id, rows, args):
    """Worker process for one GPU."""
    import inspect
    if not hasattr(inspect, 'ArgSpec'):
        inspect.ArgSpec = inspect.FullArgSpec

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    _add_infinitetalk_to_path()
    from wan.modules.t5 import T5EncoderModel

    logger.info("GPU %d: Loading T5 to %s (%d samples)", gpu_id, device, len(rows))
    t0 = time.time()
    t5_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.weights_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.weights_dir, "google", "umt5-xxl"),
    )
    logger.info("GPU %d: T5 loaded in %.1fs", gpu_id, time.time() - t0)

    success = 0
    skipped = 0
    for idx, row in enumerate(rows):
        video_rel = row["video_path"]
        text = row["text"]
        sample_name = os.path.splitext(video_rel.replace("/", "_"))[0]
        sample_dir = os.path.join(args.output_dir, sample_name)
        text_path = os.path.join(sample_dir, "text_embeds.pt")

        if os.path.isfile(text_path):
            skipped += 1
            continue

        os.makedirs(sample_dir, exist_ok=True)

        try:
            t5_encoder.model.to(device)
            ids, mask = t5_encoder.tokenizer(
                [text], return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                context = t5_encoder.model(ids, mask)
            torch.save(context.float().cpu(), text_path)
            t5_encoder.model.cpu()
            success += 1
        except Exception as e:
            logger.error("GPU %d: Failed %s: %s", gpu_id, sample_name, e)

        if (idx + 1) % 100 == 0:
            logger.info("GPU %d: %d/%d done (%d skipped)",
                        gpu_id, idx + 1, len(rows), skipped)

    logger.info("GPU %d: Complete. success=%d, skipped=%d, total=%d",
                gpu_id, success, skipped, len(rows))


def main():
    parser = argparse.ArgumentParser(
        description="Precompute T5 text embeddings only (multi-GPU)")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Process only first N samples (0 = all)")
    parser.add_argument("--neg_prompt", type=str, default=DEFAULT_NEG_PROMPT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read CSV
    with open(args.csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.num_samples > 0:
        rows = rows[:args.num_samples]
    logger.info("Total samples: %d", len(rows))

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info("Available GPUs: %d", num_gpus)

    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    # Precompute negative prompt (single GPU, shared file)
    neg_path = os.path.join(args.output_dir, "neg_text_embeds.pt")
    if not os.path.isfile(neg_path):
        logger.info("Computing negative-prompt T5 embedding...")
        _add_infinitetalk_to_path()
        from wan.modules.t5 import T5EncoderModel
        t5 = T5EncoderModel(
            text_len=512, dtype=torch.bfloat16,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(args.weights_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(args.weights_dir, "google", "umt5-xxl"),
        )
        t5.model.to("cuda:0")
        ids, mask = t5.tokenizer([args.neg_prompt], return_mask=True, add_special_tokens=True)
        with torch.no_grad():
            neg = t5.model(ids.to("cuda:0"), mask.to("cuda:0"))
        torch.save(neg.float().cpu(), neg_path)
        del t5
        torch.cuda.empty_cache()
        logger.info("Saved neg_text_embeds.pt")

    # Shard rows across GPUs
    shard_size = (len(rows) + num_gpus - 1) // num_gpus
    shards = []
    for i in range(num_gpus):
        start = i * shard_size
        end = min((i + 1) * shard_size, len(rows))
        if start < len(rows):
            shards.append(rows[start:end])

    logger.info("Launching %d workers (shard_size=%d)", len(shards), shard_size)

    # Launch workers
    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id, shard in enumerate(shards):
        p = mp.Process(target=worker, args=(gpu_id, shard, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = sum(1 for p in processes if p.exitcode != 0)
    if failed:
        logger.warning("%d worker(s) failed", failed)
    else:
        logger.info("All workers completed successfully")


if __name__ == "__main__":
    main()
