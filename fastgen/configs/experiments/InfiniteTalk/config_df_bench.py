"""Benchmark config — configurable BS, grad_accum, LoRA rank/alpha via env vars."""
import os
from fastgen.configs.experiments.InfiniteTalk.config_df_test import create_config as create_base

def create_config():
    config = create_base()
    config.dataloader_train.batch_size = int(os.environ.get("BENCH_BS", "1"))
    config.trainer.grad_accum_rounds = int(os.environ.get("BENCH_ACCUM", "1"))
    config.trainer.max_iter = int(os.environ.get("BENCH_ITERS", "5"))
    # LoRA overrides
    lora_rank = os.environ.get("BENCH_LORA_RANK")
    lora_alpha = os.environ.get("BENCH_LORA_ALPHA")
    if lora_rank is not None:
        config.model.net.lora_rank = int(lora_rank)
    if lora_alpha is not None:
        config.model.net.lora_alpha = int(lora_alpha)
    return config
