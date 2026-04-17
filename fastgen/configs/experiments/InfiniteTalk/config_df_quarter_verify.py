"""Final verification: BS=4 quarter-res, 5 iterations, 2 GPUs, lazy encoding, cleaned CSV."""
import os
os.environ.setdefault("NUM_GPUS", "2")

from fastgen.configs.experiments.InfiniteTalk.config_df_quarter_prod import create_config as create_base
from fastgen.callbacks.callback import Callback
from fastgen.utils import LazyCall as L

class MemoryCallback(Callback):
    def on_training_step_end(self, model=None, data_batch=None, output_batch=None, loss_dict=None, iteration=None, **kwargs):
        import torch
        if loss_dict:
            parts = [f"{k}={float(v):.6f}" for k, v in loss_dict.items() if hasattr(v, '__float__')]
            peak = torch.cuda.max_memory_allocated() / 1024**3
            alloc = torch.cuda.memory_allocated() / 1024**3
            print(f"  [iter {iteration}] peak={peak:.1f}GB alloc={alloc:.1f}GB {', '.join(parts)}", flush=True)

def create_config():
    config = create_base()
    config.trainer.max_iter = 5
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 999999
    config.trainer.validation_iter = 999999
    config.trainer.skip_iter0_validation = True
    config.trainer.callbacks = {"mem": L(MemoryCallback)()}
    config.log_config.wandb_mode = "disabled"
    return config
