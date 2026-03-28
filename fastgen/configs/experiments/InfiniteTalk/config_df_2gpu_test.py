# SPDX-License-Identifier: Apache-2.0
"""Minimal test: validation visualization only. No training visuals."""
import os
from fastgen.configs.experiments.InfiniteTalk.config_df import create_config as create_base_config
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.callbacks.infinitetalk_wandb import InfiniteTalkWandbCallback
from fastgen.callbacks.train_profiler import TrainProfilerCallback
from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.utils import LazyCall as L

WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
TEST_LIST = os.environ.get("INFINITETALK_DATA_LIST", "data/test_precomputed/sample_list.txt")
NEG_EMB = os.environ.get("INFINITETALK_NEG_TEXT_EMB", "data/test_precomputed/neg_text_embeds.pt")

def create_config():
    config = create_base_config()
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64
    config.model.net_optimizer.lr = 1e-5
    config.trainer.fsdp = False
    config.trainer.ddp = True
    config.trainer.max_iter = 3
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 3
    config.trainer.grad_accum_rounds = 2
    # Val at iter 2, skip iter 0
    config.trainer.validation_iter = 2
    config.trainer.skip_iter0_validation = True
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 3}]  # only 3 val samples
    config.trainer.callbacks = {
        "wandb": L(InfiniteTalkWandbCallback)(
            sample_logging_iter=999999,    # NO training visuals
            validation_logging_step=1,     # visual for each val sample
            audio_fps=25,
        ),
        "train_profiler": L(TrainProfilerCallback)(every_n=1),
        "grad_clip": L(GradClipCallback)(grad_norm=10.0, model_key="net"),
    }
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=TEST_LIST, neg_text_emb_path=NEG_EMB,
        batch_size=1, load_ode_path=False,
        expected_latent_shape=config.model.input_shape, num_workers=0,
    )
    config.dataloader_val = L(InfiniteTalkDataLoader)(
        data_list_path=TEST_LIST, neg_text_emb_path=NEG_EMB,
        batch_size=1, load_ode_path=False,
        expected_latent_shape=config.model.input_shape, num_workers=0,
    )
    config.log_config.project = "DF_InfiniteTalk"
    config.log_config.group = "infinitetalk_df"
    config.log_config.name = "val_vis_test"
    config.log_config.wandb_mode = "online"
    return config
