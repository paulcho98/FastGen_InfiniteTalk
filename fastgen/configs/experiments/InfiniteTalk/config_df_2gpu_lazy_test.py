# SPDX-License-Identifier: Apache-2.0
"""2-GPU DDP test with lazy encoding — verifies GPU offload works during training."""
import os
from fastgen.configs.experiments.InfiniteTalk.config_df import create_config as create_base_config
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.callbacks.infinitetalk_wandb import InfiniteTalkWandbCallback
from fastgen.callbacks.train_profiler import TrainProfilerCallback
from fastgen.callbacks.grad_clip import GradClipCallback
from fastgen.utils import LazyCall as L

WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
TEST_LIST = "data/test_lazy_ddp_list.txt"
NEG_EMB = os.environ.get("INFINITETALK_NEG_TEXT_EMB", "data/precomputed_talkvid/neg_text_embeds.pt")
CSV_PATH = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv"
RAW_DATA_ROOT = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/"
WAV2VEC_DIR = os.environ.get(
    "INFINITETALK_WAV2VEC_DIR",
    "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base",
)

def create_config():
    config = create_base_config()
    config.model.net.lora_rank = 128
    config.model.net.lora_alpha = 64
    config.model.net_optimizer.lr = 1e-5
    config.trainer.fsdp = False
    config.trainer.ddp = True
    config.trainer.max_iter = 4
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 999
    config.trainer.grad_accum_rounds = 1
    # No validation — just test training with lazy samples
    config.trainer.validation_iter = 999
    config.trainer.skip_iter0_validation = True
    config.trainer.callbacks = {
        "wandb": L(InfiniteTalkWandbCallback)(
            sample_logging_iter=999999,
            validation_logging_step=1,
            audio_fps=25,
        ),
        "train_profiler": L(TrainProfilerCallback)(every_n=1),
        "grad_clip": L(GradClipCallback)(grad_norm=10.0, model_key="net"),
    }
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=TEST_LIST, neg_text_emb_path=NEG_EMB,
        batch_size=1, load_ode_path=False,
        expected_latent_shape=config.model.input_shape,
        num_video_frames=93,
        num_latent_frames=21,
        raw_data_root=RAW_DATA_ROOT,
        csv_path=CSV_PATH,
        weights_dir=WEIGHTS_DIR,
        wav2vec_dir=WAV2VEC_DIR,
        encode_device="cuda",
        num_workers=0,
    )
    # No val dataloader needed for this test
    config.dataloader_val = None
    config.log_config.project = "DF_InfiniteTalk"
    config.log_config.group = "infinitetalk_df"
    config.log_config.name = "lazy_encode_ddp_test"
    config.log_config.wandb_mode = "disabled"
    return config
