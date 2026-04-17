import os
os.environ.setdefault("NUM_GPUS", "2")

from fastgen.configs.experiments.InfiniteTalk.config_df_quarter_prod import create_config as create_base
from fastgen.callbacks.callback import Callback
from fastgen.utils import LazyCall as L
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader

class StdoutCallback(Callback):
    def on_training_step_end(self, model=None, data_batch=None, output_batch=None, loss_dict=None, iteration=None, **kwargs):
        import torch
        if loss_dict:
            parts = [f"{k}={float(v):.6f}" for k, v in loss_dict.items() if hasattr(v, '__float__')]
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  [iter {iteration}] peak_mem={mem:.1f}GB {', '.join(parts)}", flush=True)

def create_config():
    config = create_base()
    config.trainer.max_iter = 3
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 999999
    config.trainer.validation_iter = 999999
    config.trainer.skip_iter0_validation = True
    config.trainer.callbacks = {"stdout": L(StdoutCallback)()}
    config.log_config.wandb_mode = "disabled"

    # Override dataloader with lazy encoding enabled
    WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path="/tmp/lazy_test_samples.txt",
        neg_text_emb_path="data/precomputed_talkvid/neg_text_embeds.pt",
        batch_size=4,
        load_ode_path=False,
        quarter_res=True,
        expected_latent_shape=None,  # Don't filter — let lazy encoding handle it
        num_video_frames=93,
        num_latent_frames=21,
        raw_data_root=os.environ.get("INFINITETALK_RAW_DATA_ROOT",
                                      "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/"),
        csv_path=os.environ.get("INFINITETALK_CSV_PATH",
                                 "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list.csv"),
        weights_dir=WEIGHTS_DIR,
        wav2vec_dir=os.environ.get("INFINITETALK_WAV2VEC_DIR",
                                    "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"),
        encode_device="cuda",
        num_workers=0,
    )
    return config
