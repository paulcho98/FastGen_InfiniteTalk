# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment config for InfiniteTalk Self-Forcing distillation (Stage 2).

14B teacher (bidirectional, frozen) -> 14B student (causal, LoRA) using Self-Forcing.
Also uses a 14B fake_score (bidirectional, LoRA) for VSD loss.

Unlike OmniAvatar (14B->1.3B), InfiniteTalk uses the same 14B architecture for
all three roles. LoRA adapters make training feasible:
  - Teacher:    base Wan I2V + InfiniteTalk ckpt + LoRA merged, fully frozen
  - Student:    causal variant with LoRA adapters (trainable)
  - Fake score: bidirectional with LoRA adapters (trainable)

3-call CFG with separate text (5.0) and audio (4.0) guidance scales.
"""

import os
import fastgen.configs.methods.config_infinitetalk_sf as config_sf_default

from fastgen.utils import LazyCall as L

from fastgen.networks.InfiniteTalk.network import InfiniteTalkWan
from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader
from fastgen.callbacks.infinitetalk_sf_wandb import InfiniteTalkSFWandbCallback
from fastgen.callbacks.wandb import WandbCallback
from fastgen.callbacks.train_profiler import TrainProfilerCallback
from fastgen.callbacks.grad_clip import GradClipCallback

# ---- Paths (override via CLI or env) ----
WEIGHTS_DIR = os.environ.get("INFINITETALK_WEIGHTS_DIR", "")
BASE_MODEL_PATHS = ",".join([
    f"{WEIGHTS_DIR}/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
    for i in range(1, 8)
])
INFINITETALK_CKPT = os.environ.get("INFINITETALK_CKPT", "")
TEACHER_LORA_CKPT = os.environ.get("INFINITETALK_TEACHER_LORA_CKPT", "")
STUDENT_LORA_CKPT = os.environ.get("INFINITETALK_STUDENT_LORA_CKPT", "")
DATA_ROOT = os.environ.get("INFINITETALK_DATA_ROOT", "")
NEG_TEXT_EMB = os.environ.get(
    "INFINITETALK_NEG_TEXT_EMB",
    "data/precomputed_talkvid/neg_text_embeds.pt",
)
AUDIO_DATA_ROOT = os.environ.get(
    "INFINITETALK_AUDIO_ROOT",
    "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data",
)

# Lazy caching paths (for samples with only text_embeds.pt)
CSV_PATH = os.environ.get(
    "INFINITETALK_CSV_PATH",
    "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/video_list_cleaned.csv",
)
RAW_DATA_ROOT = os.environ.get(
    "INFINITETALK_RAW_DATA_ROOT",
    "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/",
)
WAV2VEC_DIR = os.environ.get(
    "INFINITETALK_WAV2VEC_DIR",
    "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base",
)

# ---- Pretrained Stage 1 (DF) checkpoint ----
# Set by run_sf_training_quarter.sh (auto-detects latest), or override via env var.
PRETRAINED_DF_CKPT = os.environ.get("INFINITETALK_DF_CKPT", "")

# ---- Network configs ----

# Teacher: 14B bidirectional, frozen (base + InfiniteTalk ckpt + LoRA merged)
InfiniteTalk_14B_Teacher: dict = L(InfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path=TEACHER_LORA_CKPT,
    lora_rank=32,
    lora_alpha=32,
    apply_lora_adapters=False,  # Teacher: merge LoRA, freeze all
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)

# Fake score: 14B bidirectional with runtime LoRA adapters (trainable)
InfiniteTalk_14B_FakeScore: dict = L(InfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path="",  # No pre-trained LoRA for fake score
    lora_rank=128,
    lora_alpha=64,
    apply_lora_adapters=True,  # Fake score: runtime LoRA, trainable
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)

# Student: 14B causal with LoRA adapters (trainable)
# rank=128, alpha=64 must match the Stage 1 DF checkpoint being loaded
CausalInfiniteTalk_14B_Student: dict = L(CausalInfiniteTalkWan)(
    base_model_paths=BASE_MODEL_PATHS,
    infinitetalk_ckpt_path=INFINITETALK_CKPT,
    lora_ckpt_path=STUDENT_LORA_CKPT,
    lora_rank=128,
    lora_alpha=64,
    chunk_size=3,
    total_num_frames=21,
    local_attn_size=-1,   # -1 = attend to everything (default, safe)
    sink_size=0,
    use_dynamic_rope=False,
    net_pred_type="flow",
    schedule_type="rf",
    shift=7.0,
)


def create_config():
    config = config_sf_default.create_config()

    # Learning rates
    config.model.net_optimizer.lr = 1e-5
    config.model.net_optimizer.betas = (0.0, 0.999)  # No momentum (matches reference SF)
    config.model.fake_score_optimizer.lr = 2e-6
    config.model.fake_score_optimizer.betas = (0.0, 0.999)

    # Precision
    config.model.precision = "bfloat16"
    # precision_fsdp: parameter storage dtype for FSDP shards.
    # float32 is recommended for numerical stability, but bfloat16 halves CPU RAM
    # during init (3x 14B x 8 ranks). Use bfloat16 to avoid OOM on 1TB nodes.
    config.model.precision_fsdp = "bfloat16"

    # Input shape — overridden to quarter-res below (kept here as full-res reference)
    # Full res: [16, 21, 56, 112] (448x896), Quarter: [16, 21, 28, 56] (224x448)
    config.model.input_shape = [16, 21, 28, 56]
    config.model.fake_score_pred_type = "x0"

    # 3-call CFG with separate text and audio guidance scales
    # Note: guidance_scale is set to None to disable the base 2-call CFG path;
    # the 3-call CFG is handled entirely by InfiniteTalkSelfForcingModel._apply_classifier_free_guidance
    config.model.guidance_scale = 5.0  # Triggers CFG in base class; our override replaces the logic
    config.model.text_guide_scale = 5.0
    config.model.audio_guide_scale = 4.0

    # Networks: 14B teacher + 14B student (causal LoRA) + 14B fake_score (bidir LoRA)
    config.model.net = CausalInfiniteTalk_14B_Student
    config.model.net.total_num_frames = config.model.input_shape[1]
    config.model.teacher = InfiniteTalk_14B_Teacher
    config.model.fake_score_net = InfiniteTalk_14B_FakeScore

    # GAN disabled by default to save VRAM (matching T2V 14B teacher config pattern)
    config.model.gan_loss_weight_gen = 0
    # 1:5 ratio — critic updates every step, student every 5th (matching original SF paper)
    config.model.student_update_freq = 5

    # Student weights: let the network's own __init__ handle loading (base + ckpt + LoRA).
    # Do NOT copy teacher weights onto student (both are 14B but teacher is bidir, student is causal).
    config.model.load_student_weights = False

    # Timestep schedule -- shift=7.0 matches InfiniteTalk's 480p distribution
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.shift = 7.0
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999
    # t_list derived from shift=7.0: new_t = 7*t / (1 + 6*t) applied to linspace(1,0,5)
    config.model.sample_t_cfg.t_list = [0.999, 0.955, 0.875, 0.700, 0.0]

    # Self-Forcing specific
    config.model.enable_gradient_in_rollout = True
    config.model.start_gradient_frame = 0
    config.model.same_step_across_blocks = True
    config.model.context_noise = 0.0

    # ---- Training dataloader (with lazy caching, matching DF prod) ----
    train_list = os.environ.get(
        "INFINITETALK_TRAIN_LIST",
        "data/precomputed_talkvid/train_excl_val30.txt",
    )
    config.dataloader_train = L(InfiniteTalkDataLoader)(
        data_list_path=train_list,
        neg_text_emb_path=NEG_TEXT_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=[16, 21, 28, 56],
        num_video_frames=93,
        num_latent_frames=21,
        quarter_res=True,
        # Lazy caching: encode VAE/CLIP/audio on-the-fly for samples with only text_embeds.pt
        raw_data_root=RAW_DATA_ROOT,
        csv_path=CSV_PATH,
        weights_dir=WEIGHTS_DIR,
        wav2vec_dir=WAV2VEC_DIR,
        encode_device="cuda",
        num_workers=0,  # required for lazy caching (encoders can't cross process boundaries)
    )

    # ---- Validation dataloader (matching DF quarter prod) ----
    val_list = os.environ.get(
        "INFINITETALK_VAL_LIST",
        "data/precomputed_talkvid/val_quarter_30.txt",
    )
    config.dataloader_val = L(InfiniteTalkDataLoader)(
        data_list_path=val_list,
        neg_text_emb_path=NEG_TEXT_EMB,
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=[16, 21, 28, 56],
        num_video_frames=93,
        num_latent_frames=21,
        audio_data_root=AUDIO_DATA_ROOT,
        quarter_res=True,
        num_workers=0,
    )

    # ---- FSDP (required for 3x 14B models) ----
    config.trainer.fsdp = True
    config.model.fsdp_meta_init = False
    # Sequential init: build each 14B model → FSDP-shard to GPU → free CPU → build next.
    # Peak CPU RAM: ~1 model × 8 ranks ≈ 310 GB (vs ~900 GB if all built simultaneously).
    config.model.fsdp_sequential_init = True

    # ---- Gradient accumulation ----
    # BS=1/GPU × 4 accum = effective BS 4*N_GPUs
    config.trainer.grad_accum_rounds = 4
    # Mirror to model config for combined step loss scaling
    config.model.grad_accum_rounds = 4

    # ---- Pretrained Stage 1 (DF) checkpoint ----
    config.trainer.checkpointer.pretrained_ckpt_path = PRETRAINED_DF_CKPT
    config.trainer.checkpointer.pretrained_ckpt_key_map = {"net": "net"}

    # ---- Training schedule ----
    config.trainer.max_iter = 5000
    config.trainer.logging_iter = 1
    config.trainer.save_ckpt_iter = 100
    config.trainer.validation_iter = 100
    # Skip validation at iteration 0 to avoid FlexAttention torch.compile hang
    config.trainer.skip_iter0_validation = True
    config.trainer.global_vars_val = [{"MAX_VAL_STEPS": 10}]

    # ---- Callbacks ----
    # InfiniteTalkSFWandbCallback handles wandb.init (via on_app_begin) + loss logging
    # + deferred validation video generation (DDP-safe).
    # Do NOT include the base WandbCallback — it tries to call gen_rand()/VAE decode
    # during on_training_step_end which crashes (WanVAE is not nn.Module).
    config.trainer.callbacks = {
        "infinitetalk_sf_wandb": L(InfiniteTalkSFWandbCallback)(
            sample_logging_iter=999999,    # val handles visuals
            validation_logging_step=1,     # visual for each val sample
            audio_fps=25,
        ),
        "train_profiler": L(TrainProfilerCallback)(every_n=100),
        "grad_clip": L(GradClipCallback)(grad_norm=10.0, model_key="net"),
    }

    # ---- Student anchor mode ----
    # Default: anchor always (I2V mode). Override via STUDENT_ANCHOR_EVAL_ONLY env var.
    config.model.student_anchor_eval_only = bool(os.environ.get("STUDENT_ANCHOR_EVAL_ONLY", ""))

    # ---- Fake score anchor mode ----
    # Default: anchor always. Override via FAKE_SCORE_ANCHOR_EVAL_ONLY env var.
    # When set, fake_score uses soft conditioning during training (no anchor),
    # matching the student distribution. Teacher always anchors (target distribution).
    config.model.fake_score_anchor_eval_only = bool(os.environ.get("FAKE_SCORE_ANCHOR_EVAL_ONLY", ""))

    # ---- Logging ----
    config.log_config.project = "SF_InfiniteTalk"
    config.log_config.group = "infinitetalk_sf"
    import time
    timestamp = time.strftime("%m%d_%H%M")
    run_name = os.environ.get("FASTGEN_RUN_NAME", "")
    if not run_name:
        if config.model.student_anchor_eval_only and config.model.fake_score_anchor_eval_only:
            anchor_tag = "softboth"
        elif config.model.student_anchor_eval_only:
            anchor_tag = "softanchor"
        else:
            anchor_tag = "i2v"
        run_name = f"sf_quarter_{anchor_tag}_freq5_lr1e5_accum4_{timestamp}"
    config.log_config.name = run_name
    config.log_config.wandb_mode = "online"

    return config
