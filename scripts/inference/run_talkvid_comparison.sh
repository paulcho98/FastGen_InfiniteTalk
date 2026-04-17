#!/bin/bash
# Run DF (no anchor) and SF (soft cond, output anchor) on TalkVid val set
# for comparison against training validation logs.
set -e

WEIGHTS_DIR="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
BASE_SHARDS=$(ls "${WEIGHTS_DIR}"/diffusion_pytorch_model-*.safetensors | tr '\n' ',' | sed 's/,$//')
IT_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH="${WEIGHTS_DIR}/Wan2.1_VAE.pth"
AUDIO_ROOT="/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data"

COMMON_ARGS="--precomputed_list data/precomputed_talkvid/val_quarter_30.txt \
    --base_model_paths $BASE_SHARDS \
    --infinitetalk_ckpt $IT_CKPT \
    --vae_path $VAE_PATH \
    --audio_data_root $AUDIO_ROOT \
    --lora_rank 128 --lora_alpha 64 \
    --chunk_size 3 --num_latent_frames 21 \
    --seed 42 --context_noise 0.0 \
    --quarter_res"

DF_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/FASTGEN_OUTPUT/DF_InfiniteTalk/infinitetalk_df_quarter/quarter_r128_bs4_accum1_8gpu_0402_0836/checkpoints/0004300.pth"
SF_CKPT="/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/FastGen_InfiniteTalk/FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/sf_quarter_freq5_lr1e5_accum4_0406_0833/checkpoints/0000400_net_consolidated.pth"

echo "============================================"
echo "TalkVid Val Comparison"
echo "============================================"

# Run 1: DF no anchor
echo ""
echo ">>> Run 1: DF iter 4300, NO anchoring"
python -u -c "
import sys, os
sys.path.insert(0, '.')
import fastgen.networks.InfiniteTalk.network_causal as nc
_orig = nc.CausalInfiniteTalkWan.__init__
def _p(self, *a, **kw):
    _orig(self, *a, **kw)
    self._enable_first_frame_anchor = False
nc.CausalInfiniteTalkWan.__init__ = _p
from scripts.inference.inference_causal import main
sys.argv = ['inference_causal.py',
    '--precomputed_list', 'data/precomputed_talkvid/val_quarter_30.txt',
    '--output_dir', 'EVAL_OUTPUT/talkvid_df_no_anchor/iter_0004300',
    '--ckpt_path', os.environ['DF_CKPT'],
    '--base_model_paths', os.environ['BASE_SHARDS'],
    '--infinitetalk_ckpt', os.environ['IT_CKPT'],
    '--vae_path', os.environ['VAE_PATH'],
    '--audio_data_root', os.environ['AUDIO_ROOT'],
    '--lora_rank', '128', '--lora_alpha', '64',
    '--chunk_size', '3', '--num_latent_frames', '21',
    '--seed', '42', '--context_noise', '0.0',
    '--quarter_res', '--no_anchor_first_frame',
]
main()
"
echo ">>> Run 1 done"

# Run 2: SF soft conditioning with output anchor
echo ""
echo ">>> Run 2: SF soft cond iter 400, output anchoring"
python -u scripts/inference/inference_causal.py \
    --precomputed_list data/precomputed_talkvid/val_quarter_30.txt \
    --output_dir EVAL_OUTPUT/talkvid_sf_soft_cond/iter_0000400 \
    --ckpt_path "$SF_CKPT" \
    $COMMON_ARGS

echo ""
echo ">>> All runs complete"
echo "Compare:"
echo "  EVAL_OUTPUT/talkvid_df_no_anchor/iter_0004300/  (DF, no anchor)"
echo "  EVAL_OUTPUT/talkvid_sf_soft_cond/iter_0000400/  (SF soft cond, output anchor)"
