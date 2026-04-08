#!/usr/bin/env python3
"""Sequential batch: consolidate checkpoints, precompute Option B HDTF, run inference.

Order:
  1. Consolidate soft-anchor checkpoints (100, 200, 300)
  2. Precompute Option B (shrink-square) HDTF conditioning
  3. Inference: Option B HDTF with I2V iter 400
  4. Inference: TalkVid I2V iters 700, 800, 900
  5. Inference: TalkVid soft-anchor iters 100, 200, 300
"""

import gc
import os
import sys
import subprocess
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTGEN_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(FASTGEN_ROOT)
sys.path.insert(0, FASTGEN_ROOT)

# ── Paths ──
WEIGHTS_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
IT_CKPT = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
VAE_PATH = f"{WEIGHTS_DIR}/Wan2.1_VAE.pth"
AUDIO_ROOT = "/data/karlo-research_715/workspace/kinemaar/paul/datasets/TalkVid/data"
CLIP_PATH = f"{WEIGHTS_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
WAV2VEC_PATH = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/chinese-wav2vec2-base"

BASE_SHARDS = ",".join(sorted([
    os.path.join(WEIGHTS_DIR, f)
    for f in os.listdir(WEIGHTS_DIR)
    if f.startswith("diffusion_pytorch_model-") and f.endswith(".safetensors")
]))

I2V_CKPT_DIR = os.path.join(
    FASTGEN_ROOT,
    "FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/"
    "sf_quarter_i2v_freq5_lr1e5_accum4_0407_0100/checkpoints",
)
SOFTANCHOR_CKPT_DIR = os.path.join(
    FASTGEN_ROOT,
    "FASTGEN_OUTPUT/SF_InfiniteTalk/infinitetalk_sf/"
    "sf_quarter_softanchor_freq5_lr1e5_accum4_0408_0034/checkpoints",
)

HDTF_VIDEO_DIR = "/data/karlo-research_715/workspace/kinemaar/datasets/HDTF_original_testset_81frames/videos_cfr"
TALKVID_LIST = "data/precomputed_talkvid/val_quarter_30.txt"
HDTF_SHRINK_DIR = "data/precomputed_hdtf_quarter_shrink_square"

TARGET_H, TARGET_W = 224, 448


# =====================================================================
# Step 1: Consolidate checkpoints
# =====================================================================

def consolidate(ckpt_dir, iters):
    """Consolidate FSDP sharded checkpoints to single .pth files."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    for it in iters:
        src = os.path.join(ckpt_dir, f"{it:07d}.net_model")
        dst = os.path.join(ckpt_dir, f"{it:07d}_net_consolidated.pth")
        if os.path.exists(dst):
            # Verify it's valid
            try:
                sd = torch.load(dst, map_location="cpu", weights_only=False)
                print(f"  iter {it}: already consolidated ({len(sd)} keys), skip")
                del sd
                gc.collect()
                continue
            except Exception:
                print(f"  iter {it}: corrupt file, reconsolidating...")
                os.remove(dst)

        if not os.path.isdir(src):
            print(f"  iter {it}: shard dir not found, skip")
            continue

        print(f"  iter {it}: consolidating...")
        dcp_to_torch_save(src, dst)
        sd = torch.load(dst, map_location="cpu", weights_only=False)
        lora_keys = [k for k in sd if "lora" in k]
        print(f"    -> {len(sd)} tensors, {len(lora_keys)} LoRA keys")
        del sd
        gc.collect()


# =====================================================================
# Step 2: Precompute Option B (shrink-square) HDTF conditioning
# =====================================================================

def precompute_hdtf_shrink_square():
    """Precompute first_frame_cond + clip_features for shrink-square HDTF."""
    from fastgen.datasets.infinitetalk_dataloader import _add_infinitetalk_to_path
    _add_infinitetalk_to_path()

    from scripts.inference.inference_causal import (
        load_vae, load_clip, load_wav2vec,
        encode_clip, encode_audio_from_file,
        extract_first_frame_and_audio,
        compute_generation_length,
    )
    from PIL import Image
    import torchvision.transforms as T

    os.makedirs(HDTF_SHRINK_DIR, exist_ok=True)

    video_files = sorted([
        os.path.join(HDTF_VIDEO_DIR, f)
        for f in os.listdir(HDTF_VIDEO_DIR)
        if f.endswith(".mp4")
    ])
    print(f"  Found {len(video_files)} videos")

    # Load encoders
    print("  Loading VAE...")
    vae = load_vae(VAE_PATH, device="cpu")
    print("  Loading CLIP...")
    clip_model = load_clip(CLIP_PATH, device="cuda")
    print("  Loading wav2vec2...")
    wav2vec_fe, wav2vec_model = load_wav2vec(WAV2VEC_PATH, device="cuda")

    sample_list = []

    for vi, vpath in enumerate(video_files):
        vname = os.path.splitext(os.path.basename(vpath))[0]
        sample_dir = os.path.join(HDTF_SHRINK_DIR, vname)

        if (os.path.isfile(os.path.join(sample_dir, "first_frame_cond.pt"))
                and os.path.isfile(os.path.join(sample_dir, "clip_features.pt"))
                and os.path.isfile(os.path.join(sample_dir, "audio_emb.pt"))):
            print(f"  [{vi+1}/{len(video_files)}] {vname}: exists, skip")
            sample_list.append(sample_dir)
            continue

        print(f"  [{vi+1}/{len(video_files)}] Encoding {vname}...")
        os.makedirs(sample_dir, exist_ok=True)

        try:
            pil_image, audio_path = extract_first_frame_and_audio(vpath)
            num_latent, num_video = compute_generation_length(audio_path, 21, 3, 25)

            # Option B: shrink original square to 112x112, center-pad to 224x448
            orig_w, orig_h = pil_image.size
            small_size = min(orig_w, orig_h) // (min(orig_w, orig_h) // 112)
            # Simple: resize to 112x112 (or proportional if not square)
            small_h = orig_h * 112 // max(orig_w, orig_h)
            small_w = orig_w * 112 // max(orig_w, orig_h)
            # For square videos: 112x112. For landscape: 112xN or Nx112
            small_h = max(small_h, 1)
            small_w = max(small_w, 1)
            small_img = pil_image.resize((small_w, small_h), Image.LANCZOS)
            padded_img = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
            paste_x = (TARGET_W - small_w) // 2
            paste_y = (TARGET_H - small_h) // 2
            padded_img.paste(small_img, (paste_x, paste_y))

            # Save preview
            padded_img.save(os.path.join(sample_dir, "preview.png"))

            # VAE encode the padded frame
            transform = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
            img_tensor = transform(padded_img)  # [3, 224, 448]
            num_video_frames = 1 + (num_latent - 1) * 4
            video_padded = torch.zeros(1, 3, num_video_frames, TARGET_H, TARGET_W, dtype=torch.float32)
            video_padded[0, :, 0] = img_tensor

            vae.model = vae.model.to("cuda")
            vae.mean = vae.mean.to("cuda")
            vae.std = vae.std.to("cuda")
            vae.scale = [vae.mean, 1.0 / vae.std]
            vae_dtype = getattr(vae, "dtype", None) or next(vae.model.parameters()).dtype
            video_for_vae = video_padded[0].to(device="cuda", dtype=vae_dtype)

            with torch.no_grad():
                latent = vae.encode([video_for_vae])
            if isinstance(latent, (list, tuple)):
                latent = torch.stack(latent)
            ffc = latent.to(torch.bfloat16).cpu()
            if ffc.dim() == 4:
                ffc = ffc.unsqueeze(0)
            ffc = ffc[:, :, :num_latent]
            torch.save(ffc, os.path.join(sample_dir, "first_frame_cond.pt"))

            # CLIP encode the padded frame
            cf = encode_clip(clip_model, padded_img, device="cuda",
                             target_h=TARGET_H, target_w=TARGET_W)
            torch.save(cf.cpu(), os.path.join(sample_dir, "clip_features.pt"))

            # Audio encode
            ae = encode_audio_from_file(
                wav2vec_fe, wav2vec_model, audio_path, num_video, device="cuda")
            torch.save(ae.cpu(), os.path.join(sample_dir, "audio_emb.pt"))

            # Text (zeros) + negative text
            text_embeds = torch.zeros(1, 1, 512, 4096, dtype=torch.bfloat16)
            torch.save(text_embeds, os.path.join(sample_dir, "text_embeds.pt"))
            neg_text_embeds = torch.zeros(1, 1, 512, 4096, dtype=torch.bfloat16)
            torch.save(neg_text_embeds, os.path.join(sample_dir, "neg_text_embeds.pt"))

            # Save audio path for muxing
            with open(os.path.join(sample_dir, "audio_path.txt"), "w") as f:
                f.write(audio_path)
            os.chmod(os.path.join(sample_dir, "audio_path.txt"), 0o644)

            sample_list.append(sample_dir)
            print(f"    ffc: {list(ffc.shape)}, clip: {list(cf.shape)}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        torch.cuda.empty_cache()

    # Write sample list
    list_path = os.path.join(HDTF_SHRINK_DIR, "sample_list.txt")
    with open(list_path, "w") as f:
        for sd in sample_list:
            f.write(sd + "\n")
    print(f"  Done: {len(sample_list)} samples. List: {list_path}")

    # Free encoders
    del vae, clip_model, wav2vec_fe, wav2vec_model
    torch.cuda.empty_cache()
    gc.collect()


# =====================================================================
# Step 3: Run inference
# =====================================================================

def run_inference(precomputed_list, output_dir, ckpt_path, quarter_res=True,
                  no_anchor=False, extra_args=None):
    """Run inference_causal.py as subprocess."""
    cmd = [
        sys.executable, "-u", "scripts/inference/inference_causal.py",
        "--precomputed_list", precomputed_list,
        "--output_dir", output_dir,
        "--ckpt_path", ckpt_path,
        "--base_model_paths", BASE_SHARDS,
        "--infinitetalk_ckpt", IT_CKPT,
        "--vae_path", VAE_PATH,
        "--audio_data_root", AUDIO_ROOT,
        "--lora_rank", "128", "--lora_alpha", "64",
        "--chunk_size", "3", "--num_latent_frames", "21",
        "--seed", "42", "--context_noise", "0.0",
    ]
    if quarter_res:
        cmd.append("--quarter_res")
    if no_anchor:
        cmd.append("--no_anchor_first_frame")
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Command: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, cwd=FASTGEN_ROOT)
    if result.returncode != 0:
        print(f"  WARNING: inference exited with code {result.returncode}")
    return result.returncode


# =====================================================================
# Main
# =====================================================================

def main():
    # ── 1. Consolidate soft-anchor checkpoints ──
    print("\n" + "=" * 60)
    print("STEP 1: Consolidate soft-anchor checkpoints")
    print("=" * 60)
    consolidate(SOFTANCHOR_CKPT_DIR, [100, 200, 300])

    # Also verify I2V 700/800/900 are consolidated
    print("\n  Verifying I2V checkpoints...")
    consolidate(I2V_CKPT_DIR, [700, 800, 900])

    # ── 2. Precompute Option B HDTF ──
    print("\n" + "=" * 60)
    print("STEP 2: Precompute shrink-square HDTF conditioning")
    print("=" * 60)
    precompute_hdtf_shrink_square()

    # ── 3. Inference: Option B HDTF with I2V iter 400 ──
    print("\n" + "=" * 60)
    print("STEP 3: Inference — HDTF shrink-square, I2V iter 400")
    print("=" * 60)
    i2v_400_ckpt = os.path.join(I2V_CKPT_DIR, "0000400_net_consolidated.pth")
    hdtf_shrink_list = os.path.join(HDTF_SHRINK_DIR, "sample_list.txt")
    run_inference(
        hdtf_shrink_list,
        "EVAL_OUTPUT/hdtf_shrink_square_sf_i2v/iter_0000400",
        i2v_400_ckpt,
    )

    # ── 4. Inference: TalkVid I2V iters 700, 800, 900 ──
    for it in [700, 800, 900]:
        print("\n" + "=" * 60)
        print(f"STEP 4.{it//100}: Inference — TalkVid I2V iter {it}")
        print("=" * 60)
        ckpt = os.path.join(I2V_CKPT_DIR, f"{it:07d}_net_consolidated.pth")
        if not os.path.isfile(ckpt):
            print(f"  SKIP: {ckpt} not found")
            continue
        run_inference(
            TALKVID_LIST,
            f"EVAL_OUTPUT/talkvid_sf_i2v/iter_{it:07d}",
            ckpt,
        )

    # ── 5. Inference: TalkVid soft-anchor iters 100, 200, 300 ──
    for it in [100, 200, 300]:
        print("\n" + "=" * 60)
        print(f"STEP 5.{it//100}: Inference — TalkVid soft-anchor iter {it}")
        print("=" * 60)
        ckpt = os.path.join(SOFTANCHOR_CKPT_DIR, f"{it:07d}_net_consolidated.pth")
        if not os.path.isfile(ckpt):
            print(f"  SKIP: {ckpt} not found")
            continue
        run_inference(
            TALKVID_LIST,
            f"EVAL_OUTPUT/talkvid_sf_softanchor/iter_{it:07d}",
            ckpt,
        )

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
