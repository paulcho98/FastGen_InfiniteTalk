#!/usr/bin/env python3
"""Compare precomputed conditions against ground truth from original pipeline.

Uses InfiniteTalk's exact preprocessing functions to generate ground truth
conditions from a TalkVid video, then compares tensor-by-tensor against
our precomputed versions.

Usage:
    python scripts/compare_conditions_with_original.py \
        --video_path /path/to/video.mp4 \
        --audio_path /path/to/audio.wav \
        --text "prompt text" \
        --precomputed_dir data/test_precomputed/sample_name/ \
        --weights_dir /path/to/Wan2.1-I2V-14B-480P \
        --infinitetalk_ckpt /path/to/infinitetalk.safetensors \
        --wav2vec_dir /path/to/chinese-wav2vec2-base \
        --device cuda:0
"""

import sys
import os
import math
import inspect
import json
import argparse

# Python 3.12 compat: ArgSpec removed
if not hasattr(inspect, 'ArgSpec'):
    inspect.ArgSpec = inspect.FullArgSpec

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Add InfiniteTalk to path
IT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "InfiniteTalk")
sys.path.insert(0, IT_ROOT)

# Mock xformers and other heavy deps BEFORE any wan imports
import types
import importlib.machinery

def _ensure_mock(module_path):
    """Create a mock module at module_path if absent."""
    parts = module_path.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[:i + 1])
        if partial not in sys.modules:
            mod = types.ModuleType(partial)
            if i < len(parts) - 1:
                mod.__path__ = []
            mod.__spec__ = importlib.machinery.ModuleSpec(partial, None)
            sys.modules[partial] = mod

class _MockModule(types.ModuleType):
    """A mock module that returns dummy callables for unknown attributes."""
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None

for mock_pkg in [
    "xformers", "xformers.ops",
    "xfuser", "xfuser.core", "xfuser.core.distributed",
    "sageattn",
    "decord",
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

# Provide dummy callables that diffusers' xformers check expects
sys.modules["xformers"].ops = sys.modules["xformers.ops"]

def _memory_efficient_attention(q, k, v, attn_bias=None, op=None):
    """Drop-in for xformers MEA using flash_attn_func. [B, M, H, K] layout."""
    from flash_attn import flash_attn_func
    return flash_attn_func(q, k, v)

sys.modules["xformers.ops"].memory_efficient_attention = _memory_efficient_attention

# Mock src.vram_management (but NOT src itself — it's a real namespace package
# containing audio_analysis which we need)
vm_mock = _MockModule("src.vram_management")
vm_mock.__spec__ = importlib.machinery.ModuleSpec("src.vram_management", None)
vm_mock.AutoWrappedQLinear = None
vm_mock.AutoWrappedLinear = None
vm_mock.AutoWrappedModule = None
vm_mock.enable_vram_management = lambda *a, **k: None
sys.modules["src.vram_management"] = vm_mock


def compare_tensors(name, ours, theirs):
    """Compare two tensors and print diagnostics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Our shape:    {ours.shape}  dtype={ours.dtype}")
    print(f"  GT shape:     {theirs.shape}  dtype={theirs.dtype}")

    if ours.shape != theirs.shape:
        print(f"  *** SHAPE MISMATCH ***")
        return False

    # Cast to float32 for comparison
    a = ours.float()
    b = theirs.float()

    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_diff = (diff / (b.abs() + 1e-8)).mean().item()

    print(f"  Our range:    [{a.min().item():.4f}, {a.max().item():.4f}], std={a.std().item():.4f}")
    print(f"  GT range:     [{b.min().item():.4f}, {b.max().item():.4f}], std={b.std().item():.4f}")
    print(f"  Max diff:     {max_diff:.6f}")
    print(f"  Mean diff:    {mean_diff:.6f}")
    print(f"  Rel diff:     {rel_diff:.6f}")

    match = max_diff < 1e-3
    if match:
        print(f"  MATCH (max_diff < 1e-3)")
    else:
        # Check if it's just a scale/offset issue
        corr = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
        print(f"  MISMATCH (max_diff={max_diff:.4f})")
        print(f"  Correlation:  {corr:.6f}")

        # Show where biggest differences are
        flat_diff = diff.flatten()
        top_k = min(5, flat_diff.numel())
        top_vals, top_idx = flat_diff.topk(top_k)
        print(f"  Top-{top_k} diffs: {[f'{v:.4f}' for v in top_vals.tolist()]}")

    return match


def resize_and_centercrop_original(cond_image, target_size):
    """Exact copy of resize_and_centercrop from multitalk.py (PIL path)."""
    if isinstance(cond_image, torch.Tensor):
        _, orig_h, orig_w = cond_image.shape
    else:
        orig_h, orig_w = cond_image.height, cond_image.width

    target_h, target_w = target_size
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    scale = max(scale_h, scale_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)

    if isinstance(cond_image, torch.Tensor):
        if len(cond_image.shape) == 3:
            cond_image = cond_image[None]
        resized_tensor = F.interpolate(
            cond_image, size=(final_h, final_w), mode='nearest'
        ).contiguous()
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor.squeeze(0)
    else:
        resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
        resized_image = np.array(resized_image)
        resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
        cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
        cropped_tensor = cropped_tensor[:, :, None, :, :]

    return cropped_tensor


def main():
    parser = argparse.ArgumentParser(
        description="Compare precomputed conditions with original pipeline ground truth")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--precomputed_dir", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--infinitetalk_ckpt", type=str, default=None)
    parser.add_argument("--wav2vec_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--frame_count", type=int, default=81)
    parser.add_argument("--neg_prompt", type=str, default=(
        "bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low "
        "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards"
    ))
    args = parser.parse_args()

    device = torch.device(args.device)
    target_h = args.resolution
    target_w = args.resolution

    results = {}

    # ====================================================================
    # Step 1: Image preprocessing comparison
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Image preprocessing (no encoder needed)")
    print("=" * 70)

    # Original: extract_specific_frames uses decord to read frame 0 as PIL Image.
    # We use av (decord not installed) — both produce identical RGB arrays.
    import av
    container = av.open(args.video_path)
    for frame_av in container.decode(video=0):
        frame = frame_av.to_ndarray(format="rgb24")
        break
    container.close()
    cond_image_pil = Image.fromarray(frame)

    print(f"  Original frame size: {cond_image_pil.size} (WxH)")

    # Original pipeline processing (PIL path):
    # resize_and_centercrop returns [1, C, 1, H, W] for PIL input
    cond_image_orig = resize_and_centercrop_original(cond_image_pil, (target_h, target_w))
    cond_image_orig = cond_image_orig / 255.0
    cond_image_orig = (cond_image_orig - 0.5) * 2.0
    # cond_image_orig: [1, C, 1, H, W]
    print(f"  Original cond_image shape: {cond_image_orig.shape}")
    print(f"  Original cond_image range: [{cond_image_orig.min():.4f}, {cond_image_orig.max():.4f}]")

    # Our preprocessing: load all frames, resize with bilinear, take first
    import av
    container = av.open(args.video_path)
    frames_list = []
    for i, frame_av in enumerate(container.decode(video=0)):
        if i >= args.frame_count:
            break
        frames_list.append(frame_av.to_ndarray(format="rgb24"))
    container.close()
    frames = np.stack(frames_list, axis=0)

    # Our resize_and_center_crop (from precompute script)
    t_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
    _, _, h, w = t_tensor.shape
    scale_h = target_h / h
    scale_w = target_w / w
    scale = max(scale_h, scale_w)
    new_h = math.ceil(scale * h)
    new_w = math.ceil(scale * w)
    t_tensor = F.interpolate(t_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
    crop_top = (new_h - target_h) // 2
    crop_left = (new_w - target_w) // 2
    t_tensor = t_tensor[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]
    t_tensor = (t_tensor / 255.0 - 0.5) * 2.0
    our_video_tensor = t_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]

    # Compare first frame
    our_first_frame = our_video_tensor[:, 0:1, :, :]  # [C, 1, H, W]
    orig_first_frame = cond_image_orig[0, :, :, :, :]  # [C, 1, H, W] from [1,C,1,H,W]

    print(f"\n  Comparing first frame preprocessing:")
    print(f"  Our first frame: shape={our_first_frame.shape}, range=[{our_first_frame.min():.4f}, {our_first_frame.max():.4f}]")
    print(f"  Orig first frame: shape={orig_first_frame.shape}, range=[{orig_first_frame.min():.4f}, {orig_first_frame.max():.4f}]")
    diff = (our_first_frame.float() - orig_first_frame.float()).abs()
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")
    if diff.max().item() > 1e-3:
        print(f"  *** IMAGE PREPROCESSING DIFFERS ***")
        print(f"  This is expected: original uses PIL BILINEAR resize then center_crop")
        print(f"  Our precompute uses torch F.interpolate(mode='bilinear', align_corners=False)")
        results["image_preprocess"] = False
    else:
        print(f"  Image preprocessing MATCHES")
        results["image_preprocess"] = True

    # ====================================================================
    # Step 2: Load encoders
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Loading encoders")
    print("=" * 70)

    # --- VAE ---
    print("  Loading VAE...")
    from wan.modules.vae import WanVAE
    vae_pth = os.path.join(args.weights_dir, "Wan2.1_VAE.pth")
    vae = WanVAE(vae_pth=vae_pth, device=device)

    # --- CLIP ---
    print("  Loading CLIP...")
    from wan.modules.clip import CLIPModel
    clip_model = CLIPModel(
        dtype=torch.float16,
        device=device,
        checkpoint_path=os.path.join(args.weights_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        tokenizer_path=os.path.join(args.weights_dir, "xlm-roberta-large"),
    )

    # --- T5 ---
    print("  Loading T5...")
    from wan.modules.t5 import T5EncoderModel
    t5_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.weights_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.weights_dir, "google", "umt5-xxl"),
    )

    # --- wav2vec2 ---
    print("  Loading wav2vec2...")
    from transformers import Wav2Vec2FeatureExtractor
    from src.audio_analysis.wav2vec2 import Wav2Vec2Model
    audio_encoder = Wav2Vec2Model.from_pretrained(
        args.wav2vec_dir, local_files_only=True, attn_implementation="eager"
    )
    audio_encoder = audio_encoder.eval()
    wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir, local_files_only=True)

    # ====================================================================
    # Step 3: Generate ground truth conditions (original pipeline)
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Generating ground truth conditions")
    print("=" * 70)

    param_dtype = torch.bfloat16
    cond_image_device = cond_image_orig.to(device)

    # --- CLIP (original way) ---
    print("  CLIP encoding (original)...")
    clip_model.model.to(device)
    with torch.no_grad():
        # Original: clip.visual(cond_image[:, :, -1:, :, :])
        # cond_image is [1, C, 1, H, W]
        gt_clip = clip_model.visual(cond_image_device[:, :, -1:, :, :]).to(param_dtype)
    gt_clip_cpu = gt_clip.cpu()
    clip_model.model.cpu()
    torch.cuda.empty_cache()
    print(f"  GT CLIP shape: {gt_clip_cpu.shape}")

    # --- VAE first_frame_cond (original way) ---
    print("  VAE encoding (original)...")
    with torch.no_grad():
        # Original: torch.concat([cond_image, video_frames], dim=2) then vae.encode()
        # cond_image: [1, C, 1, H, W]
        frame_num = args.frame_count
        video_frames = torch.zeros(
            1, cond_image_device.shape[1], frame_num - cond_image_device.shape[2],
            target_h, target_w, device=device
        )
        padding_frames = torch.cat([cond_image_device, video_frames], dim=2)
        # padding_frames: [1, C, frame_num, H, W] → need [C, T, H, W] for vae
        gt_first_frame_cond = vae.encode(padding_frames)
        gt_first_frame_cond = torch.stack(gt_first_frame_cond).to(param_dtype)
        # gt_first_frame_cond: [1, C_lat, T_lat, H_lat, W_lat]
    gt_first_frame_cond_cpu = gt_first_frame_cond[0].cpu()
    print(f"  GT first_frame_cond shape: {gt_first_frame_cond_cpu.shape}")

    # Also encode full video the original way
    print("  VAE encoding full video (our preprocessing for comparison)...")
    with torch.no_grad():
        # Our way: video_tensor [C, T, H, W] → vae.encode([video_tensor])
        our_vid = our_video_tensor.to(device)
        gt_vae_from_our_preprocess = vae.encode([our_vid])
        gt_vae_from_our_preprocess = gt_vae_from_our_preprocess[0].cpu()

    # Also encode first frame cond our way for comparison
    with torch.no_grad():
        c, t, h_v, w_v = our_video_tensor.shape
        first_frame_ours = our_video_tensor[:, :1, :, :].to(device)
        padding_ours = torch.zeros(c, t - 1, h_v, w_v, device=device)
        first_frame_padded_ours = torch.cat([first_frame_ours, padding_ours], dim=1)
        gt_ffc_our_way = vae.encode([first_frame_padded_ours])
        gt_ffc_our_way = gt_ffc_our_way[0].cpu()

    # Now encode first frame the EXACT original way (using original-preprocessed image)
    # Compare VAE encode of original vs our first frame
    with torch.no_grad():
        # Original first frame: cond_image_device is [1, C, 1, H, W]
        orig_ff = cond_image_device  # [1, C, 1, H, W]
        orig_padding = torch.zeros(
            1, orig_ff.shape[1], frame_num - 1, target_h, target_w, device=device
        )
        orig_ff_padded = torch.cat([orig_ff, orig_padding], dim=2)  # [1, C, 81, H, W]
        # vae.encode expects list of [C, T, H, W] or a [B, C, T, H, W]
        gt_ffc_orig = vae.encode(orig_ff_padded)
        gt_ffc_orig = torch.stack(gt_ffc_orig).to(param_dtype)
        gt_ffc_orig_cpu = gt_ffc_orig[0].cpu()

    torch.cuda.empty_cache()

    # --- T5 text (original way) ---
    print("  T5 encoding (original)...")
    with torch.no_grad():
        # Original: self.text_encoder([input_prompt, n_prompt], self.device)
        # Returns list of 2 tensors
        t5_encoder.model.to(device)
        context_list = t5_encoder([args.text, args.neg_prompt], device)
        gt_text = context_list[0].unsqueeze(0).cpu()  # [1, seq_len, 4096]
        gt_neg_text = context_list[1].unsqueeze(0).cpu()
        t5_encoder.model.cpu()
    print(f"  GT text_embeds shape: {gt_text.shape}")
    print(f"  GT neg_text shape: {gt_neg_text.shape}")

    # But wait - original pipeline gets TRIMMED embeddings from T5EncoderModel.__call__
    # Our precompute uses t5_encoder.model(ids, mask) which returns FULL padded [1,512,4096]
    # Let's also get the padded version
    with torch.no_grad():
        t5_encoder.model.to(device)
        ids, mask = t5_encoder.tokenizer([args.text], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        gt_text_padded = t5_encoder.model(ids, mask).float().cpu()  # [1, 512, 4096]

        ids_neg, mask_neg = t5_encoder.tokenizer([args.neg_prompt], return_mask=True, add_special_tokens=True)
        ids_neg = ids_neg.to(device)
        mask_neg = mask_neg.to(device)
        gt_neg_text_padded = t5_encoder.model(ids_neg, mask_neg).float().cpu()  # [1, 512, 4096]
        t5_encoder.model.cpu()
    print(f"  GT text_embeds (padded) shape: {gt_text_padded.shape}")

    # --- Audio (original way) ---
    print("  Audio encoding (original)...")
    import librosa
    import pyloudnorm as pyln

    speech, sr = librosa.load(args.audio_path, sr=16000)
    # Original loudness_norm:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(speech)
    if abs(loudness) <= 100:
        speech = pyln.normalize.loudness(speech, loudness, -23.0)

    audio_duration = len(speech) / sr
    video_length = audio_duration * 25

    audio_feature = np.squeeze(wav2vec_fe(speech, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0)

    with torch.no_grad():
        embeddings = audio_encoder(
            audio_feature, seq_len=int(video_length), output_hidden_states=True
        )
    from einops import rearrange
    gt_audio = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    gt_audio = rearrange(gt_audio, "b s d -> s b d")
    gt_audio_cpu = gt_audio.cpu()
    print(f"  GT audio shape: {gt_audio_cpu.shape}")

    # Now the original pipeline uses SLIDING WINDOW on audio in the generate loop
    # Lines 522-533 of multitalk.py
    # This windowing happens AFTER the audio embeddings are loaded
    # The raw audio_emb we precompute should match gt_audio before windowing
    # Let's trim to frame_count
    if gt_audio_cpu.shape[0] > args.frame_count:
        gt_audio_trimmed = gt_audio_cpu[:args.frame_count]
    else:
        gt_audio_trimmed = gt_audio_cpu
    print(f"  GT audio (trimmed to {args.frame_count}): {gt_audio_trimmed.shape}")

    # ====================================================================
    # Step 4: Load precomputed and compare
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Comparing with precomputed data")
    print("=" * 70)

    precomp = {}
    for name in ["vae_latents", "first_frame_cond", "clip_features", "text_embeds", "audio_emb"]:
        path = os.path.join(args.precomputed_dir, f"{name}.pt")
        if os.path.exists(path):
            precomp[name] = torch.load(path, map_location="cpu", weights_only=True)
        else:
            print(f"  WARNING: {path} not found")

    # Also load shared neg_text
    neg_path = os.path.join(os.path.dirname(args.precomputed_dir), "neg_text_embeds.pt")
    if os.path.exists(neg_path):
        precomp["neg_text_embeds"] = torch.load(neg_path, map_location="cpu", weights_only=True)

    # --- Compare CLIP ---
    if "clip_features" in precomp:
        results["clip"] = compare_tensors("CLIP features", precomp["clip_features"], gt_clip_cpu)

    # --- Compare first_frame_cond ---
    if "first_frame_cond" in precomp:
        # Compare our precomputed vs original-preprocessed
        results["first_frame_cond_vs_orig"] = compare_tensors(
            "first_frame_cond (ours vs orig image preprocess)",
            precomp["first_frame_cond"], gt_ffc_orig_cpu
        )
        # Also compare our precomputed vs our-preprocessed VAE
        results["first_frame_cond_vs_ours"] = compare_tensors(
            "first_frame_cond (precomputed vs freshly encoded with our preprocess)",
            precomp["first_frame_cond"], gt_ffc_our_way
        )

    # --- Compare VAE latents ---
    if "vae_latents" in precomp:
        results["vae_latents"] = compare_tensors(
            "VAE latents (precomputed vs freshly encoded)",
            precomp["vae_latents"], gt_vae_from_our_preprocess
        )

    # --- Compare text ---
    if "text_embeds" in precomp:
        # Our precompute saves the PADDED version [1, 512, 4096]
        # Original pipeline uses TRIMMED version from T5EncoderModel.__call__
        results["text_padded"] = compare_tensors(
            "T5 text_embeds (padded)",
            precomp["text_embeds"], gt_text_padded
        )
        # Also check how trimmed compares
        print(f"\n  Note: Original pipeline uses TRIMMED text (shape {gt_text.shape})")
        print(f"  Our precompute saves PADDED text (shape {precomp['text_embeds'].shape})")

    if "neg_text_embeds" in precomp:
        results["neg_text"] = compare_tensors(
            "T5 neg_text_embeds",
            precomp["neg_text_embeds"], gt_neg_text_padded
        )

    # --- Compare audio ---
    if "audio_emb" in precomp:
        our_audio = precomp["audio_emb"]
        # Trim GT to match our shape
        gt_audio_compare = gt_audio_trimmed
        if our_audio.shape[0] != gt_audio_compare.shape[0]:
            min_len = min(our_audio.shape[0], gt_audio_compare.shape[0])
            print(f"\n  Audio length mismatch: ours={our_audio.shape[0]}, GT={gt_audio_compare.shape[0]}, comparing first {min_len}")
            our_audio = our_audio[:min_len]
            gt_audio_compare = gt_audio_compare[:min_len]
        results["audio"] = compare_tensors("wav2vec2 audio_emb", our_audio, gt_audio_compare)

    # ====================================================================
    # Step 5: Key comparison — original image preprocessing vs ours through VAE
    # ====================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Impact of image preprocessing on VAE output")
    print("=" * 70)

    compare_tensors(
        "first_frame_cond: original preprocess vs our preprocess (through VAE)",
        gt_ffc_orig_cpu, gt_ffc_our_way
    )

    # ====================================================================
    # Summary
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, match in results.items():
        status = "MATCH" if match else "MISMATCH"
        print(f"  {name:40s}  {status}")

    mismatches = [k for k, v in results.items() if not v]
    if mismatches:
        print(f"\n  MISMATCHES FOUND: {mismatches}")
        print(f"  These need to be fixed in precompute_infinitetalk_data.py")
    else:
        print(f"\n  ALL MATCH! Preprocessing is correct.")


if __name__ == "__main__":
    main()
