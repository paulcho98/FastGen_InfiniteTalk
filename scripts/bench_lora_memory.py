#!/usr/bin/env python3
"""Quick memory benchmark for different LoRA ranks.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_lora_memory.py --rank 32 --alpha 32
    CUDA_VISIBLE_DEVICES=1 python scripts/bench_lora_memory.py --rank 128 --alpha 64
"""
import argparse
import os
import sys
import time

# Ensure fastgen is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import torch

WEIGHTS_DIR = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/InfiniteTalk/weights/Wan2.1-I2V-14B-480P"
INFINITETALK_CKPT = "/data/karlo-research_715/workspace/kinemaar/paul/AR_diffusion/reference_FastGen_InfiniteTalk/weights/InfiniteTalk/single/infinitetalk.safetensors"
BASE_MODEL_PATHS = ",".join([
    f"{WEIGHTS_DIR}/diffusion_pytorch_model-0000{i}-of-00007.safetensors"
    for i in range(1, 8)
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"LoRA Rank={args.rank}, Alpha={args.alpha}")
    print(f"{'='*60}\n")

    from fastgen.networks.InfiniteTalk.network_causal import CausalInfiniteTalkWan
    from fastgen.datasets.infinitetalk_dataloader import InfiniteTalkDataLoader

    torch.cuda.reset_peak_memory_stats()

    # Build network directly
    print("Building CausalInfiniteTalkWan...")
    t0 = time.time()
    net = CausalInfiniteTalkWan(
        base_model_paths=BASE_MODEL_PATHS,
        infinitetalk_ckpt_path=INFINITETALK_CKPT,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        chunk_size=3,
        total_num_frames=21,
        net_pred_type="flow",
        schedule_type="rf",
        shift=7.0,
    )
    t_build = time.time() - t0
    print(f"  Built in {t_build:.1f}s")

    # Count params
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Total params:     {total_params/1e9:.3f}B")
    print(f"  Trainable params: {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.2f}%)")
    print(f"  Frozen params:    {frozen_params/1e9:.3f}B")

    # LoRA param breakdown
    lora_params = 0
    audio_proj_params = 0
    for name, p in net.named_parameters():
        if p.requires_grad:
            if 'lora_' in name:
                lora_params += p.numel()
            elif 'audio_proj' in name:
                audio_proj_params += p.numel()
    print(f"  LoRA params:      {lora_params/1e6:.1f}M")
    print(f"  AudioProj params: {audio_proj_params/1e6:.1f}M")

    device = torch.device("cuda")
    net = net.to(device).to(torch.bfloat16)
    net.train()

    mem_model = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  GPU mem (model on GPU): {mem_model:.2f} GB")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=1e-5,
    )
    mem_optim = torch.cuda.max_memory_allocated() / 1e9
    print(f"  GPU mem (+ optimizer):  {mem_optim:.2f} GB")

    # Build dataloader
    dl = InfiniteTalkDataLoader(
        data_list_path="data/test_precomputed/sample_list.txt",
        neg_text_emb_path="data/test_precomputed/neg_text_embeds.pt",
        batch_size=1,
        load_ode_path=False,
        expected_latent_shape=[16, 21, 56, 112],
        num_workers=0,
    )
    data_iter = iter(dl)

    # Training iterations
    print(f"\nRunning {args.iters} iterations...")
    iter_times = []
    for i in range(args.iters):
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time()

        data = next(data_iter)
        real = data["real"].to(device).to(torch.bfloat16)
        B, C, T, H, W = real.shape

        # Random per-frame timesteps
        t_rand = torch.rand(B, T, device=device, dtype=torch.bfloat16) * 0.998 + 0.001

        # Noisy input (all bf16)
        noise = torch.randn_like(real)
        t_exp = t_rand[:, None, :, None, None]  # [B, 1, T, 1, 1]
        x_t = t_exp * real + (1 - t_exp) * noise

        condition = {
            "text_embeds": data["text_embeds"].to(device).to(torch.bfloat16),
            "first_frame_cond": data["first_frame_cond"].to(device).to(torch.bfloat16),
            "clip_features": data["clip_features"].to(device).to(torch.bfloat16),
            "audio_emb": data["audio_emb"].to(device).to(torch.bfloat16),
        }

        # Forward (autocast like the training framework does)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = net(x_t, t_rand, condition=condition)
            mem_fwd = torch.cuda.max_memory_allocated() / 1e9

            # Loss + backward
            loss = ((pred - real) ** 2).mean()
        loss.backward()
        mem_bwd = torch.cuda.max_memory_allocated() / 1e9

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        mem_step = torch.cuda.max_memory_allocated() / 1e9

        t_iter = time.time() - t_start
        iter_times.append(t_iter)

        print(f"  [iter {i+1}] loss={loss.item():.6f} | "
              f"fwd={mem_fwd:.1f}GB bwd={mem_bwd:.1f}GB step={mem_step:.1f}GB | "
              f"time={t_iter:.1f}s")

    # Final summary
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n{'='*60}")
    print(f"SUMMARY — LoRA Rank={args.rank}, Alpha={args.alpha}")
    print(f"{'='*60}")
    print(f"  Trainable params:  {trainable_params/1e6:.1f}M")
    print(f"  LoRA params:       {lora_params/1e6:.1f}M")
    print(f"  Peak GPU memory:   {peak:.2f} GB")
    print(f"  Model on GPU:      {mem_model:.2f} GB")
    print(f"  + Optimizer:       {mem_optim:.2f} GB")
    print(f"  Avg iter time:     {sum(iter_times)/len(iter_times):.1f}s")
    if len(iter_times) > 1:
        print(f"  Avg iter (no 1st): {sum(iter_times[1:])/len(iter_times[1:]):.1f}s")
    print()


if __name__ == "__main__":
    main()
