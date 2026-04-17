#!/usr/bin/env python3
"""Verify quarter-resolution precomputed data against full-resolution.

Checks:
  1. Dimension correctness: quarter latents have same channels & temporal dim,
     spatial dims are exactly halved compared to full-res.
  2. Shared tensor integrity: CLIP/T5/audio exist with no _quarter variant,
     prints shapes.
  3. VAE decode visual verification (optional, requires --vae_path): load
     WanVAE, decode first 5 latent frames from both resolutions, save PNGs.
  4. Dataloader integration: simulate _load_sample, verify shapes/dtypes and
     consistency between vae_latents and first_frame_cond.

Usage:
    python scripts/verify_quarter_res.py \\
        --sample_dir data/precomputed_talkvid/sample_0001 \\
        --vae_path /path/to/Wan2.1_VAE.pth \\
        --output_dir /tmp/quarter_res_verify
"""

import argparse
import importlib.machinery
import os
import shutil
import sys
import types

import torch


# ===================================================================
# Mock heavy dependencies before any InfiniteTalk imports
# ===================================================================

class _MockModule(types.ModuleType):
    """A mock module that returns dummy callables for unknown attributes
    while preserving standard module attributes."""
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


def _ensure_mock(module_path):
    """Create a mock module at *module_path* (e.g. 'a.b.c') if absent."""
    parts = module_path.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[: i + 1])
        if partial not in sys.modules:
            mod = _MockModule(partial)
            if i < len(parts) - 1:
                mod.__path__ = []
            mod.__spec__ = importlib.machinery.ModuleSpec(partial, None)
            sys.modules[partial] = mod


def _install_mocks():
    """Install mocks for xfuser, xformers, optimum.quanto, decord."""
    for mock_mod in [
        "xfuser",
        "xfuser.core",
        "xfuser.core.distributed",
        "xformers",
        "xformers.ops",
        "optimum",
        "optimum.quanto",
        "optimum.quanto.nn",
        "optimum.quanto.nn.qlinear",
        "decord",
    ]:
        _ensure_mock(mock_mod)

    # Provide dummy symbols that t5.py imports from optimum.quanto
    oq = sys.modules["optimum.quanto"]
    oq.quantize = lambda *a, **k: None
    oq.freeze = lambda *a, **k: None
    oq.qint8 = None
    oq.requantize = lambda *a, **k: None

    # Provide specific dummy functions that attention.py imports by name
    xfuser_dist = sys.modules["xfuser.core.distributed"]
    xfuser_dist.get_sequence_parallel_rank = lambda: 0
    xfuser_dist.get_sequence_parallel_world_size = lambda: 1
    xfuser_dist.get_sp_group = lambda: None


def _add_infinitetalk_to_path():
    """Add InfiniteTalk source root to sys.path and install the wan
    package stub (same pattern as precompute script)."""
    import importlib as _importlib

    it_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "../../InfiniteTalk")
    )
    if not os.path.isdir(it_root):
        # Fallback: try sibling of the FastGen repo
        it_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../InfiniteTalk")
        )
    if it_root not in sys.path:
        sys.path.insert(0, it_root)

    # Create wan package stub so we can import wan.modules.vae without
    # triggering the heavy wan/__init__.py imports.
    wan_pkg = types.ModuleType("wan")
    wan_pkg.__path__ = [os.path.join(it_root, "wan")]
    wan_pkg.__package__ = "wan"
    wan_pkg.__file__ = os.path.join(it_root, "wan", "__init__.py")
    sys.modules["wan"] = wan_pkg

    for sub in ("wan.modules", "wan.utils", "wan.configs"):
        _importlib.import_module(sub)

    return it_root


# ===================================================================
# Helpers
# ===================================================================

def _load_tensor(path):
    """Load a .pt file and unwrap dict format if needed."""
    t = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(t, dict):
        t = next(v for v in t.values() if isinstance(v, torch.Tensor))
    return t


# ===================================================================
# Check 1: Dimension correctness
# ===================================================================

def check_dimensions(sample_dir):
    """Verify quarter-res latents have same C and T, spatial dims halved."""
    print("\n" + "=" * 60)
    print("CHECK 1: Dimension Correctness")
    print("=" * 60)

    results = {}

    for name in ["vae_latents", "first_frame_cond", "motion_frame"]:
        full_path = os.path.join(sample_dir, f"{name}.pt")
        quarter_path = os.path.join(sample_dir, f"{name}_quarter.pt")

        if not os.path.exists(full_path):
            print(f"  SKIP  {name}: full-res file not found")
            continue
        if not os.path.exists(quarter_path):
            print(f"  FAIL  {name}: quarter-res file not found")
            results[name] = False
            continue

        full = _load_tensor(full_path)
        quarter = _load_tensor(quarter_path)

        print(f"\n  {name}:")
        print(f"    Full-res shape:    {list(full.shape)}  dtype={full.dtype}")
        print(f"    Quarter-res shape: {list(quarter.shape)}  dtype={quarter.dtype}")

        ok = True

        # Check channels (dim 0)
        if full.shape[0] != quarter.shape[0]:
            print(f"    FAIL  channels differ: {full.shape[0]} vs {quarter.shape[0]}")
            ok = False
        else:
            print(f"    PASS  channels match: {full.shape[0]}")

        # Check temporal dim (dim 1)
        if full.shape[1] != quarter.shape[1]:
            print(f"    FAIL  temporal dim differs: {full.shape[1]} vs {quarter.shape[1]}")
            ok = False
        else:
            print(f"    PASS  temporal dim matches: {full.shape[1]}")

        # Check spatial dims are halved (dims 2, 3)
        if full.ndim >= 4:
            expected_h = full.shape[2] // 2
            expected_w = full.shape[3] // 2
            if quarter.shape[2] != expected_h or quarter.shape[3] != expected_w:
                print(
                    f"    FAIL  spatial dims not halved: "
                    f"expected [{expected_h}, {expected_w}], "
                    f"got [{quarter.shape[2]}, {quarter.shape[3]}]"
                )
                ok = False
            else:
                print(
                    f"    PASS  spatial dims halved: "
                    f"[{full.shape[2]}, {full.shape[3]}] -> "
                    f"[{quarter.shape[2]}, {quarter.shape[3]}]"
                )

        results[name] = ok

    return results


# ===================================================================
# Check 2: Shared tensor integrity
# ===================================================================

def check_shared_tensors(sample_dir):
    """Verify CLIP/T5/audio exist and no _quarter variant exists."""
    print("\n" + "=" * 60)
    print("CHECK 2: Shared Tensor Integrity")
    print("=" * 60)

    results = {}

    for name in ["clip_features", "text_embeds", "audio_emb"]:
        path = os.path.join(sample_dir, f"{name}.pt")
        quarter_path = os.path.join(sample_dir, f"{name}_quarter.pt")

        # Check that no _quarter variant exists
        if os.path.exists(quarter_path):
            print(
                f"\n  WARN  {name}_quarter.pt exists but should not "
                f"(these are resolution-independent)"
            )

        if not os.path.exists(path):
            print(f"\n  FAIL  {name}.pt not found")
            results[name] = False
            continue

        t = _load_tensor(path)
        print(f"\n  {name}:")
        print(f"    Shape: {list(t.shape)}  dtype={t.dtype}")
        print(f"    PASS  exists (no _quarter variant needed)")
        results[name] = True

    return results


# ===================================================================
# Check 3: VAE decode visual verification
# ===================================================================

def decode_and_save_previews(sample_dir, vae_path, output_dir):
    """Decode first 5 latent frames from both resolutions, save PNGs."""
    print("\n" + "=" * 60)
    print("CHECK 3: VAE Decode Visual Verification")
    print("=" * 60)

    if not vae_path:
        print("  SKIP  --vae_path not provided")
        return
    if not os.path.exists(vae_path):
        print(f"  SKIP  VAE weights not found at: {vae_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Set up InfiniteTalk imports (mocks already installed)
    _add_infinitetalk_to_path()

    from wan.modules.vae import WanVAE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading WanVAE on {device}...")
    vae = WanVAE(vae_pth=vae_path, device=device)

    from PIL import Image

    for suffix, label in [("", "full"), ("_quarter", "quarter")]:
        lat_path = os.path.join(sample_dir, f"vae_latents{suffix}.pt")
        if not os.path.exists(lat_path):
            print(f"  SKIP  {label}: {lat_path} not found")
            continue

        latents = _load_tensor(lat_path)
        print(f"\n  Decoding {label} latents {list(latents.shape)}...")

        # Take first 5 latent frames to limit memory usage
        num_frames = min(5, latents.shape[1])
        short_lat = latents[:, :num_frames].float().to(device)

        with torch.no_grad():
            decoded = vae.decode([short_lat])

        # decode returns a list of tensors
        decoded_tensor = decoded[0]  # [C, T, H, W]

        # Save first frame as PNG
        frame0 = ((decoded_tensor[:, 0].cpu().clamp(-1, 1) + 1) / 2 * 255).byte()
        frame0 = frame0.permute(1, 2, 0).numpy()  # [H, W, 3]
        out_path = os.path.join(output_dir, f"decoded_{label}_frame0.png")
        Image.fromarray(frame0).save(out_path)
        print(f"    Saved: {out_path}  ({frame0.shape[1]}x{frame0.shape[0]} pixels)")

    # Copy the pixel preview PNG if it exists
    preview_path = os.path.join(sample_dir, "preview_quarter.png")
    if os.path.exists(preview_path):
        dst = os.path.join(output_dir, "pixel_preview_quarter.png")
        shutil.copy2(preview_path, dst)
        print(f"\n  Copied pixel preview: {dst}")

    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n  VAE decode verification complete.")


# ===================================================================
# Check 4: Dataloader integration
# ===================================================================

def check_dataloader_integration(sample_dir):
    """Simulate what the dataloader's _load_sample does at quarter-res."""
    print("\n" + "=" * 60)
    print("CHECK 4: Dataloader Integration (simulated _load_sample)")
    print("=" * 60)

    suffix = "_quarter"
    results = {}

    # Mapping: file stem -> (expected ndim, description)
    file_specs = [
        (f"vae_latents{suffix}", 4, "VAE latents [C, T, H, W]"),
        (f"first_frame_cond{suffix}", 4, "First-frame cond [C, T, H, W]"),
        ("clip_features", 3, "CLIP features [1, 257, 1280]"),
        ("audio_emb", 3, "Audio embeddings [T, 12, 768]"),
        ("text_embeds", 3, "Text embeddings [1, 512, 4096]"),
    ]

    tensors = {}

    for name, expected_ndim, desc in file_specs:
        path = os.path.join(sample_dir, f"{name}.pt")
        if not os.path.exists(path):
            print(f"\n  FAIL  {name}.pt not found")
            results[name] = False
            continue

        t = _load_tensor(path)

        # Simulate dtype conversion the dataloader does
        t_bf16 = t.to(torch.bfloat16)

        # Handle text_embeds 2D -> 3D unsqueeze
        if name == "text_embeds" and t_bf16.dim() == 2:
            t_bf16 = t_bf16.unsqueeze(0)

        ndim_ok = t_bf16.ndim == expected_ndim
        status = "PASS" if ndim_ok else "FAIL"

        print(f"\n  {name}:")
        print(f"    Shape: {list(t_bf16.shape)}  dtype={t_bf16.dtype}")
        print(f"    ndim={t_bf16.ndim} (expected {expected_ndim}): {status}")
        print(f"    Description: {desc}")

        tensors[name] = t_bf16
        results[name] = ndim_ok

    # Check consistency between vae_latents and first_frame_cond
    vae_key = f"vae_latents{suffix}"
    ffc_key = f"first_frame_cond{suffix}"
    if vae_key in tensors and ffc_key in tensors:
        vae_shape = tensors[vae_key].shape
        ffc_shape = tensors[ffc_key].shape

        print(f"\n  Spatial consistency (vae_latents vs first_frame_cond):")

        # Channels must match
        channels_ok = vae_shape[0] == ffc_shape[0]
        # Spatial dims must match
        spatial_ok = vae_shape[2:] == ffc_shape[2:]

        if channels_ok and spatial_ok:
            print(
                f"    PASS  C={vae_shape[0]}, H={vae_shape[2]}, W={vae_shape[3]}"
            )
            results["spatial_consistency"] = True
        else:
            print(
                f"    FAIL  vae={list(vae_shape)}, ffc={list(ffc_shape)}"
            )
            results["spatial_consistency"] = False

        # Temporal: ffc should have >= vae's temporal dim
        print(f"    Temporal: vae T={vae_shape[1]}, ffc T={ffc_shape[1]}")

    return results


# ===================================================================
# Summary
# ===================================================================

def print_summary(all_results):
    """Print a final PASS/FAIL summary across all checks."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for section_name, results in all_results:
        if not results:
            print(f"\n  {section_name}: (no results)")
            total_skip += 1
            continue

        print(f"\n  {section_name}:")
        for name, ok in results.items():
            status = "PASS" if ok else "FAIL"
            print(f"    {name}: {status}")
            if ok:
                total_pass += 1
            else:
                total_fail += 1

    print(f"\n  Totals: {total_pass} PASS, {total_fail} FAIL, {total_skip} skipped")

    if total_fail == 0 and total_pass > 0:
        print("\n  Overall: ALL PASS")
    elif total_fail > 0:
        print("\n  Overall: SOME FAILED")
    else:
        print("\n  Overall: NO CHECKS RAN")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify quarter-resolution precomputed data against full-resolution."
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Path to a single sample directory containing both full-res (.pt) "
             "and quarter-res (*_quarter.pt) files.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="",
        help="Path to Wan2.1_VAE.pth for decode verification (optional). "
             "If not provided, the VAE decode check is skipped.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/quarter_res_verify",
        help="Directory for saving decoded PNG previews (default: /tmp/quarter_res_verify).",
    )
    args = parser.parse_args()

    print(f"Sample directory: {args.sample_dir}")
    if not os.path.isdir(args.sample_dir):
        print(f"ERROR: sample_dir does not exist: {args.sample_dir}")
        sys.exit(1)

    # Install mocks before any InfiniteTalk imports
    _install_mocks()

    # Run checks
    all_results = []

    dim_results = check_dimensions(args.sample_dir)
    all_results.append(("Dimension Correctness", dim_results))

    shared_results = check_shared_tensors(args.sample_dir)
    all_results.append(("Shared Tensor Integrity", shared_results))

    dl_results = check_dataloader_integration(args.sample_dir)
    all_results.append(("Dataloader Integration", dl_results))

    if args.vae_path:
        decode_and_save_previews(args.sample_dir, args.vae_path, args.output_dir)

    print_summary(all_results)


if __name__ == "__main__":
    main()
