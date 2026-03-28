#!/usr/bin/env python3
"""Run the ORIGINAL InfiniteTalk pipeline end-to-end as ground truth.

Patches ONLY what's needed for Python 3.12 compatibility:
  1. inspect.ArgSpec -> FullArgSpec
  2. decord -> av fallback for frame extraction
  3. xformers -> flash_attn_func (same kernel family)
  4. src.vram_management -> no-op (not using CPU offload)
  5. ffprobe -> skip codec check (assume h264)

NO functional changes to the pipeline logic itself.
"""

import sys
import os
import inspect
import types
import importlib.machinery
import argparse
import time

# ── Python 3.12 compat ──
if not hasattr(inspect, 'ArgSpec'):
    inspect.ArgSpec = inspect.FullArgSpec

# ── Paths ──
IT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../InfiniteTalk"))
sys.path.insert(0, IT_ROOT)
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── Mock heavy deps (minimal, no functional changes) ──
class _MockModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None

# xformers -> flash_attn (same [B,M,H,K] layout)
for p in ["xformers", "xformers.ops", "xformers.ops.fmha", "xformers.ops.fmha.attn_bias"]:
    parts = p.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[:i+1])
        if partial not in sys.modules:
            m = _MockModule(partial)
            if i < len(parts) - 1:
                m.__path__ = []
            m.__spec__ = importlib.machinery.ModuleSpec(partial, None)
            sys.modules[partial] = m

def _memory_efficient_attention(q, k, v, attn_bias=None, op=None):
    from flash_attn import flash_attn_func
    return flash_attn_func(q, k, v)

class _MockBlockDiag:
    @staticmethod
    def from_seqlens(*a, **k):
        return None

sys.modules["xformers"].ops = sys.modules["xformers.ops"]
sys.modules["xformers.ops"].memory_efficient_attention = _memory_efficient_attention
sys.modules["xformers.ops.fmha.attn_bias"].BlockDiagonalMask = _MockBlockDiag
sys.modules["xformers.ops.fmha"].attn_bias = sys.modules["xformers.ops.fmha.attn_bias"]

# xfuser
for p in ["xfuser", "xfuser.core", "xfuser.core.distributed"]:
    parts = p.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[:i+1])
        if partial not in sys.modules:
            m = _MockModule(partial)
            if i < len(parts) - 1:
                m.__path__ = []
            m.__spec__ = importlib.machinery.ModuleSpec(partial, None)
            sys.modules[partial] = m

# sageattn
m = _MockModule("sageattn")
m.__spec__ = importlib.machinery.ModuleSpec("sageattn", None)
sys.modules["sageattn"] = m

# src.vram_management (no-op, we don't use CPU offload)
vm = _MockModule("src.vram_management")
vm.__spec__ = importlib.machinery.ModuleSpec("src.vram_management", None)
vm.AutoWrappedQLinear = None
vm.AutoWrappedLinear = None
vm.AutoWrappedModule = None
vm.enable_vram_management = lambda *a, **k: None
sys.modules["src.vram_management"] = vm

# ── Patch decord with av ──
import av as _av

class _FakeCPU:
    def __call__(self, n=0):
        return None
_fake_cpu = _FakeCPU()

class _FakeVideoReader:
    def __init__(self, path, ctx=None):
        self._path = path
        self._container = _av.open(path)
        self._frames = []
        for frame in self._container.decode(video=0):
            self._frames.append(frame)
        self._container.close()
        self._num_frame = len(self._frames)

    def __getitem__(self, idx):
        frame = self._frames[idx]
        class _Arr:
            def __init__(self, f):
                self._f = f
            def asnumpy(self):
                return self._f.to_ndarray(format="rgb24")
        return _Arr(frame)

    def __del__(self):
        pass

decord_mod = _MockModule("decord")
decord_mod.__spec__ = importlib.machinery.ModuleSpec("decord", None)
decord_mod.VideoReader = _FakeVideoReader
decord_mod.cpu = _fake_cpu
sys.modules["decord"] = decord_mod

# ── Patch wan.utils.utils BEFORE importing multitalk ──
# multitalk.py does `from wan.utils.utils import get_video_codec, ...` at import time,
# so we must patch the module first, then let multitalk import the patched functions.
import wan.utils.utils as wan_utils
wan_utils.get_video_codec = lambda path: "h264"
wan_utils.convert_video_to_h264 = lambda *a, **k: None

# ── Now import the pipeline (it will pick up patched functions) ──
# Actually, `from X import Y` copies the reference at import time.
# So we need to patch AFTER import too, on the multitalk module directly.
from wan.multitalk import InfiniteTalkPipeline
import wan.multitalk as _mt
_mt.get_video_codec = lambda path: "h264"
_mt.convert_video_to_h264 = lambda *a, **k: None
from wan.configs import WAN_CONFIGS
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor
import torch
import numpy as np
import librosa
import pyloudnorm as pyln
from easydict import EasyDict
from PIL import Image


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def get_embedding(speech_array, wav2vec_fe, audio_encoder, sr=16000, device='cpu'):
    from einops import rearrange
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25
    audio_feature = np.squeeze(wav2vec_fe(speech_array, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device).unsqueeze(0)
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.cpu().detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--infinitetalk_ckpt", type=str, required=True)
    parser.add_argument("--wav2vec_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/test_precomputed/visualizations")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--shift", type=float, default=7.0)
    parser.add_argument("--text_guide_scale", type=float, default=5.0)
    parser.add_argument("--audio_guide_scale", type=float, default=4.0)
    args = parser.parse_args()

    device_id = int(args.device.split(":")[-1]) if ":" in args.device else 0

    # ── Build pipeline (original code) ──
    print("Building InfiniteTalk pipeline...")
    config = WAN_CONFIGS["infinitetalk-14B"]

    pipeline = InfiniteTalkPipeline(
        config=config,
        checkpoint_dir=args.weights_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=False,
        lora_dir=None,
        infinitetalk_dir=args.infinitetalk_ckpt,
    )
    print("Pipeline loaded")

    # ── Prepare audio embedding (same as generate_infinitetalk.py) ──
    print("Encoding audio...")
    wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_dir, local_files_only=True)
    audio_encoder = Wav2Vec2Model.from_pretrained(
        args.wav2vec_dir, local_files_only=True, attn_implementation="eager"
    ).eval()

    speech, sr = librosa.load(args.audio_path, sr=16000)
    speech = loudness_norm(speech, sr)
    audio_emb = get_embedding(speech, wav2vec_fe, audio_encoder, sr=16000, device='cpu')

    # Save audio embedding to temp file (pipeline loads from file)
    import tempfile
    emb_path = os.path.join(tempfile.mkdtemp(), "audio_emb.pt")
    torch.save(audio_emb, emb_path)

    # ── Build input_data (same format as generate_infinitetalk.py) ──
    input_data = {
        "prompt": args.prompt,
        "cond_video": args.video_path,
        "cond_audio": {"person1": emb_path},
    }

    # ── Extra args (disable TeaCache and APG) ──
    extra_args = EasyDict({
        "use_teacache": False,
        "use_apg": False,
        "size": "infinitetalk-480",
    })

    # ── Generate ──
    print(f"Generating with seed={args.seed}, shift={args.shift}, "
          f"text_scale={args.text_guide_scale}, audio_scale={args.audio_guide_scale}")
    t0 = time.time()
    video = pipeline.generate_infinitetalk(
        input_data,
        size_buckget="infinitetalk-480",
        frame_num=args.frame_num,
        shift=args.shift,
        sampling_steps=40,
        text_guide_scale=args.text_guide_scale,
        audio_guide_scale=args.audio_guide_scale,
        seed=args.seed,
        offload_model=False,
        max_frames_num=args.frame_num,
        extra_args=extra_args,
    )
    print(f"Generated in {time.time()-t0:.0f}s")
    print(f"Output shape: {video.shape}")

    # ── Save frames as PNGs ──
    os.makedirs(args.output_dir, exist_ok=True)
    for t_idx in [0, 20, 40, 60, 80]:
        if t_idx >= video.shape[1]:
            continue
        frame = video[:, t_idx].clamp(-1, 1)
        frame = ((frame + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
        path = os.path.join(args.output_dir, f"original_pipeline_frame{t_idx}.png")
        Image.fromarray(frame).save(path)
        print(f"Saved {path}")

    # ── Save full video with audio ──
    import imageio_ffmpeg
    import subprocess
    import tempfile

    # video tensor: [C, T, H, W] in [-1, 1] → [T, H, W, 3] uint8
    frames_np = video.clamp(-1, 1).permute(1, 2, 3, 0)  # [T, H, W, C]
    frames_np = ((frames_np + 1) / 2 * 255).byte().cpu().numpy()
    num_frames = frames_np.shape[0]
    h, w = frames_np.shape[1], frames_np.shape[2]
    fps = 25  # InfiniteTalk default

    # Write video without audio first
    video_noaudio = os.path.join(args.output_dir, "original_pipeline_noaudio.mp4")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    writer_cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        video_noaudio
    ]
    proc = subprocess.Popen(writer_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(num_frames):
        proc.stdin.write(frames_np[i].tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Saved video (no audio): {video_noaudio} ({num_frames} frames, {fps}fps)")

    # Mux audio
    video_path_out = os.path.join(args.output_dir, "original_pipeline_output.mp4")
    duration = num_frames / fps
    mux_cmd = [
        ffmpeg_exe, "-y",
        "-i", video_noaudio,
        "-i", args.audio_path,
        "-c:v", "copy", "-c:a", "aac",
        "-t", f"{duration:.3f}",  # trim audio to video length
        "-shortest",
        video_path_out
    ]
    result = subprocess.run(mux_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Saved video with audio: {video_path_out}")
        os.remove(video_noaudio)  # clean up intermediate
    else:
        print(f"Audio mux failed (video still saved without audio): {video_noaudio}")
        print(f"  stderr: {result.stderr[:200]}")

    # Clean up
    os.remove(emb_path)
    print("Done!")


if __name__ == "__main__":
    main()
