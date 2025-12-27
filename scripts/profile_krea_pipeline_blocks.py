#!/usr/bin/env python3
"""
Benchmark the Krea realtime pipeline and (optionally) emit per-block timings.

This is intended for comparing B200 vs B300 vs H100 runs at the same settings.

Examples:
  # Just FPS (synchronized)
  uv run python scripts/profile_krea_pipeline_blocks.py --iters 10

  # Per-block profile (printed at exit) + JSON output
  PROFILE_PIPELINE_BLOCKS=1 PROFILE_PIPELINE_BLOCKS_JSON=/tmp/krea_blocks.json \\
    uv run python scripts/profile_krea_pipeline_blocks.py --iters 10

  # B300 (SM103) often needs a newer ptxas for Triton/Inductor
  TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas \\
    uv run python scripts/profile_krea_pipeline_blocks.py --iters 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Krea realtime pipeline")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--iters", type=int, default=10, help="Number of pipeline calls to time")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N timed iterations")
    parser.add_argument(
        "--pre-iters",
        type=int,
        default=0,
        help="Warm iterations outside timing loop (useful for switching to V2V mode)",
    )
    parser.add_argument("--kv-cache-attention-bias", type=float, default=0.3)
    parser.add_argument(
        "--input-mode",
        choices=["text", "video"],
        default="text",
        help="Benchmark text-to-video (text) or video-to-video (video).",
    )
    parser.add_argument(
        "--video-source",
        choices=["cpu", "gpu"],
        default="cpu",
        help="For V2V: provide input frames as CPU list (realistic server path) or a prepacked GPU tensor (skips preprocessing/HtoD).",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=None,
        help="For V2V: number of input frames to provide. Default uses num_frame_per_block*vae_temporal_downsample_factor (+1 for first block).",
    )
    parser.add_argument(
        "--video-input-height",
        type=int,
        default=None,
        help="For V2V + --video-source cpu: input frame height before preprocessing/resize (defaults to --height).",
    )
    parser.add_argument(
        "--video-input-width",
        type=int,
        default=None,
        help="For V2V + --video-source cpu: input frame width before preprocessing/resize (defaults to --width).",
    )
    parser.add_argument(
        "--denoising-step-list",
        type=int,
        nargs="+",
        default=None,
        help="Optional override for denoising_step_list (e.g. --denoising-step-list 1000 500).",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.7,
        help="For V2V: blend factor between encoded latents and fresh noise (0=preserve input, 1=ignore input).",
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile attention blocks")
    parser.add_argument(
        "--quantization",
        choices=["fp8_e4m3fn", "none"],
        default="fp8_e4m3fn",
        help="Match the common 32GB setup (fp8) or run unquantized (none).",
    )
    parser.add_argument("--prompt", type=str, default="a majestic sunset")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark (can improve conv performance).",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write per-iter FPS to JSON")
    parser.add_argument(
        "--profile-blocks",
        action="store_true",
        help="Enable per-block profiling (same as PROFILE_PIPELINE_BLOCKS=1).",
    )
    parser.add_argument(
        "--profile-blocks-json",
        type=Path,
        default=None,
        help="Write per-block profile to JSON (same as PROFILE_PIPELINE_BLOCKS_JSON=...).",
    )
    return parser.parse_args()


def _cuda_unavailable_hint(torch) -> str:
    import os as _os

    dev_checks = []
    for path in ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm"):
        try:
            fd = _os.open(path, _os.O_RDWR)
            _os.close(fd)
            dev_checks.append(f"{path}=rw_ok")
        except FileNotFoundError:
            dev_checks.append(f"{path}=missing")
        except PermissionError:
            dev_checks.append(f"{path}=rw_denied")
        except Exception as e:
            dev_checks.append(f"{path}=err:{type(e).__name__}")

    return (
        "CUDA not available.\n"
        f"- torch: {torch.__version__} (cuda={torch.version.cuda})\n"
        f"- /dev access: {', '.join(dev_checks)}\n"
        "If /dev/* is rw_denied, you likely don't have GPU *compute* permissions "
        "(device-cgroup / container missing GPU access). If NVML is wedged "
        "(nvidia-smi errors), a host driver restart/reboot may be required."
    )


def main() -> int:
    args = _parse_args()

    if args.profile_blocks or args.profile_blocks_json is not None:
        os.environ["PROFILE_PIPELINE_BLOCKS"] = "1"
    if args.profile_blocks_json is not None:
        os.environ["PROFILE_PIPELINE_BLOCKS_JSON"] = str(args.profile_blocks_json)

    import torch
    from omegaconf import OmegaConf

    from scope.core.config import get_model_file_path, get_models_dir
    from scope.core.pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline
    from scope.core.pipelines.utils import Quantization

    # Attention profiling logs are emitted via a module logger; attach a handler
    # here so PROFILE_ATTENTION=1 produces output without changing global loggers.
    if os.getenv("PROFILE_ATTENTION", "0") == "1":
        attn_logger = logging.getLogger(
            "scope.core.pipelines.krea_realtime_video.modules.causal_model"
        )
        attn_logger.setLevel(logging.INFO)
        if not attn_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
        attn_logger.addHandler(handler)
        attn_logger.propagate = False

    if not torch.cuda.is_available():
        raise SystemExit(_cuda_unavailable_hint(torch))

    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"torch: {torch.__version__} (cuda={torch.version.cuda})")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    quantization = (
        Quantization.FP8_E4M3FN if args.quantization == "fp8_e4m3fn" else None
    )

    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path(
                    "krea-realtime-video/krea-realtime-video-14b.safetensors"
                )
            ),
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "vae_path": str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")),
            "model_config": OmegaConf.load(
                Path(__file__).resolve().parents[1]
                / "src/scope/core/pipelines/krea_realtime_video/model.yaml"
            ),
            "height": args.height,
            "width": args.width,
        }
    )

    pipeline = KreaRealtimeVideoPipeline(
        config,
        quantization=quantization,
        compile=args.compile,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

    # Reset profiling counters after pipeline init/warmup so JSON reflects timed iters only.
    try:
        from scope.core.pipelines.krea_realtime_video import modular_blocks as _modular_blocks

        _modular_blocks.reset_pipeline_block_profile()
    except Exception:
        pass
    try:
        from scope.core.pipelines.wan2_1.blocks import denoise as _denoise_block

        _denoise_block.reset_denoise_step_profile()
    except Exception:
        pass
    try:
        from scope.core.pipelines.wan2_1.components import generator as _wan_generator

        _wan_generator.reset_generator_step_profile()
    except Exception:
        pass
    try:
        from scope.core.pipelines.wan2_1.vae import wan as _wan_vae

        _wan_vae.reset_wanvae_decode_profile()
    except Exception:
        pass
    try:
        from scope.core.pipelines.wan2_1.vae import wan as _wan_vae

        _wan_vae.reset_wanvae_encode_profile()
    except Exception:
        pass
    try:
        from scope.core.pipelines.wan2_1.vae.modules import vae as _wan_vae_modules

        _wan_vae_modules.reset_wanvae_decode_inner_profile()
    except Exception:
        pass
    try:
        from scope.core.pipelines.krea_realtime_video.modules import causal_model as _causal_model

        _causal_model.reset_attention_profile()
    except Exception:
        pass

    prompts = [{"text": args.prompt, "weight": 100}]
    per_iter = []

    video = None
    if args.input_mode == "video":
        num_frame_per_block = int(getattr(pipeline.components.config, "num_frame_per_block", 3))
        vae_temporal_downsample_factor = int(
            getattr(pipeline.components.config, "vae_temporal_downsample_factor", 4)
        )
        # First V2V block needs +1 frame due to VAE behavior.
        default_frames = (num_frame_per_block * vae_temporal_downsample_factor) + 1
        num_frames = int(args.video_frames) if args.video_frames is not None else default_frames
        if args.video_source == "gpu":
            video_u8 = torch.randint(
                0,
                256,
                (1, 3, num_frames, args.height, args.width),
                device="cuda",
                dtype=torch.uint8,
            )
            video = (video_u8.to(dtype=torch.bfloat16) / 255.0) * 2.0 - 1.0
        else:
            in_h = int(args.video_input_height) if args.video_input_height is not None else args.height
            in_w = int(args.video_input_width) if args.video_input_width is not None else args.width
            # CPU list of frames (T=1) in 0..255, matching the server's frame path.
            video = [
                torch.randint(
                    0,
                    256,
                    (1, in_h, in_w, 3),
                    device="cpu",
                    dtype=torch.uint8,
                )
                for _ in range(num_frames)
            ]

    call_kwargs: dict[str, object] = {
        "prompts": prompts,
        "kv_cache_attention_bias": args.kv_cache_attention_bias,
    }
    if args.denoising_step_list is not None:
        call_kwargs["denoising_step_list"] = list(args.denoising_step_list)
    if video is not None:
        call_kwargs["video"] = video
        call_kwargs["noise_scale"] = float(args.noise_scale)

    if args.pre_iters:
        print(f"Pre-warm (outside timing): {args.pre_iters} iteration(s)")
        for _ in range(args.pre_iters):
            pipeline(**call_kwargs)
        torch.cuda.synchronize()

    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipeline(**call_kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        frames = int(out.shape[0])
        fps = (frames / dt) if dt > 0 else 0.0
        per_iter.append({"iter": i, "frames": frames, "seconds": dt, "fps": fps})
        print(f"iter={i:03d} frames={frames:3d} seconds={dt:.4f} fps={fps:.2f}")

        # If we're skipping early iterations for steady-state averages, also reset profiling
        # counters so the JSON artifacts reflect steady-state behavior (not warmup/cache fill).
        if args.skip and i == args.skip - 1:
            try:
                from scope.core.pipelines.krea_realtime_video import modular_blocks as _modular_blocks

                _modular_blocks.reset_pipeline_block_profile()
            except Exception:
                pass
            try:
                from scope.core.pipelines.wan2_1.blocks import denoise as _denoise_block

                _denoise_block.reset_denoise_step_profile()
            except Exception:
                pass
            try:
                from scope.core.pipelines.wan2_1.components import generator as _wan_generator

                _wan_generator.reset_generator_step_profile()
            except Exception:
                pass
            try:
                from scope.core.pipelines.wan2_1.vae import wan as _wan_vae

                _wan_vae.reset_wanvae_decode_profile()
            except Exception:
                pass
            try:
                from scope.core.pipelines.wan2_1.vae import wan as _wan_vae

                _wan_vae.reset_wanvae_encode_profile()
            except Exception:
                pass
            try:
                from scope.core.pipelines.wan2_1.vae.modules import vae as _wan_vae_modules

                _wan_vae_modules.reset_wanvae_decode_inner_profile()
            except Exception:
                pass
            try:
                from scope.core.pipelines.krea_realtime_video.modules import causal_model as _causal_model

                _causal_model.reset_attention_profile()
            except Exception:
                pass

    kept = per_iter[args.skip :] if args.skip else per_iter
    if kept:
        avg_fps = sum(x["fps"] for x in kept) / len(kept)
        print(f"\nAvg FPS (skip={args.skip}): {avg_fps:.2f}")

    if args.json is not None:
        args.json.write_text(
            json.dumps(per_iter, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote per-iter JSON: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
