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
    parser.add_argument("--kv-cache-attention-bias", type=float, default=0.3)
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


def main() -> int:
    args = _parse_args()

    if args.profile_blocks:
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
        raise SystemExit("CUDA not available")

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
        from scope.core.pipelines.wan2_1.vae.modules import vae as _wan_vae_modules

        _wan_vae_modules.reset_wanvae_decode_inner_profile()
    except Exception:
        pass

    prompts = [{"text": args.prompt, "weight": 100}]
    per_iter = []

    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipeline(prompts=prompts, kv_cache_attention_bias=args.kv_cache_attention_bias)
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
                from scope.core.pipelines.wan2_1.vae.modules import vae as _wan_vae_modules

                _wan_vae_modules.reset_wanvae_decode_inner_profile()
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
