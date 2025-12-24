#!/usr/bin/env python3
"""
Microbenchmark WanVAE stream_decode() for B200/B300 comparisons.

Example:
  PYTHONPATH=src python scripts/bench_wanvae_stream_decode.py --height 320 --width 576 --t 3 --cudnn-benchmark
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WanVAE stream_decode")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--t", type=int, default=3, help="Number of frames decoded per call")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Warmup calls before timing (>=30 recommended to reach steady-state streaming caches).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Compute dtype for the VAE decode benchmark.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark (can improve conv performance).",
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=None,
        help="Path to Wan2.1 VAE checkpoint. Defaults to repo model path.",
    )
    return parser.parse_args()


def _bench_stream_decode(
    *,
    vae,
    zs,
    scale,
    iters: int,
    warmup: int,
    label: str,
) -> float:
    import torch

    vae.clear_cache()

    for _ in range(warmup):
        _ = vae.model.stream_decode(zs, scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = vae.model.stream_decode(zs, scale)
    end.record()
    end.synchronize()

    ms = start.elapsed_time(end) / float(iters)
    print(f"{label}: {ms:.3f} ms/call (t={zs.shape[2]})")
    return ms


def main() -> int:
    args = _parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    vae_path = args.vae_path
    if vae_path is None:
        from scope.core.config import get_model_file_path

        vae_path = Path(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))

    from scope.core.pipelines.wan2_1.vae.wan import WanVAEWrapper

    print(f"torch: {torch.__version__} (cuda={torch.version.cuda}, cudnn={torch.backends.cudnn.version()})")
    print(f"device: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"vae_path: {vae_path}")

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise SystemExit("height/width must be divisible by 8 (WanVAE latent downsample).")

    latent_h = args.height // 8
    latent_w = args.width // 8

    vae = WanVAEWrapper(vae_path=str(vae_path)).to(device="cuda", dtype=dtype).eval()

    # Latent input to decode_to_pixel is [B, T, C, H, W], with C=z_dim=16.
    latent = torch.randn(
        args.batch,
        args.t,
        16,
        latent_h,
        latent_w,
        device="cuda",
        dtype=dtype,
    )
    zs = latent.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    scale = vae._get_scale(device=latent.device, dtype=latent.dtype)

    print(f"latent: {tuple(latent.shape)} {latent.dtype} {latent.device}")
    print(f"zs: {tuple(zs.shape)} strides={zs.stride()}")

    _bench_stream_decode(
        vae=vae,
        zs=zs.contiguous(),
        scale=scale,
        iters=args.iters,
        warmup=args.warmup,
        label="channels_first",
    )
    _bench_stream_decode(
        vae=vae,
        zs=zs.contiguous(memory_format=torch.channels_last_3d),
        scale=scale,
        iters=args.iters,
        warmup=args.warmup,
        label="channels_last_3d",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
