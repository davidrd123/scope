#!/usr/bin/env python3
"""
Microbenchmarks for the main memory-movement primitives in the attention path.

This is intended to quantify the upside of Level-6 work (post-projection pack /
layout transforms / KV-cache write fusion) on B200/B300-like shapes.

By default it benchmarks:
  1) [B,S,H,D] -> [B,H,S,D] via transpose+contiguous (flex_attention inputs)
  2) KV cache writes from contiguous vs strided sources
"""

from __future__ import annotations

import argparse
import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark attention packing primitives")
    p.add_argument("--b", type=int, default=1)
    p.add_argument("--s", type=int, default=2160)
    p.add_argument("--h", type=int, default=40)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--kv-cache-size", type=int, default=4320)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def _dtype_from_str(torch, s: str):
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {s}")


def _bytes_per_elem(torch_dtype) -> int:
    # torch.bfloat16/float16 = 2 bytes, float32 = 4 bytes
    return 2 if str(torch_dtype).endswith(("bfloat16", "float16")) else 4


def _time_cuda(fn, *, warmup: int, iters: int) -> float:
    # Warmup
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, iters)):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = float(start.elapsed_time(end))
    return ms / max(1, iters)


def _fmt_bw_gbs(bytes_moved: int, ms: float) -> str:
    if ms <= 0:
        return "inf"
    gbs = (bytes_moved / 1e9) / (ms / 1e3)
    return f"{gbs:7.1f} GB/s"


def _print_header(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dtype = _dtype_from_str(torch, args.dtype)
    bytes_per = _bytes_per_elem(dtype)

    b, s, h, d = args.b, args.s, args.h, args.d
    kv_cache_size = args.kv_cache_size

    print(f"device: {torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
    print(f"torch: {torch.__version__} (cuda={torch.version.cuda})")
    print(f"shape: B={b} S={s} H={h} D={d} dtype={dtype}")
    print(f"iters: warmup={args.warmup} iters={args.iters}")

    # Simulate fused-projection output where Q/K get materialized (e.g. via RMSNorm)
    # but V remains a strided view into the packed QKV buffer.
    with torch.no_grad():
        packed = torch.randn((b, s, 3, h, d), device="cuda", dtype=dtype)
    q = packed[:, :, 0].contiguous()
    k = packed[:, :, 1].contiguous()
    v_strided = packed[:, :, 2]  # intentionally non-contiguous
    v_contig = v_strided.contiguous()

    # 1) Transpose+contiguous: [B,S,H,D] -> [B,H,S,D]
    _print_header("1) Transpose+contiguous ([B,S,H,D] -> [B,H,S,D])")
    bytes_one = int(b * s * h * d * bytes_per) * 2  # read+write

    def pack_bhsd(x):
        return x.transpose(1, 2).contiguous()

    q_ms = _time_cuda(lambda: pack_bhsd(q), warmup=args.warmup, iters=args.iters)
    k_ms = _time_cuda(lambda: pack_bhsd(k), warmup=args.warmup, iters=args.iters)
    v_ms = _time_cuda(lambda: pack_bhsd(v_contig), warmup=args.warmup, iters=args.iters)
    print(f"q: {q_ms:7.3f} ms  ({_fmt_bw_gbs(bytes_one, q_ms)})")
    print(f"k: {k_ms:7.3f} ms  ({_fmt_bw_gbs(bytes_one, k_ms)})")
    print(f"v: {v_ms:7.3f} ms  ({_fmt_bw_gbs(bytes_one, v_ms)})")
    print(f"q+k+v total: {(q_ms + k_ms + v_ms):7.3f} ms")

    # 2) KV-cache writes: contiguous vs strided sources into a contiguous cache window.
    _print_header("2) KV-cache write (src contiguous vs strided)")
    cache = torch.empty((b, kv_cache_size, h, d), device="cuda", dtype=dtype)
    bytes_cache = int(b * s * h * d * bytes_per) * 2  # read+write

    def write_k(src):
        cache[:, :s].copy_(src)

    # Ensure slice exists (avoid bounds issues)
    if s > kv_cache_size:
        raise SystemExit(f"--s ({s}) must be <= --kv-cache-size ({kv_cache_size})")

    v_contig_ms = _time_cuda(lambda: write_k(v_contig), warmup=args.warmup, iters=args.iters)
    v_strided_ms = _time_cuda(lambda: write_k(v_strided), warmup=args.warmup, iters=args.iters)

    print(f"copy_ from contiguous src: {v_contig_ms:7.3f} ms  ({_fmt_bw_gbs(bytes_cache, v_contig_ms)})")
    print(f"copy_ from strided src:   {v_strided_ms:7.3f} ms  ({_fmt_bw_gbs(bytes_cache, v_strided_ms)})")
    if v_contig_ms > 0:
        print(f"slowdown (strided/contig): {v_strided_ms / v_contig_ms:5.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
