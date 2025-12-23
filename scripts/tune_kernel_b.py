#!/usr/bin/env python3
"""
Tiny tuning harness for Kernel B (piecewise bias attention).

Example:
  uv run python scripts/tune_kernel_b.py
  uv run python scripts/tune_kernel_b.py --iters 100 --warmup 10
"""

import argparse
import math
import os
import time

import torch

if "TRITON_PTXAS_PATH" not in os.environ:
    for _candidate in (
        "/usr/local/cuda-13.1/bin/ptxas",
        "/usr/local/cuda-13.0/bin/ptxas",
        "/usr/local/cuda-12.9/bin/ptxas",
        "/usr/local/cuda/bin/ptxas",
    ):
        if os.path.exists(_candidate):
            os.environ["TRITON_PTXAS_PATH"] = _candidate
            break

import triton
from triton.runtime import errors as triton_errors

import triton_sdpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Kernel B launch parameters.")
    parser.add_argument("--b", type=int, default=1, help="Batch size")
    parser.add_argument("--h", type=int, default=16, help="Num heads")
    parser.add_argument("--lq", type=int, default=4680, help="Query length")
    parser.add_argument("--lk", type=int, default=9360, help="Key/value length")
    parser.add_argument("--d", type=int, default=128, help="Head dimension")
    parser.add_argument("--frame-seqlen", type=int, default=1560, help="Tokens per frame")
    parser.add_argument("--num-frame-per-block", type=int, default=3, help="Frames per block")
    parser.add_argument("--beta", type=float, default=0.3, help="Bias factor (beta)")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="Input dtype")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--top", type=int, default=5, help="Show top-N configs")
    return parser.parse_args()


def build_configs() -> list[dict[str, int]]:
    configs = []
    for block_m in (64, 128):
        for block_n in (64, 128):
            for warps in (4, 8):
                for stages in (2, 3):
                    configs.append(
                        {
                            "BLOCK_M": block_m,
                            "BLOCK_N": block_n,
                            "num_warps": warps,
                            "num_stages": stages,
                        }
                    )
    return configs


def run_config(
    kernel,
    args,
    grid,
    meta,
    warmup: int,
    iters: int,
) -> float:
    kernel.run(*args, grid=grid, warmup=True, **meta)
    for _ in range(warmup):
        kernel.run(*args, grid=grid, warmup=False, **meta)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        kernel.run(*args, grid=grid, warmup=False, **meta)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000.0


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this tuning harness.")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(0)

    b, h, lq, lk, d = args.b, args.h, args.lq, args.lk, args.d
    frame_seqlen = args.frame_seqlen
    current_block_start = lk - frame_seqlen * args.num_frame_per_block
    if current_block_start < 0:
        raise SystemExit("current_block_start < 0; check frame_seqlen/num-frame-per-block/lk.")

    log_bias = math.log(args.beta)
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(b, h, lq, d, device="cuda", dtype=dtype)
    k = torch.randn(b, h, lk, d, device="cuda", dtype=dtype)
    v = torch.randn(b, h, lk, d, device="cuda", dtype=dtype)
    o = torch.empty(b, h, lq, d, device="cuda", dtype=dtype)

    kernel = triton_sdpa.kernel_b_bias_attention.fn

    base_args = (
        q,
        k,
        v,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lq, lk, d, scale,
        frame_seqlen, current_block_start, log_bias,
    )

    print("Kernel B tuning (B200 target shape)")
    print(f"shape: B={b} H={h} Lq={lq} Lk={lk} D={d} dtype={args.dtype}")
    print(f"frame_seqlen={frame_seqlen} current_block_start={current_block_start} beta={args.beta}")
    print(f"configs: {len(build_configs())} iters={args.iters} warmup={args.warmup}\n")

    results = []
    for cfg in build_configs():
        grid = (triton.cdiv(lq, cfg["BLOCK_M"]), b * h)
        meta = {
            "BLOCK_M": cfg["BLOCK_M"],
            "BLOCK_N": cfg["BLOCK_N"],
            "BLOCK_D": d,
            "num_warps": cfg["num_warps"],
            "num_stages": cfg["num_stages"],
        }
        try:
            ms = run_config(kernel, base_args, grid, meta, args.warmup, args.iters)
        except triton_errors.OutOfResources as exc:
            print(
                f"SKIP  BLOCK_M={cfg['BLOCK_M']:>3} BLOCK_N={cfg['BLOCK_N']:>3} "
                f"warps={cfg['num_warps']} stages={cfg['num_stages']} -> {exc}"
            )
            continue
        except Exception as exc:  # Keep the sweep going even if a config fails.
            print(
                f"FAIL  BLOCK_M={cfg['BLOCK_M']:>3} BLOCK_N={cfg['BLOCK_N']:>3} "
                f"warps={cfg['num_warps']} stages={cfg['num_stages']} -> {exc}"
            )
            continue

        results.append((ms, cfg))
        print(
            f"BLOCK_M={cfg['BLOCK_M']:>3} BLOCK_N={cfg['BLOCK_N']:>3} "
            f"warps={cfg['num_warps']} stages={cfg['num_stages']} -> {ms:.3f} ms"
        )

    if results:
        results.sort(key=lambda x: x[0])
        top = results[: max(1, args.top)]
        print("\nBest configs:")
        for ms, cfg in top:
            print(
                f"  {ms:.3f} ms  BLOCK_M={cfg['BLOCK_M']} BLOCK_N={cfg['BLOCK_N']} "
                f"warps={cfg['num_warps']} stages={cfg['num_stages']}"
            )
    else:
        print("\nNo valid configs completed.")


if __name__ == "__main__":
    main()
