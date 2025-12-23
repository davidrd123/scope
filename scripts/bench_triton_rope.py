#!/usr/bin/env python3
"""
Benchmark Triton RoPE (Step 1) vs PyTorch fallback.

Example:
  uv run python scripts/bench_triton_rope.py
  uv run python scripts/bench_triton_rope.py --f 3 --h 30 --w 52 --iters 50
  uv run python scripts/bench_triton_rope.py --f 3 --h 30 --w 52 --pad-to-multiple 128
"""

import argparse
import math
import os
import time

import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Triton RoPE vs PyTorch fallback.")
    parser.add_argument("--f", type=int, default=3, help="Frames (F)")
    parser.add_argument("--h", type=int, default=30, help="Height tokens (H)")
    parser.add_argument("--w", type=int, default=52, help="Width tokens (W)")
    parser.add_argument("--heads", type=int, default=16, help="Num heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dim (D)")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame (causal)")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="Input dtype")
    parser.add_argument("--pad", type=int, default=0, help="Extra tail tokens (right pad).")
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        default=0,
        help="Right-pad so (seq_len+pad) is a multiple of this value (added on top of --pad).",
    )
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument(
        "--disable-triton-rotary",
        action="store_true",
        help="Set SCOPE_DISABLE_TRITON_ROTARY=1 (must happen before imports).",
    )
    return parser.parse_args()


def build_freqs(max_seq_len: int, head_dim: int, device: torch.device, rope_params_fn) -> torch.Tensor:
    """Build freqs like CausalWanModel (rope_params only)."""
    d = head_dim
    c0_dim = d - 4 * (d // 6)
    c1_dim = 2 * (d // 6)
    c2_dim = 2 * (d // 6)
    freqs = torch.cat(
        [
            rope_params_fn(max_seq_len, c0_dim),
            rope_params_fn(max_seq_len, c1_dim),
            rope_params_fn(max_seq_len, c2_dim),
        ],
        dim=1,
    )
    return freqs.to(device)


def rope_torch(x_slice: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch RoPE using pre-materialized cos/sin."""
    seq_len, n, d = x_slice.shape[1], x_slice.shape[2], x_slice.shape[3]
    c = d // 2
    if cos.dim() == 2:
        cos = cos.unsqueeze(1)
    if sin.dim() == 2:
        sin = sin.unsqueeze(1)
    x_i = x_slice.reshape(1, seq_len, n, c, 2)
    x0, x1 = x_i[..., 0], x_i[..., 1]
    x0_new = x0 * cos - x1 * sin
    x1_new = x0 * sin + x1 * cos
    x_i = torch.stack([x0_new, x1_new], dim=-1).flatten(3)
    return x_i


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    if args.disable_triton_rotary:
        os.environ["SCOPE_DISABLE_TRITON_ROTARY"] = "1"

    from scope.core.pipelines.krea_realtime_video.modules.model import (
        get_rope_cos_sin,
        rope_apply,
        rope_params,
        USE_TRITON_ROTARY,
    )
    from scope.core.pipelines.krea_realtime_video.modules.causal_model import (
        causal_rope_apply,
    )

    try:
        from scope.core.kernels.triton_rotary import apply_rotary as triton_apply_rotary
        TRITON_AVAILABLE = True
    except ImportError:
        triton_apply_rotary = None
        TRITON_AVAILABLE = False

    # Optional future kernel (Step 2) – harness should work the moment it lands.
    try:
        from scope.core.kernels.triton_rope_fused import rope_fused_3way as triton_rope_fused_3way
        TRITON_FUSED_AVAILABLE = True
    except Exception:
        triton_rope_fused_3way = None
        TRITON_FUSED_AVAILABLE = False

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device("cuda")

    f, h, w = args.f, args.h, args.w
    seq_len = f * h * w
    pad_to_multiple = int(args.pad_to_multiple)
    pad_mult = 0
    if pad_to_multiple > 0:
        pad_mult = math.ceil(seq_len / pad_to_multiple) * pad_to_multiple - seq_len
    b, n, d = 1, args.heads, args.head_dim
    c = d // 2

    freqs = build_freqs(1024, d, device, rope_params)
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    cos, sin = get_rope_cos_sin(
        freqs_split, f, h, w, args.start_frame, dtype, device, c
    )
    cos_fa = cos.squeeze(1).contiguous()
    sin_fa = sin.squeeze(1).contiguous()

    pad_total = int(args.pad) + int(pad_mult)
    x = torch.randn(b, seq_len + pad_total, n, d, device=device, dtype=dtype)
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long, device="cpu")

    print(f"USE_TRITON_ROTARY={USE_TRITON_ROTARY}")
    print(
        f"shape: B={b} L={seq_len}+{pad_total} (pad={args.pad} pad_mult={pad_mult}) H={n} D={d} dtype={args.dtype}"
    )
    print(f"TRITON_AVAILABLE={TRITON_AVAILABLE} TRITON_FUSED_AVAILABLE={TRITON_FUSED_AVAILABLE}")

    # Correctness check
    torch_out = rope_torch(x[:, :seq_len], cos, sin)
    if TRITON_AVAILABLE:
        triton_out = triton_apply_rotary(x[:, :seq_len], cos_fa, sin_fa, interleaved=True)
        max_err = (triton_out - torch_out).abs().max().item()
        mean_err = (triton_out - torch_out).abs().mean().item()
        print(f"Correctness: max_err={max_err:.6f} mean_err={mean_err:.6f}")
    else:
        print("Triton rotary not available; skipping correctness against Triton.")

    # Benchmark
    def bench(fn, label: str) -> float:
        for _ in range(args.warmup):
            _ = fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(label)
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / args.iters * 1000.0
        torch.cuda.nvtx.range_pop()
        print(f"{label}: {ms:.3f} ms")
        return ms

    # Kernel-only baselines (prefix only)
    bench(lambda: rope_torch(x[:, :seq_len], cos, sin), "PyTorch (prefix only)")
    if TRITON_AVAILABLE:
        bench(
            lambda: triton_apply_rotary(x[:, :seq_len], cos_fa, sin_fa, interleaved=True),
            "Triton rotary (prefix only)",
        )
    else:
        print("Triton rotary not available; skipping Triton benchmark.")

    # End-to-end wrappers (include cache + tail handling)
    bench(lambda: rope_apply(x, grid_sizes, freqs), "rope_apply (end-to-end)")
    bench(
        lambda: causal_rope_apply(x, grid_sizes, freqs, start_frame=args.start_frame),
        f"causal_rope_apply(sf={args.start_frame}) (end-to-end)",
    )

    # Future: Step 2 fused wrapper direct call (if/when it exists)
    if TRITON_FUSED_AVAILABLE and triton_rope_fused_3way is not None:
        bench(
            lambda: triton_rope_fused_3way(x, grid_sizes, freqs, start_frame=args.start_frame, inplace=None),
            "triton_rope_fused_3way (direct)",
        )
    else:
        print("Triton fused RoPE wrapper not available; skipping Step-2 benchmark.")


if __name__ == "__main__":
    main()
