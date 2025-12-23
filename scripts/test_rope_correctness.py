#!/usr/bin/env python3
"""
Correctness checks for RoPE paths (PyTorch reference vs rope_apply/causal_rope_apply).

Examples:
  uv run python scripts/test_rope_correctness.py
  SCOPE_DISABLE_TRITON_ROTARY=1 uv run python scripts/test_rope_correctness.py
  uv run python scripts/test_rope_correctness.py --f 3 --h 30 --w 52 --pad 64
  uv run python scripts/test_rope_correctness.py --f 3 --h 30 --w 52 --pad-to-multiple 128
"""

import argparse
import math
import os
import sys

import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoPE correctness checks.")
    parser.add_argument("--f", type=int, default=3, help="Frames (F)")
    parser.add_argument("--h", type=int, default=30, help="Height tokens (H)")
    parser.add_argument("--w", type=int, default=52, help="Width tokens (W)")
    parser.add_argument("--heads", type=int, default=16, help="Num heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dim (D)")
    parser.add_argument("--start-frames", type=int, nargs="*", default=[0, 2], help="start_frame values to test")
    parser.add_argument("--pad", type=int, default=0, help="Extra tail tokens to preserve")
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        default=0,
        help="Right-pad L so (seq_len+pad) is a multiple of this value (added on top of --pad).",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="Input dtype")
    parser.add_argument("--atol", type=float, default=None, help="Max error tolerance (abs)")
    parser.add_argument(
        "--disable-triton-rotary",
        action="store_true",
        help="Set SCOPE_DISABLE_TRITON_ROTARY=1 (must happen before imports).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    if args.disable_triton_rotary:
        os.environ["SCOPE_DISABLE_TRITON_ROTARY"] = "1"

    # Import after env var is set (caller can also set SCOPE_DISABLE_TRITON_ROTARY=1)
    from scope.core.pipelines.krea_realtime_video.modules.model import (
        get_rope_cos_sin,
        rope_apply,
        rope_params,
        USE_TRITON_ROTARY,
    )
    from scope.core.pipelines.krea_realtime_video.modules.causal_model import (
        causal_rope_apply,
    )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = torch.device("cuda")
    atol = args.atol
    if atol is None:
        atol = 5e-2 if dtype is torch.bfloat16 else 2e-2

    f, h, w = args.f, args.h, args.w
    seq_len = f * h * w
    pad_to_multiple = int(args.pad_to_multiple)
    pad_mult = 0
    if pad_to_multiple > 0:
        pad_mult = math.ceil(seq_len / pad_to_multiple) * pad_to_multiple - seq_len
    b, n, d = 1, args.heads, args.head_dim

    def build_freqs(max_seq_len: int, head_dim: int) -> torch.Tensor:
        """Match the model's freqs construction (rope_params only)."""
        dim = head_dim
        c0_dim = dim - 4 * (dim // 6)
        c1_dim = 2 * (dim // 6)
        c2_dim = 2 * (dim // 6)
        freqs_local = torch.cat(
            [
                rope_params(max_seq_len, c0_dim),
                rope_params(max_seq_len, c1_dim),
                rope_params(max_seq_len, c2_dim),
            ],
            dim=1,
        )
        return freqs_local.to(device)

    def rope_ref(
        x_tensor: torch.Tensor,
        freqs_tensor: torch.Tensor,
        f_val: int,
        h_val: int,
        w_val: int,
        start_frame: int,
    ) -> torch.Tensor:
        """Reference RoPE using cached cos/sin tables."""
        nheads, c = x_tensor.size(2), x_tensor.size(3) // 2
        freqs_split = freqs_tensor.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        cos, sin = get_rope_cos_sin(
            freqs_split,
            f_val,
            h_val,
            w_val,
            start_frame,
            x_tensor.dtype,
            x_tensor.device,
            c,
        )

        seq_len_local = f_val * h_val * w_val
        x_i = x_tensor[:, :seq_len_local].reshape(
            x_tensor.size(0), seq_len_local, nheads, c, 2
        )
        x0, x1 = x_i[..., 0], x_i[..., 1]
        x0_new = x0 * cos - x1 * sin
        x1_new = x0 * sin + x1 * cos
        x_i = torch.stack([x0_new, x1_new], dim=-1).flatten(3)
        return torch.cat([x_i, x_tensor[:, seq_len_local:]], dim=1)

    freqs = build_freqs(1024, d)
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)

    pad_total = int(args.pad) + int(pad_mult)
    x = torch.randn(b, seq_len + pad_total, n, d, device=device, dtype=dtype)

    print(f"USE_TRITON_ROTARY={USE_TRITON_ROTARY}")
    print(
        f"shape: B={b} L={seq_len}+{pad_total} (pad={args.pad} pad_mult={pad_mult}) H={n} D={d} dtype={args.dtype}"
    )

    # rope_apply (start_frame=0)
    ref = rope_ref(x, freqs, f, h, w, start_frame=0)
    out = rope_apply(x, grid_sizes, freqs)
    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()
    print(f"rope_apply: max_err={max_err:.6f} mean_err={mean_err:.6f}")
    if max_err > atol:
        print("rope_apply: FAIL")
        sys.exit(1)

    # causal_rope_apply for each start_frame
    for sf in args.start_frames:
        ref = rope_ref(x, freqs, f, h, w, start_frame=sf)
        out = causal_rope_apply(x, grid_sizes, freqs, start_frame=sf)
        max_err = (out - ref).abs().max().item()
        mean_err = (out - ref).abs().mean().item()
        print(f"causal_rope_apply(start_frame={sf}): max_err={max_err:.6f} mean_err={mean_err:.6f}")
        if max_err > atol:
            print("causal_rope_apply: FAIL")
            sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
