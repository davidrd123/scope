Got it - I've updated the plan to focus on drafting the fused kernel, wrapper, and feature flag integration only.


Below is a concrete “Option 1” drop-in: **new fused Triton kernel module + wrapper + feature-flag integration** for **D=128 fast-path only**, with:

* **3-way axis lookup** (time/height/width) inside the kernel (no `(seq_len, C)` materialization)
* **module-scope axis-table cache** (shared between model + causal model)
* **in-place rotation** when `not torch.is_grad_enabled()` **and** `seq_len == L`
* **tail preservation** via `out = x.clone()` only when needed (`seq_len < L`)
* **guardrails**: layout trap, chunk-size checks, start_frame bounds, and safe fallback.

---

## 1) Add new module: `src/scope/core/kernels/triton_rope_fused.py`

Create this file:

```py
# src/scope/core/kernels/triton_rope_fused.py
# Copyright (c) 2025
# Triton fused 3-way RoPE (time/height/width) lookup + rotate.
#
# Fast-path target: D=128 (C=64 pairs).
# Avoids materializing (seq_len, C) cos/sin tables.

from __future__ import annotations

import os
from typing import Dict, Tuple, Optional, Sequence, Union

import torch

import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Module-scope cache: per-axis tables keyed by (freqs_ptr, device_index, c0,c1,c2)
# This is intentionally module scope (like your existing _ROPE_CACHE).
# -----------------------------------------------------------------------------
_ROPE_AXIS_CACHE: Dict[
    Tuple[int, int, int, int, int],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def clear_rope_axis_cache() -> None:
    _ROPE_AXIS_CACHE.clear()


def get_rope_axis_tables(
    freqs: torch.Tensor,
    c0: int,
    c1: int,
    c2: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split freqs[:, :C] into 3 semantic chunks and cache contiguous float32 cos/sin tables:
      - time chunk:  [max_seq, c0]
      - height chunk:[max_seq, c1]
      - width chunk: [max_seq, c2]

    Returns:
      cos_f, sin_f, cos_h, sin_h, cos_w, sin_w (all float32, contiguous, on `device`)
    """
    if freqs.device != device:
        freqs = freqs.to(device)

    if freqs.dtype not in (torch.complex64, torch.complex128):
        raise TypeError(f"freqs must be complex, got {freqs.dtype}")

    if device.type != "cuda":
        raise RuntimeError("get_rope_axis_tables requires CUDA device for fused kernel path")

    dev_index = device.index if device.index is not None else 0
    key = (int(freqs.data_ptr()), int(dev_index), int(c0), int(c1), int(c2))
    cached = _ROPE_AXIS_CACHE.get(key)
    if cached is not None:
        return cached

    # Split into semantic chunks (time/height/width)
    f0, f1, f2 = freqs.split([c0, c1, c2], dim=1)

    # .real/.imag are views; contiguous() materializes dense float32 tables.
    cos_f = f0.real.contiguous().to(dtype=torch.float32, device=device)
    sin_f = f0.imag.contiguous().to(dtype=torch.float32, device=device)

    cos_h = f1.real.contiguous().to(dtype=torch.float32, device=device)
    sin_h = f1.imag.contiguous().to(dtype=torch.float32, device=device)

    cos_w = f2.real.contiguous().to(dtype=torch.float32, device=device)
    sin_w = f2.imag.contiguous().to(dtype=torch.float32, device=device)

    cached = (cos_f, sin_f, cos_h, sin_h, cos_w, sin_w)
    _ROPE_AXIS_CACHE[key] = cached
    return cached


# -----------------------------------------------------------------------------
# Triton kernel: grid = (B*H, ceil_div(seq_len, BLOCK_L))
# Applies 3-way RoPE lookup + rotate in 3 contiguous chunks.
# x is interleaved pairs: (2*j, 2*j+1)
# -----------------------------------------------------------------------------
@triton.jit
def rope_fused_3way_kernel(
    X_ptr,
    OUT_ptr,
    COS_F_ptr,
    SIN_F_ptr,
    COS_H_ptr,
    SIN_H_ptr,
    COS_W_ptr,
    SIN_W_ptr,
    # x strides: [B, L, H, D]
    stride_xb,
    stride_xl,
    stride_xh,
    stride_xd,
    # out strides: [B, L, H, D]
    stride_ob,
    stride_ol,
    stride_oh,
    stride_od,
    # cos/sin strides (all 2D): [max_seq, c_axis]
    stride_cf_l,
    stride_cf_c,
    stride_ch_l,
    stride_ch_c,
    stride_cw_l,
    stride_cw_c,
    # sizes / params
    L,          # padded length (x.shape[1])
    HN,         # num heads (x.shape[2])
    HW,         # grid_h * grid_w
    W,          # grid_w
    START_FRAME,
    SEQ_LEN,    # f * grid_h * grid_w
    # meta
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_l = tl.program_id(1)

    b = pid_bh // HN
    h = pid_bh - b * HN

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < SEQ_LEN

    # Compute token -> (f_idx, h_idx, w_idx) without modulo ops:
    # f_local = t // (H*W)
    # rem = t - f_local*(H*W)
    # h_idx = rem // W
    # w_idx = rem - h_idx*W
    t = offs_l
    f_local = t // HW
    rem = t - f_local * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    f_idx = START_FRAME + f_local

    x_base = X_ptr + b * stride_xb + h * stride_xh
    o_base = OUT_ptr + b * stride_ob + h * stride_oh

    # ----------------------------
    # Chunk 0 (time): pairs [0:C0]
    # ----------------------------
    offs_c0 = tl.arange(0, C0)

    cos0 = tl.load(
        COS_F_ptr + f_idx[:, None] * stride_cf_l + offs_c0[None, :] * stride_cf_c,
        mask=mask_l[:, None],
        other=1.0,
    ).to(tl.float32)
    sin0 = tl.load(
        SIN_F_ptr + f_idx[:, None] * stride_cf_l + offs_c0[None, :] * stride_cf_c,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)

    x0_0 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * offs_c0)[None, :] * stride_xd,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)
    x1_0 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * offs_c0 + 1)[None, :] * stride_xd,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)

    y0_0 = x0_0 * cos0 - x1_0 * sin0
    y1_0 = x0_0 * sin0 + x1_0 * cos0

    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * offs_c0)[None, :] * stride_od,
        y0_0.to(OUT_ptr.dtype.element_ty),
        mask=mask_l[:, None],
    )
    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * offs_c0 + 1)[None, :] * stride_od,
        y1_0.to(OUT_ptr.dtype.element_ty),
        mask=mask_l[:, None],
    )

    # -----------------------------
    # Chunk 1 (height): pairs [C0:C0+C1]
    # -----------------------------
    offs_c1 = tl.arange(0, C1)
    pair1 = C0 + offs_c1

    cos1 = tl.load(
        COS_H_ptr + h_idx[:, None] * stride_ch_l + offs_c1[None, :] * stride_ch_c,
        mask=mask_l[:, None],
        other=1.0,
    ).to(tl.float32)
    sin1 = tl.load(
        SIN_H_ptr + h_idx[:, None] * stride_ch_l + offs_c1[None, :] * stride_ch_c,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)

    x0_1 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * pair1)[None, :] * stride_xd,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)
    x1_1 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * pair1 + 1)[None, :] * stride_xd,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)

    y0_1 = x0_1 * cos1 - x1_1 * sin1
    y1_1 = x0_1 * sin1 + x1_1 * cos1

    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * pair1)[None, :] * stride_od,
        y0_1.to(OUT_ptr.dtype.element_ty),
        mask=mask_l[:, None],
    )
    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * pair1 + 1)[None, :] * stride_od,
        y1_1.to(OUT_ptr.dtype.element_ty),
        mask=mask_l[:, None],
    )

    # -----------------------------
    # Chunk 2 (width): pairs [C0+C1:C0+C1+C2]
    # -----------------------------
    offs_c2 = tl.arange(0, C2)
    pair2 = C0 + C1 + offs_c2

    cos2 = tl.load(
        COS_W_ptr + w_idx[:, None] * stride_cw_l + offs_c2[None, :] * stride_cw_c,
        mask=mask_l[:, None],
        other=1.0,
    ).to(tl.float32)
    sin2 = tl.load(
        SIN_W_ptr + w_idx[:, None] * stride_cw_l + offs_c2[None, :] * stride_cw_c,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)

    x0_2 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * pair2)[None, :] * stride_xd,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)
    x1_2 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * pair2 + 1)[None, :] * stride_xd,
        mask=mask_l[:, None],
        other=0.0,
    ).to(tl.float32)

    y0_2 = x0_2 * cos2 - x1_2 * sin2
    y1_2 = x0_2 * sin2 + x1_2 * cos2

    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * pair2)[None, :] * stride_od,
        y0_2.to(OUT_ptr.dtype.element_ty),
        mask=mask_l[:, None],
    )
    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * pair2 + 1)[None, :] * stride_od,
        y1_2.to(OUT_ptr.dtype.element_ty),
        mask=mask_l[:, None],
    )


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------
def _as_int3(
    grid_sizes: Union[torch.Tensor, Sequence[int]],
    batch: int,
) -> Tuple[int, int, int]:
    """
    Returns (f, gh, gw). Only supports:
      - [3] tensor/list
      - [1,3] tensor
      - [B,3] tensor where all rows are identical
    """
    if isinstance(grid_sizes, (tuple, list)):
        if len(grid_sizes) != 3:
            raise ValueError(f"grid_sizes must be len==3, got {len(grid_sizes)}")
        f, gh, gw = grid_sizes
        return int(f), int(gh), int(gw)

    if not isinstance(grid_sizes, torch.Tensor):
        raise TypeError(f"grid_sizes must be Tensor or sequence, got {type(grid_sizes)}")

    if grid_sizes.ndim == 1:
        if grid_sizes.numel() != 3:
            raise ValueError(f"grid_sizes[ndim=1] must have 3 elements, got {grid_sizes.numel()}")
        f, gh, gw = grid_sizes.tolist()
        return int(f), int(gh), int(gw)

    if grid_sizes.ndim == 2:
        if grid_sizes.shape[1] != 3:
            raise ValueError(f"grid_sizes[ndim=2] must be [*,3], got {tuple(grid_sizes.shape)}")
        # B==1 fast path
        if grid_sizes.shape[0] == 1:
            f, gh, gw = grid_sizes[0].tolist()
            return int(f), int(gh), int(gw)

        # Only fuse when identical grid sizes across batch
        # (avoids per-sample Python loops, which killed earlier gains)
        first = grid_sizes[0]
        if not torch.all(grid_sizes == first):
            raise NotImplementedError("Fused RoPE only supports identical grid_sizes across batch")
        f, gh, gw = first.tolist()
        return int(f), int(gh), int(gw)

    raise ValueError(f"Unsupported grid_sizes.ndim={grid_sizes.ndim}")


def rope_fused_3way(
    x: torch.Tensor,
    grid_sizes: Union[torch.Tensor, Sequence[int]],
    freqs: torch.Tensor,
    *,
    start_frame: int = 0,
    inplace: Optional[bool] = None,
    block_l: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> torch.Tensor:
    """
    Fused 3-way RoPE (time/height/width) for x in [B, L, H, D] interleaved-pair layout.

    Fast-path constraints:
      - CUDA only
      - D == 128 (so C == 64)
      - freqs is complex, shape [>=start_frame+f, 64]

    Tail handling:
      - if seq_len < L and not inplace: out = x.clone() then masked stores (preserves tail)
      - if seq_len == L and not inplace: out = empty_like(x), kernel writes all tokens
      - if inplace: writes into x for t < seq_len; tail untouched

    In-place policy (default):
      - if inplace is None: allow inplace only when (not torch.is_grad_enabled()) and (seq_len == L)
    """
    if x.device.type != "cuda":
        raise RuntimeError("rope_fused_3way requires CUDA tensor")

    if x.ndim != 4:
        raise ValueError(f"x must be 4D [B,L,H,D], got shape={tuple(x.shape)}")

    B, L, HN, D = x.shape

    # D=128 only per your decision.
    if D != 128:
        raise NotImplementedError(f"rope_fused_3way only supports D=128, got D={D}")

    if D % 2 != 0:
        raise ValueError(f"head_dim D must be even, got D={D}")

    f, gh, gw = _as_int3(grid_sizes, batch=B)
    seq_len = int(f) * int(gh) * int(gw)

    # Layout guard: expects L dimension at x.shape[1]
    if seq_len > L:
        # This is the SAGEATTN layout trap case if someone passed [B,H,L,D],
        # or simply inconsistent padding.
        raise ValueError(
            f"Expected x as [B,L,H,D] with L>=seq_len. Got L={L}, seq_len={seq_len}, shape={tuple(x.shape)}"
        )

    # Chunk sizes are semantic. Compute and assert.
    C = D // 2
    c1 = C // 3
    c2 = C // 3
    c0 = C - c1 - c2  # == C - 2*(C//3)

    if c0 != C - 2 * (C // 3):
        raise AssertionError("Chunk size formula mismatch")

    if (c0 + c1 + c2) != C:
        raise AssertionError(f"Chunk sizes don't sum to C: {c0}+{c1}+{c2} != {C}")

    # Bounds guard for causal offset (only time axis gets start_frame)
    start_frame = int(start_frame)
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if (start_frame + f) > freqs.shape[0]:
        raise ValueError(
            f"start_frame+f out of bounds: start_frame={start_frame}, f={f}, freqs_len={freqs.shape[0]}"
        )
    if freqs.shape[1] != C:
        raise ValueError(f"freqs second dim must be C={C}, got {freqs.shape[1]}")

    # Default inplace policy (inference only)
    if inplace is None:
        inplace = (not torch.is_grad_enabled()) and (seq_len == L)

    # Output allocation / tail preservation
    if inplace:
        out = x
    else:
        if seq_len == L:
            out = torch.empty_like(x)
        else:
            out = x.clone()

    # Axis tables (cached, float32 contiguous)
    cos_f, sin_f, cos_h, sin_h, cos_w, sin_w = get_rope_axis_tables(freqs, c0, c1, c2, x.device)

    # Kernel launch params (overrideable via args/env)
    if block_l is None:
        block_l = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_L", "128"))
    if num_warps is None:
        num_warps = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_NUM_WARPS", "4"))
    if num_stages is None:
        num_stages = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_NUM_STAGES", "2"))

    hw = int(gh) * int(gw)

    grid = (B * HN, triton.cdiv(seq_len, block_l))

    # Mirror triton_rotary’s device guard to avoid launching on cuda:0 accidentally.
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(rope_fused_3way_kernel)[grid](
            x,
            out,
            cos_f,
            sin_f,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            cos_f.stride(0),
            cos_f.stride(1),
            cos_h.stride(0),
            cos_h.stride(1),
            cos_w.stride(0),
            cos_w.stride(1),
            L,
            HN,
            hw,
            int(gw),
            start_frame,
            seq_len,
            C0=c0,
            C1=c1,
            C2=c2,
            BLOCK_L=block_l,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return out
```

---

## 2) Integrate feature flag + callsite in `krea_realtime_video/modules/model.py`

Near your existing Triton rotary import block in:

`src/scope/core/pipelines/krea_realtime_video/modules/model.py`

Add this **right next to** the `triton_rotary` import / flags:

```py
# Try to import Triton fused 3-way RoPE kernel
try:
    from scope.core.kernels.triton_rope_fused import rope_fused_3way as triton_rope_fused_3way
    TRITON_ROPE_FUSED_AVAILABLE = True
except Exception:
    TRITON_ROPE_FUSED_AVAILABLE = False
    triton_rope_fused_3way = None

# Env var to disable fused RoPE (for debugging/comparison)
USE_TRITON_ROPE_FUSED = TRITON_ROPE_FUSED_AVAILABLE and os.environ.get("SCOPE_DISABLE_TRITON_ROPE_FUSED", "0") != "1"
if TRITON_ROPE_FUSED_AVAILABLE:
    print(f"TRITON_ROPE_FUSED_AVAILABLE: {TRITON_ROPE_FUSED_AVAILABLE}, USE_TRITON_ROPE_FUSED: {USE_TRITON_ROPE_FUSED}")
```

Then, **inside `rope_apply(...)`**, add the fused fast-path at the top (before building `freqs_i` / calling `get_rope_cos_sin`):

```py
def rope_apply(x, grid_sizes, freqs):
    # --- Fused Triton 3-way RoPE fast path (D=128 only) ---
    if USE_TRITON_ROPE_FUSED and x.is_cuda and x.ndim == 4 and x.shape[-1] == 128:
        try:
            # In-place allowed automatically by wrapper only for inference (grad disabled) + seq_len==L
            return triton_rope_fused_3way(x, grid_sizes, freqs, start_frame=0, inplace=None)
        except Exception:
            # Optional strict mode to surface kernel issues early:
            if os.environ.get("SCOPE_TRITON_ROPE_FUSED_STRICT", "0") == "1":
                raise
            # Otherwise fall back to existing implementation
            pass

    # --- existing rope_apply implementation continues below ---
    ...
```

That’s it for `rope_apply`: you’re now fused on D=128 when enabled, and otherwise you fall back cleanly.

---

## 3) Integrate into `krea_realtime_video/modules/causal_model.py`

In:

`src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

Add the same import/flag block near the top (after `import os`):

```py
try:
    from scope.core.kernels.triton_rope_fused import rope_fused_3way as triton_rope_fused_3way
    TRITON_ROPE_FUSED_AVAILABLE = True
except Exception:
    TRITON_ROPE_FUSED_AVAILABLE = False
    triton_rope_fused_3way = None

USE_TRITON_ROPE_FUSED = TRITON_ROPE_FUSED_AVAILABLE and os.environ.get("SCOPE_DISABLE_TRITON_ROPE_FUSED", "0") != "1"
if TRITON_ROPE_FUSED_AVAILABLE:
    print(f"TRITON_ROPE_FUSED_AVAILABLE: {TRITON_ROPE_FUSED_AVAILABLE}, USE_TRITON_ROPE_FUSED: {USE_TRITON_ROPE_FUSED}")
```

Then **inside `causal_rope_apply(x, grid_sizes, freqs, start_frame)`**, add:

```py
def causal_rope_apply(x, grid_sizes, freqs, start_frame: int):
    # --- Fused Triton 3-way RoPE fast path (D=128 only) ---
    if USE_TRITON_ROPE_FUSED and x.is_cuda and x.ndim == 4 and x.shape[-1] == 128:
        try:
            return triton_rope_fused_3way(x, grid_sizes, freqs, start_frame=int(start_frame), inplace=None)
        except Exception:
            if os.environ.get("SCOPE_TRITON_ROPE_FUSED_STRICT", "0") == "1":
                raise
            pass

    # --- existing causal_rope_apply implementation continues below ---
    ...
```

This preserves the “**start_frame offsets only the time axis**” rule, because the wrapper/kernel do exactly:
`f_idx = start_frame + t // (h*w)`.

---

## Notes on expectations / behavior

* **D=128-only**: Wrapper raises `NotImplementedError` for other D; the integration catches it and falls back.
* **SAGEATTN layout trap**: If someone passes `[B,H,L,D]`, wrapper sees `seq_len > x.shape[1]` and raises → you fall back. (Today SAGEATTN is false, but this prevents silent corruption later.)
* **Tail preservation**:

  * `seq_len == L`: no clone, out is fully written
  * `seq_len < L`: wrapper does `out = x.clone()` and kernel writes only the prefix
  * `inplace=True`: only prefix written, tail untouched
* **Cache**: axis tables are cached at **module scope** in `triton_rope_fused.py`, keyed by `(freqs.data_ptr(), device_index, c0,c1,c2)`.

---

## How to run with your existing harnesses

Enable fused path (default is enabled if import succeeds and env var not set):

```bash
# run correctness
python scripts/test_rope_correctness.py

# run benchmark
python scripts/bench_triton_rope.py
```

Disable fused path for A/B:

```bash
SCOPE_DISABLE_TRITON_ROPE_FUSED=1 python scripts/bench_triton_rope.py
```

Optional: make kernel errors crash loudly (instead of falling back):

```bash
SCOPE_TRITON_ROPE_FUSED_STRICT=1 python scripts/test_rope_correctness.py
```

Tune launch params quickly without editing code:

```bash
SCOPE_TRITON_ROPE_FUSED_BLOCK_L=128 \
SCOPE_TRITON_ROPE_FUSED_NUM_WARPS=8 \
SCOPE_TRITON_ROPE_FUSED_NUM_STAGES=2 \
python scripts/bench_triton_rope.py
```

---

If you want the *next* iteration after this lands: I’d target the two biggest performance levers on B200:

1. **row-based grid** to reduce per-element div/mod (your kernel skeleton uses per-token division), and/or
2. **fuse Q+K** into one kernel call (cuts launches + repeats of index math and table loads).
