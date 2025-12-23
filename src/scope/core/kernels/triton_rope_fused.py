# src/scope/core/kernels/triton_rope_fused.py
# Copyright (c) 2025
# Triton fused 3-way RoPE (time/height/width) lookup + rotate.
#
# Fast-path target: D=128 (C=64 pairs).
# Avoids materializing (seq_len, C) cos/sin tables.

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

import triton
import triton.language as tl

# -----------------------------------------------------------------------------
# Module-scope cache: per-axis tables keyed by (freqs_ptr, device_index, c0,c1,c2)
# This is intentionally module scope (like the existing _ROPE_CACHE).
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
    L,  # padded length (x.shape[1])
    HN,  # num heads (x.shape[2])
    HW,  # grid_h * grid_w
    W,  # grid_w
    START_FRAME,
    SEQ_LEN,  # f * grid_h * grid_w
    # meta - actual chunk sizes
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    # meta - padded power-of-2 sizes for arange
    C0_PAD: tl.constexpr,
    C1_PAD: tl.constexpr,
    C2_PAD: tl.constexpr,
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
    # Use padded power-of-2 arange with mask
    # ----------------------------
    offs_c0 = tl.arange(0, C0_PAD)
    mask_c0 = offs_c0 < C0

    cos0 = tl.load(
        COS_F_ptr + f_idx[:, None] * stride_cf_l + offs_c0[None, :] * stride_cf_c,
        mask=mask_l[:, None] & mask_c0[None, :],
        other=1.0,
    ).to(tl.float32)
    sin0 = tl.load(
        SIN_F_ptr + f_idx[:, None] * stride_cf_l + offs_c0[None, :] * stride_cf_c,
        mask=mask_l[:, None] & mask_c0[None, :],
        other=0.0,
    ).to(tl.float32)

    x0_0 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * offs_c0)[None, :] * stride_xd,
        mask=mask_l[:, None] & mask_c0[None, :],
        other=0.0,
    ).to(tl.float32)
    x1_0 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * offs_c0 + 1)[None, :] * stride_xd,
        mask=mask_l[:, None] & mask_c0[None, :],
        other=0.0,
    ).to(tl.float32)

    y0_0 = x0_0 * cos0 - x1_0 * sin0
    y1_0 = x0_0 * sin0 + x1_0 * cos0

    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * offs_c0)[None, :] * stride_od,
        y0_0,
        mask=mask_l[:, None] & mask_c0[None, :],
    )
    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * offs_c0 + 1)[None, :] * stride_od,
        y1_0,
        mask=mask_l[:, None] & mask_c0[None, :],
    )

    # -----------------------------
    # Chunk 1 (height): pairs [C0:C0+C1]
    # -----------------------------
    offs_c1 = tl.arange(0, C1_PAD)
    mask_c1 = offs_c1 < C1
    pair1 = C0 + offs_c1

    cos1 = tl.load(
        COS_H_ptr + h_idx[:, None] * stride_ch_l + offs_c1[None, :] * stride_ch_c,
        mask=mask_l[:, None] & mask_c1[None, :],
        other=1.0,
    ).to(tl.float32)
    sin1 = tl.load(
        SIN_H_ptr + h_idx[:, None] * stride_ch_l + offs_c1[None, :] * stride_ch_c,
        mask=mask_l[:, None] & mask_c1[None, :],
        other=0.0,
    ).to(tl.float32)

    x0_1 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * pair1)[None, :] * stride_xd,
        mask=mask_l[:, None] & mask_c1[None, :],
        other=0.0,
    ).to(tl.float32)
    x1_1 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * pair1 + 1)[None, :] * stride_xd,
        mask=mask_l[:, None] & mask_c1[None, :],
        other=0.0,
    ).to(tl.float32)

    y0_1 = x0_1 * cos1 - x1_1 * sin1
    y1_1 = x0_1 * sin1 + x1_1 * cos1

    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * pair1)[None, :] * stride_od,
        y0_1,
        mask=mask_l[:, None] & mask_c1[None, :],
    )
    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * pair1 + 1)[None, :] * stride_od,
        y1_1,
        mask=mask_l[:, None] & mask_c1[None, :],
    )

    # -----------------------------
    # Chunk 2 (width): pairs [C0+C1:C0+C1+C2]
    # -----------------------------
    offs_c2 = tl.arange(0, C2_PAD)
    mask_c2 = offs_c2 < C2
    pair2 = C0 + C1 + offs_c2

    cos2 = tl.load(
        COS_W_ptr + w_idx[:, None] * stride_cw_l + offs_c2[None, :] * stride_cw_c,
        mask=mask_l[:, None] & mask_c2[None, :],
        other=1.0,
    ).to(tl.float32)
    sin2 = tl.load(
        SIN_W_ptr + w_idx[:, None] * stride_cw_l + offs_c2[None, :] * stride_cw_c,
        mask=mask_l[:, None] & mask_c2[None, :],
        other=0.0,
    ).to(tl.float32)

    x0_2 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * pair2)[None, :] * stride_xd,
        mask=mask_l[:, None] & mask_c2[None, :],
        other=0.0,
    ).to(tl.float32)
    x1_2 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * pair2 + 1)[None, :] * stride_xd,
        mask=mask_l[:, None] & mask_c2[None, :],
        other=0.0,
    ).to(tl.float32)

    y0_2 = x0_2 * cos2 - x1_2 * sin2
    y1_2 = x0_2 * sin2 + x1_2 * cos2

    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * pair2)[None, :] * stride_od,
        y0_2,
        mask=mask_l[:, None] & mask_c2[None, :],
    )
    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * pair2 + 1)[None, :] * stride_od,
        y1_2,
        mask=mask_l[:, None] & mask_c2[None, :],
    )


# -----------------------------------------------------------------------------
# v2 kernel: unified 64-pair arange (no per-chunk padding)
# Grid: (H/BLOCK_H, seq_len/BLOCK_M, B)
# Key insight: C=64 is already power-of-2, so use unified rk=0..63 with masks
# -----------------------------------------------------------------------------
@triton.jit
def rope_fused_3way_kernel_v2(
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
    # cos/sin strides (2D): [max_seq, c_axis]
    stride_cf_l,
    stride_cf_c,
    stride_ch_l,
    stride_ch_c,
    stride_cw_l,
    stride_cw_c,
    # sizes / params
    SEQ_LEN,  # f * gh * gw (active tokens)
    HN,  # num heads
    HW,  # gh * gw
    W,  # gw
    START_FRAME,
    # meta
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_h = tl.program_id(0)  # head blocks
    pid_m = tl.program_id(1)  # token blocks
    pid_b = tl.program_id(2)  # batch

    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rm < SEQ_LEN
    mask_h = rh < HN

    # token -> (f_idx, h_idx, w_idx)
    t = rm
    f_local = t // HW
    rem = t - f_local * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W
    f_idx = START_FRAME + f_local

    # Build cos/sin for all 64 pairs (C=64 is power-of-two for D=128)
    rk = tl.arange(0, 64)
    cos = tl.full((BLOCK_M, 64), 1.0, tl.float32)
    sin = tl.full((BLOCK_M, 64), 0.0, tl.float32)

    # time chunk [0:C0)
    m0 = rk < C0
    cos0 = tl.load(
        COS_F_ptr + f_idx[:, None] * stride_cf_l + rk[None, :] * stride_cf_c,
        mask=mask_m[:, None] & m0[None, :],
        other=1.0,
    ).to(tl.float32)
    sin0 = tl.load(
        SIN_F_ptr + f_idx[:, None] * stride_cf_l + rk[None, :] * stride_cf_c,
        mask=mask_m[:, None] & m0[None, :],
        other=0.0,
    ).to(tl.float32)
    cos = tl.where(m0[None, :], cos0, cos)
    sin = tl.where(m0[None, :], sin0, sin)

    # height chunk [C0:C0+C1)
    m1 = (rk >= C0) & (rk < (C0 + C1))
    idx1 = rk - C0
    idx1 = tl.where(m1, idx1, 0)  # keep pointers sane for masked lanes
    cos1 = tl.load(
        COS_H_ptr + h_idx[:, None] * stride_ch_l + idx1[None, :] * stride_ch_c,
        mask=mask_m[:, None] & m1[None, :],
        other=1.0,
    ).to(tl.float32)
    sin1 = tl.load(
        SIN_H_ptr + h_idx[:, None] * stride_ch_l + idx1[None, :] * stride_ch_c,
        mask=mask_m[:, None] & m1[None, :],
        other=0.0,
    ).to(tl.float32)
    cos = tl.where(m1[None, :], cos1, cos)
    sin = tl.where(m1[None, :], sin1, sin)

    # width chunk [C0+C1:C0+C1+C2)
    m2 = rk >= (C0 + C1)
    idx2 = rk - (C0 + C1)
    idx2 = tl.where(m2, idx2, 0)  # keep pointers sane for masked lanes
    cos2 = tl.load(
        COS_W_ptr + w_idx[:, None] * stride_cw_l + idx2[None, :] * stride_cw_c,
        mask=mask_m[:, None] & m2[None, :],
        other=1.0,
    ).to(tl.float32)
    sin2 = tl.load(
        SIN_W_ptr + w_idx[:, None] * stride_cw_l + idx2[None, :] * stride_cw_c,
        mask=mask_m[:, None] & m2[None, :],
        other=0.0,
    ).to(tl.float32)
    cos = tl.where(m2[None, :], cos2, cos)
    sin = tl.where(m2[None, :], sin2, sin)

    # Load X as [BLOCK_H, BLOCK_M, 128] (interleaved pairs)
    rk_d = tl.arange(0, 128)

    x_base = X_ptr + pid_b * stride_xb
    o_base = OUT_ptr + pid_b * stride_ob

    x_ptrs = (
        x_base
        + rh[:, None, None] * stride_xh
        + rm[None, :, None] * stride_xl
        + rk_d[None, None, :] * stride_xd
    )
    o_ptrs = (
        o_base
        + rh[:, None, None] * stride_oh
        + rm[None, :, None] * stride_ol
        + rk_d[None, None, :] * stride_od
    )

    mask = mask_h[:, None, None] & mask_m[None, :, None]  # broadcast over rk_d

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, 64, 2]))

    cos_b = cos[None, :, :]
    sin_b = sin[None, :, :]

    y0 = x0 * cos_b - x1 * sin_b
    y1 = x0 * sin_b + x1 * cos_b

    y = tl.reshape(tl.join(y0, y1), [BLOCK_H, BLOCK_M, 128])
    tl.store(o_ptrs, y, mask=mask)


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


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


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

    Tail handling (CRITICAL - avoids full tensor clone):
      - if seq_len < L and not inplace: out = empty_like(x), kernel writes prefix,
        then tiny tail copy: out[:, seq_len:] = x[:, seq_len:]
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

    # D=128 only per decision.
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

    # Default inplace policy (inference only, full tensor)
    if inplace is None:
        inplace = (not torch.is_grad_enabled()) and (seq_len == L)

    # Output allocation / tail preservation
    # CRITICAL FIX: Never clone full tensor. Use empty_like + tiny tail copy.
    if inplace:
        out = x
    else:
        out = torch.empty_like(x)
        # Kernel will write prefix; copy tail if needed
        if seq_len < L:
            out[:, seq_len:] = x[:, seq_len:]

    # Axis tables (cached, float32 contiguous)
    cos_f, sin_f, cos_h, sin_h, cos_w, sin_w = get_rope_axis_tables(freqs, c0, c1, c2, x.device)

    # A/B switch: v1 (padded) vs v2 (unified 64-pair)
    impl = os.environ.get("SCOPE_TRITON_ROPE_FUSED_IMPL", "v2").lower()

    if num_warps is None:
        num_warps = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_NUM_WARPS", "4"))
    if num_stages is None:
        num_stages = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_NUM_STAGES", "2"))

    hw = int(gh) * int(gw)

    # Mirror triton_rotary's device guard to avoid launching on cuda:0 accidentally.
    with torch.cuda.device(x.device.index):
        if impl == "v1":
            # v1: padded kernel (for A/B comparison - causes regression)
            if block_l is None:
                block_l = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_L", "128"))

            c0_pad = _next_power_of_2(c0)
            c1_pad = _next_power_of_2(c1)
            c2_pad = _next_power_of_2(c2)

            grid = (B * HN, triton.cdiv(seq_len, block_l))

            rope_fused_3way_kernel[grid](
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
                C0_PAD=c0_pad,
                C1_PAD=c1_pad,
                C2_PAD=c2_pad,
                BLOCK_L=block_l,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            # v2: unified 64-pair kernel (default - fixes regression)
            block_m = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_M", "8"))
            block_h = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_H", "2"))

            grid = (
                triton.cdiv(HN, block_h),
                triton.cdiv(seq_len, block_m),
                B,
            )

            rope_fused_3way_kernel_v2[grid](
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
                seq_len,
                HN,
                hw,
                int(gw),
                start_frame,
                C0=c0,
                C1=c1,
                C2=c2,
                BLOCK_H=block_h,
                BLOCK_M=block_m,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return out
