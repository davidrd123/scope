### Diagnosis: why Step‑2 regressed even though microbench looked OK

Your current fused kernel in `src/scope/core/kernels/triton_rope_fused.py` is doing **extra channel work** because it pads each chunk’s channel arange to the next power‑of‑2:

* Real chunk sizes at **D=128** (C=64 pairs): **C0/C1/C2 = 22/21/21**
* Kernel uses: **C0_PAD/C1_PAD/C2_PAD = 32/32/32**
* That is **96 pairs vs 64 pairs ⇒ +50% lanes** (and worse: +50% register footprint for the per‑program 2D tensors)

Even with masked loads/stores, Triton still builds and carries around tensors shaped like:

* `BLOCK_L x C0_PAD` = `128 x 32` = **4096** elements per intermediate (x0, x1, cos, sin, y0, y1…)
* repeated 3× (time/height/width)

That’s a *lot* of live state and very likely pushes you into **register pressure / spills / lower occupancy** on SM100, which is exactly the kind of thing that can show up as an end‑to‑end FPS regression even when a single‑kernel microbench looks “fine”.

Tail handling is **not** the culprit anymore (you already fixed clone → `empty_like + tiny tail copy`), so the remaining dominant risk is exactly the **padded channel lanes**.

---

## Confirm the fused path is actually executing (and with what shapes)

You already have the right hooks; you just need to run them in “no‑fallback” mode and print the real runtime shapes once.

### 1) Force fused path, crash on fallback

```bash
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
SCOPE_DISABLE_TRITON_ROTARY=1 \
uv run python scripts/test_rope_correctness.py --pad-to-multiple 128
```

If that passes, you’ve confirmed:

* fused import works
* fused wrapper executes for your test shape
* fused kernel is correct on padded L cases (tail preserved)

### 2) Benchmark fused directly vs rope_apply wrappers

```bash
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
```

This prints `TRITON_FUSED_AVAILABLE` and includes:

* `rope_apply (end-to-end)`
* `causal_rope_apply (end-to-end)`
* `triton_rope_fused_3way (direct)` (if available)

### 3) Add a “print once” debug to the wrapper (recommended)

Add to `rope_fused_3way()` (near the end, after computing `c0/c1/c2` and `c*_pad`):

```py
_DEBUG = os.environ.get("SCOPE_TRITON_ROPE_FUSED_DEBUG", "0") == "1"
global _ROPE_FUSED_DEBUG_PRINTED
try:
    _ROPE_FUSED_DEBUG_PRINTED
except NameError:
    _ROPE_FUSED_DEBUG_PRINTED = False

if _DEBUG and not _ROPE_FUSED_DEBUG_PRINTED:
    _ROPE_FUSED_DEBUG_PRINTED = True
    print(
        "[rope_fused_3way] ",
        f"B={B} L={L} HN={HN} D={D} | grid=(f={f},h={gh},w={gw}) seq_len={seq_len} start_frame={start_frame} ",
        f"| chunks c0/c1/c2={c0}/{c1}/{c2} pads={c0_pad}/{c1_pad}/{c2_pad} ",
        f"| block_l={block_l} warps={num_warps} stages={num_stages} inplace={inplace}",
    )
```

Then run:

```bash
SCOPE_TRITON_ROPE_FUSED_DEBUG=1 \
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
```

This will conclusively show the **real runtime (f,h,w), L vs seq_len**, and the **pad inflation**.

---

## Verify “padding is the dominant overhead” with NCU/NSYS (what to look for)

Once you have a capture on B200:

* **Register count / spills**: the padded kernel should show notably higher `regs/thread` and possibly local memory traffic.
* **Occupancy / waves per SM**: padded kernel should show lower occupancy / fewer active warps.
* **Instruction count / math**: padded kernel executes ~1.5× more math on channels.

If you keep two variants (old padded vs fixed), this becomes an easy A/B.

---

# Fix: remove chunk padding by switching to a power‑of‑2 pair dimension (C=64) and a FlashAttn‑like tiling

Key observation: for **D=128**, the total number of pairs **C=64 is already power‑of‑two**.

So don’t build three separate aranges at sizes 22/21/21 (and then pad to 32/32/32).
Instead:

1. Use `rk_half = tl.arange(0, 64)` (power‑of‑two, vectorizable)
2. Build a full per‑token cos/sin vector of length 64 by selecting from:

   * `cos_f[f_idx, :c0]`
   * `cos_h[h_idx, :c1]` into slots `[c0:c0+c1]`
   * `cos_w[w_idx, :c2]` into slots `[c0+c1:]`
3. Rotate using the same proven layout strategy as `triton_rotary.py`:

   * `BLOCK_M=8` tokens
   * `BLOCK_H=2` heads
   * `BLOCK_K=128` headdim

This eliminates the **96‑pair inflation** entirely and also fixes the **giant 128×32 intermediates** problem.

---

## Concrete implementation (drop‑in)

Below is a patch-style rewrite of the kernel + launch while preserving:

* wrapper API
* tail copy semantics
* `inplace` semantics
* fallback behavior (unchanged in callers)

### 1) Replace the current kernel with a v2 kernel

In `src/scope/core/kernels/triton_rope_fused.py`, add this new kernel (you can keep the old one around temporarily for A/B, but default to this):

```py
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
    SEQ_LEN,     # f * gh * gw (active tokens)
    HN,          # num heads
    HW,          # gh * gw
    W,           # gw
    START_FRAME,
    # meta
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_h = tl.program_id(0)   # head blocks
    pid_m = tl.program_id(1)   # token blocks
    pid_b = tl.program_id(2)   # batch

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
    idx2 = tl.where(m2, idx2, 0)
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

    x_ptrs = x_base + rh[:, None, None] * stride_xh + rm[None, :, None] * stride_xl + rk_d[None, None, :] * stride_xd
    o_ptrs = o_base + rh[:, None, None] * stride_oh + rm[None, :, None] * stride_ol + rk_d[None, None, :] * stride_od

    mask = mask_h[:, None, None] & mask_m[None, :, None]  # broadcast

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, 64, 2]))

    cos_b = cos[None, :, :]
    sin_b = sin[None, :, :]

    y0 = x0 * cos_b - x1 * sin_b
    y1 = x0 * sin_b + x1 * cos_b

    y = tl.reshape(tl.join(y0, y1), [BLOCK_H, BLOCK_M, 128])
    tl.store(o_ptrs, y, mask=mask)
```

### 2) Update the wrapper to launch v2 and delete the C*_PAD path

In `rope_fused_3way()`:

* **Remove** `_next_power_of_2`, `c*_pad`, and the `C0_PAD/C1_PAD/C2_PAD` meta args
* Change token tiling from “BLOCK_L” to “BLOCK_M”, and add `BLOCK_H`

Example wrapper launch section:

```py
    # --- Kernel launch params ---
    # Keep old env var name as an alias, but default to a FlashAttn-like token tile.
    if block_l is None:
        block_m = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_M",
                        os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_L", "8")))
    else:
        block_m = int(block_l)

    block_h = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_BLOCK_H", "2"))

    if num_warps is None:
        num_warps = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_NUM_WARPS", "4"))
    if num_stages is None:
        num_stages = int(os.environ.get("SCOPE_TRITON_ROPE_FUSED_NUM_STAGES", "2"))

    hw = int(gh) * int(gw)

    grid = (
        triton.cdiv(HN, block_h),
        triton.cdiv(seq_len, block_m),
        B,
    )

    with torch.cuda.device(x.device.index):
        rope_fused_3way_kernel_v2[grid](
            x,
            out,
            cos_f,
            sin_f,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            cos_f.stride(0), cos_f.stride(1),
            cos_h.stride(0), cos_h.stride(1),
            cos_w.stride(0), cos_w.stride(1),
            seq_len,
            HN,
            hw,
            int(gw),
            start_frame,
            C0=c0, C1=c1, C2=c2,
            BLOCK_H=block_h,
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
```

Everything else in the wrapper (bounds checks, tail copy, cache) stays the same.

---

## Why this fixes the regression

* **No chunk padding** → you rotate **exactly 64 pairs**, not 96.
* You avoid the **huge `[128, 32]` intermediates** entirely.
* The headdim arange is **128** and the pair arange is **64** (both power‑of‑two), so you preserve the vectorization motivation without inflating work.
* Tiling (`BLOCK_M`, `BLOCK_H`) matches the proven FlashAttention rotary kernel structure, which is typically very well‑behaved on modern NVIDIA SMs.

---

## Validation checklist

### Correctness

Run:

```bash
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/test_rope_correctness.py --pad-to-multiple 128
```

Also test causal offsets:

```bash
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/test_rope_correctness.py --pad-to-multiple 128 --start-frames 0 2 5
```

### Performance (micro + wrapper)

```bash
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
```

Try quick tile tuning:

```bash
SCOPE_TRITON_ROPE_FUSED_BLOCK_M=8  uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
SCOPE_TRITON_ROPE_FUSED_BLOCK_M=16 uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
SCOPE_TRITON_ROPE_FUSED_BLOCK_M=32 uv run python scripts/bench_triton_rope.py --pad-to-multiple 128
```

### Pipeline confirmation

Run your end-to-end FPS test with:

```bash
unset SCOPE_DISABLE_TRITON_ROPE_FUSED
SCOPE_TRITON_ROPE_FUSED_STRICT=1 \
SCOPE_TRITON_ROPE_FUSED_DEBUG=1 \
<your normal run command>
```

You should see one debug line confirming `pads` are gone (or you can delete pad reporting entirely after the fix).

---

## If you want a minimal-risk A/B switch (optional but useful)

Keep the old padded kernel as `rope_fused_3way_kernel_padded` and add an env var:

* `SCOPE_TRITON_ROPE_FUSED_IMPL=padded|v2` (default `v2`)

That makes it trivial to prove:

* regression correlates with padded implementation
* v2 removes it

---

### Summary

* The current regression is overwhelmingly consistent with the **32/32/32 padding path** creating **50% extra lane work** plus **bad register pressure** from giant `[BLOCK_L, C_PAD]` tensors.
* The fix is to stop doing per-chunk power‑of‑two padding and instead operate on the natural power‑of‑two dimensions you already have at D=128 (**pairs=64, headdim=128**), using FlashAttn‑style tiling.
* The proposed kernel preserves correctness, tail handling, and fallback behavior, while removing the dominant extra work.

If you want, I can also suggest the exact Nsight Compute sections/counters to compare between `padded` and `v2` so you get a clean “this was the regression lever” proof in one capture.
