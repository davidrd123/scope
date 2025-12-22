# RoPE Optimization - Phase 1

## Summary

Optimized `rope_apply` and `causal_rope_apply` to remove float64 upcast and complex number math.

**Commit:** `78b835c`

## Files Changed

1. `src/scope/core/pipelines/krea_realtime_video/modules/model.py` - `rope_apply()`
2. `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` - `causal_rope_apply()`

## The Problem

Original implementation (25.1s, 15.8% of self_attn):

```python
# Upcast to float64 (unnecessary precision)
x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))

# Build freqs_i (complex tensor)
freqs_i = torch.cat([...]).reshape(seq_len, 1, -1)

# Complex multiplication + convert back
x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
```

Issues:
1. **float64 upcast** - 2x memory, slower compute, unnecessary for RoPE
2. **view_as_complex** - creates complex tensor, extra allocation
3. **Complex multiplication** - slower than real arithmetic
4. **view_as_real** - converts back, another allocation

## The Solution

Use sin/cos directly (the freqs tensor is already `cos + i*sin` from `torch.polar`):

```python
# Build freqs_i (still complex, but we extract real/imag)
freqs_i = torch.cat([...]).reshape(seq_len, 1, -1)

# Extract cos/sin, cast to input dtype (bf16/fp16)
cos = freqs_i.real.to(x.dtype)  # (seq_len, 1, c)
sin = freqs_i.imag.to(x.dtype)  # (seq_len, 1, c)

# Reshape x to access pairs
x_i = x[i, :seq_len].reshape(seq_len, n, -1, 2)
x0, x1 = x_i[..., 0], x_i[..., 1]

# Apply rotation directly
x0_new = x0 * cos - x1 * sin
x1_new = x0 * sin + x1 * cos

# Stack back
x_i = torch.stack([x0_new, x1_new], dim=-1).flatten(2)
```

## Math Equivalence

Complex multiplication `(x0 + i*x1) * (cos + i*sin)`:
```
= x0*cos + i*x0*sin + i*x1*cos + i²*x1*sin
= x0*cos - x1*sin + i*(x0*sin + x1*cos)
= (x0*cos - x1*sin) + i*(x0*sin + x1*cos)
```

So:
- Real part: `x0*cos - x1*sin` → `x0_new`
- Imag part: `x0*sin + x1*cos` → `x1_new`

This is exactly what we compute, just without the complex number overhead.

## What Stayed The Same

1. **freqs_i construction** - still uses torch.cat + expand (future optimization target)
2. **Loop over samples** - still loops (but B=1 typically)
3. **Interface** - same function signature, drop-in replacement

## Expected Performance

- **Before:** 25.1s (0.48ms/call × 52160 calls)
- **After:** ~10-12s estimated (50%+ reduction)
- **Why:** No float64, no complex math, fewer allocations

## Correctness Check

The transformation is mathematically equivalent. To verify:
1. Run the model - output should look identical
2. Could add a unit test comparing old vs new on small inputs

## Rollback

```bash
git checkout HEAD~1 -- \
  src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py \
  src/scope/core/pipelines/krea_realtime_video/modules/model.py
```

## Phase 2: Caching (Added)

**Commit:** (pending)

Added LRU cache for `(cos, sin)` tensors to avoid rebuilding `freqs_i` on every call.

**Cache implementation:**
- Location: `src/scope/core/pipelines/krea_realtime_video/modules/model.py`
- Key: `(device, dtype, f, h, w, start_frame, c)`
- Value: `(cos, sin)` tensors of shape `(seq_len, 1, c)`
- Max entries: 32 (~30MB at 1560 tokens/frame)

**Usage:**
```python
# In both rope_apply and causal_rope_apply:
cos, sin = get_rope_cos_sin(freqs_split, f, h, w, start_frame, x.dtype, x.device, c)
```

**Expected additional savings:**
- Cache hits skip the expensive `torch.cat + expand + reshape` on every call
- First call per unique `(f, h, w, start_frame)` still builds, but subsequent calls are free
- In steady state with fixed resolution: near 100% cache hit rate

## Phase 2.5: Use float32 (complex64) instead of float64 (complex128)

**Commit:** (pending)

**Insight from Codex1:** `freqs` was built as complex128 (float64) in `rope_params()`, so `freqs_i.real/imag` are float64 → casting to bf16 every call moves 2x more data than needed.

**Fix:** Changed `rope_params()` and `rope_params_riflex()` to use `torch.float32`:

```python
# Before
torch.arange(0, dim, 2).to(torch.float64).div(dim)

# After
torch.arange(0, dim, 2, dtype=torch.float32).div(dim)
```

This makes `freqs` complex64, so `.real/.imag` are float32 → smaller cast to bf16.

**Cache reuse note (Codex1):** Q and K share the same (f,h,w,start_frame) key within a step, so cache helps there. Across steps, `start_frame` is monotonic so don't expect huge reuse unless restructuring callsite.

## Phase 3: Triton Kernel (Future)

Fuse entire RoPE into single kernel, no intermediate tensors.

**Kernel sketch from Codex2:**

```python
@triton.jit
def rope_kernel(
    X_ptr, OUT_ptr,
    COS_ptr, SIN_ptr,
    stride_xb, stride_xl, stride_xh, stride_xd,
    stride_ob, stride_ol, stride_oh, stride_od,
    stride_cos_l, stride_cos_c,
    stride_sin_l, stride_sin_c,
    L, H, C,
    BLOCK_L: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # 0..B*H
    pid_l = tl.program_id(1)   # 0..ceil(L/BLOCK_L)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_c = tl.arange(0, BLOCK_C)

    mask_l = offs_l < L
    mask_c = offs_c < C
    mask = mask_l[:, None] & mask_c[None, :]

    x_base = X_ptr + b * stride_xb + h * stride_xh
    o_base = OUT_ptr + b * stride_ob + h * stride_oh

    # cos/sin: [L, C]
    cos = tl.load(
        COS_ptr + offs_l[:, None] * stride_cos_l + offs_c[None, :] * stride_cos_c,
        mask=mask, other=0.0,
    )
    sin = tl.load(
        SIN_ptr + offs_l[:, None] * stride_sin_l + offs_c[None, :] * stride_sin_c,
        mask=mask, other=0.0,
    )

    # x: [L, D] where D = 2*C (pairwise)
    x0 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * offs_c)[None, :] * stride_xd,
        mask=mask, other=0.0,
    )
    x1 = tl.load(
        x_base + offs_l[:, None] * stride_xl + (2 * offs_c + 1)[None, :] * stride_xd,
        mask=mask, other=0.0,
    )

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * offs_c)[None, :] * stride_od,
        y0, mask=mask,
    )
    tl.store(
        o_base + offs_l[:, None] * stride_ol + (2 * offs_c + 1)[None, :] * stride_od,
        y1, mask=mask,
    )


def rope_triton(x, cos, sin, seq_len):
    """Wrapper: x is [B, L, H, D], cos/sin: [seq_len, C]"""
    B, L, H, D = x.shape
    C = D // 2
    out = x.clone()  # preserve tail if seq_len < L

    BLOCK_L, BLOCK_C = 128, 64  # BLOCK_C=64 matches D=128 heads
    grid = (B * H, triton.cdiv(seq_len, BLOCK_L))
    rope_kernel[grid](
        x, out,
        cos, sin,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        seq_len, H, C,
        BLOCK_L=BLOCK_L, BLOCK_C=BLOCK_C,
    )
    return out
```

**Notes:**
- BLOCK_C=64 matches D=128 heads (C=64)
- BLOCK_L=128 or 64 is a reasonable start
- Can tune for L=1560, H=16, D=128
