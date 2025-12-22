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

Initial expectation was optimistic; the real speedup is modest without eliminating `freqs_i` materialization.

Measured (post-Phase-1 profiling):

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Per-call time (`rope_apply` timer) | 0.48ms | 0.42ms | -12.5% |
| % of `self_attn` | 15.8% | 14.0% | -1.8pp |

Notes:
- Absolute seconds aren’t comparable across different run lengths; ms/call and within-`self_attn` share are the stable signals.
- This confirms Phase 1 is a real win, but the remaining cost is still dominated by `freqs_i` materialization (`expand + cat + reshape`) and dtype casts.

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

**Commit:** `eb280ba`

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
- Cache hits skip the expensive `torch.cat + expand + reshape` on that exact `(f, h, w, start_frame)` key.
- In streaming (bias-path), `start_frame` is typically monotonic, so cross-step reuse is low; the main guaranteed win is within a self-attn call because Q and K share the same key.
- In non-causal `rope_apply` (`start_frame=0`), reuse is high.

## Phase 2.5: Use float32 (complex64) instead of float64 (complex128)

**Commit:** `eba19ce`

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

## Phase 2.6: Avoid `freqs_i` materialization (no kernel) - ROLLED BACK

**Status:** Attempted 2025-12-22, rolled back due to performance regression.

**Idea:** Exploit 3-way split structure with broadcast cos/sin:
- reshape `x[:seq_len]` to `(f, h, w, n, c, 2)`
- apply rotation per chunk with broadcast shapes: `(f,1,1,1,c_f)`, `(1,h,1,1,c_h)`, `(1,1,w,1,c_w)`

**Result:** FPS dropped from 18.6-18.8 → 17 (regression)

**Lesson learned:**
- Traded memory traffic for kernel launch overhead - wrong tradeoff on GPU
- 3 separate rotation blocks + concat added more overhead than the cache-based approach
- Complex broadcast patterns `(f,h,w,n,c) * (f,1,1,1,c)` less efficient than simple `(seq_len,n,c) * (seq_len,1,c)`
- Cache-based Phase 2.5 is the sweet spot before going full Triton fusion

**Codex2 also noted potential bugs:**
- Device mismatch: `freqs.real.to(x.dtype)` missing `device=x.device`
- Layout assumption: assumes (F, H, W) order with W fastest

**Conclusion:** Skip to Phase 3 (Triton kernel) for next RoPE optimization

## Phase 3: Triton Kernel (Future)

Fuse entire RoPE into single kernel, no intermediate tensors.

**Implementation notes (pre-work for Opus):**
- **Data layout:** `x` is `[B, L, H, D]` with `D = 2*C` (pairwise). Pairs are contiguous: `x[..., 2*j]`, `x[..., 2*j+1]`.
- **Token -> grid mapping:** `seq_len = f*h*w`.
  - `f_idx = start_frame + t // (h*w)`
  - `h_idx = (t % (h*w)) // w`
  - `w_idx = t % w`
- **Freqs layout:** `freqs` is complex and split into 3 chunks:
  - `C0 = C - 2*(C//3)` (time), `C1 = C//3` (height), `C2 = C//3` (width).
  - For D=128 -> C=64 -> chunks 22/21/21.
  - Current Python impl concatenates chunks (not multiply), so the angle is a per-axis lookup.
- **Start frame:** only offsets the time axis (`freqs_split[0][start_frame : start_frame+f]`).
- **Kernel sketch (fast path):**
  - Launch per sample (grid sizes vary across batch).
  - Grid over `(ceil_div(L, BLOCK_L), H)` or `(H, ceil_div(L, BLOCK_L))`.
  - Compute `(f_idx, h_idx, w_idx)` per token; apply rotation in three contiguous chunks:
    - chunk0: `freqs_time[f_idx, :]` size `C0`
    - chunk1: `freqs_height[h_idx, :]` size `C1`
    - chunk2: `freqs_width[w_idx, :]` size `C2`
  - Use cos/sin real tensors (no complex ops in Triton).
  - Keep tail (`t >= seq_len`) unchanged.
- **Gotchas:**
  - Variable `grid_sizes` -> easiest is per-sample kernel launch (B usually 1).
  - Ensure `start_frame + f <= freqs_time.shape[0]`.
  - Consider fp32 math in-kernel, cast back to bf16/fp16.
- **Suggested initial params (B200, D=128):** `BLOCK_L=128 or 256`, `num_warps=4 or 8`, `num_stages=2`.
- **Validation:** compare output vs `rope_apply` / `causal_rope_apply` for `start_frame=0` and `start_frame>0`.

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

---

## Phase 3 Implementation Plan

**Goal:** Single Triton kernel that:
1. Takes `x[B, L, H, D]` and 3 freqs tables (not pre-materialized cos/sin)
2. Computes `(f_idx, h_idx, w_idx)` from token index inside kernel
3. Looks up cos/sin from 3 small tables, applies rotation
4. Returns rotated output in-place or to output buffer

### Step 1: Simplest working kernel (use pre-materialized cos/sin)

Start with Codex2's sketch verbatim:
- Wrapper builds `cos/sin[seq_len, C]` using existing `get_rope_cos_sin()`
- Kernel just does the rotation (no 3-way logic yet)
- Validate correctness against PyTorch implementation
- Measure baseline Triton overhead

**Why:** Derisks Triton mechanics (launch, strides, masking) before adding complexity.

### Step 2: Fuse 3-way lookup into kernel

Modify kernel to:
- Take 3 freqs tables: `freqs_f[max_f, C0]`, `freqs_h[max_h, C1]`, `freqs_w[max_w, C2]`
- Pass `h, w, start_frame` as scalar args
- Compute per-token indices:
  ```
  f_idx = start_frame + t // (h * w)
  h_idx = (t % (h * w)) // w
  w_idx = t % w
  ```
- Load cos/sin from 3 tables, apply to 3 chunks of C

**Complexity:** Integer division in kernel, 3 separate loads per token.

### Step 3: Optimize memory access

- Consider storing freqs as real cos/sin (not complex) at model init
- Coalesce loads if possible
- Tune BLOCK_L for L=1560 (likely 128 or 256)

### Validation

For each step:
```python
# Compare outputs
ref = causal_rope_apply(x, grid_sizes, freqs, start_frame)
out = rope_triton(x, ...)
assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3)
```

### Target shapes

| Param | Value |
|-------|-------|
| B | 1 |
| L | 1560-4680 (1-3 frames) |
| H | 16 |
| D | 128 (C=64) |
| C0/C1/C2 | 22/21/21 |
| f | 1-3 |
| h | 20 (for 320px) or 30 (for 480px) |
| w | 36 (for 576px) or 52 (for 832px) |

### File locations

- Kernel: `src/scope/core/kernels/triton_rope.py` (new file)
- Integration: Replace `rope_apply` / `causal_rope_apply` calls
- Tests: Add to existing test suite or standalone validation script

### Critical implementation notes (from Codex2)

**Layout assumption:**
- `rope_apply`/`causal_rope_apply` expect `x` shaped `[B, L, H, D]` (pairs in last dim)
- SAGEATTN path passes `[B, H, L, D]` (see `model.py`) - would break strict kernel
- **Action:** Gate kernel behind layout assert OR normalize layout before calling

**Preserve tail:**
- `seq_len = f*h*w` and `x` may be padded beyond that
- Kernel must leave `x[:, seq_len:, ...]` untouched
- **Easiest:** Write into a copy, only update first `seq_len` tokens

**start_frame only offsets time:**
- Causal path: only time axis gets `start_frame`; height/width always start at 0
- This logic must survive the kernel fuse

**Strides matter:**
- `x` is NOT guaranteed contiguous
- Kernel should use passed strides (don't assume contiguous unless adding `.contiguous()` and accepting copy cost)

**Chunk sizes must match:**
- `c0 = c - 2*(c//3)`, `c1 = c//3`, `c2 = c//3`
- Assert `c0 + c1 + c2 == c` in wrapper to avoid silent mis-rotation if head dim changes

**Precision choice:**
- For safety: compute in fp32, cast back to bf16/fp16
- Validate against Phase 2.5 Python path with `torch.allclose`

**Per-sample launch is fine:**
- `grid_sizes` can vary per batch element
- For B=1 this is trivial and avoids complexity

### Reference implementations

Vendored examples to study:
- `vendored/rope/flash_attn_triton_rotary.py` - Triton rotary example
- `vendored/rope/vllm_rotary_embedding_common.py` - vLLM rotary logic

### Minimal test harness

Before tuning, validate with tiny f/h/w:
```python
import torch
from src.scope.core.pipelines.krea_realtime_video.modules.model import rope_apply
from src.scope.core.kernels.triton_rope import rope_triton  # new kernel

# Tiny test case
B, H, D = 1, 16, 128
f, h, w = 2, 4, 6  # small grid
L = f * h * w  # 48 tokens
x = torch.randn(B, L, H, D, device='cuda', dtype=torch.bfloat16)
grid_sizes = torch.tensor([[f, h, w]], device='cuda')
freqs = ...  # from model

# Compare
ref = rope_apply(x.clone(), grid_sizes, freqs)
out = rope_triton(x.clone(), ...)

print(f"Max diff: {(ref - out).abs().max().item()}")
assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3), "Mismatch!"
```
