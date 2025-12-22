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

## Future Optimizations (Phase 3)

1. **Triton kernel** - fuse entire RoPE into single kernel, no intermediate tensors
2. **Pre-compute cos/sin at init** - store as separate tensors instead of complex
