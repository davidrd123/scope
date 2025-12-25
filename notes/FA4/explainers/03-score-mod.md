# score_mod: Custom Attention Logic in FA4/CUTE

> **What this is:** An explainer for how to inject custom logic into FlashAttention 4's attention kernel using CUTE DSL's `score_mod` mechanism.
> **Context:** We used this to implement KV-cache attention bias, achieving ~1.89x speedup over Triton (steady-state Kernel B on B200 @ `320x576`, KV-bias `0.3`).
> **Updated:** 2025-12-25

---

## Overview

### The Problem

Standard attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

But sometimes you need to **modify the attention scores** before softmax. Common examples:
- Causal masking (set future positions to -inf)
- Relative position bias (add learned position offsets)
- **KV-cache bias** (reduce attention to cached frames to prevent error accumulation)

### The Challenge

FlashAttention achieves its speed by:
1. **Tiling:** Processing small blocks that fit in SRAM
2. **Fusion:** One kernel does everything (no intermediate memory writes)
3. **Online softmax:** Computing softmax incrementally as tiles are processed

If you want to modify scores, you need to do it **inside** the kernel. Writing your own fused attention kernel is hard. Can we inject custom logic into FA's existing kernel?

### The Solution: score_mod

FA4/CUTE provides `score_mod` - a hook that lets you inject a function that modifies attention scores **inside** the fused kernel:

```python
@cute.jit
def my_score_mod(scores, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    # Modify scores here
    return modified_scores

output, lse = _flash_attn_fwd(q, k, v, score_mod=my_score_mod)
```

---

## The Concept: What score_mod Does

### Where It Fits in the Attention Pipeline

```
Standard FlashAttention:

1. Load Q tile (M tokens)
2. For each K,V tile (N tokens):
   a. Compute S = Q @ K^T (scores)
   b. Scale: S = S / √d
   c. ─────────────────────────────
      │ score_mod injects here!  │  ← YOUR CODE
      ─────────────────────────────
   d. Online softmax update
   e. Accumulate: O += softmax(S) @ V
3. Store output O
```

The `score_mod` function receives the raw scores (after scaling) and returns modified scores. It runs **on the GPU, inside the attention kernel**, operating on tiles.

### What score_mod Receives

```python
def score_mod(
    tSrS_ssa,       # Vector fragment of (scaled) scores; SSA value in CUTE
    b_idx,          # Batch index (constant for the tile)
    h_idx,          # Head index (constant for the tile)
    q_idx,          # Query token indices for each element in tSrS_ssa (vector)
    kv_idx,         # Key/Value token indices for each element in tSrS_ssa (vector)
    seqlen_info,    # Sequence length info (for varlen)
    aux_tensors,    # Additional tensors you can pass in
):
    # Return modified scores
    return modified_scores
```

**Key insight:** `q_idx` and `kv_idx` are **vectors**, not scalars. In the current CuTe kernels, `score_mod` is applied in a vectorized loop over the score tile, so your function is called repeatedly on small score fragments (e.g. `vec_size=2`) rather than once per full `M×N` tile.

---

## The Implementation: KV-Cache Bias

### The Problem We Solved

In streaming video generation with KV-cache:

```
KV-Cache Layout:
┌─────────────┬─────────────┬─────────────┐
│ Frame 0     │ Frames 1-N  │ Current     │
│ (anchor)    │ (cached)    │ (generating)│
└─────────────┴─────────────┴─────────────┘
     ↑              ↑              ↑
 Full attention   0.3x bias    Full attention
```

- **Frame 0 (anchor):** Full attention - it's the reference
- **Cached frames:** Reduced attention (0.3x) - prevents error accumulation
- **Current block:** Full attention - we're generating this

### The score_mod Implementation

```python
@cute.jit
def score_mod_kv_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    # Closure constants (captured at JIT compile time)
    frame_seqlen_tensor = cute.full_like(kv_idx, _frame_seqlen)
    block_start_tensor = cute.full_like(kv_idx, _block_start)
    bias_tensor = cute.full_like(tSrS_ssa, _log_bias)

    # Apply bias to past-frame region: kv_idx in [frame_seqlen, block_start)
    #
    # Region 0: [0, frame_seqlen)        → No bias (anchor frame)
    # Region 1: [frame_seqlen, block_start) → Apply log_bias
    # Region 2: [block_start, end)       → No bias (current block)

    biased = cute.where(
        operator.ge(kv_idx, frame_seqlen_tensor),  # kv_idx >= frame_seqlen
        tSrS_ssa + bias_tensor,                     # Add log(0.3) ≈ -1.2
        tSrS_ssa,                                   # Keep original
    )
    return cute.where(
        operator.lt(kv_idx, block_start_tensor),    # kv_idx < block_start
        biased,                                      # Use biased scores
        tSrS_ssa,                                   # Keep original
    )
```

### Why Closure Constants?

```python
def _get_fa4_score_mod(frame_seqlen: int, block_start: int, log_bias: float):
    # Capture as Python values (become compile-time constants in CUTE)
    _frame_seqlen = frame_seqlen
    _block_start = max(0, int(block_start))
    _log_bias = log_bias

    @cute.jit
    def score_mod_kv_bias(...):
        # _frame_seqlen, _block_start, _log_bias are closure constants
        # They're baked into the compiled kernel!
```

**Why not use `aux_tensors`?**

Using `aux_tensors` (runtime values) is supported, but it can cost performance: it forces more dynamic indexing/bounds handling in the score-mod plumbing and can reduce the vectorization the kernel chooses (we keep `vec_size=2` by capturing small scalars as closure constants).

**The tradeoff:** Each unique `(frame_seqlen, block_start, log_bias)` tuple compiles a new kernel. That's why we cache:

```python
cache_key = (int(frame_seqlen), int(block_start), round(log_bias, 6))
if cache_key in _fa4_score_mod_cache:
    return _fa4_score_mod_cache[cache_key]
```

---

## Key Design Decisions

### 1. Nested `cute.where` Instead of Boolean AND

You might expect:
```python
# DON'T DO THIS
mask = (kv_idx >= frame_seqlen) & (kv_idx < block_start)
return cute.where(mask, tSrS_ssa + bias, tSrS_ssa)
```

But boolean AND on CUTE SSA values can cause MLIR issues in some cutlass-dsl builds. Nested `where` is safer:

```python
# DO THIS
biased = cute.where(ge(kv_idx, frame_seqlen), tSrS_ssa + bias, tSrS_ssa)
return cute.where(lt(kv_idx, block_start), biased, tSrS_ssa)
```

### 2. The Vendored Path Problem

Many `flash_attn` wheels **don’t expose score_mod support** (or expose a version that differs from what we need). In this repo we assume you will use either:
1. A vendored version with score_mod patches
2. A local build from source

The code handles this with path manipulation:
```python
def _extend_flash_attn_path_for_score_mod():
    # Look for vendored sources
    vendored = parent / "vendored" / "flash_attn_cute_score_mod" / "flash_attn"
    if vendored.is_dir():
        _flash_attn.__path__.insert(0, vendored_str)  # Prefer vendored
```

And validates at import time:
```python
if "score_mod" not in inspect.signature(_fa4_fwd).parameters:
    raise ImportError("FA4/CUTE score_mod requires vendored sources...")
```

### 3. torch.compile Compatibility

CUTE uses DLPack and Python glue that `torch._dynamo` can't trace. Solution:

```python
# Make the call opaque to Dynamo
@torch._dynamo.disable
def _fa4_fwd_opaque(*args, **kwargs):
    return _fa4_fwd(*args, **kwargs)
```

This lets you compile the surrounding transformer while keeping the FA4 call as a black box.

### 4. The B=1 Stride Normalization Trick

When K/V come from a larger cache tensor (sliced view), the batch stride can confuse CUTE's layout inference:

```python
# K has shape [1, L, H, D] but stride(0) reflects the full cache size
# CUTE fails: "Can't deduce the leading dimension..."

# Fix: normalize the view
if k_fa4.shape[0] == 1:
    q_fa4 = q_fa4[0].unsqueeze(0)
    k_fa4 = k_fa4[0].unsqueeze(0)
    v_fa4 = v_fa4[0].unsqueeze(0)  # New view with normalized stride(0)
```

### 5. Fallback Strategy

The code implements a graceful degradation chain:

```
fa4 (score_mod)     ← Fastest (1.89x)
    ↓ (if fails)
flash (segment-combine) ← Safe on SM103
    ↓ (if fails)
triton (Kernel B)   ← Fallback on SM80/90/100; **avoid on SM103** unless you’ve verified it’s on a fast path
    ↓ (if fails)
flex_attention      ← Last resort
```

With one-shot trip wires:
```python
if not _fa4_bias_tripped:
    try:
        # ... FA4 path
    except Exception as e:
        _fa4_bias_tripped = True  # Never try again this session
        logger.warning("FA4 failed; falling back...")
```

**SM103 (B300) note:** in our current environment/toolchain, the Triton KV-bias backend can fall onto a catastrophically slow scalar path. For B300, “safe fallback” usually means `flash` (segment-combine), not Triton. (See `notes/FA4/b300/session-state.md` for the practical backend guidance.)

---

## Performance Results

Example steady-state Kernel B numbers (B200 @ `320x576`, BF16, KV-bias `0.3`). See `notes/FA4/docs/kernel-optimization-guide.md`.

| Backend | Time (ms/call) | vs Triton |
|---------|----------------|-----------|
| Triton Kernel B | 1.02 | 1.00x |
| flex_attention | 1.14 | 0.89x (slower) |
| FA4/CUTE score_mod | **0.54** | **1.89x** |

The 1.89x speedup comes from:
1. FA4's optimized memory access patterns
2. Better tensor core utilization
3. No intermediate memory traffic (fully fused)

---

## How to Use score_mod

### Step 1: Get a score_mod-capable flash_attn

Either vendor the CUTE sources or build from source with score_mod support.

### Step 2: Write Your score_mod Function

```python
import operator
import cutlass.cute as cute

@cute.jit
def my_score_mod(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    # Example: Add +1 to all scores (useless but demonstrates the pattern)
    return tSrS_ssa + cute.full_like(tSrS_ssa, 1.0)
```

### Step 3: Call FA4 with Your score_mod

```python
from flash_attn.cute.interface import _flash_attn_fwd

output, lse = _flash_attn_fwd(
    q, k, v,
    score_mod=my_score_mod,
    causal=False,
    return_lse=False,
)
```

### Step 4: Cache Compiled Kernels

```python
_cache = {}

def get_score_mod(param1, param2):
    key = (param1, param2)
    if key in _cache:
        return _cache[key]

    _p1, _p2 = param1, param2  # Closure constants

    @cute.jit
    def score_mod(...):
        # Use _p1, _p2 here
        ...

    _cache[key] = score_mod
    return score_mod
```

---

## Questions & Opportunities

### What's Unclear

1. **How deep does score_mod go?** Can it access the Q/K values themselves, or only the computed scores?

2. **Can we do RoPE in score_mod?** RoPE modifies Q and K **before** the dot product. score_mod is **after** the dot product. So no, score_mod can't do RoPE directly. We'd need a `q_mod`/`k_mod` hook.

3. **What are the compile time costs?** Each unique score_mod compiles a new kernel (~seconds). How does this scale?

### Opportunities

1. **Fusing more operations:** If we had `q_mod`/`k_mod` hooks, we could fuse RoPE into the attention kernel.

2. **Learned position biases:** score_mod could implement ALiBi, relative position encodings, etc.

3. **Sparse attention patterns:** score_mod could implement block-sparse attention by returning -inf for masked positions.

4. **Softcapping:** CuTe `_flash_attn_fwd` exposes `softcap`; internally it's implemented via a built-in score_mod and is mutually exclusive with a custom `score_mod`.

---

## References

### Code Files

| File | Description |
|------|-------------|
| `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` | Our KV-bias implementation |
| `flash_attn/cute/interface.py` | FA4/CUTE API entry point (installed package; exact contents depend on wheel version) |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py` | Vendored sources used for score_mod experimentation |

### Related Docs

- [CUTLASS Python DSL](https://github.com/NVIDIA/cutlass/tree/main/python)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- `notes/FA4/kernel-dev-log.md` - Full development chronicle
- `notes/FA4/docs/kernel-optimization-guide.md` - Technical deep dive
