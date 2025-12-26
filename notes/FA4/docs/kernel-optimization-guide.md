# Kernel Optimization Guide: From Profiling to FlashAttention 4

This guide documents the journey of optimizing the KREA Realtime Video Pipeline's attention kernels, from initial profiling to integrating FlashAttention 4 with CUTE DSL.

## Summary

- Bottleneck: self-attention was ~51% of the pipeline, but the KV-bias attention kernel was only ~27% of self-attn time (QKV + RoPE were another ~37%).
- Biggest wins: FA4/CUTE `score_mod` for the KV-bias path (0.54ms vs 1.02ms Triton) plus RoPE cleanup/fusion (0.48ms → 0.38ms).
- End-to-end: ~15 FPS → ~20 FPS on B200 and 8.8 FPS → 15 FPS on B300 (cu130), measured at `320x576` with quality-preserving settings.
- Compile/FP8 note (B300): `torch.compile` is high-upside but has known SM103 footguns (cudagraph “output overwritten” in `reduce-overhead`). TorchAO FP8 + compile is **blocked upstream** by `torchao.quantization.Float8Tensor` missing `aten.as_strided.default`, but is unblocked locally via a PerTensor-only monkeypatch (`src/scope/core/compat/torchao_float8_as_strided.py`; disable with `SCOPE_TORCHAO_PATCH_FLOAT8_AS_STRIDED=0`). See `notes/issues/torchao-as-strided-dispatch.md`.
- SM103 note: avoid “silent fallback” benchmarking. On B300, some toolchain/backends can be *catastrophically* slow; prefer `fa4` when available, otherwise `flash`, and treat `triton` as an explicit experiment.

---

## Table of Contents

0. [Summary](#summary)
1. [The Problem](#1-the-problem)
2. [Understanding the Pipeline](#2-understanding-the-pipeline)
3. [Profiling: Where Does Time Go?](#3-profiling-where-does-time-go)
4. [The Attention Kernel](#4-the-attention-kernel)
5. [Optimization Approach](#5-optimization-approach)
6. [Triton Kernel Development](#6-triton-kernel-development)
7. [FlashAttention 4 Integration](#7-flashattention-4-integration)
8. [RoPE Optimization](#8-rope-optimization)
9. [Results and Lessons](#9-results-and-lessons)
10. [Environment Setup](#10-environment-setup)

---

## 1. The Problem

**Baseline (pre-optimization):** the KREA Realtime Video Pipeline was generating video at ~`15 FPS` on a B200 GPU (measured at `320x576`, 4 denoise steps, KV-bias `0.3`).

**Goal:** understand *why* it was slow, then make it materially faster without taking “quality shortcuts.”

**Key questions:**
- Where is time being spent?
- Which operations are GPU-limited vs memory-limited?
- Can we write custom kernels that beat the defaults?

---

## 2. Understanding the Pipeline

### Pipeline Architecture

```
Input Prompts → Text Encoder → UMT5-XXL embeddings
                                    ↓
                           ┌────────────────────┐
                           │  Denoising Loop    │
                           │  (4 steps)         │
                           │                    │
                           │  For each step:    │
                           │  ├─ Self-Attention │ ← 51% of time
                           │  ├─ Cross-Attention│
                           │  └─ FFN            │
                           └────────────────────┘
                                    ↓
                              VAE Decode
                                    ↓
                            Output Frames (3)
```

Each call to the pipeline:
1. Encodes text prompts (once, then cached)
2. Runs 4 denoising steps through 32 transformer blocks
3. Decodes latents to pixels via VAE
4. Outputs 3 video frames

### Self-Attention: The Bottleneck

Self-attention in this pipeline is special:

1. **Causal Attention**: Each frame block can only attend to previous frames
2. **KV-Cache**: Key-value pairs from previous frames are cached
3. **Attention Bias**: Past frames get reduced attention weight (0.3x)

```python
# Pseudocode for the attention pattern
def self_attention(query, key, value):
    # Q: current block (3 frames = 4680 tokens)
    # K, V: cached from previous + current (6 frames = 9360 tokens)

    scores = query @ key.T / sqrt(d)

    # Apply bias: past frames get 0.3x attention
    scores[:, past_tokens] += log(0.3)  # ≈ -1.2

    attention = softmax(scores)
    output = attention @ value
    return output
```

---

## 3. Profiling: Where Does Time Go?

### Enabling Profiling

```bash
PROFILE_ATTENTION=1 uv run daydream-scope
```

This adds CUDA event timing around key operations.

### Initial Profile Results

```
=== Attention Profiling Report ===
Component           Time      % Total   Calls   ms/call
───────────────────────────────────────────────────────
self_attn (outer)   73.3s     51.2%     34720   2.11
self_attn_kv_bias   25.6s     17.9%     27840   0.92
ffn                 21.4s     14.9%     34720   0.62
cross_attn          19.8s     13.9%     34720   0.57
self_attn_block_mask 3.0s     2.1%      6880    0.44
```

### Breakdown Within Self-Attention

```
self_attn breakdown (159.3s total):
───────────────────────────────────
self_attn_kv_bias    43.7s   27.4%   ← The attention kernel
qkv_projection       33.1s   20.8%   ← QKV GEMM
rope_apply           25.1s   15.8%   ← Rotary Position Embedding
output_projection    13.2s    8.3%
transpose_contiguous  8.6s    5.4%
self_attn_block_mask  5.4s    3.4%
cache_update          3.3s    2.1%
```

**Key insight:** The attention kernel is only 27% of self-attention time. QKV projection and RoPE combined are 37%!

### Amdahl's Law Reality Check

Even a 2x speedup on the 27% kernel yields:
```
New time = (1 - 0.27) + 0.27/2 = 0.73 + 0.135 = 0.865
Speedup = 1/0.865 = 1.16 (16% faster)
```

This means kernel optimization alone won't double our FPS. We need to optimize multiple components.

---

## 4. The Attention Kernel

### Two Attention Paths

The pipeline has two distinct attention patterns:

**Kernel A (Recompute Path)**
- Used once per frame for KV-cache initialization
- Block-causal mask: each block attends to itself + all previous blocks
- Shape: L=9360 (6 frames), symmetric Q=K

**Kernel B (Bias Path)**
- Used 4x per frame (once per denoising step)
- KV-cache attention with bias
- Shape: Lq=4680 (current block), Lk=9360 (cached)
- Past frames get 0.3x attention weight

### Which Path Matters?

```
Path             Time Share   Calls/Frame
───────────────────────────────────────────
Kernel B (bias)    89.5%        4
Kernel A (recomp)  10.5%        1
```

**Conclusion:** Optimize Kernel B first. Kernel A is only 10% of attention time.

---

## 5. Optimization Approach

### The Backend Options

| Backend | Description | Flexibility | Performance |
|---------|-------------|-------------|-------------|
| PyTorch SDPA | Built-in attention | Low | Baseline |
| flex_attention | PyTorch's flexible attention | High (score_mod, block_mask) | Good |
| Triton | Custom GPU kernel language | High | Better |
| FlashAttention 2/4 | Optimized attention library | Medium | Best |
| FA4/CUTE | FA4 with CUTE DSL score_mod | High | **Best** |

### The Challenge

Standard FlashAttention doesn't support our attention bias pattern. We need:
1. KV-cache (Q != K lengths)
2. Score modification (bias on past tokens)

Options:
1. **flex_attention**: Supports score_mod, but not as fast as FA
2. **Triton Kernel B**: Custom kernel, handles our pattern
3. **FA4/CUTE**: FlashAttention 4 with CUTE DSL score_mod

---

## 6. Triton Kernel Development

### What is Triton?

Triton is a Python-like language for writing GPU kernels. It compiles to PTX (GPU assembly) and handles:
- Thread block management
- Memory coalescing
- Register allocation

### Kernel B Implementation

```python
@triton.autotune(configs=[...], key=["Lq", "Lk", "D"])
@triton.jit
def kernel_b_bias_attention(
    Q, K, V, Out,
    Lq, Lk, D,
    frame_seqlen, current_block_start, log_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each block processes BLOCK_M query tokens
    start_m = tl.program_id(0) * BLOCK_M
    head_idx = tl.program_id(1)

    # Initialize accumulator and max for softmax
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Load Q block
    q = tl.load(Q + ...)  # [BLOCK_M, D]

    # Iterate over K,V blocks
    for start_n in range(0, Lk, BLOCK_N):
        # Load K, V blocks
        k = tl.load(K + ...)  # [BLOCK_N, D]
        v = tl.load(V + ...)  # [BLOCK_N, D]

        # Compute attention scores
        scores = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]

        # Apply bias to past frames (not first frame, not current block)
        kv_indices = start_n + tl.arange(0, BLOCK_N)
        is_past = (kv_indices >= frame_seqlen) & (kv_indices < current_block_start)
        scores = tl.where(is_past, scores + log_bias, scores)

        # Online softmax update
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p, v)
        m_i = m_new

    # Normalize
    out = acc / l_i[:, None]
    tl.store(Out + ..., out)
```

### Tuning Results

```
Config                                    Time (ms)
──────────────────────────────────────────────────
BLOCK_M=64  BLOCK_N=64  warps=8 stages=2   1.022  ← Winner
BLOCK_M=128 BLOCK_N=64  warps=4 stages=2   1.054
BLOCK_M=64  BLOCK_N=64  warps=8 stages=3   1.068
BLOCK_M=64  BLOCK_N=64  warps=4 stages=3   1.196
```

**Result:** Triton Kernel B: 1.023ms vs flex_attention: 1.144ms (**10.7% faster**)

---

## 7. FlashAttention 4 Integration

### What is FlashAttention?

FlashAttention is an optimized attention implementation that:
1. **Tiles** the computation to fit in SRAM
2. **Recomputes** attention during backward pass (trading compute for memory)
3. **Fuses** operations into a single kernel

### FA4 with CUTE DSL

FlashAttention 4 adds **score_mod support** via NVIDIA's CUTE DSL. This lets you inject custom logic into the attention kernel without breaking the fusion.

```python
import operator
import cutlass.cute as cute

def make_kv_bias_score_mod(frame_seqlen: int, current_block_start: int, log_bias: float):
    # Capture constants in closure for best perf (and cache compiled score_mods in Python).
    _frame_seqlen = int(frame_seqlen)
    _block_start = max(0, int(current_block_start))
    _log_bias = float(log_bias)

    @cute.jit
    def score_mod_kv_bias(tSrS_ssa, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        frame_seqlen_tensor = cute.full_like(kv_idx, _frame_seqlen)
        block_start_tensor = cute.full_like(kv_idx, _block_start)
        bias_tensor = cute.full_like(tSrS_ssa, _log_bias)

        # Apply bias only to Region 2: kv_idx in [frame_seqlen, block_start)
        biased = cute.where(
            operator.ge(kv_idx, frame_seqlen_tensor),
            tSrS_ssa + bias_tensor,
            tSrS_ssa,
        )
        return cute.where(operator.lt(kv_idx, block_start_tensor), biased, tSrS_ssa)

    return score_mod_kv_bias
```

### Integration

```python
from flash_attn.cute.interface import _flash_attn_fwd

score_mod = make_kv_bias_score_mod(frame_seqlen, current_block_start, log_bias)
out, _ = _flash_attn_fwd(q, k, v, score_mod=score_mod, causal=False, return_lse=False)
```

### Compilation, Caching, and Gotchas

- CUTE `score_mod` is JIT-compiled; the first call is slow. Cache compiled `score_mod`s keyed by `(frame_seqlen, current_block_start, log_bias)` to avoid repeated compile hits.
- Capturing constants in the closure is fastest; use `aux_tensors` only if those values truly vary every call and you need to avoid recompilation.
- Keep `score_mod` logic simple (nested `cute.where`); some cutlass-dsl builds hit MLIR issues on complex boolean expressions.
- If you use `torch.compile`, wrap the CuTe call with `torch._dynamo.disable` so Dynamo doesn’t try to trace CuTe’s Python/DLPack glue.
- If `B=1` and K/V are sliced views from a larger cache tensor, you may need to normalize batch stride (e.g., `k = k[0].unsqueeze(0)`) to avoid leading-dim inference failures.

### Results

```
Backend          Time (ms)   vs Triton
────────────────────────────────────────
Triton Kernel B   1.022      1.00x
FA4/CUTE          0.540      1.89x ← Winner
```

**Result:** FA4/CUTE is **1.89x faster** than our hand-written Triton kernel!

---

## 8. RoPE Optimization

### What is RoPE?

Rotary Position Embedding (RoPE) encodes position information by rotating the query and key vectors:

```python
def rope(x, cos, sin):
    x0, x1 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.cat([
        x0 * cos - x1 * sin,
        x0 * sin + x1 * cos,
    ], dim=-1)
    return x_rotated
```

### The Problem

The original RoPE implementation:
1. Upcasts to float64 (slow on GPUs!)
2. Uses complex multiplication (creates intermediate tensors)
3. Materializes full position tables

### Optimization Phases

**Phase 1: Remove float64**
```python
# Before: uses complex64 which promotes to float64 for multiply
x = x * freqs_cis  # complex64 * complex64 → float64 intermediate

# After: use sin/cos directly in float32
x_rot = x * cos - x_flip * sin  # stays in float32
```

**Phase 2: Cache cos/sin tables**
```python
@lru_cache(maxsize=32)
def get_rope_cos_sin(device, dtype, f, h, w, start_frame, c):
    # Compute once, reuse many times
    ...
```

**Phase 3: Triton fused kernel**
```python
@triton.jit
def triton_rope_kernel(X, Cos, Sin, Out, ...):
    # Load X, Cos, Sin
    # Apply rotation in registers
    # Store result
    # No intermediate tensors!
```

### Results

| Phase | Time (ms/call) | Speedup |
|-------|----------------|---------|
| Baseline | 0.48 | 1.0x |
| Phase 1 (no float64) | 0.45 | 1.07x |
| Phase 2 (+caching) | 0.42 | 1.14x |
| Phase 3 (Triton) | 0.38 | 1.26x |

---

## 9. Results and Lessons

### Final Performance

| GPU | Before | After | Improvement |
|-----|--------|-------|-------------|
| B200 | ~15 FPS | ~20 FPS | **1.33x** |
| B300 | ~8.8 FPS (repo-default stack) | ~15 FPS (cu130 stack) | **1.70x** |

### Benchmarking Notes

- For comparable perf numbers, use `320x576` and avoid accuracy-changing shortcuts (e.g., KV-cache recompute skipping).
- Warm up first (FA4/CuTe and Triton both JIT); measure steady-state (e.g., median over many iterations).

### Lessons Learned

1. **Profile first**: Without profiling, we would have optimized the wrong things
2. **Amdahl's Law is real**: Even 2x on 27% only gives 16%
3. **Multiple components matter**: Kernel + RoPE + environment
4. **Environment matters**: B300 needed cu130 runtime, not code changes
5. **Libraries beat hand-written**: FA4/CUTE > our Triton kernel

### What We'd Do Differently

1. Start with full pipeline profiling, not kernel microbenchmarks
2. Try existing optimized libraries before writing custom kernels
3. Test on all target GPUs early

---

## 10. Environment Setup

### B200 (SM100)

```bash
# Just works
uv run daydream-scope
```

### B300 (SM103)

```bash
# Requires CUDA 12.9+ for SM103 support
export TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas

# Verify cu130 runtime
python -c "import torch; print(torch.version.cuda)"  # Should be 13.0+

uv run daydream-scope
```

### Profiling

```bash
# Attention profiling
PROFILE_ATTENTION=1 uv run daydream-scope

# Pipeline block profiling (more overhead)
PROFILE_PIPELINE_BLOCKS=1 PROFILE_PIPELINE_BLOCKS_JSON=/tmp/blocks.json uv run daydream-scope
```

### Backend Selection

```bash
# Auto (default): SM103 → flash, otherwise → triton
uv run daydream-scope

# FA4/CUTE (best perf when available)
SCOPE_KV_BIAS_BACKEND=fa4 uv run daydream-scope

# FlashAttention (SM103-safe fallback; no score_mod)
SCOPE_KV_BIAS_BACKEND=flash uv run daydream-scope

# Triton (SM100-friendly; SM103: only if validated, can be catastrophically slow)
SCOPE_KV_BIAS_BACKEND=triton uv run daydream-scope
```

Note: `SCOPE_KV_BIAS_BACKEND` is read at import time; restart the python process when switching it.

SM103 note: the `flash` segment-combine backend uses the stable FlashAttention varlen op by default on B300 (some cutlass-dsl builds ICE on the FA4 `return_lse` path). To explicitly experiment with FA4 LSE on SM103, set `SCOPE_FLASH_COMBINE_USE_FA4_LSE=1` (and restart if you want to clear any one-time “tripped” fallback state).

### Validation

```bash
# FA4/CUTE score_mod correctness + microbench
uv run python scripts/test_fa4_kv_bias.py
```

---

## Files Referenced

| File | Description |
|------|-------------|
| `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` | Attention implementation |
| `src/scope/core/kernels/triton_attention.py` | Triton Kernel B |
| `src/scope/core/kernels/triton_rotary.py` | Triton RoPE kernel |
| `scripts/triton_sdpa.py` | Kernel development harness |
| `scripts/test_fa4_kv_bias.py` | FA4 integration tests |
| `notes/FA4/kernel-dev-log.md` | Full development chronicle |

---

## Further Reading

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention 2 Paper](https://arxiv.org/abs/2307.08691)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [CUTE DSL Documentation](https://github.com/NVIDIA/cutlass/tree/main/python)
- [CausVid Paper](https://arxiv.org/abs/2412.07772) (the streaming architecture)
