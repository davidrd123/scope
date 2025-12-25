# How FA4 Attention Works: The Complete Picture

> **Explainer #1** - Overview and synthesis of the FA4/CUTE architecture.
> This is the "umbrella" explainer that ties together all the building blocks.

---

## Overview

FlashAttention 4 (FA4) is a CUDA kernel for computing attention efficiently on NVIDIA GPUs. The implementation in `flash_attn/cute/` uses NVIDIA's CuTe-DSL (a Python-based DSL for writing CUDA kernels) and supports three GPU architectures:

| Architecture | Class | GPU Generation |
|--------------|-------|----------------|
| SM80 | `FlashAttentionForwardSm80` | Ampere (A100) |
| SM90 | `FlashAttentionForwardSm90` | Hopper (H100) |
| SM100 | `FlashAttentionForwardSm100` | Blackwell (B200) |

**The core insight:** Attention is memory-bound. FA4 achieves speed by:
1. **Tiling** - Never materialize the full attention matrix
2. **Online softmax** - Compute softmax incrementally as tiles arrive
3. **Warp specialization** - Different warps do different jobs (load, compute, store)
4. **Hardware-specific optimizations** - TMA, tensor cores, TMEM (Blackwell)

---

## The Architecture

### Entry Point: `interface.py`

The public API routes through `_flash_attn_fwd()`:

```python
# interface.py:102-133
def _flash_attn_fwd(
    q, k, v,
    cu_seqlens_q=None, cu_seqlens_k=None,  # Variable length support
    softmax_scale=None,
    causal=False,
    score_mod=None,    # Custom attention modifications (see Explainer #3)
    mask_mod=None,     # Custom masking
    ...
)
```

The dispatcher selects the architecture-specific kernel:

```python
# interface.py:430-476
if compute_capability == 9:  # Hopper
    fa_fwd = FlashAttentionForwardSm90(...)
elif compute_capability == 10:  # Blackwell
    fa_fwd = FlashAttentionForwardSm100(...)
```

### Kernel Structure

All forward implementations share a common structure:

```
┌─────────────────────────────────────────────────────────────┐
│                        PROLOGUE                             │
│  • Allocate shared memory for Q, K, V tiles                 │
│  • Load first Q tile                                        │
│  • Initialize pipeline (prefetch K/V tiles)                 │
│  • Set up tile scheduler                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        MAINLOOP                             │
│  for each N tile (K/V blocks):                              │
│    1. S = Q @ K^T  (scores)           [MMA]                 │
│    2. Apply score_mod (if any)        [See Explainer #3]    │
│    3. Apply mask (causal/local)                             │
│    4. P = online_softmax(S)           [See Explainer #7]    │
│    5. O += P @ V  (accumulate output) [MMA]                 │
│    6. Rescale O if max changed        [See Explainer #7]    │
│    7. Load next K/V tiles             [See Explainer #5]    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        EPILOGUE                             │
│  • Finalize softmax (normalize by row_sum)                  │
│  • Write O to global memory                                 │
│  • Write LSE (log-sum-exp) for backward pass                │
└─────────────────────────────────────────────────────────────┘
```

---

## The Building Blocks (Other Explainers)

Each component has its own deep-dive explainer:

### 1. Memory Loading (Explainer #5: TMA and Memory Loading)

**SM80/SM90:** Uses `cp.async` - asynchronous copies from global → shared memory
**SM100:** Uses TMA (Tensor Memory Accelerator) - hardware DMA engine

```python
# flash_fwd.py - SM80 uses cp.async
atom_async_copy = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    ...
)

# flash_fwd_sm100.py - SM100 uses TMA
tma_load_K, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, ...)
```

### 2. Tile Scheduling (Explainer #6: Tile Scheduling and Pipelining)

Four scheduler types balance work across SMs:

| Scheduler | Use Case |
|-----------|----------|
| `SingleTileScheduler` | Basic (one tile per block) |
| `StaticPersistentTileScheduler` | Persistent (reuse blocks) |
| `SingleTileLPTScheduler` | Causal (longest-first for load balance) |
| `SingleTileVarlenScheduler` | Variable length sequences |

```python
# tile_scheduler.py:347 - LPT reversal
block = params.num_block - 1 - block  # Process larger blocks first
```

### 3. Online Softmax (Explainer #7: Online Softmax)

The key to FlashAttention - compute softmax incrementally:

```python
# Per tile (simplified):
max_new = max(max_old, tile_max)
scale = exp2((max_old - max_new) * LOG2_E)
row_sum = row_sum * scale + sum(exp2(S - max_new))
O = O * scale + P @ V
```

### 4. Custom Attention Logic (Explainer #3: score_mod)

User-defined score modifications (like KV-bias):

```python
# User code
def kv_bias_score_mod(score, b, h, q_idx, kv_idx, aux):
    bias = aux[0][b, h, q_idx, kv_idx]
    return score + bias
```

### 5. Blackwell-Specific Features (Explainer #2: The Blackwell Path)

SM100 introduces:
- **tcgen05** - 5th generation tensor core instructions
- **TMEM** - Tensor Memory for MMA accumulators
- **Elect-one pattern** - Single thread issues MMA
- **TS mode** - Read A operand from TMEM (for P @ V)

```python
# flash_fwd_sm100.py:148-154 - Warp assignments
self.softmax0_warp_ids = (0, 1, 2, 3)
self.softmax1_warp_ids = (4, 5, 6, 7)
self.correction_warp_ids = (8, 9, 10, 11)
self.mma_warp_id = 12
self.epilogue_warp_ids = (13,)
self.load_warp_ids = (14,)
self.empty_warp_ids = (15,)
```

---

## Architecture Comparison

| Feature | SM80 (Ampere) | SM90 (Hopper) | SM100 (Blackwell) |
|---------|---------------|---------------|-------------------|
| **Memory Load** | `cp.async` | TMA | TMA |
| **Synchronization** | barriers | mbarrier | mbarrier |
| **MMA** | warp MMA | warpgroup MMA | tcgen05.mma |
| **Accumulators** | Registers | Registers | TMEM |
| **Warp Specialization** | No | Limited | Full (16 warps) |
| **Threads/CTA** | 128-384 | 128-384 | 512 (16 warps) |

---

## The Mainloop in Detail

### SM80/SM90 Mainloop (`flash_fwd.py:951-1010`)

```python
# Mainloop iteration structure
mask_fn = partial(mask.apply_mask, ...)

# First iteration: needs seqlen masking
compute_one_n_block(n_block, mask_fn=partial(mask_fn, mask_seqlen=True))

# Causal iterations: needs causal masking
for n_tile in range(n_block_max - 1 - n_block_min_causal):
    compute_one_n_block(n_block, mask_fn=partial(mask_fn, mask_seqlen=False))

# Remaining iterations: no masking needed
for n_tile in range(remaining):
    compute_one_n_block(n_block)  # No mask_fn
```

The key insight: masking is expensive, so we separate the loop into:
1. **First tile** - Always needs boundary checks
2. **Causal tiles** - Need causal mask
3. **Full tiles** - No masking, fastest path

### SM100 Mainloop (Warp-Specialized)

On Blackwell, different warps run different code paths:

```python
# flash_fwd_sm100.py (conceptual)
if warp_idx in self.load_warp_ids:
    load_loop()  # TMA loads K/V
elif warp_idx == self.mma_warp_id:
    mma_loop()   # Issue tcgen05.mma
elif warp_idx in self.softmax0_warp_ids:
    softmax_loop(stage=0)  # Online softmax
elif warp_idx in self.correction_warp_ids:
    correction_loop()  # Rescale O
elif warp_idx in self.epilogue_warp_ids:
    epilogue_loop()  # Write output
```

---

## Data Flow Through the Kernel

```
                      GLOBAL MEMORY
┌────────────────────────────────────────────────────────────────┐
│    Q [B, Sq, H, D]     K [B, Sk, Hkv, D]     V [B, Sk, Hkv, Dv]│
└─────────┬───────────────────┬──────────────────────┬───────────┘
          │                   │                      │
          │ TMA/cp.async      │ TMA                  │ TMA
          ▼                   ▼                      ▼
┌────────────────────────────────────────────────────────────────┐
│                      SHARED MEMORY                             │
│    sQ [M, D]         sK [N, D, stages]      sV [N, Dv, stages] │
└─────────┬───────────────────┬──────────────────────┬───────────┘
          │ ldmatrix          │ ldmatrix             │ ldmatrix
          ▼                   ▼                      ▼
┌────────────────────────────────────────────────────────────────┐
│                    REGISTERS / TMEM                            │
│    rQ [M, D]         rK [N, D]              rV [N, Dv]         │
│                                                                │
│    S = Q @ K^T  ────▶  acc_S [M, N]  ────▶  P (softmax)        │
│                                                                │
│    O = P @ V    ────▶  acc_O [M, Dv]                           │
└─────────────────────────────────────────────────────────────────┘
          │
          │ store (via smem)
          ▼
┌────────────────────────────────────────────────────────────────┐
│                      GLOBAL MEMORY                             │
│    O [B, Sq, H, Dv]                     LSE [B, H, Sq]         │
└────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why Tile in M and N, Not K?

The head dimension (K) is fixed and small (64-256). But sequence length (M, N) can be huge. By tiling M and N:
- Each tile computes a partial attention for a block of queries against a block of keys
- Online softmax allows incremental accumulation

### 2. Why Online Softmax?

Standard softmax requires knowing the max over all elements first. Online softmax tracks running max/sum and rescales when a new max is found. This enables:
- Single pass over K/V
- No materialization of full attention matrix
- O(n) memory instead of O(n²)

### 3. Why Warp Specialization (SM100)?

Blackwell's TMEM and tcgen05 require specific programming patterns:
- MMA is issued by a single thread (elect-one)
- Different warps can run independent code paths
- Load, compute, and writeback can overlap

### 4. Why Multiple Schedulers?

Different workloads have different characteristics:
- **Causal attention** - Later M blocks have more work (LPT scheduler)
- **Variable length** - Sequences have different lengths (varlen scheduler)
- **Large batch** - Persistent scheduling reuses blocks

---

## Files Reference

| File | Purpose |
|------|---------|
| `interface.py` | Public API, architecture dispatch |
| `flash_fwd.py` | SM80/SM90 forward kernels |
| `flash_fwd_sm100.py` | SM100 (Blackwell) forward kernel |
| `softmax.py` | Online softmax implementation |
| `tile_scheduler.py` | Tile scheduling strategies |
| `pipeline.py` | Pipeline state management |
| `blackwell_helpers.py` | SM100-specific MMA helpers |
| `mma_sm100_desc.py` | SM100 descriptor encoding |
| `copy_utils.py` | TMA and memory copy utilities |
| `mask.py` | Attention masking (causal, local) |

---

## How the Explainers Fit Together

```
                    ┌─────────────────────────┐
                    │  #1 How FA4 Works       │  ◄── You are here
                    │  (This overview)        │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ #2 Blackwell  │     │ #3 score_mod    │     │ #4 RoPE         │
│ Path (SM100)  │     │                 │     │                 │
└───────┬───────┘     └────────┬────────┘     └─────────────────┘
        │                      │
        │              (modifies scores)
        │                      │
        ▼                      ▼
┌───────────────────────────────────────────────────────────────┐
│                         MAINLOOP                              │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐ │
│  │ #5 TMA/Memory  │──▶│ #6 Tile Sched  │──▶│ #7 Online      │ │
│  │ Loading        │   │ & Pipelining   │   │ Softmax        │ │
│  └────────────────┘   └────────────────┘   └────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

---

## Summary

FlashAttention 4 is a carefully orchestrated dance of:

1. **Memory efficiency** - Tile the computation to fit in SRAM
2. **Compute efficiency** - Use tensor cores for matrix multiply
3. **Pipelining** - Overlap load/compute/store
4. **Hardware adaptation** - Architecture-specific optimizations

The key innovations:
- **Online softmax** makes single-pass attention possible
- **Warp specialization** (SM100) maximizes hardware utilization
- **TMA** (SM90+) offloads memory management to hardware
- **TMEM** (SM100) provides dedicated accumulator storage

Understanding these pieces gives you the foundation to:
- Debug attention performance issues
- Add custom score modifications
- Reason about when to use which scheduler
- Eventually, write your own fused attention variants

---

## References

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - Original algorithm
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691) - Improved parallelism
- [CUTLASS Examples](https://github.com/NVIDIA/cutlass/tree/main/examples) - Reference implementations
- Other explainers in this series for deep dives
