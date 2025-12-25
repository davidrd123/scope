# TMA and Memory Loading in FA4/CUTE

> **What this is:** An explainer for how FlashAttention 4 loads tensor tiles using TMA (Tensor Memory Accelerator) and where a hypothetical Q/K modifier hook would slot in for RoPE fusion (in SM100 tcgen05, that likely means a shared-memory stage modifier).
> **Context:** We're exploring Level 5/6 optimization. After understanding score_mod (explainer #3) and RoPE (explainer #4), this explainer answers: "Where in the kernel would we inject RoPE?"
> **Updated:** 2025-12-25

---

## Overview

### The Memory Hierarchy Problem

In attention, we need to:
1. Load Q, K, V tiles from global memory (HBM)
2. Compute QK^T (scores)
3. Apply softmax
4. Compute O = softmax(S) × V
5. Store O back to global memory

The bottleneck is often **memory bandwidth**, not compute. Each tensor core operation is incredibly fast, but feeding it data fast enough is the challenge.

### TMA: The Hardware Solution

TMA (Tensor Memory Accelerator) is NVIDIA's hardware for async memory transfers:

| Feature | Traditional Load | TMA Load |
|---------|------------------|----------|
| Threads needed | All threads cooperate | 1 elected thread issues |
| Address calculation | Manual, per-thread | Hardware handles |
| Overlap with compute | Limited | Full async |
| First available | - | Hopper (SM90) and newer (e.g. Blackwell SM100/SM103) |

TMA uses `cp.async.bulk.tensor` PTX instructions and can load 2D-5D tensor tiles directly into shared memory.

### What This Explainer Covers

1. How TMA works conceptually
2. How FA4/CUTE sets up TMA for Q, K, V
3. The producer/consumer pipeline with mbarriers
4. Warp specialization in the kernel
5. **Where RoPE fusion would inject** (the key question from #4)

---

## TMA Concepts

### Tensor Maps

TMA requires a **tensor map** (CUtensorMap) set up on the host:

```cpp
// Host-side setup (conceptual)
CUtensorMap Q_tmap;
cuTensorMapEncodeTiled(
    &Q_tmap,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,  // dtype
    rank=1..5,                           // tile rank (depends on how you encode the tensor)
    Q_ptr,                               // global memory base
    globalDim,                           // full tensor shape
    globalStrides,                       // strides in bytes
    boxDim,                              // tile shape (e.g., 128x128)
    elementStrides,                      // typically {1, 1}
    ...
);
```

The tensor map encodes:
- Global tensor shape and strides
- Tile (box) dimensions
- Swizzling pattern for efficient access

### Issuing TMA Loads

In the kernel, CuTe arranges for **one elected thread** in the producer group to do the book-keeping (e.g. `expect_tx`). The actual TMA copy is expressed as a `cute.copy(...)` on a TMA copy atom; the underlying lowering ensures only the required lanes issue the PTX instruction(s).

```cpp
// Conceptual pattern (matches the CuTe code structure)
elect_one_thread_in_producer_group();
mbarrier.arrive.expect_tx(bytes_to_transfer);
cp.async.bulk.tensor... mbarrier::complete_tx::bytes (...);  // issued by TMA "producer" lane(s)
```

### mbarrier: Synchronization

TMA uses **mbarriers** (memory barriers) to signal completion:

```
┌─────────────────────────────────────────────────────────────┐
│ Producer (Load Warp)              Consumer (Compute Warps)  │
│                                                             │
│ 1. Issue TMA loads                                          │
│ 2. mbarrier.arrive.expect_tx ────────────────────────────>  │
│                                   3. mbarrier.wait(phase)   │
│         (TMA hardware transfers)                            │
│         (TMA completes, decrements tx-count)                │
│                                   4. Data ready in smem!    │
│                                   5. Compute with data      │
│                                   6. mbarrier.arrive ──────>│
│ 7. mbarrier.wait (buffer empty)                             │
│ 8. Load next tile                                           │
└─────────────────────────────────────────────────────────────┘
```

**Phase parity:** mbarriers use 0/1 phase bits to track odd/even iterations:
```python
# In CuTe kernels this is typically a 2-sided handshake with *two* barrier arrays:
# - "full" barriers: producer -> consumer (data ready)
# - "empty" barriers: consumer -> producer (buffer reusable)
#
# Each stage has its own barrier, and both sides track a parity bit to avoid races.
producer_phase = 1
consumer_phase = 0
```

---

## FA4/CUTE TMA Setup

### Creating TMA Atoms

In `flash_fwd_sm100.py`, TMA atoms are created for Q, K, V:

```python
# flash_fwd_sm100.py lines 489-519
tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

# TMA for Q
tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
    tma_load_op,
    mQ,                              # global tensor
    cute.select(sQ_layout, ...),     # shared memory layout
    self.mma_tiler_qk,               # (M_BLOCK, N_BLOCK, head_dim)
    tiled_mma_qk,
    self.cluster_layout_vmnk.shape,
)

# TMA for K (if using TMA for KV)
tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
    tma_load_op,
    mK,
    cute.select(sK_layout, ...),
    self.mma_tiler_qk,
    tiled_mma_qk,
    self.cluster_layout_vmnk.shape,
)

# TMA for V
tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
    tma_load_op,
    mV,
    cute.select(sV_layout, ...),
    self.mma_tiler_pv,
    tiled_mma_pv,
    self.cluster_layout_vmnk.shape,
)
```

### TMA Partition

The TMA atoms are partitioned into thread-local views:

```python
# flash_fwd_sm100.py lines 1176-1188
tKsK, tKgK = cpasync.tma_partition(
    tma_atom_K,
    0,  # no multicast
    cute.make_layout(1),
    cute.group_modes(sK, 0, 3),    # shared memory view
    cute.group_modes(tSgK, 0, 3),  # global memory view
)

tVsV, tVgV = cpasync.tma_partition(
    tma_atom_V,
    0,
    cute.make_layout(1),
    cute.group_modes(sV, 0, 3),
    cute.group_modes(tOgV, 0, 3),
)
```

### Load Function Partials

The actual load functions are created as partials:

```python
# flash_fwd_sm100.py lines 1212-1242
load_Q = partial(
    self.load_Q,
    load_Q_fn,
    mbar_ptr + self.mbar_load_q_full_offset,
    mbar_ptr + self.mbar_load_q_empty_offset,
    phase=q_producer_phase,
)

load_K = partial(
    self.load_KV,
    tma_atom_K,
    tKgK,
    tKsK,
    paged_kv_manager,
    sK,
    mbar_ptr + self.mbar_load_kv_full_offset,
    mbar_ptr + self.mbar_load_kv_empty_offset,
    K_or_V="K",
)

load_V = partial(
    self.load_KV,
    tma_atom_V,
    tVgV,
    tVsV,
    ...
    K_or_V="V",
)
```

---

## Pipeline and Warp Specialization

### Warp Roles

FA4/CUTE uses **warp specialization** - different warps do different jobs:

```python
# flash_fwd_sm100.py lines 148-154
self.softmax0_warp_ids = (0, 1, 2, 3)    # Softmax for stage 0
self.softmax1_warp_ids = (4, 5, 6, 7)    # Softmax for stage 1
self.correction_warp_ids = (8, 9, 10, 11) # Online softmax correction
self.mma_warp_id = 12                     # Tensor core MMAs
self.epilogue_warp_ids = (13,)            # Output handling
self.load_warp_ids = (14,)                # TMA loads
self.empty_warp_ids = (15,)               # Available
```

Notes:
- If `use_tma_KV=False` (paged KV path), FA4 widens the producer group (`load_warp_ids=(14, 15)`) because it uses `cp.async`-style loads instead of TMA.
- For some varlen configurations, epilogue warps can be reassigned (e.g. epilogue work moved onto the correction warps).

### Pipeline Stages

Multiple tiles are in-flight simultaneously:

```python
self.kv_stage = 4 if self.q_dtype.width == 8 else 3  # 3-4 KV stages
self.q_stage = 2   # 2 Q stages
self.acc_stage = 1 # Accumulator
self.epi_stage = 2 # Epilogue
```

### The Loading Sequence

```python
# flash_fwd_sm100.py lines 1249-1279 (simplified)
# Stage 0
load_Q(block=m_block*2 + 0, stage=0)    # Q0
load_K(block=n_block_max-1, ...)         # K0
load_Q(block=m_block*2 + 1, stage=1)    # Q1
load_V(block=n_block_max-1, ...)         # V0

# Loop for remaining tiles
for i in range(n_block_max - 1 - n_block_min):
    n_block = n_block_max - 2 - i
    load_K(block=n_block, ...)           # Ki
    load_V(block=n_block, ...)           # Vi
```

This overlaps:
1. Loading tile N+1
2. Computing on tile N
3. Writing results from tile N-1

---

## Where score_mod Injects (Review)

From explainer #3, `score_mod` is applied **after** QK^T scores are computed:

```python
# flash_fwd_sm100.py lines 1888-1904
# Wait for scores Si to be ready
cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, ...)

# Load scores from tensor memory to registers
tSrS_t2r = cute.make_fragment(...)
cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

# >>> score_mod is applied HERE <<<
if cutlass.const_expr(self.score_mod is not None):
    self.apply_score_mod(
        tSrS_t2r,      # Score fragment in registers
        ...
        batch_idx, head_idx, m_block, n_block,
        ...
    )
```

The scores have already been computed - `score_mod` modifies them in-place.

---

## Where RoPE Would Inject

### The Challenge

RoPE needs to modify Q and K **before** QK^T:
- score_mod: `S = QK^T; S = score_mod(S)` ✓
- RoPE: `Q = rope(Q); K = rope(K); S = QK^T` - different hook point!

### Option 1: Modify Shared Memory After TMA

```
TMA loads Q tile → sQ (shared memory)
TMA loads K tile → sK (shared memory)
                   ↓
            [ Apply RoPE to sQ, sK ]  ← INJECT HERE
                   ↓
MMA: S = sQ @ sK^T (tensor memory)
```

**Pros:**
- Simple conceptually
- Data already in shared memory

**Cons:**
- Extra shared memory traffic (read-modify-write)
- Need an extra synchronization step between "TMA done" and "rotated ready"
- May cause pipeline stalls

### Option 2: Modify During Smem→Tmem/Register Copy

This is a common fusion pattern in kernels that explicitly stage operands into registers.

**But:** in the SM100 tcgen05 path used by FA4/CUTE, Q/K are consumed from **shared memory** by the GEMM, and the intermediates land in **TMEM**. There isn't a distinct "smem → regs" copy for Q/K you can hook without rewriting the operand path.

```
(hypothetical)
sQ (shared) → [ Apply RoPE ] → tQ (register tile)
sK (shared) → [ Apply RoPE ] → tK (register tile)
                                ↓
                MMA: S = tQ @ tK^T
```

**Pros:**
- No extra shared memory traffic
- Data already in registers for modification
- Can fuse with the copy operation

**Cons:**
- Deeper in the pipeline
- Need to understand smem→tmem copy path
- More complex hook interface

### Option 3: Fused TMA + RoPE (Hardware Dream)

If TMA could apply transformations during load:

```
Global Q → [ TMA + RoPE ] → sQ (already rotated)
```

This would require hardware support that doesn't exist (yet).

---

## The Ideal Hook API

Given the SM100 tcgen05 operand path, a practical hook would be a **shared-memory modifier** that runs after the TMA stage is known-complete but before the GEMM consumes that stage:

```python
@cute.jit
def qk_smem_mod(sX_stage, positions, cos, sin):
    """
    Apply RoPE to a Q or K *shared-memory stage*.

    Args:
        sX_stage: stage view into the swizzled shared-memory operand tile
        positions: position indices for the logical rows being rotated
        cos, sin: (max_pos, rotary_dim//2) tables

    Returns:
        (typically) nothing: modifies the stage in-place
    """
    ...
```

And the call site would need to ensure the GEMM consumer waits on a "rotated-ready" signal, not on the raw TMA-complete barrier.

```python
# Conceptual integration point (SM100):
# - TMA fills sQ/sK stage buffers and signals the existing "..._full" mbarriers.
# - A RoPE stage waits on those barriers, updates sQ/sK in-place, then signals a second barrier.
# - The MMA warp waits on the *second* barrier before running GEMM that consumes sQ/sK.
```

---

## Practical Path Forward

### Step 1: Find the Actual Producer→Consumer Boundary

On SM100, the key boundary for "where can I modify Q/K?" is:

```python
# Producer: TMA into shared memory (sQ/sK/sV), tracked by mbarriers
load_Q(...)
load_K(...)

# Consumer (warp 12): tcgen05 GEMM reads A/B from shared memory and writes S/O into TMEM
tSrQ = tiled_mma_qk.make_fragment_A(sQ)
tSrK = tiled_mma_qk.make_fragment_B(sK)
gemm_Si(..., sA=sQ[...], sB=sK[...], tCrA=tSrQ[...], tCrB=tSrK[...])
```

There isn't a separate "smem → registers" staging step for Q/K that you can conveniently intercept; the tcgen05 GEMM path sources A/B from shared memory and produces intermediates in TMEM.

### Step 2: Identify the Modification Point

For this SM100 kernel:
- **Operands:** Q/K/V live in **shared memory** (often with swizzled layouts tuned for tcgen05).
- **Intermediates:** scores/probabilities/output partials are staged through **TMEM** (see the `tmem_*_offset` bookkeeping).

So a RoPE hook must either:
1. **Write rotated Q/K into the exact shared-memory layouts the GEMMs consume**, or
2. Change the GEMM operand path itself (much more invasive).

### Step 3: Add Sync Points

The naive "do RoPE in shared memory then `__syncthreads()`" sketch is **not safe** in this kernel: warps are specialized and don't all reach the same control-flow points, so a full-threadblock barrier can deadlock.

If you add a shared-memory transform stage, it needs to be synchronized using the same kind of mechanisms the kernel already uses:
- **mbarriers** (e.g. introduce a separate "rope_ready" barrier per stage), or
- **named barriers** / warp-group scoped barriers that only include the participating warps.

```python
# Conceptual (one possible design):
# 1) wait for the existing TMA barrier ("raw tile is in smem")
# 2) run the transform with a defined participating group
# 3) signal a second barrier ("rotated tile is ready")
# 4) have the GEMM consumer wait on (3), not on (1)
```

---

## Key Insights

### 1. TMA is Async and Efficient

TMA loads tiles with minimal thread involvement. The key is synchronization via mbarriers.

### 2. score_mod is Post-Dot-Product

`score_mod` injects after QK^T, modifying scores. It can't do RoPE.

### 3. RoPE Needs Pre-Dot-Product Hook

RoPE modifies Q/K vectors before QK^T. Would need:
- Access to shared memory after TMA load
- Position information for each tile
- cos/sin tables

### 4. Blackwell MMA Reads from Shared Memory

In this SM100 tcgen05 kernel, the GEMMs source A/B from shared memory; there isn't a distinct Q/K register-tile staging step to hook. Any RoPE fusion has to respect the swizzled shared-memory layouts and the existing barrier protocol.

### 5. Pipeline Complexity

The multi-stage pipeline (load N+1, compute N, write N-1) makes hook injection tricky - need to maintain synchronization.

---

## Questions for Further Investigation

1. **What's the overhead of smem modification?**
   - How many cycles to apply RoPE in-place in shared memory?
   - Does it break the pipeline?

2. **Can we overlap RoPE with other work?**
   - While MMA computes on tile N, can we RoPE tile N+1?
   - Would need careful sync

3. **Position tracking:**
   - How do we know the positions for each Q/K tile?
   - For KV-cache, Q has different offsets than K

4. **cos/sin table access:**
   - Store in shared memory? (extra smem usage)
   - Recompute on-the-fly? (extra ALU)
   - Pass as TMA-loaded texture?

---

## References

### Code Files

| File | Description |
|------|-------------|
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py` | Main SM100 forward pass, TMA setup |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/copy_utils.py` | TMA utility functions |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/pipeline.py` | Pipeline and mbarrier abstractions |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/blackwell_helpers.py` | Blackwell-specific helpers |

### Blogs & Docs

| Resource | What It Covers |
|----------|----------------|
| `notes/research/2025-12-24/incoming/perf/blogs/gau-nerst-tcgen05.md` | TMA, mbarrier, tcgen05 tutorial |
| `notes/research/2025-12-24/incoming/perf/blogs/thunderkittens-blackwell.md` | Warp specialization patterns |
| `notes/research/2025-12-24/incoming/perf/blogs/warp-specialization.md` | Additional warp specialization notes |
| [PTX docs - tcgen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05) | Official tcgen05 reference |
| [PTX docs - TMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor) | TMA instructions |

### Related Explainers

| # | Topic | Relevance |
|---|-------|-----------|
| 3 | [score_mod](03-score-mod.md) | How post-QK^T hooks work |
| 4 | [RoPE](04-rope.md) | What we want to fuse |
