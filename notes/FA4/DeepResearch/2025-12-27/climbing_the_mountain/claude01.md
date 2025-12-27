# SM103/Blackwell Kernel Development: Layout & Constraint Reference

Primary-source documentation for Level-6+ kernel development targeting self-attention paths (post-QKV → RoPE/pack → KV write) on NVIDIA B300 (SM103).

---

## 1. PTX ISA (CUDA 13.x / PTX ISA 8.7–9.1)

**Link:** https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html  
**PDF:** https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf (v8.7) / PTX ISA 9.1 for latest

### 1.1 cp.async.bulk.tensor.* (TMA) Instructions

**Relevant Section:** 9.7.8.25 "Data Movement and Conversion Instructions: cp.async.bulk.tensor"

```
cp.async.bulk.tensor.{1d,2d,3d,4d,5d}.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_addr], [tensorMap, tensorCoords], [mbar_addr];
```

**Key excerpts from PTX ISA:**

> The `cp.async.bulk.tensor` instruction initiates an asynchronous copy of a tensor tile from global memory to shared memory. The copy is performed by the Tensor Memory Accelerator (TMA) unit.

**Operand constraints:**
- `smem_addr`: Must be aligned to 16 bytes in shared memory
- `tensorMap`: 64-bit address of CUtensorMap descriptor (must be 64-byte aligned)
- `tensorCoords`: Array of 32-bit coordinates, one per dimension
- `mbar_addr`: 64-bit address of mbarrier object in shared memory (8-byte aligned)

**Completion mechanism:**
```
mbarrier::complete_tx::bytes  // Decrements tx-count by bytes transferred
```

### 1.2 mbarrier.* Instructions

**Relevant Sections:** 9.7.12.15 "Parallel Synchronization: mbarrier"

```ptx
// Initialization (once per barrier)
mbarrier.init.shared::cta.b64 [mbar_addr], arrival_count;

// Set expected transaction bytes (before TMA)
mbarrier.arrive.expect_tx.shared::cta.b64 state, [mbar_addr], tx_count;

// Wait for phase completion
mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [mbar_addr], phaseParity;
```

**Phase semantics:**
- mbarrier tracks both **arrival count** and **tx-count** (bytes transferred)
- Phase bit flips when: `arrivals >= expected_arrivals AND tx_count == 0`
- Initial phase after `mbarrier.init` is always **0**
- Consumer waits on current phase; producer arrives and flips phase

**Key constraints:**
- mbarrier object: 64-bit (8 bytes), **8-byte aligned** in shared memory
- `arrival_count`: Number of threads/warps expected to arrive (typically 1 for single-thread TMA issuer)
- `tx_count`: Exact number of bytes the TMA will transfer

### 1.3 tcgen05.* Instructions (Blackwell SM100+)

**Relevant Section:** 9.7.16 "Tensorcore 5th Generation Instructions"

**tcgen05.mma (Matrix Multiply-Accumulate):**
```ptx
tcgen05.mma.cta_group::{1|2}.kind::{f16|bf16|tf32|mxf8|mxf4|nvf4mxf4}
    [tmem_addr], desc_a, desc_b;
```

**Key forms:**
| Instruction | Description |
|------------|-------------|
| `tcgen05.mma.cta_group::1` | Single-CTA MMA (simpler scheduling) |
| `tcgen05.mma.cta_group::2` | 2-CTA cooperative MMA (higher throughput, complex sync) |
| `tcgen05.mma.sp` | Sparse MMA variant |
| `tcgen05.mma.ws` | Warp-specialized variant |

**tcgen05.alloc/dealloc (Tensor Memory):**
```ptx
tcgen05.alloc.cta_group::1.sync.aligned.b32 tmem_addr, num_cols;
tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, num_cols;
tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;
```

**TMEM constraints:**
- SM100 TMEM capacity: 228KB (minus 1KB overhead)
- Minimum allocation: 4 columns
- Column granularity for allocation

**tcgen05.cp (SMEM ↔ TMEM copy):**
```ptx
tcgen05.cp.cta_group::1 [tmem_addr], [smem_addr];
```

**tcgen05.wait/commit:**
```ptx
tcgen05.wait.cta_group::1;           // Wait for prior tcgen05 ops
tcgen05.commit.cta_group::{1|2};     // Commit ops to mbarrier
```

### 1.4 wgmma.* (Hopper, for comparison)

**Note:** wgmma is **SM90 (Hopper)** — replaced by tcgen05.mma on SM100+

```ptx
wgmma.mma_async.sync.aligned.m64n128k16.f16.f16.f16
    {d0..d63}, desc_a, desc_b, scale_d, imm_scale_a, imm_scale_b, neg_imm_a, neg_imm_b;
```

**Key difference on SM100:**
- wgmma requires **warpgroup coordination** (4 warps)
- tcgen05.mma can be issued by **single warp/thread**
- tcgen05.mma outputs to **TMEM** (not registers)

---

## 2. CUDA C++ Programming Guide

**Link:** https://docs.nvidia.com/cuda/cuda-programming-guide/  
**Async Copies Section:** https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html

### 2.1 Tensor Memory Accelerator / Tensor Maps

**Alignment/Stride Rules (from CUDA Programming Guide):**

| Constraint | Requirement |
|-----------|-------------|
| `tensorMap` address | 64-byte aligned |
| `globalAddress` | 16-byte aligned (32-byte if interleave=32B or subbyte types) |
| `globalStrides[i]` | Multiple of 16 bytes, < 2^40 |
| `boxDim[0] * elemSize` | Multiple of 16 bytes |
| Each `boxDim[i]` | ≤ 256 elements |
| `tensorRank` | 3–5 dimensions |

**Swizzle modes and constraints:**

| Swizzle Mode | Inner Dim Constraint | Use Case |
|-------------|---------------------|----------|
| `SWIZZLE_NONE` | No constraint | Simple layouts |
| `SWIZZLE_32B` | boxDim[0]*elemSize ≤ 32 | Small tiles |
| `SWIZZLE_64B` | boxDim[0]*elemSize ≤ 64 | Medium tiles |
| `SWIZZLE_128B` | boxDim[0]*elemSize ≤ 128 | Large tiles, avoid bank conflicts |
| `SWIZZLE_128B_ATOM_32B` | boxDim[0]*elemSize ≤ 128 | 32B granularity swizzle |
| `SWIZZLE_128B_ATOM_64B` | boxDim[0]*elemSize ≤ 128 | 64B granularity swizzle (store only) |

**OOB (Out-of-Bounds) behavior:**
```c
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE           // Zero fill
CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA  // NaN that becomes 0 in FMA
```

### 2.2 Async Transaction Barriers

**mbarrier + TMA integration pattern:**

```cpp
#include <cuda/barrier>

// In shared memory
__shared__ cuda::barrier<cuda::thread_scope_block> bar;

// Initialize (once)
if (threadIdx.x == 0) {
    init(&bar, 1);  // 1 arrival expected
}
__syncthreads();

// Before TMA: set expected bytes
cuda::device::barrier_arrive_tx(bar, 1, num_bytes);

// Issue TMA (single thread)
if (threadIdx.x == 0) {
    cp.async.bulk.tensor... [mbar = &bar]
}

// Wait for completion (all threads)
bar.wait(bar.arrive());
```

### 2.3 Blackwell/SM10x Warp Specialization Notes

**From CUTLASS/CuTe patterns:**

**FA4-style 5-stage pipeline:**
1. **TMA warp** (1 warp): Issues `cp.async.bulk.tensor` for Q, K, V tiles
2. **MMA warp** (1 warp): Issues `tcgen05.mma` instructions
3. **Softmax warps** (2 warpgroups/8 warps): Compute softmax in parallel
4. **Epilogue warps** (4 warps): Move results from TMEM to global

**Key sync points:**
- TMA → MMA: mbarrier with `complete_tx::bytes`
- MMA → Epilogue: `tcgen05.commit` + mbarrier
- Pipeline staging: Phase-based double/triple buffering

---

## 3. cuTensorMapEncode API

**Link:** https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

### 3.1 cuTensorMapEncodeTiled Signature

```c
CUresult cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap,           // OUT: 64-byte aligned descriptor
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,            // 3-5
    void* globalAddress,              // 16-byte aligned (32B for interleave/subbyte)
    const cuuint64_t* globalDim,      // [tensorRank] sizes
    const cuuint64_t* globalStrides,  // [tensorRank-1] in BYTES
    const cuuint32_t* boxDim,         // [tensorRank] tile sizes (≤256 each)
    const cuuint32_t* elementStrides, // [tensorRank] iteration steps (≤8)
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill
);
```

### 3.2 Hard Constraints Summary

| Parameter | Constraint |
|-----------|-----------|
| `tensorMap` | 64-byte aligned |
| `globalAddress` | 16-byte aligned (32B for interleave=32B or subbyte) |
| `globalDim[i]` | > 0, ≤ 2^32 |
| `globalStrides[i]` | Multiple of 16 bytes, < 2^40 |
| `boxDim[i]` | > 0, ≤ 256 |
| `boxDim[0] * elemSize` | Multiple of 16 bytes |
| `elementStrides[i]` | > 0, ≤ 8 |
| `tensorRank` | 3–5 |

### 3.3 Data Type Enum

```c
typedef enum CUtensorMapDataType_enum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,    // 1 byte
    CU_TENSOR_MAP_DATA_TYPE_UINT16,       // 2 bytes
    CU_TENSOR_MAP_DATA_TYPE_UINT32,       // 4 bytes
    CU_TENSOR_MAP_DATA_TYPE_INT32,        // 4 bytes
    CU_TENSOR_MAP_DATA_TYPE_UINT64,       // 8 bytes
    CU_TENSOR_MAP_DATA_TYPE_INT64,        // 8 bytes
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,      // 2 bytes (fp16)
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32,      // 4 bytes (fp32)
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64,      // 8 bytes (fp64)
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,     // 2 bytes (bf16)
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,  // 4 bytes
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,     // 4 bytes (tf32)
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ, // 4 bytes
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, // 4 bits (packed)
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,// 4 bits (packed, gaps)
    CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B // 6 bits (packed, gaps)
} CUtensorMapDataType;
```

### 3.4 Swizzle Enum

```c
typedef enum CUtensorMapSwizzle_enum {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B,            // 16B chunks in 32B span
    CU_TENSOR_MAP_SWIZZLE_64B,            // 16B chunks in 64B span
    CU_TENSOR_MAP_SWIZZLE_128B,           // 16B chunks in 128B span
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,  // 32B chunks in 128B span
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B, // + swap 8B in 16B
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B   // 64B chunks in 128B span
} CUtensorMapSwizzle;
```

---

## 4. CUTLASS / CuTe Documentation

### 4.1 Primary Docs

**Blackwell Functionality:**
- https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md
- https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md

**CuTe Core:**
- https://github.com/NVIDIA/cutlass/tree/main/include/cute

**TMA Tutorial:**
- https://research.colfax-intl.com/tutorial-hopper-tma/

**Blackwell Clusters Tutorial:**
- https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/

### 4.2 Key Example Files

| Example | Path | Focus |
|---------|------|-------|
| Basic Blackwell GEMM | `examples/73_blackwell_gemm_preferred_cluster/` | Dynamic clusters, tcgen05 |
| Stream-K Blackwell | `examples/74_blackwell_gemm_streamk/` | Load balancing |
| FMHA Blackwell | `examples/python/CuTeDSL/blackwell/fmha.py` | Attention kernel |
| SM100 Mainloop | `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp` | Warp specialization |
| Blackwell Pipeline | `include/cute/arch/sm100_mma_umma.hpp` | tcgen05 atoms |

### 4.3 PipelineTmaAsync Pattern (Producer/Consumer)

**From CUTLASS sm100_mma_warpspecialized.hpp:**

```cpp
// Pipeline with TMA + mbarrier
template <int Stages>
struct PipelineTmaAsync {
    using BarrierType = cutlass::arch::ClusterTransactionBarrier;
    
    // Producer side (TMA warp)
    CUTLASS_DEVICE void producer_acquire(int stage) {
        // Wait for consumer to release this stage's buffer
        barrier_[stage].wait(phase_[stage]);
    }
    
    CUTLASS_DEVICE void producer_commit(int stage, uint32_t bytes) {
        // Signal TMA completion
        barrier_[stage].arrive_and_expect_tx(bytes);
    }
    
    // Consumer side (MMA warp)
    CUTLASS_DEVICE void consumer_wait(int stage) {
        barrier_[stage].wait(phase_[stage] ^ 1);
    }
    
    CUTLASS_DEVICE void consumer_release(int stage) {
        barrier_[stage].arrive();
        phase_[stage] ^= 1;  // Flip phase
    }
};
```

### 4.4 CuTe make_tma_copy Pattern

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>

// Host: Create TMA descriptor
auto tma_load_a = make_tma_copy(
    SM90_TMA_LOAD{},           // TMA operation type
    gmem_tensor_a,             // Global memory tensor
    smem_layout_a,             // Shared memory layout
    tile_shape,                // Tile dimensions
    cluster_shape              // Cluster shape
);

// Device: Issue TMA
auto [tAgA, tAsA] = tma_partition(tma_load_a, ...);
copy(tma_load_a.with(mbarrier, mcast_mask), tAgA, tAsA);
```

### 4.5 tcgen05 MMA Atom (CUTLASS)

**From include/cute/atom/mma_traits_sm100.hpp:**

```cpp
// SM100 tcgen05.mma tile shapes
using MmaTileShape_MNK = Shape<_256, _128, _64>;  // Typical tile

// MMA atom for bf16
using TiledMma = TiledMMA<
    MMA_Atom<SM100_MMA_F16BF16_SS>,  // tcgen05.mma.f16/bf16
    Layout<Shape<_1, _1, _1>>,       // Atom layout
    Tile<_256, _128, _64>            // Tile shape
>;
```

---

## 5. FlashAttention-4 / CuTe API Contract

### 5.1 flash_attn_varlen_func Requirements

**Source:** https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py

```python
def flash_attn_varlen_func(
    q,                    # (total_q, num_heads, head_dim)
    k,                    # (total_k, num_heads_k, head_dim)  
    v,                    # (total_k, num_heads_k, head_dim)
    cu_seqlens_q,         # (batch_size + 1,) cumulative sequence lengths
    cu_seqlens_k,         # (batch_size + 1,)
    max_seqlen_q,         # int
    max_seqlen_k,         # int
    dropout_p=0.0,
    softmax_scale=None,   # Default: 1/sqrt(head_dim)
    causal=False,
    window_size=(-1, -1), # (left, right) for sliding window
    softcap=0.0,          # Softmax capping
    alibi_slopes=None,    # (num_heads,) or (batch, num_heads)
    deterministic=False,
    return_attn_probs=False,
):
```

### 5.2 Layout Constraints

| Tensor | Expected Layout | Alignment |
|--------|----------------|-----------|
| Q | `(total_q, num_heads, head_dim)` | Contiguous, head_dim innermost |
| K | `(total_k, num_heads_k, head_dim)` | Contiguous |
| V | `(total_k, num_heads_k, head_dim)` | Contiguous |
| O (output) | Same as Q | Same as Q |

**Head dimension constraints:**
- All head dimensions up to 256 supported
- Head dim > 192 backward requires A100/H100+
- Head dim must be multiple of 8

**Dtype support:**
- `fp16` (all architectures)
- `bf16` (Ampere+ / SM80+)

### 5.3 SM100/Blackwell Specifics

**From FA4 implementation analysis:**

| Aspect | FA4 on SM100 |
|--------|--------------|
| Compute | `tcgen05.mma.cta_group::1` |
| Memory | TMA for Q/K/V loads |
| Accumulators | TMEM (tensor memory) |
| Pipeline | 5-stage warp-specialized |
| Dtypes | BF16 only (currently) |
| FP4/FP8 | Not yet (planned) |
| 2-CTA MMA | Not yet (planned) |

**Current limitations (FA4 beta):**
- Forward pass only (no backward yet)
- BF16 only
- No FP4/FP8 support
- No 2-CTA cooperative matmuls

### 5.4 score_mod / Custom Bias Hooks

**Not yet exposed in FA4.** For custom attention biases:
- Use `alibi_slopes` parameter for ALiBi
- Use `softcap` for softmax capping
- For arbitrary bias: requires kernel modification or `flex_attention` (PyTorch 2.5+)

---

## Key Constraints Summary (Quick Reference)

### Alignment Requirements

| Object | Alignment |
|--------|-----------|
| CUtensorMap descriptor | **64 bytes** |
| Global memory base address | **16 bytes** (32 for interleave/subbyte) |
| Global strides | Multiple of **16 bytes** |
| Shared memory mbarrier | **8 bytes** |
| SMEM tile (for TMA) | **16 bytes** |

### Stride Rules

```
globalStrides[0] = globalDim[0] * sizeof(element) + padding[0]
globalStrides[i] = globalStrides[i-1] * (globalDim[i] + padding[i])

// Constraint: each stride must be multiple of 16 bytes
```

### Swizzle Selection Guide

| Tile Inner Dim (bytes) | Recommended Swizzle |
|-----------------------|---------------------|
| ≤ 32 | `SWIZZLE_32B` |
| ≤ 64 | `SWIZZLE_64B` |
| ≤ 128 | `SWIZZLE_128B` |
| Any (no bank conflicts) | `SWIZZLE_NONE` |

### tcgen05 vs wgmma Quick Comparison

| Aspect | wgmma (SM90) | tcgen05 (SM100) |
|--------|--------------|-----------------|
| Issuing unit | Warpgroup (4 warps) | Single warp/thread |
| Output location | Registers | TMEM |
| Sync mechanism | wgmma.commit_group + mbarrier | tcgen05.commit + mbarrier |
| Performance | ~2x vs Ampere | ~2-4x vs Hopper |
