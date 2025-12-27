# Blackwell Primitives Cheatsheet

> Status: Complete (integrated from doc_ref_guide)
> Priority: Medium — quick reference for kernel writing
> Date: 2025-12-26
> Source: [`claude01.md`](../DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/claude01.md)

## Purpose

Tight reference for "what is TMA / mbarrier / WGMMA / TMEM and where it shows up in CuTe/CUTLASS" — so Level 6 kernel work is grounded in the actual primitives.

---

## Blackwell Architecture Overview

| Property | Value | Notes |
|----------|-------|-------|
| Compute Capability | 10.0 (B200), **10.3 (B300)**, 12.0 (RTX 5090) | Different fast paths |
| CUDA minimum | 12.8+ (12.9 recommended) | For SM103 support |
| cuDNN | 9.13+ | Blackwell-optimized |
| Key new features | TMA, WGMMA, TMEM, warp specialization | |

---

## TMA (Tensor Memory Accelerator)

### What It Is

Async bulk copy engine for structured tensor data. Moves tiles between global memory and shared memory with hardware-managed addressing.

### When to Use

- Tiled loads with predictable patterns (conv, attention)
- Large shared memory tiles
- When you want to overlap memory with compute

### PTX Instructions

| Instruction | Purpose | PTX ISA Section |
|-------------|---------|-----------------|
| `cp.async.bulk.shared::cta.global` | Global → Shared (1D) | §9.7.9.25.4.1 |
| `cp.async.bulk.tensor.2d.shared::cluster.global` | Global → Shared (2D-5D) | §9.7.9.27.1.2 |
| `cp.async.bulk.tensor.*.shared.global` | Shared → Global | §9.7.9.27 |
| `cp.async.bulk.commit_group` | Commit pending copies | |
| `cp.async.bulk.wait_group` | Wait for completion | |

### PTX Syntax Examples

```ptx
// TMA 1D copy
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
    [dstMem], [srcMem], size, [smem_bar];

// TMA 2D tensor copy
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_ptr], [tensor_map, {coord0, coord1}], [mbar_ptr];
```

### CuTe API

```cpp
// TMA descriptor setup (host-side, before kernel launch)
auto tma_load = make_tma_copy(
    SM90_TMA_LOAD{},
    tensor_gmem,
    layout_smem
);

// Issue TMA load (device-side)
copy(tma_load, thr_copy_gmem, thr_copy_smem);
```

### Key Constraints / Gotchas

| # | Constraint | Details |
|---|------------|---------|
| 1 | **16-byte alignment mandatory** | Both source and destination must be 16-byte aligned |
| 2 | **Size must be multiple of 16** | The size operand for cp.async.bulk |
| 3 | **16-byte stride alignment** | All strides except stride-1 dim must be multiples of 16 bytes |
| 4 | **TensorMap is 64B-aligned** | `CUtensorMap` descriptor address must be 64-byte aligned |
| 5 | **Descriptors host-created** | Use `cuTensorMapEncodeTiled` (driver API) / `cuTensorMapEncode` before kernel launch |
| 6 | **Descriptor encodes everything** | Base ptr, dtype, dims, strides, SMEM box size, swizzle, OOB behavior |
| 7 | **`*_a` targets unlock features** | Some PTX “architecture-accelerated” features require `sm_90a`/`sm_100a`/`sm_103a` targets (not just `sm_103`) |

---

## mbarrier (Memory Barrier)

### What It Is

Hardware synchronization primitive for producer-consumer patterns. Enables efficient warp specialization.

### When to Use

- Producer/consumer warp patterns
- Async completion tracking (TMA)
- Fine-grained synchronization

### PTX Instructions

| Instruction | Purpose | PTX ISA Section |
|-------------|---------|-----------------|
| `mbarrier.init` | Initialize barrier | §9.7.13.15 |
| `mbarrier.arrive` | Signal arrival | §9.7.13.15.11 |
| `mbarrier.wait` | Wait for phase | §9.7.13.15.13 |
| `mbarrier.arrive.expect_tx` | Arrive with expected bytes | §9.7.13.15.12 |

### Pipeline Pattern (Two Barrier Arrays)

```
Pipeline uses TWO barrier arrays:
  - full_barrier:  producer → consumer ("data ready")
  - empty_barrier: consumer → producer ("buffer free")

Each barrier has:
  - Phase bit (toggles each completion)
  - Arrival count
  - Transaction count (tx-count)
```

### CuTe/CUTLASS API

```cpp
// From cutlass/pipeline/pipeline.hpp
using MainloopPipeline = typename cutlass::PipelineTmaAsync<Stages>;
MainloopPipeline pipeline(shared_storage.pipeline, params.pipeline);

// Producer side
pipeline.producer_acquire(state);
// ... issue TMA loads ...
pipeline.producer_commit(state);

// Consumer side
pipeline.consumer_wait(state);
// ... use data ...
pipeline.consumer_release(state);
```

### Key Constraints / Gotchas

| # | Constraint | Details |
|---|------------|---------|
| 1 | **Phase completion requires TWO conditions** | Arrival count = 0 AND tx-count = 0. Missing either → hang |
| 2 | **expect_tx ACCUMULATES** | Multiple calls ADD to tx-count, don't replace. Track total carefully |
| 3 | **TMA producer_commit is NOOP** | TMA hardware auto-updates tx-count. Call for API consistency only |
| 4 | **Pipeline init critical** | Must call `make_producer_start_state()` so first `producer_acquire()` succeeds |
| 5 | **Phase bit semantics** | Even/odd phases; must match between arrive and wait |
| 6 | **mbarrier is `.b64` (8B)** | PTX `mbarrier.*.b64` uses an 8-byte object in shared memory with 8-byte alignment (don’t confuse with 64-byte `CUtensorMap` alignment) |

---

## WGMMA (Warpgroup Matrix Multiply-Accumulate)

### What It Is

Tensor Core operations at warpgroup granularity (4 warps = 128 threads). Higher throughput than per-warp MMA. **Deprecated on Blackwell** — use tcgen05.mma instead.

### When to Use

- Large matrix multiplies (Hopper SM90)
- When you can fill 128-thread workgroups
- Attention, GEMM

### PTX Instructions

| Instruction | Purpose | PTX ISA Section |
|-------------|---------|-----------------|
| `wgmma.mma_async.*` | Async warpgroup MMA | §9.7.15.5 |
| `wgmma.fence.*` | Memory fence | §9.7.15.5 |
| `wgmma.commit_group` | Commit pending ops | |
| `wgmma.wait_group` | Wait for completion | |

### Supported Shapes

| M | N | K (bytes) | Types | Notes |
|---|---|-----------|-------|-------|
| **64** (fixed) | 8-256 (×8) | 32 | FP16, BF16 | MN-major or K-major |
| **64** (fixed) | 8-256 (×8) | 32 | FP8, TF32, INT8 | **K-major only** |

### CuTe API

```cpp
// From cute/atom/mma_traits_sm90_gmma.hpp
using TiledMma = TiledMMA<
    MMA_Atom<SM90_64x128x16_F16F16F16_SS>,  // SS = both operands from SMEM
    Layout<Shape<_2, _1, _1>>>;              // 2 warpgroups

// Execute
gemm(tiled_mma, accum, tA, tB, accum);
```

### Key Constraints / Gotchas

| # | Constraint | Details |
|---|------------|---------|
| 1 | **M dimension always 64** | Fixed; N ranges 8-256 in multiples of 8 |
| 2 | **Requires 128 threads** | 4 contiguous warps = 1 warpgroup |
| 3 | **Operand B always from SMEM** | Operand A: SMEM (SS mode) or registers (RS mode) |
| 4 | **Accumulator in registers** | C/D always in registers |
| 5 | **Non-16-bit types need K-major** | FP8/TF32/INT8 must use K-major layout |
| 6 | **wgmma.fence required** | Before first WGMMA if registers were modified |
| 7 | **Consecutive WGMMAs** | Can skip intermediate fences if same accumulator shape |

---

## TMEM (Tensor Memory)

### What It Is

Large distributed shared memory accessible by tensor cores. ~256KB per SM. Extends shared memory capacity for large tiles. **Blackwell-only (SM100+)**.

### When to Use

- Very large attention tiles
- When shared memory is the bottleneck
- ThunderKittens-style kernels

### PTX Instructions

| Instruction | Purpose | PTX ISA Section |
|-------------|---------|-----------------|
| `tcgen05.alloc` | Allocate TMEM | §9.7.16.7.1 |
| `tcgen05.dealloc` | Deallocate TMEM | §9.7.16.7.1 |
| `tcgen05.ld.*` | Load from TMEM | §9.7.16 |
| `tcgen05.st.*` | Store to TMEM | §9.7.16 |
| `tcgen05.relinquish_alloc_permit` | Allow future CTAs to queue | §9.7.16.7.1 |

### PTX Syntax

```ptx
// Allocate TMEM
tcgen05.alloc.cta_group::1 [tmem_addr], nCols;

// Deallocate (only warp that allocated can deallocate)
tcgen05.dealloc.cta_group::1 [tmem_addr], nCols;
```

### Key Constraints / Gotchas

| # | Constraint | Details |
|---|------------|---------|
| 1 | **Address space is 6** | Distinct from SMEM (address space 3) |
| 2 | **~256KB per SM** | Organized as 512 columns × 128 rows of 32-bit cells |
| 3 | **Address format** | Bits 31-16 = lane ID, bits 15-0 = column |
| 4 | **Only allocator can deallocate** | The warp that called `tcgen05.alloc` must call `tcgen05.dealloc` |
| 5 | **Must relinquish permit** | Call `tcgen05.relinquish_alloc_permit` or SM starvation |

---

## tcgen05 (5th Gen Tensor Core) — Blackwell MMA

### What It Is

Blackwell's replacement for WGMMA. **2-4× faster** than wgmma.mma_async. Uses TMEM for operands.

### When to Use

- All matrix multiplies on Blackwell (SM100/SM103)
- Replaces WGMMA for new code

### PTX Instructions

| Instruction | Purpose | PTX ISA Section |
|-------------|---------|-----------------|
| `tcgen05.mma` | Matrix multiply-accumulate | §9.7.16.10.9 |
| `tcgen05.mma.cta_group::1` | Standard MMA | §9.7.16.10.9 |
| `tcgen05.mma.ws` | Warp-specialized variant | §9.7.16.10.9 |

### PTX Syntax

```ptx
// Blackwell MMA
tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc,
    disable_output_lane, enable_input_d, scale_input_d;
```

### Key Constraints / Gotchas

| # | Constraint | Details |
|---|------------|---------|
| 1 | **Replaces WGMMA** | wgmma.mma_async is deprecated on Blackwell |
| 2 | **2-4× performance** | Over WGMMA on equivalent workloads |
| 3 | **Requires SM100a/SM103a** | PTX 8.6+, compile with `-arch=sm_100a` |
| 4 | **Uses TMEM** | Operands come from/go to Tensor Memory |
| 5 | **No forward PTX compat** | "a" suffix means arch-specific, not portable |

---

## Warp Specialization Pattern

### The Idea

Split warps into specialized roles:
- **Producer warps:** Issue TMA loads, stage tiles (few registers)
- **Consumer warps:** Compute (WGMMA/tcgen05, math) (many registers)
- **Store warps:** Write results back

### Why It's Faster

- Hides memory latency with compute
- Reduces contention
- Better instruction-level parallelism
- Register allocation optimized per role

### Typical Structure

```
Warpgroup (128 threads = 4 warps):
  ├── Warp 0:   Producer (TMA load A, B) — 24 registers
  ├── Warp 1-2: Consumer (WGMMA/tcgen05) — 240 registers
  └── Warp 3:   Store (TMA store C)      — 240 registers
```

### Register Allocation

```ptx
// Set max registers per warp (PTX §9.7.19.5)
setmaxnreg.inc.sync.aligned.u32 240;  // Consumer warps
setmaxnreg.dec.sync.aligned.u32 24;   // Producer warps
```

| Role | Typical Registers | Rationale |
|------|-------------------|-----------|
| Producer | 24 | Just enough for TMA control |
| Consumer | 240 | Maximize accumulator space |
| Store | 240 | May also do epilogue math |

### CuTe / CUTLASS Support

```cpp
// From CUTLASS persistent kernel examples
// See: test/unit/gemm/device/sm100_*

// Warp specialization with TMA pipeline
template <class MainloopPipeline>
struct WarpSpecializedMainloop {
    static constexpr int NumMmaWarpGroups = 2;
    static constexpr int NumProducerThreads = 128;  // 1 warpgroup
    // ...
};
```

### Key Examples to Study

- `examples/49_hopper_gemm_with_collective_builder/` — EVT epilogue customization
- `examples/cute/tutorial/wgmma_sm90.cu` — WGMMA patterns
- `test/unit/gemm/device/sm100_*` — Blackwell tests

---

## CUTLASS/CuTe Building Blocks

### For Custom GEMM Epilogues

| Component | Purpose | Location |
|-----------|---------|----------|
| `CollectiveMainloop` | Main GEMM loop | `cutlass/gemm/collective/` |
| `CollectiveEpilogue` | Post-GEMM ops | `cutlass/epilogue/collective/` |
| `Sm90TmaWarpSpecialized` | Hopper mainloop | `cutlass/gemm/collective/sm90_mma_tma_gmma_*.hpp` |
| EVT (Epilogue Visitor Tree) | Custom epilogue fusion | Example 49 |

### For TMA Operations

| Component | Purpose |
|-----------|---------|
| `SM90_TMA_LOAD` | TMA load operation |
| `SM90_TMA_STORE` | TMA store operation |
| `make_tma_copy()` | Create TMA copy object |

### Swizzle Modes (Bank Conflict Elimination)

| Mode | Purpose | Use When |
|------|---------|----------|
| `GMMA::Layout_MN_SW128_Atom<T>` | 128-byte swizzle, MN-major | Default for FP16/BF16 |
| `GMMA::Layout_MN_SW64_Atom<T>` | 64-byte swizzle | Smaller tiles |
| `GMMA::Layout_MN_SW32_Atom<T>` | 32-byte swizzle | Very small tiles |
| `GMMA::Layout_MN_INTER_Atom<T>` | No swizzle | Debug only |

### Critical Source Files

```
include/cute/atom/copy_traits_sm90_tma.hpp   # TMA copy traits
include/cute/atom/mma_traits_sm90_gmma.hpp   # WGMMA traits
include/cute/swizzle.hpp                      # Swizzle patterns
include/cutlass/pipeline/pipeline.hpp        # Pipeline primitives
```

---

## Quick Reference: When to Use What

| Scenario | Hopper (SM90) | Blackwell (SM100/103) |
|----------|---------------|----------------------|
| Tiled GEMM | WGMMA + TMA + mbarrier | tcgen05.mma + TMA + TMEM |
| Attention | FA3 (WGMMA) | FA4 (tcgen05) |
| Layout transform | TMA async copy | TMA async copy |
| RoPE fusion | Register ops + TMA store | Register ops + TMA store |
| KV cache write | TMA store | TMA store |

---

## SM Version Requirements Matrix

| Feature | Min SM | PTX Version | Notes |
|---------|--------|-------------|-------|
| `cp.async.bulk` (TMA) | SM_90 | 8.0+ | |
| `wgmma.mma_async` | SM_90a | 8.0+ | Deprecated on Blackwell |
| `tcgen05.*` | SM_100a/103a | 8.6+ | Blackwell-only |
| TMEM | SM_100 | 8.6+ | 256KB/SM |
| FlashAttention 3 | SM_90 | — | 75% utilization |
| FlashAttention 4 | SM_100 | — | Forward-only, BF16 |

### Compile Flags

```bash
# Hopper
nvcc -arch=sm_90a ...

# Blackwell (B300 = SM103)
nvcc -arch=sm_103a ...

# Note: "a" suffix = architecture-specific, no forward PTX compat
```

---

## Minimum Viable Reading Path

| Week | Focus | Resources |
|------|-------|-----------|
| 1 | Foundations | CuTe Quickstart (00_quickstart.md), Layout tutorial (01_layout.md) |
| 2 | Core patterns | TMA Tensors doc, Pipeline doc, Colfax tutorials |
| 3 | Integration | GEMM API 3.x, Example 49 source, EVT patterns |

---

## References

### NVIDIA Official

| Resource | URL |
|----------|-----|
| PTX ISA 9.1 | https://docs.nvidia.com/cuda/parallel-thread-execution/ |
| CUTLASS 3.x Design | https://docs.nvidia.com/cutlass/media/docs/cpp/cutlass_3x.html |
| GEMM API 3.x | https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html |
| CuTe TMA Tensors | https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html |
| Pipeline Primitives | https://docs.nvidia.com/cutlass/media/docs/cpp/pipeline.html |
| Blackwell Functionality | https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html |

### GitHub

| Resource | URL |
|----------|-----|
| CUTLASS | https://github.com/NVIDIA/cutlass |
| FlashAttention | https://github.com/Dao-AILab/flash-attention |
| ThunderKittens | https://github.com/HazyResearch/ThunderKittens |

### Research

| Resource | URL |
|----------|-----|
| FA3 Paper | https://tridao.me/publications/flash3/flash3.pdf |
| ThunderKittens Blog | https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell |
