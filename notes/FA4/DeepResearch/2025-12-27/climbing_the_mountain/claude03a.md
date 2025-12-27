# NVIDIA Blackwell SM100/SM103 kernel development reference

The fifth-generation Tensor Core (tcgen05) instructions for Blackwell GPUs are documented across PTX ISA 8.7-9.1, CUTLASS 3.8+, and FlashAttention-4. This reference extracts **verbatim specifications** with exact version pins, section numbers, and source citations for production kernel development.

## PTX ISA version matrix and SM100 target support

**PTX ISA 9.1** is the current release (December 2025), with **PTX ISA 8.8** introducing SM100/SM103 targets. The tcgen05 instruction family spans Sections 9.7.16.1 through 9.7.16.12 in the PTX ISA documentation.

| PTX ISA Version | Key SM100 Features |
|-----------------|-------------------|
| **8.8** | Adds `sm_100`, `sm_103`, `sm_100f`, `sm_103f` targets; introduces family-specific targets with 'f' suffix |
| **8.7** | Extends `tcgen05.mma` with `.kind::mxf4nvf4` and `.scale_vec::4X` qualifiers |
| **9.0** | Current documentation baseline at docs.nvidia.com/cuda/parallel-thread-execution |
| **9.1** | Adds `.volatile` qualifier with `.local` state space for `ld`/`st` |

From PTX ISA 8.8 release notes (verbatim): 
> "Introduces family-specific target architectures that are represented with 'f' suffix. PTX for family-specific targets is compatible with all subsequent targets in same family. Adds support for sm_100f, sm_101f, sm_103f, sm_120f, sm_121f."

## tcgen05.mma instruction structure and kind qualifiers

**Section 9.7.16.10.9.1** documents the `tcgen05.mma` instruction. The instruction operates on Tensor Memory (TMEM) with descriptors in shared memory.

### Kind qualifiers (Section 9.7.16.10)
The `.kind` qualifier specifies the data type and MMA operation mode:

```
.kind::tf32      - TF32 MMA Operation
.kind::f16       - F16/BF16 MMA Operation  
.kind::i8        - I8 MMA Operation
.kind::f8f6f4    - FP8/FP6/FP4 MMA Operation
.kind::mxf8f6f4  - MXF8 BlockScaled MMA Operation
.kind::mxf4      - MXF4 BlockScaled MMA Operation
.kind::mxf4nvf4  - MXF4NVF4 BlockScaled MMA Operation (PTX ISA 8.7+)
```

### CTA group qualifier
```
.cta_group::1    - Single CTA mode (one SM)
.cta_group::2    - CTA pair mode (two SMs cooperating)
```

### Block scaling qualifiers (Section 9.7.16.10.7)
Scale vector size qualifiers for block-scaled operations:
```
.scale_vec::1X   - Base scaling
.scale_vec::2X   - 2x scaling
.scale_vec::3X   - K=96 configuration
.scale_vec::4X   - 4x scaling (PTX ISA 8.7+)
.block16         - 16-element blocks
.block32         - 32-element blocks
```

### Descriptor format (Section 9.7.16.4.1)
Descriptors `desc_a` and `desc_b` reside in **shared memory** (`.shared` state space) as 64-bit packed values:
- **Base address**: 14-bit field (4 LSB not included, address >> 4)
- **Stride information**: Encoded in remaining bits
- **Alignment**: 8 bytes

From Table 43 (PTX ISA 8.8): "Shared memory descriptor layout" defines the 64-bit descriptor format with base address and stride fields for matrices A and B.

## tcgen05.commit and tcgen05.wait synchronization

**Section 9.7.16.12.1** (`tcgen05.commit`) and **Section 9.7.16.8.5** (`tcgen05.wait`) document the completion mechanism.

The documentation at Section 9.7.16.6.2.1.2 describes the "`tcgen05.wait` instruction based completion mechanism" for pipelined tcgen05 operations. From Compute Sanitizer documentation (verbatim):
> "On Blackwell GPUs, if a program is using the PTX Tensor Core 5th generation family instructions (tcgen05.*), the PTX optimizing assembler can be instructed to insert guardrails around TCMMA instructions using PTXAS flag -g-tmem-access-check."

## mbarrier phase semantics and tx-count tracking

**Section 9.7.13.15** documents mbarrier operations. Phase completion requires **both** arrival count and tx-count to reach zero.

### Phase completion (Section 9.7.13.15.6)
Verbatim from CUTLASS tutorial documentation:
> "The mbarrier object has two internal counters that are used to track completion of the current phase: a **pending arrival count in threads** and a **pending transaction count (tx-count) in bytes**. **When both counts reach 0, the phase is completed.**"

### mbarrier.arrive.expect_tx syntax (Section 9.7.13.15.11)
PTX ISA 8.0+, SM_90:
```ptx
mbarrier.arrive.expect_tx.shared::cta.b64 _, [addr], txCount;

// Standalone expect_tx forms:
mbarrier.expect_tx.relaxed.cta.shared::cta.b64     [addr], txCount;
mbarrier.expect_tx.relaxed.cluster.shared::cta.b64 [addr], txCount;
mbarrier.expect_tx.relaxed.cta.shared::cluster.b64 [addr], txCount;
mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64 [addr], txCount;
```

Semantic (verbatim from MLIR docs): "The expect-tx operation, with an $txcount argument, **increases the tx-count of an mbarrier object by the value specified by $txcount**. This makes the current phase of the mbarrier object to expect and track the completion of additional asynchronous transactions."

### mbarrier.try_wait.parity syntax (Section 9.7.13.15.16)
```ptx
// Basic form (PTX ISA 7.8, SM_90):
mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [addr], phaseParity;

// With suspend hint:
mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [addr], phaseParity, suspendTimeHint;

// With semantic and scope (PTX ISA 8.0+):
mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 waitComplete, [addr], phaseParity;
mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 waitComplete, [addr], phaseParity;

// Relaxed semantic (PTX ISA 8.6+):
mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 waitComplete, [addr], phaseParity;
```

Operand types:
| Operand | Type | Description |
|---------|------|-------------|
| `waitComplete` | `.pred` | Output: true if phase completed |
| `addr` | `uint64_t*` | mbarrier address in shared memory |
| `phaseParity` | `uint32_t` | Phase parity to wait for (0 or 1) |
| `suspendTimeHint` | `uint32_t` | Optional nanoseconds suspend hint |

From CUTLASS Tutorial (verbatim): "The `try_wait` qualifier (as opposed to `test_wait`) indicates that the wait is a blocking instruction. The `parity` qualifier, whose use entails providing a phase bit, indicates that the **thread sleeps until that phase bit of the mbarrier flips**."

## TMA store operations (shared::cta → global)

**Section 9.7.9.25.5.2** documents `cp.async.bulk.tensor` for tensor copy operations.

### STORE syntax (tile mode)
```ptx
// Direction: .global (dst) .shared::cta (src)
cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
```

### Scatter4 mode STORE (PTX ISA 86, SM_100a/SM_101a)
```ptx
cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [tensorMap, tensorCoords], [srcMem];
```

**Key constraints:**
- Completion mechanism: `.bulk_group` (stores use bulk async-group, NOT mbarrier)
- Source state space: Only `.shared::cta` (not `.shared::cluster`)
- Modes: `.tile` and `.tile::scatter4`

## cuTensorMapEncodeTiled driver API constraints

**Source:** CUDA Driver API v13.1.0, Section 6.30 "Tensor Map Object Management"  
**URL:** docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

### Function signature
```c
CUresult cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap,           // Output: 64-byte aligned
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,            // 3-5
    void* globalAddress,              // 16B aligned (32B for interleave=32B)
    const cuuint64_t* globalDim,      // Non-zero, ≤ 2^32
    const cuuint64_t* globalStrides,  // Multiple of 16B, < 2^40
    const cuuint32_t* boxDim,         // Non-zero, ≤ 256
    const cuuint32_t* elementStrides, // Non-zero, ≤ 8
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill
);
```

### Verbatim parameter constraints

**globalAddress alignment** (verbatim):
> "globalAddress, which specifies the starting address of the memory region described, must be 16 byte aligned. The following requirements need to also be met:
> - When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, globalAddress must be 32 byte aligned.
> - When tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, globalAddress must be 32 byte aligned."

**globalStrides** (verbatim):
> "globalStrides array, which specifies tensor stride of each of the lower tensorRank - 1 dimensions in bytes, must be a multiple of 16 and less than 2^40. Additionally, the following requirements need to be met:
> - When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, the strides must be a multiple of 32.
> - When tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, the strides must be a multiple of 32."

**boxDim** (verbatim):
> "boxDim array, which specifies number of elements to be traversed along each of the tensorRank dimensions, must be non-zero and less than or equal to 256. Additionally, the following requirements need to be met:
> - When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE, { boxDim[0] * elementSizeInBytes(tensorDataType) } must be a multiple of 16 bytes."

### OOB fill enum values
```c
typedef enum CUtensorMapFloatOOBfill_enum {
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
} CUtensorMapFloatOOBfill;
```

**CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE (= 0):** Out-of-bounds elements are filled with **ZERO**. From documentation: "zero... will be used to fill out-of-bound elements."

**CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA:** Out-of-bounds filled with special NaN that triggers zero behavior in FMA. Restriction (verbatim):
> "Note that CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA can only be used when tensorDataType represents a floating-point data type, and when tensorDataType is not CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, and CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B."

### Constraints summary table
| Parameter | Constraint |
|-----------|------------|
| tensorMap alignment | **64 bytes** |
| globalAddress | **16 bytes** (32B when interleave=32B or packed types) |
| globalStrides | Multiple of **16 bytes**, < 2^40 |
| globalDim | Non-zero, ≤ **2^32** |
| boxDim | Non-zero, ≤ **256**; boxDim[0]*elemSize multiple of 16B |
| elementStrides | Non-zero, ≤ **8** |
| tensorRank | **3-5** |

## CUDA C++ barrier APIs from libcudacxx

**Header:** `#include <cuda/barrier>`  
**Feature flag:** `__cccl_lib_local_barrier_arrive_tx`  
**Minimum arch:** SM_90 (Hopper+)

### cuda::device::barrier_arrive_tx (primary API)
```cpp
__device__ cuda::barrier<cuda::thread_scope_block>::arrival_token 
cuda::device::barrier_arrive_tx(
    cuda::barrier<cuda::thread_scope_block>& bar,
    ptrdiff_t arrive_count_update,      // Amount to decrement arrival count
    ptrdiff_t transaction_count_update  // Bytes to add to expected tx-count
);
```

### cuda::device::barrier_expect_tx (expect only, no arrive)
```cpp
__device__ void cuda::device::barrier_expect_tx(
    cuda::barrier<cuda::thread_scope_block>& bar,
    ptrdiff_t transaction_count_update
);
```

### Usage example (verbatim from libcudacxx docs)
```cpp
#include <cuda/barrier>
#include <cuda/std/utility>

__global__ void example_kernel() {
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    // Arrive with transaction count update of 0 bytes
    auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);
    bar.wait(cuda::std::move(token));
}
```

**Constraints:**
- Barrier must be in `__shared__` memory
- Only works with `cuda::thread_scope_block`
- Use `init(&bar, expected_count)` friend function (not constructor)
- Use `#pragma nv_diag_suppress static_var_with_dynamic_init` for `__shared__` barriers

## FlashAttention-4 SM100 implementation constraints

**Repository:** github.com/Dao-AILab/flash-attention  
**Version:** flash-attn 2.8.3 (late 2025)  
**Key files:** `flash_attn/cute/flash_fwd_sm100.py`, `flash_attn/cute/interface.py`

### TMEM capacity assertion
**File:** `flash_attn/cute/flash_fwd_sm100.py`, Line 132

```python
assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS
```

Per NVIDIA documentation, TMEM allocation columns maximum is **512** (minimum 32). The constraint failed for Gemma-3 with `head_dim=256` on B200 (GitHub Issue #1959).

### head_dim constraints
| head_dim | SM100 cute support |
|----------|-------------------|
| **64** | NOT supported |
| **128** | Supported ✓ |
| **256** | Fails TMEM capacity assertion |

From GitHub Issue #2000: "flash_attn_cute on B200 can support head_dim=128 but no head_dim=64."

### dtype restrictions
Supported: `bf16` (bfloat16), `fp16`

### Stride contiguity requirements
From `interface.py`:
```python
v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
```
V tensor must be contiguous on last dimension (stride -1 == 1) OR on dimension -3.

### Warp specialization architecture
| Warp Type | Count | Purpose |
|-----------|-------|---------|
| Load warp | 1 | Load Q, K, V from global memory |
| MMA warp | 1 | Matrix multiply-accumulate |
| Softmax warps | 8 (2 warpgroups) | Normalize attention scores |
| Correction warps | 4 | Rescale outputs |
| Epilogue warps | 1-2 | Store output to global memory |

## CUTLASS 3.8+ SM100 MMA traits

**Repository:** github.com/NVIDIA/cutlass  
**Tag:** v3.8.0 (January 25, 2025) - first Blackwell SM100 support  
**File:** `include/cute/atom/mma_traits_sm100.hpp`

### Copyright header (verbatim)
```cpp
/***************************************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 ***************************************************************************************************/
```

### Key includes
```cpp
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
```

### SmemDescriptor construction (verbatim)
```cpp
template <UMMA::Major MajorMode, class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
SmemDescriptor
make_umma_desc(Tensor<TEngine,TLayout> const& tensor)
{
  static_assert(is_smem<TEngine>::value, "UMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "UMMA Descriptors can only be constructed on rank-2 tensors.");

  SmemDescriptor desc;
  desc.version_ = 1;     // Set the version for blackwell
  desc.lbo_mode_ = 0;    // set to legacy mode by default

  constexpr UMMA::LayoutType LAYOUT_TYPE = UMMA::layout_type(u128_tensor);
  desc.layout_type_ = uint8_t(LAYOUT_TYPE);

  uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(u128_tensor.data()));
  desc.start_address_ = static_cast<uint16_t>(start_address >> 4);
  // ...
}
```

### SM100 MMA shape constraints (verbatim static_asserts)
**FP8/FP6/FP4 single SM:**
```cpp
static_assert(M == 64 || M == 128, 
    "SM100_MMA_F8F6F4_SS M-mode size should be 64 or 128 for 1 CTA cluster MMA.");
static_assert((M == 64  && (N % 8 == 0)  && (8 <= N)  && (N <= 256)) ||
              (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
    "SM100_MMA_F8F6F4_SS N-mode size should be a multiple of 8 between 8 and 256 for M=64, "
    "or a multiple of 16 between 16 and 256 for M=128.");
```

**FP8/FP6/FP4 dual SM (2x1SM):**
```cpp
static_assert(M == 128 || M == 256, 
    "SM100_MMA_F8F6F4_2x1SM_SS M-mode size should be 128 or 256.");
static_assert((N % 32 == 0) && (32 <= N) && (N <= 256), 
    "SM100_MMA_F8F6F4_2x1SM_SS N-mode size should be a multiple of 32 between 32 and 256.");
```

### Supported tile shapes summary
| Mode | M sizes | N sizes | K (bits) |
|------|---------|---------|----------|
| 1SM, M=64 | 64 | 8-256 (mult 8) | 256 |
| 1SM, M=128 | 128 | 16-256 (mult 16) | 256 |
| 2SM | 128, 256 | 32-256 (mult 32) | 256 |

### TMEM allocation modes (verbatim enum)
```cpp
enum class TmemAllocMode {
  Interleaved = 0,           // Default allocation mode
  NonInterleaved = 1,        // Prevents interleaving
  Duplicated = 2,            // Duplicates across subpartitions
  ScaleFactorDuplicated4by1 = 3, // Scale factor for 4x1 data path
  ScaleFactorDuplicated2by2 = 4  // Scale factor for 2x2 data path
};
```

### MMA execution pattern (verbatim)
```cpp
uint64_t desc_a = A[0];
uint64_t desc_b = B[0];
uint32_t tmem_c = raw_pointer_cast(D.data());
uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
SM100_MMA_F16BF16_SS::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_), idesc);
```

## Conclusion

This reference anchors Blackwell SM100/SM103 kernel development to exact specifications. Key architectural changes from Hopper include the transition from WGMMA to **tcgen05** instructions, the introduction of **Tensor Memory (TMEM)** as a first-class address space, and **2-SM cooperative MMA** via `.cta_group::2`. The cuTensorMapEncodeTiled API maintains Hopper-era constraints with the critical clarification that **CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE produces zeros** (not undefined values). FlashAttention-4's SM100 implementation currently requires **head_dim=128** due to TMEM capacity constraints, with head_dim=64 and 256 unsupported.
