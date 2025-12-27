# B300 GPU Kernel Optimization: Documentation Reference Guide

This guide consolidates primary documentation sources, key constraints, and minimum viable reading paths for FlashAttention, CUTLASS/CuTe, PTX ISA Blackwell primitives, and cuDNN—all targeting SM103 (Blackwell) optimization work.

---

## Target 1: FlashAttention layout contracts (FA2 + FA4/CuTe)

### Authoritative documentation URLs

| Source | URL |
|--------|-----|
| **FA2/FA3 Main Repo** | https://github.com/Dao-AILab/flash-attention |
| **FA3 Paper (Layout Specs)** | https://tridao.me/publications/flash3/flash3.pdf |
| **Hopper Interface (FA3)** | https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py |
| **PyTorch FlexAttention** | https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html |

### Key section headings

- `flash_attn_func` — Batch attention with fixed sequence lengths
- `flash_attn_varlen_func` — Variable-length attention (packed sequences)
- `flash_attn_with_kvcache` — Inference with KV cache
- `flash_attn_qkvpacked_func` — QKV-stacked tensor optimization
- FA3 Paper §2.2: Memory hierarchy, thread hierarchy, low-precision formats
- FA3 Paper §3.3: Low-precision with FP8 (layout transformations)

### Key constraints and gotchas

1. **Tensor shapes differ between APIs**: `flash_attn_func` expects `(B, S, H, D)` while `flash_attn_varlen_func` expects packed `(total_q, H, D)` format. Mixing these causes silent failures or crashes.

2. **Last dimension MUST be contiguous**: Assertions enforce `k_cache.stride(-1) == 1` and `v_cache.stride(-1) == 1`. Non-contiguous head dimensions trigger expensive internal copies.

3. **head_dim alignment varies by build**: FA2 generally requires **multiples of 8**; some builds (vLLM fork) enforce **multiples of 32**. FA3 rounds up internally to nearest of 64, 96, 128, 192, or **256 maximum**.

4. **cu_seqlens format is strict**: Shape must be `(batch_size + 1,)` with dtype `torch.int32`. Values are cumulative sums starting with 0: sequences [3,4,3] → `[0, 3, 7, 10]`. Must also provide `max_seqlen_q` and `max_seqlen_k`.

5. **LSE output layout differs by mode**: Batch mode returns `(B, H, S_q)` in float32; varlen mode returns `(H, total_q)` in float32.

6. **FP8 requires K-major layout**: FA3's FP8 path only accepts K-major format for WGMMA instructions. FP16/BF16 accept both MN-major and K-major.

7. **MQA/GQA head ratio constraint**: Number of Q heads must be exactly divisible by number of KV heads.

8. **FA4 (Blackwell) is forward-only BF16**: Currently targets SM100 with 5-stage pipeline using CuTe DSL. Backward pass not yet implemented; forward supports BF16 only.

9. **score_mod signature (FlexAttention)**: `def score_mod(score, batch, head, q_idx, k_idx) -> Tensor` where all index args are `torch.int` scalars. No trainable parameters allowed inside `score_mod`.

10. **Head dim >192 backward restricted**: FA2 backward with head_dim > 192 requires A100/H100; consumer GPUs only support head_dim ≤ 192 for backward.

### Minimum viable reading

1. **README docstrings** for `flash_attn_func` and `flash_attn_varlen_func` shape/stride requirements
2. **FA3 Paper §3.3** for FP8 layout transformation rules
3. **GitHub Issue #880** for cu_seqlens batch_size+1 explanation
4. **`flash_attn/cute/interface.py`** for FA4/Blackwell interface patterns

---

## Target 2: CUTLASS/CuTe extension points for custom kernels

### Authoritative documentation URLs

| Source | URL |
|--------|-----|
| **CUTLASS 3.x Design** | https://docs.nvidia.com/cutlass/media/docs/cpp/cutlass_3x.html |
| **GEMM API 3.x** | https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html |
| **CuTe TMA Tensors** | https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0z_tma_tensors.html |
| **Pipeline Primitives** | https://docs.nvidia.com/cutlass/media/docs/cpp/pipeline.html |
| **Blackwell Functionality** | https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html |
| **CuTe DSL (Python)** | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html |

### Key section headings

- **CUTLASS 3.x Design**: CuTe Layout/Tensor adoption, conceptual GEMM hierarchy
- **GEMM API 3.x**: Device/Kernel/Collective APIs, Tiled MMA, Atom API
- **CuTe TMA Tensors**: ArithTupleIterators, basis elements, stride representations
- **Pipeline**: PipelineAsync/PipelineTmaAsync, producer/consumer state machines

### Key constraints and gotchas

1. **TMA descriptors must be host-created**: Use `cuTensorMapEncode` before kernel launch. Descriptors encode base pointer, dtype, dimension sizes, strides, SMEM box size, swizzle pattern, and OOB behavior.

2. **16-byte stride alignment for TMA**: All strides except the contiguous stride-1 dimension must be multiples of **16 bytes**.

3. **WGMMA M dimension is always 64**: N ranges 8–256 (multiples of 8), K = 32 bytes (e.g., K=16 for FP16). Requires warpgroup of **128 threads** (4 contiguous warps).

4. **WGMMA operand placement rules**: Operand B **always from SMEM**; operand A can be SMEM (SS mode) or registers (RS mode). Accumulator C/D always in registers.

5. **Non-16-bit types require K-major layout**: Only FP16/BF16 support MN-major; FP8/TF32/INT8 must use K-major.

6. **Swizzle modes match access patterns**: Use `GMMA::Layout_MN_SW128_Atom<T>` for MN-major with 128-byte swizzle; variants include SW64, SW32, INTER (no swizzle). Swizzling eliminates bank conflicts.

7. **Two barrier arrays for pipeline**: `full_barrier` (producer→consumer) and `empty_barrier` (consumer→producer). Each barrier has phase bit + arrival count.

8. **TMA producer_commit is NOOP**: TMA hardware automatically updates transaction count—explicit commit unnecessary but must still call for API consistency.

9. **Pipeline initialization critical**: Call `make_producer_start_state()` to ensure first `producer_acquire()` succeeds for "empty" pipeline start.

10. **wgmma.fence required before first WGMMA**: Must issue `wgmma.fence` if registers were modified before WGMMA; consecutive WGMMAs sharing accumulator shape can skip intermediate fences.

### Key examples to study

- **Example 49**: `examples/49_hopper_gemm_with_collective_builder/` — EVT epilogue customization
- **WGMMA Tutorial**: `examples/cute/tutorial/wgmma_sm90.cu`
- **TMA Patterns**: `examples/cute/tutorial/tma_*.cu`
- **Blackwell Tests**: `test/unit/gemm/device/sm100_*`

### Minimum viable reading

| Week | Focus | Resources |
|------|-------|-----------|
| 1 | Foundations | CuTe Quickstart (00_quickstart.md), Layout tutorial (01_layout.md), Colfax TMA Tutorial |
| 2 | Core patterns | Colfax WGMMA Tutorial, CuTe TMA Tensors doc, Colfax Pipeline Tutorial |
| 3 | Integration | GEMM API 3.x, Example 49 source, Colfax EVT Tutorial |

**Critical source files**:
```
include/cute/atom/copy_traits_sm90_tma.hpp
include/cute/atom/mma_traits_sm90_gmma.hpp
include/cute/swizzle.hpp
include/cutlass/pipeline/pipeline.hpp
```

---

## Target 3: NVIDIA PTX ISA references for Blackwell primitives

### Authoritative documentation URLs

| Instruction Category | PTX ISA Section |
|---------------------|-----------------|
| **cp.async.bulk (TMA 1D)** | §9.7.9.25.4.1 |
| **cp.async.bulk.tensor (TMA 2D-5D)** | §9.7.9.27.1.2 |
| **mbarrier instructions** | §9.7.13.15 |
| **tcgen05 (5th Gen Tensor Core)** | §9.7.16 |
| **wgmma.mma_async** | §9.7.15.5 |
| **setmaxnreg** | §9.7.19.5 |
| **Tensor Memory (TMEM)** | §9.7.16.1 |

**Base URL**: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (PTX ISA 9.1)

### Key section headings

- **§9.7.9.25.1**: Completion mechanisms for async copy operations
- **§9.7.13.15.4-6**: mbarrier phases, tracking, completion conditions
- **§9.7.16.1**: Tensor Memory addressing and allocation
- **§9.7.16.7.1**: tcgen05.alloc/dealloc/relinquish_alloc_permit
- **§9.7.16.10.9**: tcgen05.mma variants (standard, sparse, warp-specialized)
- **§9.7.15.5.1**: WGMMA register fragments and SMEM layouts

### Key constraints and gotchas

1. **TMA 16-byte alignment mandatory**: Both source and destination for `cp.async.bulk` must be 16-byte aligned; size operand must be multiple of 16 bytes.

2. **mbarrier phase completion requires two conditions**: Pending arrival count reaches zero AND tx-count reaches zero. Missing either causes hangs.

3. **mbarrier.expect_tx accumulates**: Multiple calls ADD to tx-count rather than replacing it. Track total expected bytes carefully.

4. **TMEM address space is 6**: Distinct from SMEM (address space 3). TMEM is ~256KB per SM organized as **512 columns × 128 rows** of 32-bit cells.

5. **TMEM address format**: Bits 31-16 = lane ID, bits 15-0 = column. Only the warp that called `tcgen05.alloc` can call `tcgen05.dealloc`.

6. **setmaxnreg constraints**: Range 24–256 registers in multiples of 8. Typical warp-specialization: 24/240/240 for 1 producer + 2 consumer warpgroups.

7. **tcgen05.mma replaces wgmma on Blackwell**: wgmma.mma_async is deprecated; tcgen05.mma offers **2-4× performance improvement**.

8. **SM version requirements**:
   | Instruction | Min SM | PTX Version |
   |-------------|--------|-------------|
   | cp.async.bulk | SM_90 | 8.0+ |
   | wgmma.mma_async | SM_90a | 8.0+ |
   | tcgen05.* | SM_100a/103a | 8.6+ |

9. **Architecture-accelerated suffix "a"**: SM100a/SM103a features lack forward PTX compatibility guarantees. Compile with `-arch=sm_100a`.

10. **TMEM allocation permit**: Use `tcgen05.relinquish_alloc_permit` to allow future CTAs to queue for TMEM—failure to call can cause SM starvation.

### Instruction syntax examples

```ptx
// TMA 1D copy
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes 
    [dstMem], [srcMem], size, [smem_bar];

// TMA 2D tensor copy
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_ptr], [tensor_map, {coord0, coord1}], [mbar_ptr];

// Blackwell MMA
tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc,
    disable_output_lane, enable_input_d, scale_input_d;

// TMEM allocation
tcgen05.alloc.cta_group::1 [tmem_addr], nCols;

// Warp specialization register allocation
setmaxnreg.inc.sync.aligned.u32 240;
```

### Minimum viable reading

1. **TMA**: §9.7.9.25.1 (completion mechanisms) → §9.7.9.27.1.2 (tensor copy)
2. **mbarrier**: §9.7.13.15.4-6 (phase/tracking concepts) → §9.7.13.15.11-16 (instructions)
3. **tcgen05**: §9.7.16.1 (TMEM concept) → §9.7.16.10.9 (MMA variants)
4. **Practical**: Colfax tutorials for TMA, WGMMA, and pipelining patterns

---

## Target 4: cuDNN frontend/backend for Conv3d and graph capture

### Authoritative documentation URLs

| Resource | URL |
|----------|-----|
| **Frontend Developer Guide** | https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/overview.html |
| **Backend Graph API** | https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.0/developer/graph-api.html |
| **Graph Library API Ref** | https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-graph-library.html |
| **Support Matrix** | https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html |
| **cudnn-frontend GitHub** | https://github.com/NVIDIA/cudnn-frontend |

### Key section headings

- **Workflow Steps 1-11**: Create graph → Define tensors → Define ops → Validate → Build → Filter plans → Autotune → Execute
- **Core Concepts**: Tensor layouts, data types, memory alignment
- **Convolution Operations**: `conv_fprop`, `conv_dgrad`, `conv_wgrad`
- **CUDA Graphs**: `cudnnBackendPopulateCudaGraph()`, `cudnnBackendUpdateCudaGraph()`
- **Backend Descriptor Types**: `CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR`, `CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR`

### Key constraints and gotchas

1. **Frontend API preferred**: 5-10× less code than backend, includes RAII, error handling, errata filters, and autotuning. Only use backend for legacy fixed-function routines or pure C interface requirements.

2. **Conv3d uses NCDHW format**: 5D tensors use (N, C, D, H, W) ordering for channel-first. Transform to NDHWC for Tensor Core performance.

3. **NHWC/NDHWC required for Tensor Cores**: NCHW layouts incur automatic transpose overhead. All modern convolution engines prefer channels-last format.

4. **FP16/BF16 channel alignment**: Input channels must be **multiples of 8** for FP16 Tensor Cores; **multiples of 16** for INT8.

5. **128-bit memory alignment minimum**: All tensors and workspace buffers must be 128-bit aligned; **1024-bit alignment** delivers better performance.

6. **In-place operations blocked for multi-node graphs**: Input/output UIDs cannot match in graphs with >1 operation node.

7. **Virtual tensors enable fusion**: Set `set_is_virtual(true)` for intermediate tensors to enable memory I/O optimization—cuDNN may eliminate materializing these tensors entirely.

8. **Neural net heuristics disabled for 3D**: `CUDNN_HEUR_MODE_B` falls back to `CUDNN_HEUR_MODE_A` for 3D convolutions, grouped convolutions (G>1), and dilated convolutions.

9. **FFT/Winograd algorithms block graph capture**: Filter engines with `CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API` for graph-compatible execution.

10. **Runtime compilation overhead**: Engines with `CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION` enable generic fusion but have higher initial compilation time.

### Frontend API Conv3d pattern

```cpp
namespace fe = cudnn_frontend;
auto graph = fe::graph::Graph();
graph.set_io_data_type(fe::DataType_t::HALF)
     .set_compute_data_type(fe::DataType_t::FLOAT);

// 5D input: (N, C, D, H, W)
auto X = graph.tensor(fe::graph::Tensor_attributes()
    .set_dim({N, C, D, H, W})
    .set_stride({C*D*H*W, D*H*W, H*W, W, 1}));

auto conv_options = fe::graph::Conv_fprop_attributes()
    .set_padding({padD, padH, padW})
    .set_stride({strideD, strideH, strideW})
    .set_dilation({dilD, dilH, dilW});

auto Y = graph.conv_fprop(X, W, conv_options);
graph.validate().build_operation_graph(handle);
graph.create_execution_plans({fe::HeurMode_t::A});
graph.build_plans(handle);
```

### Minimum viable reading

1. **Frontend Overview**: Workflow steps 1-11 at developer/overview.html
2. **GitHub README**: Setup and sample locations
3. **Convolution Operations**: frontend/latest/operations/Convolutions.html
4. **CUDA Graphs Utility**: frontend/latest/utilities/cuda-graphs.html
5. **Samples**: `samples/cpp/conv_sample.cpp` in cudnn-frontend repo

---

## Condensed output file suggestions

### `notes/FA4/b300/layout-contracts.md`

Key content: Tensor shape tables (batch vs varlen), stride contiguity rules, head_dim alignment matrix by FA version, cu_seqlens format specification, LSE output shapes, FP8 K-major requirement.

### `notes/FA4/b300/blackwell-primitives-cheatsheet.md`

Key content: tcgen05.mma syntax and constraints, TMEM address format (lane/column bits), TMA alignment rules (16-byte), mbarrier phase completion dual-condition, setmaxnreg ranges, SM100a compile flags.

### `notes/FA4/b300/vae-decode-architecture.md` (cuDNN portions)

Key content: Conv3d NCDHW→NDHWC transform rationale, channel alignment rules (8 for FP16, 16 for INT8), virtual tensor fusion pattern, graph capture engine filtering, frontend API boilerplate for 3D convolutions.

---

## Quick reference table: SM version requirements

| Feature | Hopper (SM90) | Blackwell (SM100/103) |
|---------|---------------|----------------------|
| wgmma.mma_async | ✓ | Deprecated |
| tcgen05.mma | ✗ | ✓ |
| TMEM | ✗ | 256KB/SM |
| TMA (cp.async.bulk.tensor) | ✓ | ✓ |
| FP8 E4M3/E5M2 | ✓ | ✓ (native block-scale) |
| FlashAttention | FA3 (75% util) | FA4 (forward-only BF16) |
| CUTLASS | 3.x WGMMA | 3.8+ tcgen05 |

This reference guide provides the documentation foundation for B300 kernel optimization. For implementation, start with the minimum viable reading paths and progressively explore the constraint details as specific issues arise.
