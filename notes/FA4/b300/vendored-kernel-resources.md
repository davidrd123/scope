# Vendored Kernel Resources: In-House Reference Material

> Status: Resource index + extracted patterns
> Priority: Reference material for Level 6 kernel work
> Date: 2025-12-26

## Purpose

This doc indexes the vendored CuTe/CUTLASS code in this repo and extracts actionable patterns for custom kernel development. These are real Blackwell kernels we can learn from directly.

---

## Vendored Code Locations

| Directory | Contents | Relevance |
|-----------|----------|-----------|
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/` | FA4 SM100 attention (Tri Dao) | **High** — our actual attention backend |
| `vendored/cutlass-cute/python/CuTeDSL/blackwell/` | NVIDIA FMHA, GEMM, MLA examples | **High** — reference patterns |
| `vendored/cutlass-cute/python/CuTeDSL/utils/` | FMHA helpers, tile schedulers | Medium — utility patterns |
| `vendored/rope/` | RoPE implementations (vLLM, transformers, flash_attn) | Medium — RoPE variants |

---

## Key Files in `flash_attn_cute_score_mod`

### Core Kernels

| File | What It Does | Key Patterns |
|------|--------------|--------------|
| `flash_fwd_sm100.py` | SM100 forward attention | 5-stage pipeline, score_mod, tcgen05 |
| `flash_fwd.py` | Dispatch/interface | Shape validation, dtype handling |
| `pack_gqa.py` | GQA head packing | Pointer arithmetic for packed layouts |
| `softmax.py` | Online softmax | Numerically stable streaming softmax |

### Infrastructure

| File | What It Does | Useful For |
|------|--------------|------------|
| `copy_utils.py` | TMA copy helpers | `cvt_copy`, `load_s2r`, `make_tmem_copy` |
| `barrier.py` | Barrier management | Named barriers, sync patterns |
| `named_barrier.py` | Named barrier enum | Barrier ID conventions |
| `mma_sm100_desc.py` | MMA descriptors | tcgen05 MMA config |
| `seqlen_info.py` | Sequence length tracking | Varlen handling |
| `block_info.py` | Block metadata | Tile scheduling |
| `mask_definitions.py` | Causal/window masks | Mask patterns |
| `fast_math.py` | Fast divmod | Avoiding expensive integer division |

---

## Key Files in `cutlass-cute/blackwell`

### Reference Kernels

| File | Lines | What It Does |
|------|-------|--------------|
| `fmha.py` | ~2600 | Full FMHA with warp specialization |
| `fmha_bwd.py` | ? | Backward pass (reference) |
| `dense_gemm.py` | ~1000 | Basic Blackwell GEMM |
| `dense_gemm_software_pipeline.py` | ~1200 | GEMM with pipeline stages |
| `mla.py` | ~5200 | Multi-head Latent Attention (DeepSeek) |
| `mixed_input_gemm.py` | ~2700 | FP8/mixed precision GEMM |

### Distributed (Lower Priority)

| File | What It Does |
|------|--------------|
| `distributed_gemm_all_reduce_blackwell.py` | GEMM + all-reduce fusion |
| `distributed_all_gather_gemm_blackwell.py` | All-gather + GEMM fusion |

---

## Extracted Patterns

### Pattern 1: Warp Specialization Layout

Two concrete examples. **The exact warp IDs can differ between kernels**, so don’t assume the numbers; assume the pattern (dedicated producer/consumer roles).

**FA4 attention (Tri Dao)**
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`

```python
self.softmax0_warp_ids = (0, 1, 2, 3)       # Softmax stage 0
self.softmax1_warp_ids = (4, 5, 6, 7)       # Softmax stage 1
self.correction_warp_ids = (8, 9, 10, 11)   # Online correction
self.mma_warp_id = 12                       # MMA compute
self.epilogue_warp_ids = (13,)              # Epilogue / store
self.load_warp_ids = (14,)                  # TMA loads (can be (14, 15) in some modes)
self.empty_warp_ids = (15,)                 # Padding/unused (may be repurposed)
```

**CUTLASS/CuTe DSL FMHA example**
- `vendored/cutlass-cute/python/CuTeDSL/blackwell/fmha.py`

```python
self.softmax0_warp_ids = (0, 1, 2, 3)
self.softmax1_warp_ids = (4, 5, 6, 7)
self.correction_warp_ids = (8, 9, 10, 11)
self.mma_warp_id = 12
self.load_warp_id = 13
self.epilogue_warp_id = 14
self.empty_warp_id = 15
```

**Key insight**: Warps are statically assigned roles. Producer (load) and consumer (MMA) are separate, and these roles often drive the pipeline structure (mbarriers/TMA/TMEM).

### Pattern 2: Named Barrier Convention

```python
class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # Starts from 1 (barrier 0 = sync_threads)
```

**Key insight**: Barrier 0 is reserved for `__syncthreads()`. Custom barriers start at 1.

### Pattern 3: TMEM Allocation

From `fmha.py`:

```python
SM100_TMEM_CAPACITY_COLUMNS = 512
self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS  # Full 256KB
```

**Key insight**: TMEM is 256KB/SM, addressed as 512 columns. Allocate early in kernel.

### Pattern 4: CTA Tiler Convention

```python
# 2 Q tiles per CTA (for better occupancy)
self.cta_tiler = (2 * m_block_size, n_block_size, head_dim_padded)
self.mma_tiler_qk = (m_block_size, n_block_size, head_dim_padded)
self.mma_tiler_pv = (m_block_size, head_dim_v_padded, n_block_size)
```

**Key insight**: CTA processes 2x tiles, MMA instruction works on 1x. This 2-stage Q pipeline is common.

### Pattern 5: Head Dimension Padding

```python
hdim_multiple_of = 16
self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
self.check_hdim_oob = head_dim != self.head_dim_padded
```

**Key insight**: Head dim must be multiple of 16 for efficient TMA. Pad and track OOB predicate.

### Pattern 6: GQA Pointer Arithmetic

From `pack_gqa.py`:

```python
@cute.jit
def compute_ptr(self, tensor, cRows, tidx, block, threads_per_row, num_threads):
    # Map linear index to (m_idx, h_idx) for GQA head mapping
    idx = block * self.m_block_size + row
    m_idx = idx // self.qhead_per_kvhead
    h_idx = idx - m_idx * self.qhead_per_kvhead
    return utils.elem_pointer(tensor, ((h_idx, m_idx),)).toint()
```

**Key insight**: GQA packing requires custom pointer compute to map Q heads to KV heads.

### Pattern 7: Copy Atom Factory

From `copy_utils.py`:

```python
@dsl_user_op
def get_copy_atom(dtype, num_copy_elems, is_async=False):
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
```

**Key insight**: Copy atoms are parameterized by (dtype, bits, async). Max 128 bits per copy.

### Pattern 8: Type-Converting Copy

```python
@dsl_user_op
def cvt_copy(atom, src, dst, *, pred=None):
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_fragment_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    cute.copy(atom, src, dst, pred=pred)
```

**Key insight**: Type conversion happens in registers before copy. Use `make_fragment_like` for temp.

### Pattern 9: Persistent Kernel Tile Scheduling

From `dense_gemm_software_pipeline.py`:

```python
# is_persistent controls whether kernel loops over tiles or returns
self.is_persistent = is_persistent

# Persistent loop pattern:
while work_tile_info.is_valid():
    process_tile(work_tile_info)
    work_tile_info = scheduler.advance()
```

**Key insight**: Persistent kernels amortize launch overhead. Need tile scheduler for work distribution.

### Pattern 10: Alignment Requirements

From `dense_gemm_software_pipeline.py` constraints:

```
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned:
  - TFloat32: multiple of 4 elements
  - Float16/BFloat16: multiple of 8 elements
  - Int8/Uint8/Float8: multiple of 16 elements
```

**Key insight**: TMA requires 16-byte minimum alignment. Plan tensor dimensions accordingly.

---

## Feature Support Matrix (flash_fwd_sm100.py)

| Feature | Supported | Notes |
|---------|-----------|-------|
| BF16 | ✓ | |
| FP16 | ✓ | |
| FP8 | ✗ | "will be added later" |
| Causal | ✓ | |
| Noncausal | ✓ | |
| MHA/GQA/MQA | ✓ | `qhead_per_kvhead` param |
| hdim 64, 96, 128 | ✓ | |
| hdim 192, 256 | ✗ | "will be added later" |
| Varlen | ✓ | `is_varlen_q` |
| Sliding window | ✓ | `is_local` |
| Split-KV | ✓ | `is_split_kv` |
| score_mod | ✓ | Pure function, no trainable params |
| mask_mod | ✓ | Block sparsity support |
| Paged KV | ✓ | `paged_kv_non_tma` variant |
| Pack GQA | ✓ | Folds Q heads into M dimension |

---

## RoPE Implementations (vendored/rope/)

| File | Source | Key Difference |
|------|--------|----------------|
| `flash_attn_triton_rotary.py` | flash_attn | Triton kernel, interleaved/non-interleaved |
| `vllm_rotary_embedding_base.py` | vLLM | Base class, cos/sin caching |
| `vllm_rotary_embedding_common.py` | vLLM | Common rotary ops |
| `transformers_modeling_llama.py` | HuggingFace | Reference implementation |

**Fusion opportunity**: RoPE is separate from attention. The flash_attn Triton RoPE could be a fusion target.

---

## How to Use This Doc

1. **Before writing a pack kernel**: Read `pack_gqa.py` for GQA pointer patterns
2. **Before writing a fused kernel**: Read `flash_fwd_sm100.py` for warp specialization
3. **For TMA copy patterns**: Read `copy_utils.py`
4. **For tile scheduling**: Read `fmha_helpers.py`
5. **For GEMM epilogue patterns**: Read `dense_gemm_software_pipeline.py`

---

## Cross-References

- Layout contracts: `layout-contracts.md`
- Blackwell primitives: `blackwell-primitives-cheatsheet.md`
- Kernel experiment template: `kernel-experiment-template.md`

---

## TODO

- [ ] Extract softmax streaming patterns from `softmax.py`
- [ ] Document pipeline stage barriers from `pipeline.py`
- [ ] Extract TMA descriptor patterns from actual kernel code
- [ ] Profile which patterns are actually hot in our workload
