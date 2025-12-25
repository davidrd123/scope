# The Blackwell Path: SM100-Specific Features in FA4

> **What this is:** An explainer for Blackwell (SM100) specific features used in FlashAttention 4 - tcgen05 tensor cores, tensor memory, shared memory descriptors, and warp specialization patterns.
> **Context:** This complements the other explainers by focusing on *why* FA4 on Blackwell looks different from previous generations.
> **Scope note:** This explainer is anchored on the SM100 implementation in `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`. Other SM10x variants may differ.
> **Updated:** 2025-12-25

---

## Overview

### Why Blackwell is Different

Each GPU generation has unique tensor core instructions:

| Generation | Load | Compute | Accumulator |
|------------|------|---------|-------------|
| Ampere (SM80) | `cp.async` | `mma` | Registers |
| Hopper (SM90) | `cp.async.bulk.tensor` (TMA) | `wgmma.mma_async` | Registers |
| **Blackwell (SM100)** | `cp.async.bulk.tensor` (TMA) | `tcgen05.mma` | **Tensor Memory** |

The key Blackwell innovations:
1. **tcgen05 tensor cores** - 5th generation, new instruction format
2. **Tensor Memory (TMEM)** - Dedicated memory for accumulators
3. **Shared memory descriptors** - New way to specify smem layout to MMA
4. **CTA groups** - Cluster-aware execution

Note: the SM100 FA4 kernel has both TMA and non-TMA paths (e.g., KV can fall back to `cp.async` for paged KV non-TMA).

### What This Explainer Covers

1. tcgen05 MMA instruction format and encoding
2. Tensor Memory (TMEM) - what it is, how FA4 uses it
3. Shared memory descriptors for operands
4. The elect_one pattern for MMA issue
5. How FA4 structures the kernel around these features

---

## tcgen05: 5th Generation Tensor Cores

### The Instruction Format

Blackwell tensor cores use the `tcgen05.mma` PTX instruction:

```asm
tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, pred;
```

Where:
- `tmem_acc` - Tensor memory address for accumulator (output)
- `smem_desc_a` - 64-bit shared memory descriptor for A
- `smem_desc_b` - 64-bit shared memory descriptor for B
- `idesc` - 32-bit instruction descriptor (encodes types + major modes + M/N; K is implied by the op/dtype)
- `pred` - Predicate for accumulate (0=overwrite, 1=accumulate)

### The Instruction Descriptor

The 32-bit `idesc` encodes most of the operation (types/major modes and M/N):

```python
# mma_sm100_desc.py lines 124-162
def make_instr_desc(a_type, b_type, c_type, M, N, a_major, b_major, ...):
    a_fmt = to_UMMA_format(a_type)  # BF16=1, F16=0
    b_fmt = to_UMMA_format(b_type)
    c_fmt = to_C_format(c_type)      # F32=1

    m_dim = M >> 4  # M/16, 5 bits (M=64,128,256)
    n_dim = N >> 3  # N/8, 6 bits  (N=8..256)

    # Pack into 32-bit descriptor (simplified: the real encoding also includes negate/saturate/etc.)
    desc = 0
    desc |= (c_fmt & 0x3) << 4      # c_format [4:6)
    desc |= (a_fmt & 0x7) << 7      # a_format [7:10)
    desc |= (b_fmt & 0x7) << 10     # b_format [10:13)
    desc |= (int(a_major) & 0x1) << 15  # a_major [15:16)
    desc |= (int(b_major) & 0x1) << 16  # b_major [16:17)
    desc |= (n_dim & 0x3F) << 17    # n_dim [17:23)
    desc |= (m_dim & 0x1F) << 24    # m_dim [24:29)
    return desc
```

**Key insight:** Unlike previous generations where shape was implicit in the instruction mnemonic (e.g., `mma.m16n8k16`), Blackwell encodes shape in a runtime descriptor.

### MMA Shapes

tcgen05 supports flexible MMA shapes:

| Dimension | Valid Values |
|-----------|-------------|
| M | 64, 128, 256 |
| N | 8, 16, 24, ..., 256 (multiples of 8) |
| K | Depends on dtype (typically 16-64) |

FA4 uses two different MMA shapes in the forward pass:
- **QK^T:** `(M, N) = (m_block_size, n_block_size)`
- **P×V:** `(M, N) = (m_block_size, head_dim_v_padded)`

---

## Tensor Memory (TMEM)

### What is Tensor Memory?

TMEM is a new memory space in Blackwell, dedicated to MMA accumulators:

| Memory | Scope | Purpose in FA4 |
|--------|-------|----------------|
| Registers | Per-thread | Scalar state + control + small fragments |
| Shared Memory | Per-CTA | Staging Q/K/V/O tiles + barriers |
| **Tensor Memory (TMEM)** | Per-CTA | MMA accumulators + intermediate matrices (S/P/O) |

**Why TMEM?**
- MMA outputs don't need to be in thread-local registers
- Frees up register pressure
- Enables different warps to access MMA results via TMEM copy ops (without spilling through global memory)

### TMEM in FA4

```python
# flash_fwd_sm100.py lines 155-156
SM100_TMEM_CAPACITY_COLUMNS = 512
self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
```

Note: "columns" here is the kernel's unit for budgeting TMEM usage. The kernel tracks how many columns it needs (`self.tmem_total`) and asserts it fits.

### Allocation and Deallocation

```python
# flash_fwd_sm100.py lines 981-1015 (MMA warp)
if warp_idx == self.mma_warp_id:
    # Allocate TMEM for this CTA
    tmem_alloc_cols = Int32(self.tmem_alloc_cols)
    cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
    cute.arch.sync_warp()

    # ... MMA computation ...

    # Cleanup
    cute.arch.relinquish_tmem_alloc_permit()
    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        Float32,
        alignment=16,
        ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
    )
    cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)
```

**Key point:** TMEM allocation is done by one warp, but the result is visible to the whole CTA. The holding buffer (`tmem_holding_buf`) stores the allocated TMEM address so other warps can access it.

### MMA Output to TMEM

In the gemm functions, the accumulator is addressed as a TMEM pointer:

```python
# blackwell_helpers.py line 58
acc_tmem_addr = acc.iterator.toint()
```

This address is passed directly to the PTX instruction:
```asm
tcgen05.mma.cta_group::1.kind::f16 [$0], smem_desc_a, smem_desc_b, idesc, p;
                                   ^
                                   └── TMEM address
```

---

## Shared Memory Descriptors

### The 64-bit Descriptor

Blackwell MMA reads A and B from shared memory, but instead of passing pointers, we pass **descriptors** that encode layout information:

```python
# mma_sm100_desc.py lines 216-286
def make_smem_desc_base(layout, swizzle, major):
    layout_type = _layout_type(swizzle)  # SWIZZLE_128B, SWIZZLE_64B, etc.

    # Pack into 64-bit descriptor
    desc = 0
    desc |= (leading_byte_offset & 0x3FFF) << 16   # [16:30)
    desc |= (stride_byte_offset & 0x3FFF) << 32    # [32:46)
    desc |= (VERSION & 0x3) << 46                  # [46:48)
    desc |= (BASE_OFFSET & 0x7) << 49              # [49:52)
    desc |= (LBO_MODE & 0x1) << 52                 # [52:53)
    desc |= (int(layout_type) & 0x7) << 61         # [61:64)
    return desc
```

The descriptor encodes:
- **Swizzle pattern** - How data is laid out in shared memory
- **Strides** - Leading dimension and stride
- **Layout type** - K-major vs MN-major

### Start Address

The 64-bit descriptor is split: high 32 bits from the base descriptor, low 32 bits include the start address:

```python
# mma_sm100_desc.py lines 289-291
def make_smem_desc_start_addr(start_addr):
    # 14 bits, remove 4 LSB (bits 0-13 in desc)
    return (start_addr.toint() & 0x3FFFF) >> 4
```

```python
# blackwell_helpers.py lines 131-136
smem_desc_start_a_lo = Int32(smem_desc_base_a_lo) | make_smem_desc_start_addr(
    sA[None, None, 0].iterator
)
smem_desc_start_b_lo = Int32(smem_desc_base_b_lo) | make_smem_desc_start_addr(
    sB[None, None, 0].iterator
)
```

### Swizzle Patterns

| Layout Type | Swizzle | Use Case |
|-------------|---------|----------|
| SWIZZLE_NONE | None | Simple layouts |
| SWIZZLE_32B | 32-byte | Small tiles |
| SWIZZLE_64B | 64-byte | Medium tiles |
| SWIZZLE_128B | 128-byte | Large tiles (common) |
| SWIZZLE_128B_BASE32B | Special | MN-major with 32B base |

FA4 typically uses SWIZZLE_128B for efficient bank-conflict-free access.

---

## The Elect-One Pattern

### Why Only One Thread Issues MMA

In this implementation, each `tcgen05.mma` is guarded by **one elected thread**:

```python
# blackwell_helpers.py lines 150-174
with cute.arch.elect_one():
    llvm.inline_asm(
        None,
        [...],
        "{\n\t"
        ".reg .pred leader_thread;\n\t"
        "elect.sync _|leader_thread, -1;\n\t"
        ...
        f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], smem_desc_a, smem_desc_b, idesc, p;\n\t"
        "}\n",
        ...
    )
```

The `elect.sync` instruction:
1. Elects one thread as leader
2. Sets `leader_thread` predicate
3. Only leader executes the MMA

**Why?** The hardware broadcasts the operation to tensor cores internally. Multiple threads issuing the same MMA would be redundant (and wrong).

### The Loop Pattern

```python
# blackwell_helpers.py lines 277-317
# Unrolled K loop with elect_one
llvm.inline_asm(
    None,
    [...],
    "{\n\t"
    "elect.sync _|leader_thread, -1;\n\t"
    ...
    # First MMA
    f"@leader_thread tcgen05.mma... [tmem_acc], smem_desc_a, smem_desc_b, idesc, p;\n\t"
    # K iterations (unrolled)
    + "".join(
        f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {offset_a_diff[k-1]};\n\t"
        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {offset_b_diff[k-1]};\n\t"
        f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
        f"@leader_thread tcgen05.mma... [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
        for k in range(1, K_tiles)
    )
    + "}\n",
    ...
)
```

The K-dimension loop is unrolled at compile time, updating the descriptor offsets between MMAs.

---

## CTA Groups and Clusters

### What are CTA Groups?

Blackwell supports **CTA clusters** - groups of CTAs that can cooperate:

```python
# flash_fwd_sm100.py line 112
self.cluster_shape_mn = (1, 1)  # No clustering by default
```

The MMA instruction specifies CTA group size:
```asm
tcgen05.mma.cta_group::1.kind::f16  # cta_group::1 = single CTA
tcgen05.mma.cta_group::2.kind::f16  # cta_group::2 = pair of CTAs
```

### Pipeline with Clusters

```python
# pipeline.py lines 262-340
class PipelineTmaUmma:
    @staticmethod
    def create(..., cta_layout_vmnk, ...):
        # Consumer type is TCGen05Mma (not AsyncThread)
        consumer_type = PipelineOp.TCGen05Mma

        # CTA group derived from cta_layout_vmnk (ONE when not using clusters)
        cta_group = (
            tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1
            else tcgen05.CtaGroup.TWO
        )
```

For FA4, clustering *can* enable multicast loads where one TMA load populates shared memory for multiple CTAs. In the current `flash_fwd_sm100.py` kernel, `cluster_shape_mn=(1, 1)` and `PipelineTmaUmma.create(...)` is called without `cta_layout_vmnk`, so multicast isn't active by default.

---

## FA4 Kernel Structure on Blackwell

### Warp Specialization

16 warps with dedicated roles:

```python
# flash_fwd_sm100.py lines 148-154
self.softmax0_warp_ids = (0, 1, 2, 3)    # Softmax stage 0
self.softmax1_warp_ids = (4, 5, 6, 7)    # Softmax stage 1
self.correction_warp_ids = (8, 9, 10, 11) # Output rescaling
self.mma_warp_id = 12                     # MMA (single warp issues all MMAs)
self.epilogue_warp_ids = (13,)            # Store output
self.load_warp_ids = (14,)                # TMA loads
self.empty_warp_ids = (15,)               # Available
```

Notes:
- If `use_tma_KV=False` (paged KV non-TMA), the kernel uses two load warps: `load_warp_ids=(14, 15)` and `empty_warp_ids=()`.
- If `is_varlen_q=True`, epilogue work may be reassigned (see `use_correction_warps_for_epi` / `is_varlen_q` handling).

### Why One MMA Warp?

On Blackwell, MMA is issued by one thread, and the result goes to TMEM. Only one warp is needed to issue MMAs; other warps do useful work (softmax, correction) while MMA executes.

### The Double-Buffering Strategy

FA4 processes 2 Q tiles per CTA:

```python
# flash_fwd_sm100.py lines 103-107
self.q_stage = 2

# 2 Q tile per CTA
self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
```

This enables:
- Softmax0 warps process Q tile 0 while Softmax1 warps process Q tile 1
- Better utilization of the 16 warps
- Overlap of softmax computation with MMA

### Memory Layout

```python
# flash_fwd_sm100.py lines 615-637
@cute.struct
class SharedStorage:
    # mbarriers for pipelines
    mbar_ptr: cute.struct.MemRange[Int64, self.mbar_total]
    # TMEM holding buffer (stores allocated TMEM address)
    tmem_holding_buf: Int32
    # Row statistics (max and sum)
    sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
    # Output buffer
    sO: cute.struct.Align[cute.struct.MemRange[o_dtype, sO_size], align]
    # Q tiles
    sQ: cute.struct.Align[cute.struct.MemRange[q_dtype, sQ_size], align]
    # K tiles (multi-stage)
    sK: cute.struct.Align[cute.struct.MemRange[k_dtype, sK_size], align]
```

---

## Tensor-SMEM (TS) Mode

### A from TMEM, B from SMEM

For the P×V matmul (softmax output times V), A comes from tensor memory:

```python
# blackwell_helpers.py lines 93-94, 175-195
is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM

if is_ts:
    # A comes from TMEM (P matrix from softmax)
    llvm.inline_asm(
        ...,
        f"tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], smem_desc_b, {hex(idesc)}, p;\n\t"
        ...                                           ^
    )                                                 └── [$1] is TMEM address
else:
    # A comes from SMEM (Q matrix)
    llvm.inline_asm(
        ...,
        f"tcgen05.mma.cta_group::1.kind::f16 [$0], smem_desc_a, smem_desc_b, idesc, p;\n\t"
        ...
    )
```

**Why TS mode?**
- After softmax, P is in TMEM (never written to smem)
- P×V reads P directly from TMEM
- Avoids TMEM→SMEM→TMEM round-trip

---

## Comparison: Hopper vs Blackwell

| Aspect | Hopper (SM90) | Blackwell (SM100) |
|--------|--------------|-------------------|
| MMA instruction | `wgmma.mma_async` | `tcgen05.mma` |
| Accumulator | Registers | Tensor Memory |
| Operand source | SMEM | SMEM or TMEM |
| Shape encoding | In mnemonic | In descriptor |
| Issue pattern | Warpgroup | Elect-one |
| Pipeline | TmaAsync | TmaUmma / AsyncUmma |

### Key Differences in FA4

1. **Accumulator location:** Hopper keeps O in registers, Blackwell uses TMEM
2. **Softmax access:** On Blackwell, softmax warps read from TMEM via special copy operations
3. **P×V matmul:** Blackwell uses TS mode (P from TMEM), Hopper uses SS mode (P in SMEM)

---

## Questions for Further Investigation

1. **TMEM capacity limits:**
   - 512 columns - how does this constrain tile sizes?
   - What happens with larger head dimensions?

2. **Cluster efficiency:**
   - FA4 defaults to cluster=(1,1). When does clustering help?
   - Multicast TMA vs independent loads

3. **Register pressure:**
   - How much register pressure does TMEM relieve?
   - Impact on occupancy

4. **TS mode performance:**
   - Is TS mode always faster than copying to SMEM?
   - When would SS mode be preferred?

---

## References

### Code Files

| File | Description |
|------|-------------|
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py` | Main SM100 forward pass |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/blackwell_helpers.py` | MMA helper functions, PTX emission |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/mma_sm100_desc.py` | Instruction + smem descriptor encoding |
| `vendored/flash_attn_cute_score_mod/flash_attn/cute/pipeline.py` | UMMA pipeline helpers (`PipelineTmaUmma`, etc.) |

### External Resources

| Resource | What It Covers |
|----------|----------------|
| [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) | Excellent Blackwell matmul tutorial |
| [PTX tcgen05 docs](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05) | Official instruction reference |
| [CUTLASS Blackwell FMHA](https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha) | NVIDIA's reference implementation |

### Related Explainers

| # | Topic | Relevance |
|---|-------|-----------|
| 5 | [TMA and Memory Loading](05-tma-memory-loading.md) | How TMA loads work |
| 6 | [Tile Scheduling](06-tile-scheduling.md) | CTA/cluster scheduling |
| 7 | [Online Softmax](07-online-softmax.md) | TMEM↔softmax interaction |
