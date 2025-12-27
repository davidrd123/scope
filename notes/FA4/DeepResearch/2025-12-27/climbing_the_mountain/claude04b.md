# PTX ISA Specifications for SM103/Blackwell Development

The confusion in your documentation stems from conflating two **distinct objects**: mbarrier (8 bytes, 8-byte aligned) and CUtensorMap (64-byte aligned). Below are the exact specifications from PTX ISA documentation with verbatim excerpts where available.

## Issue 1: mbarrier object size and alignment (RESOLVED)

### Definitive specification

| Property | Value | Source |
|----------|-------|--------|
| **Size** | 8 bytes (64-bit) | `.b64` type |
| **Alignment** | **8 bytes** | PTX ISA 9.7.13.15.1 |
| **Location** | Shared memory | `.shared` state space |

**PTX ISA Version:** 9.0 (Section numbering consistent in 8.5–9.1)
**Section:** 9.7.13.15.1 "Size and alignment of mbarrier object"

**Verbatim quote** (from MLIR/LLVM documentation which directly cites PTX ISA):
> "An mbarrier object is an opaque object in shared memory with **.b64 type and an alignment of 8-bytes.**"

The PTX instruction `mbarrier.init.shared.b64` confirms the `.b64` (64-bit = 8-byte) type specification.

### Resolution of the "64-byte" confusion

The **64-byte alignment** refers to **CUtensorMap**, NOT mbarrier. These are completely different objects:

| Object | Purpose | Size | Alignment | Memory Space |
|--------|---------|------|-----------|--------------|
| **mbarrier** | Async synchronization barrier | 8 bytes | **8 bytes** | Shared |
| **CUtensorMap** | TMA tensor descriptor | Opaque (≥64B) | **64 bytes** | Constant/Param |

**Verbatim from CUDA Driver API v13.1** (cuTensorMapEncodeTiled):
> "The parameters passed are bound to the following requirements: **tensorMap address must be aligned to 64 bytes.**"

---

## Issue 2: cp.async.bulk.tensor target restrictions for sm_103/BF16/FP16

### Key findings from PTX ISA 8.8/9.0 documentation

**PTX ISA 8.8 introduces sm_103 support:**
> "Adds support for `sm_103` target architecture. Adds support for target `sm_103a` that supports architecture-specific features. Introduces family-specific target architectures that are represented with 'f' suffix. PTX for family-specific targets is compatible with all subsequent targets in same family. Adds support for `sm_100f`, `sm_101f`, `sm_103f`..."

### Target architecture requirements for cp.async.bulk.tensor

From libcudacxx PTX documentation (which mirrors PTX ISA verbatim):

**LOAD direction (.shared::cluster ← .global):**
```
// cp.async.bulk.tensor.{1d-5d}.shared::cluster.global.tile.mbarrier::complete_tx::bytes
// PTX ISA 80, SM_90
```

**LOAD direction (.shared::cta ← .global):**
```
// cp.async.bulk.tensor.{1d-5d}.shared::cta.global.tile.mbarrier::complete_tx::bytes
// PTX ISA 86, SM_90
```

**With .cta_group modifiers:**
```
// cp.async.bulk.tensor.{1d-5d}.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group
// .cta_group = { .cta_group::1, .cta_group::2 }
// PTX ISA 86, SM_100a, SM_101a
```

**Note:** sm_103 is NOT explicitly listed for `.cta_group` variants—these appear limited to SM_100a and SM_101a in current documentation.

### BF16/FP16 (2-byte types) vs subbyte types

The tensor copy restrictions distinguish between:
- **2-byte types** (f16, bf16): Standard alignment to 16-byte boundaries for bounding box
- **Subbyte types** (e4m3, e5m2, etc.): Additional alignment requirements

From CUTLASS Blackwell documentation:
> "Tensor copy instructions for **subbyte types** impose additional alignment requirements while loading narrow-precision tensors from global memory to shared memory."

For **BF16 and FP16** specifically:

| Data Type | Element Size | Alignment (elements) |
|-----------|--------------|---------------------|
| `half_t` (f16) | 2 bytes | 8 elements |
| `bfloat16_t` (bf16) | 2 bytes | 8 elements |

The PTX ISA section 9.7.9.25.5.1 "Restriction on Tensor Copy instructions" contains the full restrictions, but 2-byte floating point types (FP16/BF16) do **NOT** have the special subbyte restrictions—they use standard tensor copy alignment rules.

### sm_103 confirmation for 5th-gen TensorCore (tcgen05) with f16

From libcudacxx documentation:
```
// tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc...
// PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
// .kind = { .kind::f16, .kind::tf32 }
```

This confirms **SM_103a and SM_103f support `.kind::f16` (FP16)** for tcgen05 MMA operations.

---

## Summary table

| Item | Specification | PTX ISA Section |
|------|---------------|-----------------|
| mbarrier size | 8 bytes (.b64) | 9.7.13.15.1 |
| mbarrier alignment | 8 bytes | 9.7.13.15.1 |
| CUtensorMap alignment | 64 bytes | CUDA Driver API |
| cp.async.bulk.tensor basic | SM_90+, PTX ISA 80 | 9.7.9.25.5.2 |
| cp.async.bulk.tensor .cta_group | SM_100a, SM_101a, PTX ISA 86 | 9.7.9.25.5.2 |
| BF16/FP16 tensor copy | Standard alignment (not subbyte-restricted) | 9.7.9.25.5.1 |
| sm_103 tcgen05 f16 support | SM_103a, SM_103f confirmed | tcgen05.mma docs |

### Version notes
- **PTX ISA 8.8**: Introduced sm_103, sm_103a, sm_103f
- **PTX ISA 9.0**: Current stable version (December 2025)
- **PTX ISA 9.1**: Latest version with sm_110 additions

The mbarrier specification has been consistent since PTX ISA 7.0 (introduced with SM_80/Ampere). For exact section 9.7.9.25.5.1 "Restriction on Tensor Copy instructions" verbatim text regarding 2-byte type specifics, refer directly to the PTX ISA 9.0 PDF at `https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf`.
