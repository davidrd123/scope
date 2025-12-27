# NVIDIA mbarrier and cp.async.bulk.tensor specifications for SM103/Blackwell

The official PTX ISA documentation specifies that **mbarrier objects require 8-byte alignment** (not 64-byte), and **cp.async.bulk.tensor on sm_103/sm_103f inherits capabilities from sm_100** with architecture-specific features requiring the "a" suffix targets.

## mbarrier size and alignment: 8 bytes with 8-byte alignment

The PTX ISA (Section 9.7.13.15.1 in version 9.0/9.1) titled "Size and alignment of mbarrier object" specifies the fundamental mbarrier properties. Multiple authoritative NVIDIA-adjacent sources provide verbatim documentation:

**From MLIR NVVM Dialect documentation** (https://mlir.llvm.org/docs/Dialects/NVVMDialect/):
> "An mbarrier is a barrier created in shared memory that supports synchronizing any subset of threads within a CTA. **An mbarrier object is an opaque object in shared memory with .b64 type and an alignment of 8-bytes.**"

**From LLVM NVPTX Usage Guide** (https://llvm.org/docs/NVPTXUsage.html):
> "An mbarrier object is an opaque object in shared memory with an alignment of 8-bytes. It keeps track of:
> - Current phase of the mbarrier object
> - Count of pending arrivals for the current phase of the mbarrier object
> - Count of expected arrivals for the next phase of the mbarrier object
> - Count of pending asynchronous memory operations (or transactions) tracked by the current phase of the mbarrier object. This is also referred to as tx-count."

The **64-byte alignment confusion** likely stems from TMA data buffer requirements, not the mbarrier itself. The libcudacxx documentation at nvidia.github.io/cccl specifies data alignment separately:
> "src, dest are **16-byte aligned** and size is a multiple of 16" for TMA operations.

No architecture-specific differences between SM90 (Hopper) and SM100/SM103 (Blackwell) exist for mbarrier object alignment—all use the same **8-byte opaque .b64 type with 8-byte alignment**.

## cp.async.bulk.tensor target restrictions for sm_103/sm_103f

The PTX ISA 9.1 Release Notes document sm_103 support:
> "Adds support for sm_103 target architecture."
> "Adds support for target sm_103a that supports architecture-specific features."
> "Introduces family-specific target architectures that are represented with 'f' suffix."
> "PTX for family-specific targets is compatible with all subsequent targets in same family."

**sm_103 is not explicitly listed** in cp.async.bulk.tensor target restrictions—the instruction uses sm_90, sm_90a, sm_100, sm_100a, and sm_101a targets. As sm_103 belongs to the SM100 family (Blackwell), it inherits capabilities through forward compatibility.

**Verbatim target requirements from libcudacxx PTX documentation** (https://wmaxey.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk_tensor.html):

| Operation Pattern | Target Requirement |
|---|---|
| `.shared::cluster.global.tile.mbarrier::complete_tx::bytes` | PTX ISA 80, **SM_90** |
| `.shared::cta.global.tile.mbarrier::complete_tx::bytes` | PTX ISA 86, **SM_90** |
| `.shared::cta.global.tile.mbarrier...cta_group::1/2` | PTX ISA 86, **SM_100a, SM_101a** |
| `.global.shared::cta.tile.bulk_group` (store) | PTX ISA 80, **SM_90** |
| `.multicast::cluster` variants | PTX ISA 80, **SM_90a, SM_100a, SM_101a** |
| `.tile::gather4` (basic) | PTX ISA 86, **SM_100** |
| `.tile::gather4.cta_group` | PTX ISA 86, **SM_100a, SM_101a** |
| `.tile::scatter4.bulk_group` (store) | PTX ISA 86, **SM_100a, SM_101a** |

The changelog notes an important correction:
> "In earlier versions, cp_async_bulk_tensor_multicast was enabled for SM_90. This has been changed to SM_90a."

## Key implications for SM103 Blackwell kernel development

**The datatype (BF16/FP16) is NOT specified at the PTX instruction level**—element types are encoded in the tensorMap descriptor created via `cuTensorMapEncodeTiled`, not in the cp.async.bulk.tensor instruction itself. Therefore, no BF16/FP16-specific PTX restrictions exist.

For sm_103 targeting:
- **Basic operations work** (inheriting from sm_90 and sm_100 capabilities)
- **cta_group modifier requires sm_100a or sm_101a targets**, not sm_103 directly—architecture-accelerated features need the "a" suffix
- Multicast operations require architecture-specific "a" suffix targets

The CUTLASS documentation confirms this pattern:
> "CUDA 12.0 introduced the concept of 'architecture-accelerated features' whose PTX does not have forward compatibility guarantees. Several Hopper and Blackwell PTX instructions fall under this category of architecture-accelerated features, and thus require a sm_90a or sm100a target architecture (note the 'a' appended)."

## Summary of verbatim specifications

**mbarrier object**: `.b64` type (64-bit = 8 bytes), **8-byte alignment**, opaque structure in shared memory. The specification is consistent across SM80, SM90, and SM100/SM103 architectures.

**cp.async.bulk.tensor for sm_103**: Inherits from sm_100 family. Basic unicast tile operations require SM_90. Store operations require SM_90. Gather4/scatter4 require SM_100 or SM_100a. CTA-group modifiers and multicast require architecture-specific "a" suffix targets (SM_90a, SM_100a, SM_101a)—sm_103a may be needed for these features.
