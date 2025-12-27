# Primary-source layout and constraint notes (SM103 focus)

This file collects spec-level, primary-source constraints for:
- TMA bulk and bulk-tensor async copies (cp.async.bulk.* and cp.async.bulk.tensor.*)
- mbarrier phase parity and tx-count tracking
- tcgen05 (TMEM) alloc, mma, commit, wait
- cuTensorMapEncodeTiled (tensormap descriptor encoding)

Notes:
- All URLs are written in inline code to keep them easy to copy.
- Excerpts are kept short. For full context, jump to the cited heading in the linked doc.

---

## Parallel Thread Execution ISA (PTX ISA 9.1) (web doc, accessed 2025-12-27)

Link
- `https://docs.nvidia.com/cuda/parallel-thread-execution/`

### Relevant sections
- **9.7.9.27.1.2** Data Movement and Conversion Instructions: `cp.async.bulk.tensor` (TMA bulk-tensor load and store forms, per-target restrictions)  
- **9.7.13.15.4-9.7.13.15.6** `mbarrier` phase and tx-count tracking semantics (what causes a phase flip)  
- **9.7.13.15.11** `mbarrier.expect_tx` syntax and address-space constraints  
- **9.7.13.15.16** `mbarrier.test_wait` / `mbarrier.try_wait` including `.parity` modifier  
- **9.7.16.6.2.1.2** `tcgen05.wait` completion mechanism for `tcgen05.ld` and `tcgen05.st`  
- **9.7.16.7.1** `tcgen05.alloc` / `tcgen05.dealloc` / `tcgen05.relinquish_alloc_permit`  
- **9.7.16.10** `tcgen05.mma` forms and operand roles (`[d-tmem]`, `a-desc`, `b-desc`, etc.)  
- **(tcgen05 commit)** `tcgen05.commit.mbarrier::arrive::*` and what it tracks  

### Short excerpts

#### A. TMA bulk-tensor load and store instruction spelling (PTX)
From **9.7.9.27.1.2 `cp.async.bulk.tensor`**:

- The PTX syntax explicitly includes a **shared-to-global** form: the doc lists a `// shared::cta -> global` syntax block for `cp.async.bulk.tensor`.  
- Example spellings in the same section include:
  - `cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];`  
  - `cp.async.bulk.tensor.1d.global.shared::cta.bulk_group  [tensorMap3, {tc0}], [sMem3];` (shared to global, bulk-group completion)  

#### B. A concrete SM103 alignment/stride restriction (bulk-tensor)
From the same **`cp.async.bulk.tensor`** section, there is an explicit target restriction block for `sm_103a` with direction `.global.shared::cta` and type `.b6p2x16`. It includes these hard constraints:
- Box-Size[0] must be exactly 48B or 96B.
- Global memory base address must be 16B aligned.
- Tensor stride in every dimension must be 16B aligned.
- First coordinate in the `tensorCoords` vector must be a multiple of 64.
- Tensor-Size[0] must be a multiple of 48B.
- Supported swizzle modes: None.  

#### C. mbarrier phase parity (how to wait on the right phase)
From **9.7.13.15 `mbarrier`** and the examples for `test_wait`:
- PTX provides an explicit **phase-parity wait form**: `mbarrier.test_wait.parity...` is shown in an example where `parArg = i & 1` and the wait uses that parity argument.  

#### D. mbarrier tx-count (what gates phase completion)
From **9.7.13.15.5-9.7.13.15.6**:
- The doc defines a tx-count used to track outstanding async transactions, and describes `expect-tx` as incrementing tx-count and `complete-tx` as decrementing it.  
- Current phase completion requires both pending arrivals reaching zero and tx-count reaching zero.  

#### E. `mbarrier.expect_tx` exact spelling and operand constraints
From **9.7.13.15.11 `mbarrier.expect_tx`**:
- Syntax is documented as:
  - `mbarrier.expect_tx{.sem.scope}{.space}.b64 [addr], txCount;`  
- It also states that if the address does not fall in `.shared::cta` or `.shared::cluster`, behavior is undefined.  

#### F. tcgen05 wait spelling and required qualifiers
From **9.7.16.6.2.1.2 (tcgen05.wait based completion)** and the later `tcgen05.wait` description:
- The doc states that `tcgen05.wait::ld` and `tcgen05.wait::st` are used to track completion of `tcgen05.ld` and `tcgen05.st`.  
- It also states `.sync` and `.aligned` are mandatory qualifiers for `tcgen05.wait_operation` (warp-level agreement requirements).  

#### G. tcgen05 alloc/dealloc lifecycle constraints
From **9.7.16.7.1 (Tensor Memory Allocation and Management)**:
- The doc states that all TMEM allocated with `tcgen05.alloc` must be explicitly deallocated with `tcgen05.dealloc` before kernel exit.  
- Example instruction spellings include:
  - `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [sMemAddr1], 32;`
  - `tcgen05.dealloc.cta_group::1.sync.aligned.b32  taddr, 32;`
  - `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;`  

#### H. tcgen05 commit spelling and what it tracks
From the tcgen05 synchronization patterns and commit description:
- Example shows `tcgen05.commit.mbarrier::arrive::one [mbar]` paired with `mbarrier.try_wait...` on the same barrier.  
- Later, the doc states `tcgen05.commit` makes the given mbarrier track completion of prior asynchronous tcgen05 ops (for the chosen `.cta_group`).  
- The doc also states: all `tcgen05` instructions in a kernel must specify the same `.cta_group`.  

#### I. tcgen05 mma spellings and operand roles
From the tcgen05 MMA forms list:
- One canonical dense form is shown as:
  - `tcgen05.mma.cta_group.kind.ashift{.collector_usage} [d-tmem], [a-tmem], b-desc, idesc, { disable-output-lane }, enable-input-d {, scale-input-d};`  
- The doc enumerates `.cta_group` values `{ .cta_group::1, .cta_group::2 }` and `.kind` includes at least `{ .kind::f16, .kind::tf32, .kind::f8f6f4 }` in that block.  

### Key constraints (inferred, anchored to sections above)
- **TMA bulk-tensor store exists in PTX** (`cp.async.bulk.tensor ... shared::cta -> global`), and examples show `.global.shared::cta.bulk_group` for the completion mechanism, which is relevant if KV-write is done via TMA rather than STG.  
- **SM103-specific hard alignment/stride constraints exist** for certain element types (example: `.b6p2x16` restrictions). Do not assume Hopper-era swizzles and coordinate rules carry over unchanged.  
- **mbarrier phase completion is gated by both arrivals and tx-count**, so any pipeline that uses `expect_tx` must ensure the matching `complete_tx` accounting happens (or uses an instruction that triggers it implicitly).  
- **tcgen05 has two completion styles** in this doc set: `mbarrier` based tracking via `tcgen05.commit...mbarrier::arrive::*`, and `tcgen05.wait::*` for `tcgen05.ld/st`.  

---

## CUDA C++ Programming Guide (Release 13.1, Sep 2025) (sections on TMA, barriers, tensormaps)

Link
- `https://docs.nvidia.com/cuda/cuda-programming-guide/`
- (Direct section used below) `https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html`
- PDF: `https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf`

### Relevant sections
- **4.11 Asynchronous Data Copies** (bulk async copies, TMA usage conditions, tensormap encoding example, OOB behavior, barrier tx accounting)

### Short excerpts

#### A. When the CUDA C++ APIs actually use TMA
The guide states (same section):
- `cuda::memcpy_async` uses TMA only if src and dst addresses are 16-byte aligned and transfer size is a multiple of 16; otherwise it falls back to synchronous copies.  
- `cuda::device::memcpy_async_tx` and `cuda::ptx::cp_async_bulk` always use TMA, and violating the requirements results in undefined behavior.  

#### B. Barrier transaction accounting API names (header-backed names)
In the code path where you call low-level bulk async APIs yourself, the guide shows an explicit call:
- `cuda::device::barrier_expect_tx(cuda::device::barrier_native_handle(bar), sizeof(smem_data));`  
And the text notes that for the low-level forms you may need to explicitly call `cuda::ptx::mbarrier_expect_tx`.  

This directly addresses the “arrive + expect_tx bytes” naming question:
- Arrival: `bar.arrive()` on `cuda::barrier<cuda::thread_scope_block>` (gives an arrival token)  
- Tx-count increment: `cuda::device::barrier_expect_tx(...)` or `cuda::ptx::mbarrier_expect_tx(...)` depending on which layer you are using.  

#### C. OOB (out-of-bounds) fill behavior for tiled tensormap copies
In the tensormap construction example (same section), after showing `oobFill` set to `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`, the guide states:

> Any element that is outside of bounds will be set to zero by the TMA transfer.

This is the official semantics line that resolves the “NONE means zero fill or no fill?” ambiguity for that example.  

#### D. Alignment requirements table (bulk async)
The same section includes a table for 1D bulk-async alignment requirements:
- Global address: 16-byte aligned
- Shared address: 16-byte aligned
- Barrier address: 8-byte aligned
- Transfer size: multiple of 16 bytes  

### Key constraints (inferred, anchored to this section)
- If you route KV-write through **bulk async store** (shared to global) and want defined behavior, you must satisfy the same 16B alignment and 16B size multiple rules.  
- If you depend on barrier-based completion, make sure your code path either:
  - uses a helper (like `cuda::memcpy_async`) that performs tx accounting automatically, or
  - explicitly calls `barrier_expect_tx` / `mbarrier_expect_tx` to set the tx-count.  

---

## CUDA Driver API: Tensor Map Object Management (v13.1.0 doc set, last updated 2025-12-04)

Link
- `https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html`

### Relevant sections
- **6.30 Tensor Map Object Management**
  - `cuTensorMapEncodeTiled` signature and requirements table (alignment, stride units, bounds)  
  - Tensormap enum spellings: `CUtensorMapSwizzle`, `CUtensorMapInterleave`, `CUtensorMapL2promotion`, `CUtensorMapFloatOOBfill`  

### Short excerpts

#### A. `cuTensorMapEncodeTiled` signature (driver API)
The driver API lists the full signature of `cuTensorMapEncodeTiled`, including:
- `tensorDataType`, `tensorRank`, `globalAddress`
- `globalDim[]`, `globalStrides[]`, `boxDim[]`, `elementStrides[]`
- `interleave`, `swizzle`, `l2Promotion`, `oobFill`  

#### B. Hard constraints on address/strides/dims (what you should treat as “spec”)
The doc binds these parameters to explicit requirements, including:
- `globalAddress` aligned to 16 bytes, and points to at least 16 bytes of memory.
- `globalDim[i]` values in (0, 2^32].
- `globalStrides[i]` are multiples of 16 and in (0, 2^40].
- `boxDim[i]` values in (0, 256].
- `elementStrides[i]` values in [1, 8].  

#### C. Swizzle and OOB fill enum spellings
The same doc page includes the official enumerator spellings, including:
- Swizzle: `CU_TENSOR_MAP_SWIZZLE_NONE`, `_32B`, `_64B`, `_128B`
- Interleave: `CU_TENSOR_MAP_INTERLEAVE_NONE`, `_16B`
- OOB fill: `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`, `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA`  

### Key constraints (inferred, anchored to this section)
- For “tensormap stride units”, the requirements are written directly on `globalStrides[i]` (multiples of 16) which strongly suggests these are **byte strides** (not element strides). Treat `globalStrides` as byte-based and enforce 16B multiples unless you have a counterexample from NVIDIA docs.  
- `elementStrides[i]` has a very small allowed range ([1, 8]), so treat it as a compact per-dimension “element stride / stride factor” field rather than a byte count.  
- If you need OOB behavior beyond “zero fill” vs “special NaN constant”, note that the driver API’s `CUtensorMapFloatOOBfill` enum exposed here only lists those two choices in this doc set.  

---

## Delta vs the earlier “vibes doc” issues you flagged

### Version pinning and exact spellings
- PTX is pinned to **PTX ISA 9.1** and the instruction spellings above come from the exact numbered headings listed in that doc.  
- Driver API tensormap encoding is pinned to **CUDA Driver API v13.1.0** doc set.  
- CUDA C++ guide is pinned to **Release 13.1** (PDF and web).  

### OOB fill semantics correction
- The official CUDA C++ guide explicitly states OOB elements are set to zero in the shown `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` example.  

### Barrier API name correction
- The guide uses `cuda::device::barrier_expect_tx(...)` and points at `cuda::ptx::mbarrier_expect_tx` for explicit tx-count accounting.
