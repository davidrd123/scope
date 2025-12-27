# Blackwell primitives cheat sheet (B300, SM10x)

Goal: a tight, non-cargo-cult map from concepts to what you actually see in CuTe/CUTLASS and low-level tooling.

This is intentionally short. Link out to real references.

---

## A. Core primitives (what, why, where)

### TMA (Tensor Memory Accelerator)
- What: hardware unit that can move tensor tiles between global memory and shared memory asynchronously
- Why: overlaps memory movement with compute, avoids having threads execute per-element loads
- Where you see it:
  - CuTe/CUTLASS "pipelines with TMA"
  - kernels that have explicit load-warps and compute-warps (warp specialization)
  - Nsight Compute: large async copy regions, fewer LDGSTS instructions

Reference:
- CUTLASS pipeline overview (TMA section): https://docs.nvidia.com/cutlass/latest/pipeline.html

### mbarrier (memory barrier objects)
- What: barrier objects used for producer-consumer synchronization in shared memory pipelines
- Why: lets you coordinate async copies and compute without full CTA barriers
- Where you see it:
  - "arrive" / "wait" style calls around pipeline stages
  - CuTe pipeline code

Reference:
- PTX ISA reference (use the exact version matching your CUDA toolkit): https://docs.nvidia.com/cuda/archive/10.2/pdf/ptx_isa_6.5.pdf

### Warp specialization
- What: split a CTA into warps (or warp groups) with different roles
- Why: keeps copy scheduling predictable and compute dense, reduces contention
- Where you see it:
  - "load warps" issuing TMA ops
  - "mma warps" issuing tensor core instructions
  - separate warps for softmax / epilogue work in attention kernels

### WGMMA (warp-group MMA)
- What: matrix multiply accumulate issued at warp-group granularity (not individual warps)
- Why: higher throughput, better mapping to tensor cores on newer GPUs
- Where you see it:
  - CUTLASS/CuTe GEMM kernels for SM90+ and beyond
  - kernels that treat 4 warps (128 threads) as one scheduling unit

### TMEM / tcgen05 (tensor memory concepts and instruction families)
- What: GPU generation specific storage and instruction patterns for tensor core pipelines
- Why: reduce register and shared memory pressure in large tiles
- Where you see it:
  - advanced kernels that push tile sizes and overlap depth
  - instruction names in SASS or PTX for newer architectures

Note: verify exact mnemonics and constraints in the PTX ISA version for your toolkit.

---

## B. Practical "how to use this" mapping

When you see a performance hotspot and ask "should I do Level 6 kernel work?", look for:

1) The op is mostly format tax (copies, transposes, packing)
2) The op is memory bound and predictable (tile loads, stores)
3) There is an obvious pipeline opportunity (load next tile while computing current tile)
4) Shapes are stable enough to specialize

If any of these is false, start with layout unification and orchestration before custom kernels.

---

## C. Common pitfalls

- TMA and deep pipelining can increase register pressure, which can lower occupancy and hurt performance.
- Warp specialization can backfire if the load side starves the compute side (or vice versa).
- "Fusing" can make a previously perfect GEMM slower if you pull it out of cuBLAS and lose autotuned kernels.
- Always log which path ran, so you can detect silent fallbacks.

