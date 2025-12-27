# External doc packets for Level 6 work

Format: link + suggested section headings to read + key constraints / gotchas to extract.

This file is meant to be pasted into internal notes, then refined with the exact versions you vendored.

---

## Packet 1: FlashAttention (FA2 and FA4/CuTe) layout and API requirements

Primary links:
```text
FlashAttention repo (README and usage): https://github.com/Dao-AILab/flash-attention
FA v2.1.0 release note (causal mask alignment detail): https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0
```

Suggested sections to extract (exact headings may vary by version):
- API overview: flash_attn_func, flash_attn_qkvpacked_func
- Varlen API: cu_seqlens and max_seqlen parameters
- Supported dtypes and head dims
- Layout and contiguity constraints
- When internal repacking happens (and how to detect it)

Key constraints / gotchas to capture into layout-contracts.md:
- Exact required shapes for Q/K/V and packed QKV (padded and varlen)
- Required contiguity (especially head-dim contiguous) and alignment
- Head dim allowed values and multiple-of requirements
- Causal masking convention (bottom-right alignment changed at least once historically)
- What forces a slower path (non-contiguous, unsupported head dim, wrong dtype)

---

## Packet 2: CUTLASS/CuTe extension points (custom epilogues, TMA pipelines, WGMMA building blocks)

Primary links:
```text
CUTLASS pipeline overview (includes "pipelines with TMA"): https://docs.nvidia.com/cutlass/latest/pipeline.html
CUTLASS docs index: https://docs.nvidia.com/cutlass/latest/index.html
```

Suggested sections to extract:
- Pipeline Overview
- Pipelines with TMA
- Epilogue design (visitor tree, fusion points)
- CuTe layout and tiling primitives (how to express layouts)
- SM90+ and newer instruction families as referenced by CUTLASS

Key constraints / gotchas:
- What alignment and tile-shape constraints TMA imposes on shared memory tiles
- How pipeline stage count impacts register pressure and occupancy
- How to add a fused epilogue without breaking the main MMA pipeline
- How to structure warp specialization so load does not starve compute
- What instrumentation helps you detect fallbacks (kernel name patterns, compile-time guards)

---

## Packet 3: NVIDIA PTX ISA references for low-level primitives (cp.async.bulk*, mbarrier, WGMMA, tcgen05, TMEM)

Primary link:
```text
PTX ISA reference (pick the exact version matching your toolkit): https://docs.nvidia.com/cuda/archive/10.2/pdf/ptx_isa_6.5.pdf
```

Suggested sections to extract:
- Asynchronous copy and bulk copy instruction families (for TMA-related ops)
- Barrier instructions (mbarrier semantics, arrive/wait)
- Warp-group MMA instruction semantics and constraints
- Any tensor-memory specific instructions (if applicable to your target arch)

Key constraints / gotchas:
- Minimum `.target` / SM required for each instruction family
- Operand alignment and address space requirements
- Memory ordering guarantees (what the barrier actually guarantees)
- How to recognize these ops in SASS vs. PTX

Note: the PTX ISA evolves quickly. The archive link above is an example reference; in practice you should use the PTX ISA doc shipped with or matching your CUDA toolkit version.

---

## Packet 4: cuDNN frontend / backend API + Conv3d planning and graph capture

Primary links:
```text
cuDNN backend CNN library docs: https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html
cuDNN docs index: https://docs.nvidia.com/deeplearning/cudnn/
```

Suggested sections to extract:
- Backend descriptors needed for convolution graphs
- Execution plans and engine configs
- Workspace planning
- CUDA Graph capture constraints and recommendations
- Release notes: anything mentioning Conv3d, grouped conv, BF16/FP16/FP8, and Blackwell/SM10x

Key constraints / gotchas:
- Which tensors and strides are supported for Conv3d, and what triggers fallback kernels
- How to cache plans across runs (stable shapes)
- Which knobs matter for Conv3d throughput (algo selection, workspace)
- Graph capture: what needs to be static and preallocated

---

## Minimal integration plan

1) Fill layout-contracts.md using actual tensor dumps
2) Fill other-in-self-breakdown.md using a stack-attributed profile
3) Only then, decide if the first fusion kernel is:
   - a pure pack/layout kernel (lower risk)
   - or a projection+epilogue fusion (higher risk)

