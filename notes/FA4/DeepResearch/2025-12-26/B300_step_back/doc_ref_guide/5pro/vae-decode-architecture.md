# VAE decode architecture map (B300)

Goal: make the VAE decode path legible enough to optimize deliberately.

This doc should answer:
- What the decode graph is (ops and shapes)
- Where the time goes
- Which Conv3d can be rewritten as Conv2d (time-kernel = 1) or otherwise simplified
- What cuDNN frontend planning opportunities exist

---

## A. High-level decode flow

Write this as a linear pipeline, even if the real graph has branches.

Example format:

latent [B, C, T, H, W]
  -> op_0: conv3d (kT,kH,kW) stride (...) groups (...)
  -> op_1: groupnorm
  -> op_2: silu
  -> op_3: upsample (mode=...)
  -> ...
  -> output [B, 3, T, H_out, W_out]

---

## B. Op table (shapes, params, and "Conv3d time-kernel=1?" markers)

Fill one row per op.

| # | op | input shape | output shape | dtype | key params | time-kernel = 1? | notes |
|---:|----|-------------|--------------|------|------------|------------------|-------|
| 0 | conv3d |  |  |  | k=( , , ) s=( , , ) g= |  |  |
| 1 | groupnorm |  |  |  | groups= | n/a |  |
| 2 | silu |  |  |  |  | n/a |  |
| 3 | upsample |  |  |  | mode= | n/a |  |

---

## C. Profiling annotations

Attach:
- a torch.profiler table for decode only
- an Nsight Systems trace snippet if possible (kernel names + occupancy)

Record:
- which conv kernels are selected (implicit GEMM vs. FFT vs. Winograd, etc)
- workspace sizes
- whether graph capture is used

---

## D. Conv3d -> Conv2d rewrite candidates

For each Conv3d where kT == 1 and strideT == 1:

- Candidate rewrite: treat each time slice independently
  - Option A: reshape [B, C, T, H, W] -> [B*T, C, H, W], run Conv2d, reshape back
  - Option B: fuse into a single grouped Conv2d if weights share structure (rare)
- Risks:
  - memory layout changes and extra reshapes
  - different cuDNN kernel selection
  - numerics if using different algorithms

Write one short entry per candidate:

| op # | why candidate | expected win | risks | required layout contract changes |
|------|--------------|--------------|-------|----------------------------------|
|  |  |  |  |  |

---

## E. cuDNN frontend planning opportunities

If decode is stable-shape, cudnn frontend can let you:
- prebuild execution plans
- choose engine configs explicitly
- reduce runtime autotune overhead
- integrate with CUDA Graph capture

Write:
- which ops are stable-shape across runs
- which ops change with resolution or batch
- whether plan caching would help

---

## F. "Do not do" list

To avoid perf mirages:
- do not compare runs without warmup
- do not compare graph-capture vs. eager without noting it
- do not mix dtype modes

