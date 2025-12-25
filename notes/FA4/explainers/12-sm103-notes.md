# SM103 Notes (B300): What Carries Over from SM100, What Doesn’t

> **Explainer #12** — A practical bridge between the SM100-centric FA4/CuTe docs and B300 reality (SM103): what concepts transfer, what breaks, and what to watch for.
> **Updated:** 2025-12-25

---

## Overview

Most of the deep FA4/CuTe code in our repo is written and documented around **SM100** (Blackwell B200).
But B300 is **SM103**, and we’ve seen that “Blackwell-ish” does not mean “identical behavior”.

This explainer is a living checklist for future work:

- Which SM100 concepts apply to SM103 (usually: algorithm + structure)
- Which *implementation* details are SM100-specific (often: ISA paths, kernel availability)
- How this maps to our backend selection and fallback strategy in the realtime pipeline

---

## What carries over (concepts)

These explainers remain “true” on SM103 as mental models:

- **Online softmax math** (Explainer #7): the rescale logic is algorithmic, not SM100-specific.
- **Tiling + pipelining** (Explainer #6): scheduling and staging ideas still apply.
- **“Where would fusion live?”** (Explainer #5): the right insertion points (after load, before MMA) are conceptual.

In other words: the shape of the kernel is similar even if the exact primitives differ.

---

## What is SM100-specific (implementation)

The SM100 explainer (#2) includes multiple things that may not map 1:1 to SM103:

- **tcgen05 details** (instruction descriptor encodings, “elect-one” patterns)
- **TMEM behavior** (exact capacity limits, access patterns)
- **Cluster / CTA-group assumptions** (some heuristics may differ)

Even if SM103 supports the same broad features, small differences can change which kernels are compiled/selected.

---

## What we’ve observed in this repo

### 1) Backend “availability” and fallbacks matter more on SM103

Our production code uses a fallback ladder roughly like:

```
fa4 (score_mod) → flash (safe) → triton → flex_attention
```

On SM103, “it compiles” and “it runs fast” are separate questions. When a backend silently falls back (e.g. scalar-ish code paths), end-to-end perf can collapse.

### 2) KV-bias share is backend-dependent

Profiling has shown that the “KV-bias slice” of self-attention can look materially different depending on whether the bias is expressed as:

- an in-kernel `score_mod` (FA4), or
- a heavier-weight path (Flash segment-combine-ish)

This is why we document backend-dependent profiling expectations in the B300 notes.

### 3) Runtime stack can dominate

For B300, we’ve seen cases where overall FPS is dominated by the CUDA runtime / cuDNN / decode behavior rather than attention kernels themselves. Kernel work alone can’t fix the wrong stack.

---

## How to extend this doc (when you learn something new)

When something “mysteriously differs” on SM103, try to classify it as:

1. **Dispatch difference** (different kernel picked / feature gated)
2. **Compiler/codegen difference** (kernel exists but compiles poorly)
3. **Runtime stack difference** (cuDNN/driver behavior dominates)
4. **Workload difference** (shapes, strides, caching, quantization)

And record:

- a minimal reproduction command
- the exact GPU name + torch/cuda versions
- the observed backend (which path actually executed)

---

## References

- Explainer #2 (SM100): `notes/FA4/explainers/02-blackwell-path.md`
- B300 investigation runbook: `notes/FA4/b300/investigation-runbook.md`
- B300 session truth: `notes/FA4/b300/session-state.md`
- Kernel optimization story: `notes/FA4/docs/kernel-optimization-guide.md`

