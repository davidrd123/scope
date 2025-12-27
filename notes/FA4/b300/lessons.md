# B300 Lessons (What We’ve Actually Learned)

> **Purpose:** The practical companion to `philosophy.md`: not what we *intend* to do, but what we *now know* (including null results).
> **Created:** 2025-12-27

This is a **curated** index of “bagged patterns” with pointers to the underlying evidence.

- **Not a logbook:** raw trials live in [`experiments.md`](experiments.md).
- **Not a truth doc:** current commands / best-known config live in [`session-state.md`](session-state.md).
- **Not a roadmap:** priorities live in [`optimization-vision.md`](optimization-vision.md) + [`development-plan.md`](development-plan.md).

## How to Use This File

When something “clicks”:
1. Capture the measured result in [`experiments.md`](experiments.md).
2. Add a **1–3 bullet distillation** here with links to the experiment card / issue / codepath.

Prefer “why + where” over “what”:
- **What we learned** (one sentence)
- **Why it was true** (one sentence)
- **Where to look** (links)

## Bagged Patterns

### Measurement Hygiene

- **Always re-measure baseline before attributing wins.** Noise sources include warmup, GPU contention, and shape drift; follow [`investigation-runbook.md`](investigation-runbook.md).
- **Canonical comparisons are `320x576` with quality-preserving settings.** (Reference: [`README.md`](README.md).)
- **One-change rule prevents self-deception.** If two knobs changed, we don’t know which mattered; log it as “confounded” and re-run. (Reference: [`experiments.md`](experiments.md).)

### Quality Is a First-Class Gate

- **FP8 is currently off-limits for “real” output on B300.** Keep FP8 data as perf-only breadcrumbs; BF16 (`--quantization none`) is canonical. (Reference: [`session-state.md`](session-state.md).)
- **Perf wins that introduce visible glitches are debug-only.** Document them, don’t promote them to “truth config”. (Example: KV-cache recompute cadence skipping; see [`session-state.md`](session-state.md).)

### Toolchain / Upstream Landmines (B300)

- **`torch.compile` stability depends on avoiding known “abort” codegen paths on SM103.** We explicitly disable flex-attention compilation on SM103 to avoid tcgen05 LLVM aborts; details in [`session-state.md`](session-state.md) and [`../../issues/triton-sm103-tcgen05-llvm-abort.md`](../../issues/triton-sm103-tcgen05-llvm-abort.md).
- **CUDAGraph trees are a correctness hazard unless step-markers and constraints are respected.** The spelling matters: `torch.compiler.cudagraph_mark_step_begin()`. (Reference: [`session-state.md`](session-state.md), [`../../issues/pytorch-cudagraph-output-overwritten.md`](../../issues/pytorch-cudagraph-output-overwritten.md).)
- **TorchAO FP8 + compile can fail due to missing `aten.as_strided.default` on the quantization Float8Tensor.** There’s a PerTensor-only workaround and a paste-ready upstream issue. (Reference: [`../../issues/torchao-as-strided-dispatch.md`](../../issues/torchao-as-strided-dispatch.md).)

### “Two Worlds” on B300: Stack Matters

- **Repo-default stack and the cu130 stack behave like different systems.** Treat comparisons across stacks as invalid unless you explicitly call it out and re-establish baseline. (Reference: [`session-state.md`](session-state.md), [`setup-guide.md`](setup-guide.md).)

### Decode: The Biggest “Non-Attention” Lever We Found

- **Decode performance can collapse due to silent slow-path Conv3d / resample layout behavior.** The durable fix pattern is to keep inputs in layouts that stay on fast kernels rather than accepting implicit copies. (Reference: [`vae-decode-architecture.md`](vae-decode-architecture.md), [`session-state.md`](session-state.md).)
- **Patch-embed style conv3d→conv2d fastpaths can eliminate copy/fill storms.** This was a major real win and a general pattern: look for “Conv3d where kernel_size[0]==1” opportunities. (Reference: [`experiments.md`](experiments.md), decode map in [`vae-decode-architecture.md`](vae-decode-architecture.md).)

### Denoise: Where the Remaining Time Lives

- **FA4 score_mod helps the KV-bias slice, but “other_in_self” remains the mountain.** The remaining work is mostly QKV projection + plain attention + glue ops; don’t over-index on KV-bias once it’s no longer dominant. (Reference: [`session-state.md`](session-state.md), [`other-in-self-breakdown.md`](other-in-self-breakdown.md).)
- **Triton KV-bias backend on SM103 can be catastrophically slow due to forced fallback.** Don’t treat “it runs” as “it’s a contender”. (Reference: [`session-state.md`](session-state.md).)

### Level 6 Groundwork (Primitives + Contracts)

- **The hard part is rarely “write a kernel”; it’s “meet the contracts.”** Layout/stride/alignment constraints determine whether you stay on fast paths; document contracts before rewriting hotspots. (Reference: [`layout-contracts.md`](layout-contracts.md).)
- **Vendored kernels are the most truthful teachers.** When in doubt, learn from real implementations in-repo rather than blog summaries. (Reference: [`vendored-kernel-resources.md`](vendored-kernel-resources.md).)

## Open Questions (Keep This Short)

- What is the measured breakdown of `other_in_self` (QKV vs attention vs glue) on the current truth config? (Fill [`other-in-self-breakdown.md`](other-in-self-breakdown.md) from real stack-attributed profiles.)
- Which denoise improvements are both (a) measurable and (b) low behavioral risk (quality-preserving)?

