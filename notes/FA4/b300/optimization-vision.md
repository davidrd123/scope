# B300 Optimization Vision

> **Purpose:** Strategic roadmap for B300 performance optimization  
> **Created:** 2025-12-25 | **Updated:** 2025-12-26  
> **Current (quality-preserving):** ~19–20 FPS (BF16) / ~22–23 FPS (BF16 + `--compile`) @ `320x576` | **Target:** 24+ FPS (real-time)

**Quality gate:** On B300/SM103, FP8 quantization currently produces **garbage output** (gray/noise). Treat FP8 as perf-only debugging, not a shippable win. Canonical baseline is `--quantization none` (BF16).

---

## Executive Summary

The B300 journey has been mostly about **getting onto the right fast paths** (stack + backends) and then eliminating **hidden slow-path ops** (copy/fill storms), not “inventing a faster attention kernel”.

We now have a quality-preserving BF16 path on B300 that’s within ~5–6% of the 24 FPS target (benchmark harness):
- repo-default stack: ~8.8 FPS (decode/cuDNN dominated)
- cu130 stack + FA4 KV-bias + glue elimination: ~19–20 FPS (BF16)
- + `--compile`: ~22–23 FPS (BF16)

Near-term strategy: validate the best config **end-to-end in Daydream**, then use stack-attributed op profiling to find the next 1–2 “patch-embed style” wins (usually: dtype/layout churn or Conv3d slow paths).

### Progress Log (benchmark harness, `320x576`, steps=4, bias=0.3)

| Date | Milestone | FPS | Key Change |
|------|-----------|-----|------------|
| 2025-12-24 | Baseline | ~8.8 | repo-default runtime stack (decode slow) |
| 2025-12-25 | cu130 baseline | ~14.9 | CUDA 13 + cuDNN 9.13 (decode fast) |
| 2025-12-25 | FA4 KV-bias | ~16.7 | FA4/CuTe `score_mod` (avoid segment-combine overhead) |
| 2025-12-26 | Patch-embed fastpath | ~19.7 | Conv3d patch embed → per-frame Conv2d (kills copy/fill storm) |
| 2025-12-26 | + `--compile` (BF16) | ~22.8 | stable regional compilation around the model |

---

## 1. Where We Are (Current Truth)

### Two “Worlds” on B300

| Regime | Typical FPS @ `320x576` | What dominates | What to do |
|---|---:|---|---|
| Repo-default stack | ~8.8 | decode/cuDNN slow paths | switch to the cu130 stack first |
| cu130 stack | ~19–23 (BF16) | transformer + remaining decode conv3d | optimize glue/projections/decode; expand safe compile |

### Known-Good BF16 Configs (quality-preserving)

Numbers below are from `scripts/profile_krea_pipeline_blocks.py` in the cu130 env; use [`session-state.md`](session-state.md) for the exact command.

| Config | FPS | Notes |
|---|---:|---|
| `SCOPE_KV_BIAS_BACKEND=fa4` | ~19.7 | best no-compile baseline |
| `SCOPE_KV_BIAS_BACKEND=fa4` + `--compile` | ~22.8 | best measured BF16; longer warmup |
| `SCOPE_KV_BIAS_BACKEND=flash` | ~17.2 | stable fallback |
| `SCOPE_KV_BIAS_BACKEND=flash` + `--compile` | ~21.4 | solid fallback + compile |

### What’s Actually Left

- **Attention is still a large chunk**, but the KV-bias microkernel is now a minority of `self_attn` time (FA4 `score_mod` helped).
- The remaining “big rocks” are typically:
  - **QKV/projection GEMMs + layout/dtype glue** (`aten::copy_`, `aten::_to_copy`, small elementwise storms)
  - **WanVAE decode conv3d** (still substantial even on the cu130 stack)

This is why the next wins tend to come from eliminating slow paths (patch-embed, dtype roundtrips, Conv3d→Conv2d patterns) rather than swapping attention kernels again.

---

## 2. High-ROI Options (Ordered)

### Option A: “Fast Paths First” (Operational Guardrails)

**Goal:** Stop wasting cycles on “it got slow because we benchmarked the other backend / wrong stack”.

Work (doc + code hygiene):
- Always log a **startup banner** of chosen backends (KV-bias + non-bias attention + fallbacks).
- Keep the **cu130 env** and required knobs (`TRITON_PTXAS_PATH`, `DISABLE_FLEX_ATTENTION_COMPILE=1`) as the default path for B300.

Status: ongoing hardening; see [`development-plan.md`](development-plan.md).

### Option B: Repeat the Patch-Embed Playbook (Glue / Slow-Path Elimination)

**Proven pattern:** stack-attributed op profiling → find a single pathological source of `copy_`/`fill_` → remove it with a targeted rewrite.

Next targets to look for:
- More **Conv3d with temporal kernel size 1** that can be rewritten as per-frame Conv2d (especially inside VAE decode).
- Remaining dtype/layout churn in the transformer and decode.

Tooling:
- `scripts/profile_krea_pipeline_ops.py --with-stack --summary` to get “top op stacks” for `aten::copy_`, `aten::_to_copy`, `aten::fill_`.

### Option C: cuDNN SDPA Backend A/B (Cross-Attn Focus)

cuDNN 9.13+ includes Blackwell-focused attention kernels. A cheap experiment is to see whether cross-attn or non-bias attention can land on cuDNN SDPA.

Experiment-card idea:
- Enable cuDNN SDPA globally for a run (and record whether it was actually selected), then benchmark.
- If it’s not selected, log why (PyTorch has `can_use_cudnn_attention(..., debug=True)` helpers).

### Option D: torch.compile Expansion + Mode Experiments (BF16 only)

Default `--compile` is already a real win on B300 (BF16). Next steps are:
- Measure **warmup time** explicitly (it matters for UX).
- Try `SCOPE_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` as an experiment-only knob (it can improve kernel selection without CUDAGraphs, but may still hard-abort on SM103 due to tcgen05/LLVM issues).
- Keep `reduce-overhead` as known-bad on SM103 unless/until the “output overwritten” class is resolved.

### Option E: Cross-Resolution Scaling (One Card)

Run one higher-res experiment (e.g. `480x864` or `640x1152`) to learn whether:
- attention becomes dominant again (sequence length explosion), or
- decode/conv3d remains a major limiter.

This decides whether the ladder should fork into “attention R&D” vs “decode + glue elimination”.

### Option F: Workflow Gates (Regression + Quality)

Docs/ops improvements that prevent future churn:
- A single-command **regression gate** script that runs the canonical benchmark and checks it against a stored baseline.
- A simple **quality snapshot** routine (fixed prompt → save N frames) so “perf wins” can’t silently degrade output.

---

## 3. Recommended Path (Near-Term)

1) **Validate the best BF16 config in Daydream (end-to-end)**
   - Use `SCOPE_KV_BIAS_BACKEND=fa4`, `--quantization none`, and optionally `--compile`.
   - Record (a) steady-state FPS and (b) warmup time.

2) **Run stack-attributed op profiling on the best config**
   - Goal: pick the next 1–2 concrete call stacks to optimize (copy/to/fill hotspots).

3) **Hunt Conv3d→Conv2d opportunities in VAE decode**
   - Any Conv3d with time kernel 1 is a candidate (same pattern as patch embed).

4) **A/B cuDNN SDPA**
   - If cuDNN SDPA is usable for cross-attn/non-bias attention, it can be a low-effort win.

5) **One cross-resolution scaling card**
   - Decide if we need an “attention-first” branch for high-res.

---

## 4. Success Metrics (BF16 / Quality-Preserving)

| Milestone | FPS | What it demonstrates | Status |
|---|---:|---|---|
| Repo-default baseline | ~8.8 | “slow stack” reference | ✅ |
| cu130 baseline (flash) | ~14.9 | decode fast paths unlocked | ✅ |
| FA4 KV-bias | ~16.7 | KV-bias is no longer a tax | ✅ |
| Patch-embed fastpath | ~19.7 | “glue elimination” works | ✅ |
| + `--compile` | ~22.8 | compile is viable on SM103 | ✅ |
| Real-time target | 24+ | sustained real-time @ `320x576` | 🎯 |

---

## 5. Key References

| Purpose | File |
|---|---|
| What to run today | [`session-state.md`](session-state.md) |
| How to measure | [`investigation-runbook.md`](investigation-runbook.md) |
| One-change experiment cards | [`experiments.md`](experiments.md) |
| This strategy doc | [`optimization-vision.md`](optimization-vision.md) |
| Dev workstreams | [`development-plan.md`](development-plan.md) |
| Op profiler | `scripts/profile_krea_pipeline_ops.py` |
| Block profiler | `scripts/profile_krea_pipeline_blocks.py` |
| External links / receipts | [`blackwell-docs.md`](blackwell-docs.md) |
