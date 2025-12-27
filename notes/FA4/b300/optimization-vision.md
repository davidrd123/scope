# B300 Optimization Vision

> **Purpose:** Strategic roadmap for B300 performance optimization  
> **Created:** 2025-12-25 | **Updated:** 2025-12-27  
> **Current (quality-preserving, benchmark harness):** ~30.7–30.9 FPS (BF16 + `--compile`) @ `320x576`  
> **Daydream end-to-end:** TBD re-measure (don’t rely on historical numbers)

**Quality gate:** On B300/SM103, FP8 quantization currently produces **garbage output** (gray/noise). Treat FP8 as perf-only debugging, not a shippable win. Canonical baseline is `--quantization none` (BF16).

---

## 0. Stack Maturity Watchlist (Re-check Periodically)

This is the short list of “upstream maturity” items that can unlock real speedups on Blackwell **without** changing our core model code.

### Quantization: FP8 (today: perf-only) → FP4 (future: highly constrained)

- **TorchAO FP8 stability + quality**
  - What to watch: TorchAO/torch 2.9 compatibility, “cpp extensions skipped” warning disappearing, and *visually correct output* (not just higher FPS).
  - Where: [`session-state.md`](session-state.md) (TorchAO section) + [`torchao-as-strided-dispatch.md`](../../issues/torchao-as-strided-dispatch.md).
  - Re-test trigger: TorchAO releases that explicitly support the exact torch build we run on B300 (no “skipping import of cpp extensions”), plus a quality-clean demo run.

- **FP4 inference**
  - Reality: FP4 tends to be “weight-only” and stack-specific (TransformerEngine / TensorRT-style paths); it is not a drop-in “PyTorch + compile” switch today.
  - Re-test trigger: a supported, quality-preserving FP4 inference path for our model class (attention-heavy diffusion) that we can run end-to-end without major re-architecture.

### Attention Backends (cuDNN SDPA / FA4 / FlashAttention)

- **cuDNN SDPA on Blackwell**
  - What to watch: whether cross-attn / non-bias attn can actually land on cuDNN SDPA kernels in our shapes (and stay there).
  - Re-test trigger: cuDNN/PyTorch updates that expand cuDNN SDPA coverage on SM10x; add an experiment card when it changes.

- **FA4/FlashAttention SM10x maturity**
  - What to watch: fewer fallbacks, better varlen coverage, and stable compilation/dispatch on SM103.
  - Re-test trigger: version bumps that change kernel selection (FlashAttention/FA4/CUTLASS DSL), especially if they claim SM10x improvements.

### Compile + Codegen (Inductor / Triton / CUDAGraph)

- **Triton/Inductor SM103 tcgen05 path**
  - What to watch: removing remaining “hard abort” failure modes on SM103 and improving codegen for attention/GEMMs/glue.
  - Re-test trigger: Triton/torch releases that mention SM10x/tcgen05 stability/perf.

- **CUDAGraph Trees (“reduce-overhead”)**
  - What to watch: correctness fixes for the “output overwritten” class and stable step-marker usage.
  - Where: [`session-state.md`](session-state.md) (cudagraph section).
  - Re-test trigger: upstream fixes in PyTorch + a clear recipe that remains correct for our streaming loop.

### cuDNN / Conv3d / Decode

- **Conv3d BF16/FP16 regressions**
  - What to watch: cuDNN version guidance changing (or regressions being fixed) in PyTorch release notes / issue threads.
  - Where: [`session-state.md`](session-state.md) (Conv3d section).
  - Re-test trigger: any upgrade of `nvidia-cudnn-cu12` / PyTorch that could alter Conv3d algo selection on SM103.

---

## Executive Summary

The B300 journey has been mostly about **getting onto the right fast paths** (stack + backends) and then eliminating **hidden slow-path ops** (copy/fill storms), not “inventing a faster attention kernel”.

We now have a quality-preserving BF16 path on B300 that is ~**3.5×** the repo-default baseline in the benchmark harness:
- repo-default stack: ~8.8 FPS (decode/cuDNN dominated)
- cu130 stack + `--compile` + decode fast paths + FA4 knobs: ~30.7–30.9 FPS (BF16)

Near-term strategy: validate the best config **end-to-end in Daydream**, then use stack-attributed op profiling to focus denoise (`other_in_self`) and KV-cache recompute.

### Progress Log (benchmark harness, `320x576`, steps=4, bias=0.3)

| Date | Milestone | FPS | Key Change |
|------|-----------|-----|------------|
| 2025-12-24 | Baseline | ~8.8 | repo-default runtime stack (decode slow) |
| 2025-12-25 | cu130 baseline | ~14.9 | CUDA 13 + cuDNN 9.13 (decode fast) |
| 2025-12-25 | FA4 KV-bias | ~16.7 | FA4/CuTe `score_mod` (avoid segment-combine overhead) |
| 2025-12-26 | Patch-embed fastpath | ~19.7 | Conv3d patch embed → per-frame Conv2d (kills copy/fill storm) |
| 2025-12-26 | + `--compile` (BF16) | ~22.8 | stable regional compilation around the model |
| 2025-12-27 | Decode slow-path fix | ~29.4 | keep streaming Resample outputs contiguous (Conv3d stays on fast kernels) |
| 2025-12-27 | Best-known (BF16) | ~30.8 | FA4 varlen opt-in + fused projections + compile |

---

## 1. Where We Are (Current Truth)

### Two “Worlds” on B300

| Regime | Typical FPS @ `320x576` | What dominates | What to do |
|---|---:|---|---|
| Repo-default stack | ~8.8 | decode/cuDNN slow paths | switch to the cu130 stack first |
| cu130 stack | ~30+ (BF16 + `--compile`) | denoise + recompute_kv_cache | focus `other_in_self` and stability/quality; keep decode on fast paths |

### Known-Good BF16 Configs (quality-preserving)

Numbers below are from `scripts/profile_krea_pipeline_blocks.py` in the cu130 env; use [`session-state.md`](session-state.md) for the exact command.

| Config | FPS | Notes |
|---|---:|---|
| `SCOPE_KV_BIAS_BACKEND=fa4` + `SCOPE_ENABLE_FA4_VARLEN=1` + `--compile` + `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=1` | ~30.8 | best-known BF16; warmup cost |
| Above + `SCOPE_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` | ~30.9 | experiment-only; longer warmup/autotune |
| Same stack but `WANVAE_RESAMPLE_ENSURE_CONTIGUOUS=0` | ~21.5 | demonstrates the decode slow-path hazard |
| Historical (needs re-measure post decode fix): `SCOPE_KV_BIAS_BACKEND=fa4` (no compile) | ~19.7 | keep as historical reference only |

### What’s Actually Left

- The “big rocks” are now:
  - **`denoise`** (transformer) + **`recompute_kv_cache`**
  - Within denoise: **QKV/projection GEMMs + layout/dtype glue** (often visible as `aten::copy_`, `aten::_to_copy`, small elementwise storms)

This is why the next wins tend to come from **reducing “other_in_self” glue** and avoiding accidental slow paths, rather than continuing to swap KV-bias kernels (which is already a smaller slice).

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
- Remaining dtype/layout churn and view/contiguity hazards in **denoise** (QKV packing, projections, cache writes).
- Any remaining decode slow paths that reappear at higher resolution / different settings.

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
| Decode slow-path fix | ~29.4 | decode no longer dominates @ `320x576` | ✅ |
| Best-known BF16 | ~30.8 | B300 harness is ~3.5× baseline | ✅ |
| Daydream end-to-end | TBD | real server steady-state + UX (warmup) | 🎯 |

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
