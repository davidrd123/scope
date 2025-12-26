# B300 (SM103) Development Plan (FA4 / KREA Realtime Video)

> **Purpose:** A concrete plan for B300 performance that reflects the two hard realities we’ve observed:
> 1) the runtime stack (cuDNN/CUDA) can dominate decode performance, and
> 2) backend selection/toolchain gating on SM103 can silently destroy perf.
> **Updated:** 2025-12-26

---

## Goals / Non-Goals

**Goals**
- Make B300 fast *and* repeatable at canonical settings (`320x576`, 4 steps, bias `0.3`, quality-preserving).
- Keep the system usable under partial failures (graceful fallback ladders, clear logging).
- Move toward a credible path to **24+ FPS** once “stack correctness” is locked.

**Non-goals (for this plan)**
- “Just micro-optimize attention” without first validating stack/toolchain (we already learned this is a trap).
- Shipping quality regressions (e.g. KV recompute skipping) as “performance wins”.

---

## Current State (B300)

Two regimes exist:
- **Repo-default stack:** ~`8.8 FPS` at `320x576` (decode/cuDNN dominated).
- **SM103-native stack (cu130):** ~`19–20 FPS` baseline (BF16) and ~`22–23 FPS` with `--compile` (BF16), at `320x576` with bias `0.3`.

Truth sources:
- `notes/FA4/b300/session-state.md`
- `notes/FA4/b300/investigation-runbook.md`
- `notes/FA4/b300/optimization-vision.md`

---

## Workstreams (Ordered by “Unblockers First”)

### 1) Environment / Toolchain Reliability (1–3 days, highest priority)

**Why:** On SM103, many “performance bugs” are actually “wrong stack” or “wrong toolchain”.

**Work**
- Keep a single blessed setup path for cu130 and document it:
  - env create/fix scripts
  - required `ptxas` path for SM103
  - FlashAttention install requirements
- Maintain a small “sanity checklist” command set (versions + imports).
- Document and detect common failure states:
  - NVML/CUDA wedges (Error 304)
  - missing `flash_attn` leading to catastrophic fallbacks
  - CUTLASS DSL arch rejection (`sm_103a`) and patching steps

**Acceptance**
- A fresh worker can get to the “known good” baseline in one sitting with no guesswork.

References:
- `notes/FA4/b300/session-state.md`
- `notes/FA4/b300/fa4-patches.md`
- `scripts/patch_cutlass_sm103.sh`

---

### 2) Backend Selection Hardening (1–4 days)

**Why:** On B300, backend choice is correctness/perf-critical: a “working” fallback can be 10× slower.

**Work**
- Keep a strict, explicit fallback ladder for KV-bias:
  - `fa4` score_mod (best when available) → `flash` segment-combine (stable) → `flex` (last resort)
  - avoid Triton on SM103 unless Triton improves (historically catastrophic)
- Ensure we never accidentally mix CuTe module variants (vendored vs wheel) when `SCOPE_KV_BIAS_BACKEND=fa4`.
- Consider adding a single startup log line that prints the chosen backends (KV-bias + non-bias attention).

**Acceptance**
- If FA4 isn’t available, we fall back to a stable backend (not Triton) and we *know* it happened (logged).

Touchpoints:
- KV-bias backend selection: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- Non-bias attention selection / FA4 varlen guardrail: `src/scope/core/pipelines/wan2_1/modules/attention.py`

---

### 3) Choose the Right “Performance Baseline” (Quantization + Stack) (1–2 days)

**Why:** On B300, **output quality comes first** — FP8 is currently broken (gray/noise), so the canonical baseline must be BF16 (`--quantization none`). FP8 runs are kept only as perf-only breadcrumbs for upstream/tooling work.

**Work**
- Establish BF16 (`--quantization none`) as the single baseline for B300 perf work (so results remain quality-preserving).
- For BF16: record end-to-end FPS + warmup time + one op-level profile (stack-attributed) for the current “best” config.
- If running FP8 for debugging: require (a) working torchao C++ extensions and (b) a quality snapshot check, so we don’t mistake “fast garbage” for progress.

**Acceptance**
- We have one baseline config for future experiments (BF16), with repeatable numbers and a short “known good” command in `session-state.md`.

References:
- `notes/FA4/b300/optimization-vision.md`
- `notes/FA4/b300/session-state.md`

---

### 4) Reduce “other_in_self” (Glue + Projections) (3–10 days)

**Why:** After FA4 KV-bias, the remaining self-attn cost is often dominated by QKV/projections and memory traffic (`copy_`, `to`).

**Work (one-change cards)**
- Use op-level profiling (`scripts/profile_krea_pipeline_ops.py --with-stack --summary`) to identify the top copy/to/fill hotspots by call stack.
- Remove redundant conversions, hoist invariant reshapes, and reduce layout churn.
- Run one higher-resolution “sanity card” (e.g. `480x864`) to see whether bottlenecks shift (attention vs decode vs glue).
- Treat each fix as an experiment card; promote only stable wins.

**Acceptance**
- `PROFILE_ATTENTION=1`: `other_in_self` decreases (ms/call).
- End-to-end FPS improves at canonical settings on the cu130 stack.

Touchpoints:
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- `scripts/profile_krea_pipeline_ops.py`

---

### 5) Torch Compile on SM103: Controlled, Opt-In Experiments (3–14 days)

**Why:** Compile is high upside but has known SM103 failure modes (tcgen05 LLVM aborts, cudagraph overwrites).

**Work**
- Keep compile experiments tightly scoped (regional compile around stable regions).
- Sweep `SCOPE_TORCH_COMPILE_MODE` via experiment cards; record “known good” vs “known bad”.
- Track warmup time explicitly (compile wins can trade steady-state for cold-start).
- Consider AOT export (torch.export/AOTInductor) only if cold-start is a primary goal; keep it separate from steady-state perf work.

**Acceptance**
- At least one documented compile setting yields repeatable steady-state wins on B300 without instability.

References:
- `notes/FA4/b300/session-state.md`
- `src/scope/core/pipelines/krea_realtime_video/pipeline.py`

---

### 6) Level 6 R&D Bets (weeks, only after the above)

Candidate directions:
- **Level 6 kernel thesis: “post-projection pack”** (targets `other_in_self`)
  - Goal: delete glue by owning the boundary between QKV projection → RoPE → KV-cache write → attention.
  - Phase A (smaller): keep cuBLAS GEMMs, then run one custom kernel that applies RoPE to Q/K and writes Q and KV cache directly into the layouts downstream kernels want.
  - Phase B (harder): replace QKV projection with a CUTLASS/CuTe GEMM that has a custom epilogue (GEMM + RoPE + packing in one op).
  - Primary metric: “how many kernels/copies did we delete?” (and which stack groups disappeared), not just microbench TFLOPS.
- **Make VAE decode a planned subsystem**
  - Treat decode as an engine: stabilize algorithm selection and reduce framework overhead (cudnn-frontend planning or capture-style approaches) rather than writing conv kernels.
  - Motivation: we’ve already seen 4× swings from stack/algo selection; decode is a “big rock” worth subsystem treatment.
- **ThunderKittens / reference Blackwell kernels**
  - Use as a learning accelerator for Blackwell patterns (TMA, warpgroup pipelining, etc.) and as a potential source of drop-in kernels where shapes match.
  - Sober gate: only pursue if it integrates cleanly (doesn’t poison toolchain) and can match our runtime shapes.
- **A small “layout & glue kernel pack”**
  - One or two highly optimized pack/unpack kernels (QKV/KV-cache layouts, dtype/layout shims) that can be reused across call sites to kill `copy_/to` at the source.
- Investigate cuDNN SDPA backend choices for non-bias attention (cheap A/B; can be a “free win” if selected).
- If Triton improves on SM103, revisit warp specialization experiments (learning-first, with a safe escape hatch).

Productionization guardrails for Level 6 work (start day 1):
- Keep a hard fallback ladder (never require the new kernel to run).
- Add correctness + quality sentinels (golden clips / snapshot set).
- Add a “kernel provenance” banner at startup (what’s active vs what fell back).
- Keep Nsight-friendly minimal repro scripts per kernel.

References:
- `notes/FA4/b300/level5-level6-resources.md`
- `notes/FA4/explainers/13-optimization-bootstrapping.md`
- `notes/FA4/explainers/14-blog-patterns-to-experiments.md`

---

## Milestones (Concrete “Done” Checks)

1) **B300 Baseline Is Reproducible**
- session-state has a single blessed cu130 command
- baseline card exists in `notes/FA4/b300/experiments.md`

2) **Backend Failures Are Visible**
- chosen KV-bias backend and fallbacks are logged
- no silent fallbacks to Triton on SM103

3) **Baseline Configuration Chosen**
- `quantization none` vs fp8 decision documented with measurements

4) **Glue Reduction Round 1**
- ≥2 experiment cards that reduce copy/to hotspots and improve end-to-end FPS

5) **Compile Win (Optional)**
- one compile mode documented as stable and beneficial (or explicitly documented as not viable yet)

---

## Where to Record Work

- Experiments: `notes/FA4/b300/experiments.md`
- “What to run today”: `notes/FA4/b300/session-state.md`
- Measurement protocol: `notes/FA4/b300/investigation-runbook.md`
- Strategy options: `notes/FA4/b300/optimization-vision.md`
