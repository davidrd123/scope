# B200 (SM100) Development Plan (FA4 / KREA Realtime Video)

> **Purpose:** A concrete, executable plan for improving and hardening B200 performance (SM100) using the work we’ve already done (FA4/CuTe score_mod, RoPE work, profiling).
> **Updated:** 2025-12-26

---

## Goals / Non-Goals

**Goals**
- Improve end-to-end FPS at canonical settings (`320x576`, 4 steps, KV-bias `0.3`) without quality shortcuts.
- Make the “fast path” the default when available (minimize manual env-var babysitting).
- Keep changes reproducible: every meaningful change becomes an experiment card and/or session-state update.

**Non-goals (for this plan)**
- Big architectural rewrites of the model.
- Novel attention algorithms (that’s Level 7+ work).
- Shipping anything that changes output quality unless explicitly agreed.

---

## Current State (B200)

- B200 is the “happy path” for FA4/CuTe: fewer toolchain hazards than SM103.
- We already have the key building blocks:
  - KV-bias via FA4 `score_mod` (fast when available)
  - profiling hooks (`PROFILE_PIPELINE_BLOCKS`, `PROFILE_ATTENTION`)
  - fused QKV projections (`fuse_projections()`)

Truth sources:
- `notes/FA4/b200/session-state.md`
- `notes/FA4/docs/kernel-optimization-guide.md`
- `notes/FA4/optimization-map.md`

---

## Workstreams (Ordered by Expected ROI)

### 1) Baseline + Regression Guardrails (1 day)

**Deliverables**
- A single “blessed” canonical benchmark command in `notes/FA4/b200/session-state.md`.
- A fresh experiment card recording today’s baseline on B200.

**Acceptance**
- Anyone can run the command and get comparable numbers (same settings, same env vars).

Where:
- `notes/FA4/b200/session-state.md`
- `notes/FA4/b200/experiments.md`

---

### 2) Make the Best KV-bias Backend Automatic (1–2 days)

**Why:** On B200 we shouldn’t require `SCOPE_KV_BIAS_BACKEND=fa4` manually when FA4 score_mod is available; the code can choose.

**Work**
- Change backend selection to:
  - prefer FA4 `score_mod` when it’s importable and supports the needed signature
  - otherwise fall back to the current stable default
- Keep `SCOPE_KV_BIAS_BACKEND` as an override for experiments.

**Acceptance**
- With no env vars set, B200 uses the fastest available KV-bias backend.
- With `SCOPE_KV_BIAS_BACKEND=<x>`, we can still force a backend for experiments.

Code touchpoints:
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

---

### 3) Attack “other_in_self” (GEMMs + copies + glue) (3–7 days)

**Why:** Once KV-bias is fast, remaining self-attn time tends to be dominated by projections, casts/copies, and small glue kernels.

**Work (one-change cards)**
- Verify QKV projection fusion is active (and stays active).
- Run op-level profiling and pick the top 1–3 offenders (often `aten::copy_`, `aten::to`, `aten::_to_copy`).
- For each offender: design a single hypothesis and implement it (hoist conversion, remove redundant cast, reuse cached tensor/layout).

**Acceptance**
- `PROFILE_ATTENTION=1` shows reduced `other_in_self` ms/call.
- End-to-end FPS improves measurably at canonical settings.

Touchpoints:
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- `src/scope/core/pipelines/krea_realtime_video/pipeline.py`
- profiler scripts under `scripts/`

---

### 4) Torch Compile: Make It a Reliable, Opt-In Win on B200 (3–10 days)

**Why:** B200 is a good place to make `--compile` work consistently (SM103 has more compiler/toolchain landmines).

**Status (this repo / this machine):**
- `scripts/profile_krea_pipeline_blocks.py --compile` is a **large steady-state win** on B200 with FP8 quantization (measured `~18.36 FPS → ~29.98 FPS` at canonical settings).
- The server path supports opt-in via `SCOPE_COMPILE_KREA_PIPELINE=1`; on SM100 we allow compile to remain enabled even when FP8 quantization is on (other architectures default-disable unless explicitly overridden).

**Work**
- Use **regional compilation** (compile only repeated submodules).
- Sweep `SCOPE_TORCH_COMPILE_MODE` options as experiment cards.
- Reduce recompiles (prefer `dynamic=True` / `mark_dynamic` where appropriate).
- Ensure custom kernels (CuTe) stay opaque to Dynamo tracing.

**Acceptance**
- `--compile` improves steady-state FPS after warmup without crashes or repeated recompiles.
- Failure modes are documented as “known bad modes” (so future workers don’t rediscover them).

Touchpoints:
- `src/scope/core/pipelines/krea_realtime_video/pipeline.py` (`SCOPE_TORCH_COMPILE_MODE`)
- `notes/FA4/b200/experiments.md`

---

### 5) Level 5/6 (Optional R&D): Fuse Adjacent Work / Blackwell-Specific Kernel Work (weeks)

This is for when the above plateaus.

Candidate tracks:
- Fuse RoPE into attention prologue (not via `score_mod`; RoPE modifies Q/K vectors).
- Investigate CuTe hooks (`q_mod`/`k_mod`-style) or equivalent in our vendored CuTe.
- Evaluate ThunderKittens-style attention as a reference for “bubble-free” dataflow kernels.

References:
- `notes/FA4/b300/level5-level6-resources.md`
- `notes/FA4/explainers/13-optimization-bootstrapping.md`
- `notes/FA4/explainers/14-blog-patterns-to-experiments.md`

---

## Milestones (Concrete “Done” Checks)

1) **B200 Baseline Locked**
- baseline card exists in `notes/FA4/b200/experiments.md`
- session-state has the single blessed command

2) **Automatic KV-bias Backend Selection**
- no-env-var run uses best available backend on B200
- override still works for experiments

3) **Glue Reduction Round 1**
- at least 2 experiment cards that each remove or materially reduce a copy/to hotspot

4) **Compile Win (Optional)**
- one documented compile mode that improves steady-state FPS (and doesn’t regress correctness)

---

## Where to Record Work

- Experiments: `notes/FA4/b200/experiments.md`
- “What to run today”: `notes/FA4/b200/session-state.md`
- High-level map: `notes/FA4/optimization-map.md`
