# Bootstrapping Further Optimization (Phase 3): From Explainery Knowledge to More FPS

> **Explainer #13** — Phase 1–2 explain “how FA4/CuTe works”. This Phase 3 doc is the *optimization playbook*: how to turn those internals into a repeatable pipeline for getting real speedups (without quality shortcuts), based on what we’ve learned in this repo.
> **Updated:** 2025-12-25

---

## Why Phase 3 Exists

Phase 1–2 explainers are kernel-internals focused:
- how the forward kernel is structured
- how masking, paged KV, split-K, and backward work
- what changes (in practice) on SM103/B300

But the remaining optimization work is mostly about:
- **choosing the right bottleneck** (Amdahl + profiling discipline),
- **avoiding “silent fallbacks”** (backend/toolchain gating),
- **reducing the stuff around attention** (“other_in_self”: GEMMs + copies + glue),
- and **making improvements reproducible** (experiment cards → session-state).

This doc is written explicitly for “future worker boots in and makes progress fast.”

---

## Where We Are (The Journey So Far, in One Page)

**Canonical perf comparisons use** `320x576`, 4 denoise steps, KV-bias `0.3`, and quality-preserving settings:
- `notes/FA4/optimization-map.md`
- `notes/FA4/b300/investigation-runbook.md`

**B200 (SM100):** ~`20 FPS` class at `320x576` (happy path).

**B300 (SM103):** two different realities:
- repo-default stack: ~`8.8 FPS` (decode/cuDNN dominated)
- cu130 “SM103-native” stack: ~`15 FPS` end-to-end typical; higher best-case when compile/stack aligns

What changed the game on B300:
- decode got ~3–4× faster with the cu130 stack (cuDNN/kernel coverage)
- KV-bias got meaningfully faster when expressed as **FA4 `score_mod`** rather than segment-combine

Key truth sources (if numbers disagree):
- `notes/FA4/b300/session-state.md`
- `notes/FA4/b300/investigation-runbook.md`
- `notes/FA4/b300/optimization-vision.md`

---

## The “Optimization Loop” (What to Do Next, Every Time)

This is the loop we’ve converged on, and it’s the fastest way to not get lost.

### Step 0 — Lock a baseline

Pick *one* baseline command and keep it stable (same resolution/steps/bias/backend).

B300 cu130 example (see session state for the current blessed invocation):
- `notes/FA4/b300/session-state.md`

### Step 1 — Get a breakdown before touching code

Use the lightweight profilers we already built:
- `PROFILE_PIPELINE_BLOCKS=1` for “denoise vs decode vs recompute”
- `PROFILE_ATTENTION=1` for “self vs cross vs ffn” and “kv_bias vs other_in_self”

If you don’t know which bucket is dominant, optimizing kernels is gambling.

### Step 2 — Choose a lever using Amdahl

Don’t optimize what feels cool; optimize what moves end-to-end.

Practical heuristics from our own profiling (see `notes/FA4/docs/kernel-optimization-guide.md`):
- KV-bias kernel speedups help, but are bounded once “other_in_self” dominates.
- After FA4 `score_mod` KV-bias is in, the next wins often come from:
  - QKV/output projections (GEMMs)
  - dtype/layout conversions (`aten::copy_`, `aten::to`)
  - compile/graph capture opportunities around (not through) custom kernels

### Step 3 — Make one change

One change means:
- one env var toggle, or
- one code path swap, or
- one isolated optimization

### Step 4 — Record an experiment card

If it isn’t written down, it didn’t happen:
- `notes/FA4/b300/experiments.md`
- `notes/FA4/b200/experiments.md`

Promotion rule:
- if it becomes “how we should run by default”, update `session-state.md`
- if it changes how we measure/decide, update the runbook

---

## The Next Bottleneck (What Phase 1–2 Enabled Us To See)

Phase 1–2 explainers were necessary to make these statements confidently:

1. **KV-bias is not “just attention”**: backend choice can turn it into multiple attention calls + logaddexp combine.
   - Explainer #11: `notes/FA4/explainers/11-splitk-and-segment-combine.md`

2. **FA4 `score_mod` reduces *functional overhead***: it keeps bias inside the main kernel instead of segmenting KV.
   - Explainer #3: `notes/FA4/explainers/03-score-mod.md`

3. **Once KV-bias is fixed, “other_in_self” dominates** (QKV, projections, copies, glue).
   - Evidence in B300 vision: `notes/FA4/b300/optimization-vision.md`
   - Evidence in the broader perf story: `notes/FA4/docs/kernel-optimization-guide.md`

So Phase 3 is mostly about attacking “other_in_self”, not endlessly re-tuning KV-bias.

---

## Phase 3 Roadmap (Concrete, Repo-Grounded)

This is the recommended ordering for “future worker wants progress fast.”

### A) Make sure you’re not benchmarking a fallback

On SM103, “it runs” can still mean “it fell back to something terrible.”

Checklist:
- confirm torch/CUDA stack (cu130 matters on B300): `notes/FA4/b300/session-state.md`
- confirm `TRITON_PTXAS_PATH` points at a CUDA that knows `sm_103`: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- confirm KV-bias backend: `SCOPE_KV_BIAS_BACKEND`
- confirm you’re not mixing CuTe module variants when `SCOPE_KV_BIAS_BACKEND=fa4`:
  - `src/scope/core/pipelines/wan2_1/modules/attention.py` (`SCOPE_ENABLE_FA4_VARLEN`)

### B) Reduce conversion / copy / layout “glue”

Op-level profiles on B300 show a lot of time in:
- `aten::copy_`
- `aten::to` / `_to_copy`
- elementwise kernels
- FP8 GEMMs (`aten::_scaled_mm`) depending on quantization

This is often the highest-leverage work after KV-bias is solved, because it hits many calls.

Where to start in code:
- transformer hot path: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- attention backend dispatch: `src/scope/core/pipelines/wan2_1/modules/attention.py`

### C) Improve the “non-bias” attention story (self + cross)

KV-bias is only one part of attention.
If attention is ~70–80% of transformer time on B300 (current evidence), you want both:
- a good KV-bias path, and
- a good non-bias attention path

Candidate experiments (ordered by effort):
- compare `quantization none` vs fp8 on B300 (conversion overhead can dominate)
- benchmark cuDNN SDPA / backend selection (see B300 vision “Option C”)
- consider opt-in FA4 varlen once the CuTe module mixing risk is handled (`SCOPE_ENABLE_FA4_VARLEN=1`)

### D) Torch compile: “regional compile” around custom kernels

Working principle:
- keep CuTe calls opaque to Dynamo (we already do this for score_mod)
- compile the regions around attention to fuse norms/pointwise ops/projections

Known risks on SM103 (don’t rediscover these from scratch):
- some compile modes abort with tcgen05 LLVM intrinsic errors
- CUDAGraph modes can fail with “output overwritten” hazards

Current state and caveats:
- `notes/FA4/b300/session-state.md`
- `notes/FA4/b300/optimization-vision.md` (Option E)

### E) Level 5/6: bigger R&D bets (when the above is exhausted)

If you want to go deeper than “fix glue + GEMMs + compile,” the next tier is:
- Level 5: fuse adjacent work (RoPE-in-prologue style)
- Level 6: architecture-specific attention kernels (TMA/warp specialization)

Starting points:
- `notes/FA4/b300/level5-level6-resources.md`
- ThunderKittens (Blackwell attention) is the canonical “Level 6” reference in our notes

---

## How This Relates to the Explainers You Just Read

If you’re trying to decide “which explainer should I re-open right now?”, use this mapping:

- You are debugging **why KV-bias is slower on one backend** → #3 and #11
- You are debugging **why combine exists / how LSE should be used** → #7 and #11
- You are debugging **a weird SM103-only failure** → #12 + `notes/FA4/b300/fa4-patches.md`
- You are extending **score_mod or adding new callbacks** → #3 + `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`
- You are touching **forward scheduling / loads / pipelining** → #2, #5, #6
- You are trying to understand **why backward is expensive / structured the way it is** → #10

---

## References (Start Here)

- “You are here” map: `notes/FA4/optimization-map.md`
- Shareable perf story: `notes/FA4/docs/kernel-optimization-guide.md`
- B300 protocol: `notes/FA4/b300/investigation-runbook.md`
- B300 current best-known config: `notes/FA4/b300/session-state.md`
- B300 forward-looking options: `notes/FA4/b300/optimization-vision.md`
- Level 5/6 reading list: `notes/FA4/b300/level5-level6-resources.md`
- Phase 1–2 explainer index: `notes/FA4/explainers/README.md`

