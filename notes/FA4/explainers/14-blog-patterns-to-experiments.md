# From Blog Patterns to Experiment Cards (Phase 3)

> **Explainer #14** — A concrete “what to try next” menu, derived from our blog notes (Blackwell dataflow, warp specialization, memory-bound kernels, torch.compile/host overhead) and translated into *one-change* experiment cards for this repo.
> **Updated:** 2025-12-25

---

## The Goal

Phase 3 only works if we can repeatedly answer:

1) **What is the bottleneck in *this* stack on *this* GPU?**  
2) **What’s the smallest change that could move it?**  
3) **Did it move end-to-end (not just a microbench)?**  

This doc turns the “blog mental models” into *actionable* cards you can paste into:
- [`experiments.md`](../b300/experiments.md)
- [`experiments.md`](../b200/experiments.md)

If you haven’t read the Phase 3 playbook first:
- [`13-optimization-bootstrapping.md`](13-optimization-bootstrapping.md)

---

## Pre-Flight (Don’t Skip)

These prevent week-long rabbit holes.

### Canonical settings (so results compare)

Use:
- `320x576`
- 4 denoise steps
- KV-bias `0.3`
- no quality shortcuts

Truth sources:
- [`optimization-map.md`](../optimization-map.md)
- [`investigation-runbook.md`](../b300/investigation-runbook.md)
- [`session-state.md`](../b300/session-state.md)

### Know what your profiler does

- `PROFILE_PIPELINE_BLOCKS=1` and friends add `synchronize()` calls. Treat them as **breakdown truth**, not peak throughput truth.
- End-to-end FPS without forced sync is the final arbiter.

---

## Experiment Menu (Pick One Card, Run It, Write It Down)

Each item below is intentionally phrased as:
- **Question**
- **Change (one thing)**
- **Measurement**
- **Success signal**
- **Stop condition / risk**

### A) Backend / Dispatch Hygiene (SM103 makes this critical)

**A1 — Confirm you’re not accidentally on a fallback**

- Question: are we benchmarking the intended backend, or a silent fallback?
- Change: none (just verify).
- Measurement:
  - confirm env vars: `SCOPE_KV_BIAS_BACKEND`, `TRITON_PTXAS_PATH`
  - confirm stack versions (torch + CUDA) match the session state
- Success signal: matches the expected “good stack” checklist.
- Stop condition: if anything is off, fix it before proceeding.

References:
- [`12-sm103-notes.md`](12-sm103-notes.md)
- [`session-state.md`](../b300/session-state.md)

**A2 — KV-bias backend swap: `flash` segment-combine vs `fa4` score_mod**

- Question: does in-kernel `score_mod` still win in our current stack?
- Change: set `SCOPE_KV_BIAS_BACKEND=flash` vs `SCOPE_KV_BIAS_BACKEND=fa4`.
- Measurement:
  - end-to-end benchmark: `scripts/profile_krea_pipeline_blocks.py`
  - breakdown: `PROFILE_ATTENTION=1` (expect lower absolute FPS; look at shares/ms)
- Success signal:
  - `self_attn_kv_bias_*` ms/call drops
  - end-to-end FPS improves (or stays flat while shifting time elsewhere)
- Stop condition: if `fa4` path triggers toolchain failures, record the error + versions and fall back to `flash` to keep the system usable.

References:
- Why combine exists / what overhead looks like: [`11-splitk-and-segment-combine.md`](11-splitk-and-segment-combine.md)
- KV-bias backend implementation: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

**A3 — FA4 varlen opt-in (when KV-bias uses vendored CuTe)**

- Question: can we safely enable FA4 varlen without module-mixing hazards?
- Change: set `SCOPE_ENABLE_FA4_VARLEN=1` (only when `SCOPE_KV_BIAS_BACKEND=fa4`).
- Measurement:
  - end-to-end benchmark + `PROFILE_ATTENTION=1`
  - watch for runtime DSL/codegen errors
- Success signal:
  - small but consistent FPS win (historically ~1–2% on B300)
  - no new runtime instability
- Stop condition:
  - any CuTe import/codegen mismatch → revert; log the failure.

Reference:
- Guardrail logic: `src/scope/core/pipelines/wan2_1/modules/attention.py`

---

### B) “Other In Self” (GEMMs + Glue) — The Next Likely Lever

Blog inspiration:
- “QuACK” memory-bound kernel playbook: [`getting-mem-bound-kernals-SOL.md`](../../research/2025-12-24/incoming/perf/blogs/getting-mem-bound-kernals-SOL.md)
- Diffusers compile playbook: [`torch-compile-and-diffusers.md`](../../research/2025-12-24/incoming/perf/blogs/torch-compile-and-diffusers.md)

**B1 — Verify QKV projections are fused (they should be)**

- Question: are we paying 3 separate projections, or 1 fused projection?
- Change: none (verify); optionally add a one-card test that disables fusion temporarily (code change) if you need proof.
- Measurement:
  - `PROFILE_ATTENTION=1` and look at `qkv_projection` ms/call
  - optionally op-level profile: `scripts/profile_krea_pipeline_ops.py`
- Success signal: fused path is active by default.

Reference:
- QKV fusion happens at pipeline init: `src/scope/core/pipelines/krea_realtime_video/pipeline.py`
- Implementation: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (`CausalWanSelfAttention.fuse_projections`)

**B2 — Quantization sweep (B300 often prefers “none”)**

- Question: is fp8 actually faster on *this* stack, or is conversion/scaling dominating?
- Change: switch `--quantization none` vs fp8 in the benchmark script.
- Measurement:
  - end-to-end FPS
  - op-level profile to see if `aten::_scaled_mm` and `aten::to/copy_` dominate
- Success signal:
  - whichever mode wins becomes your “baseline for future experiments”
- Stop condition:
  - don’t mix quantization modes while comparing kernel experiments; pick one baseline per session.

Reference:
- B300 vision notes about conversion overhead: [`optimization-vision.md`](../b300/optimization-vision.md)

**B3 — Attack copy/to glue with a single hypothesis**

Examples of “one-change” hypotheses (pick exactly one):
- “This dtype cast is redundant; remove it.”
- “This tensor is re-materialized each call; cache it.”
- “This layout conversion can be hoisted out of the loop.”

Measurement:
- op-level profile (copies/to share)
- end-to-end FPS

Success signal:
- fewer copy/to kernels (count or time) *and* stable correctness

Where to look:
- transformer hot path: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

---

### C) Compile + Host Overhead (Treat as Experiments, Not Defaults)

Blog inspiration:
- [`torch-compile-and-diffusers.md`](../../research/2025-12-24/incoming/perf/blogs/torch-compile-and-diffusers.md)
- [`modal_host-overhead-inference-efficency.md`](../../research/2025-12-24/incoming/perf/blogs/modal_host-overhead-inference-efficency.md)

**C1 — Regional compile mode sweep**

- Question: can a different `torch.compile` mode move “other_in_self” without breaking SM103?
- Change: set `SCOPE_TORCH_COMPILE_MODE` (one value at a time) while using `--compile`.
- Measurement:
  - end-to-end FPS after warmup
  - “first iteration” latency if you care about cold start
- Success signal:
  - repeatable end-to-end gain without tcgen05 aborts
- Stop condition:
  - hard aborts or repeated recompilations → revert and record.

Reference:
- Compile wiring: `src/scope/core/pipelines/krea_realtime_video/pipeline.py`
- Known failures on SM103: [`session-state.md`](../b300/session-state.md)

**C2 — Look for host overhead (only once kernels get short)**

- Question: are there gaps in CUDA streams / launch overhead dominating?
- Change: none (measure).
- Measurement:
  - Nsight Systems if available, or coarse “GPU time vs wall time” comparisons
  - avoid the synchronizing profilers when diagnosing overlap
- Success signal:
  - if you find gaps: prioritize fusion / cudagraph style work over more kernel micro-optimizations

Reference:
- Host overhead mental model: [`modal_host-overhead-inference-efficency.md`](../../research/2025-12-24/incoming/perf/blogs/modal_host-overhead-inference-efficency.md)

---

### D) Level 6 R&D (Only After the “Easy Wins” Plateau)

Blog inspiration:
- Blackwell dataflow + 128×128 tiles: [`thunderkittens-blackwell.md`](../../research/2025-12-24/incoming/perf/blogs/thunderkittens-blackwell.md)
- Warp specialization compiler support: [`warp-specialization.md`](../../research/2025-12-24/incoming/perf/blogs/warp-specialization.md)
- TMA/mbarrier fundamentals: [`gau-nerst-tcgen05.md`](../../research/2025-12-24/incoming/perf/blogs/gau-nerst-tcgen05.md)

These are not “quick wins”; they’re longer projects. Still, you can structure them as experiment cards:

**D1 — Warp-specialized Triton kernel (B200-first)**
- Change: add `num_consumer_groups` / `num_buffers_warp_spec` to an autotuned Triton kernel
- Measurement: microbench + end-to-end sanity
- Stop condition: any SM103 instability → keep this B200-only until Triton/stack improves

**D2 — CuTe kernel parameter sweeps (tile/stages)**
- Change: adjust one tiling/staging knob (if exposed)
- Measurement: kernel time + end-to-end; validate no “persistence disabled” regressions
- Stop condition: register/smem blow-ups or correctness drift

---

## “What’s Next?” (How to Extend This Document)

When you discover a new high-value lever:

1) add a short “blog-inspired principle” to [`13-optimization-bootstrapping.md`](13-optimization-bootstrapping.md)
2) add a corresponding experiment card recipe here
3) run it once and link the artifact/log in the experiments file

That’s how Phase 3 stays useful instead of turning into a motivational poster.

---

## References

- Phase 3 playbook: [`13-optimization-bootstrapping.md`](13-optimization-bootstrapping.md)
- B300 protocol: [`investigation-runbook.md`](../b300/investigation-runbook.md)
- B300 state: [`session-state.md`](../b300/session-state.md)
- B300 strategy: [`optimization-vision.md`](../b300/optimization-vision.md)
- Blog notes root: `notes/research/2025-12-24/incoming/perf/blogs/`

