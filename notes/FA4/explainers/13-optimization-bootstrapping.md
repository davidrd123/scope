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
- [`optimization-map.md`](../optimization-map.md)
- [`investigation-runbook.md`](../b300/investigation-runbook.md)

**B200 (SM100):** ~`20 FPS` class at `320x576` (happy path).

**B300 (SM103):** two different realities:
- repo-default stack: ~`8.8 FPS` (decode/cuDNN dominated)
- cu130 “SM103-native” stack: ~`15 FPS` end-to-end typical; higher best-case when compile/stack aligns

What changed the game on B300:
- decode got ~3–4× faster with the cu130 stack (cuDNN/kernel coverage)
- KV-bias got meaningfully faster when expressed as **FA4 `score_mod`** rather than segment-combine

Key truth sources (if numbers disagree):
- [`session-state.md`](../b300/session-state.md)
- [`investigation-runbook.md`](../b300/investigation-runbook.md)
- [`optimization-vision.md`](../b300/optimization-vision.md)

---

## The “Optimization Loop” (What to Do Next, Every Time)

This is the loop we’ve converged on, and it’s the fastest way to not get lost.

### Step 0 — Lock a baseline

Pick *one* baseline command and keep it stable (same resolution/steps/bias/backend).

B300 cu130 example (see session state for the current blessed invocation):
- [`session-state.md`](../b300/session-state.md)

### Step 1 — Get a breakdown before touching code

Use the lightweight profilers we already built:
- `PROFILE_PIPELINE_BLOCKS=1` for “denoise vs decode vs recompute”
- `PROFILE_ATTENTION=1` for “self vs cross vs ffn” and “kv_bias vs other_in_self”

If you don’t know which bucket is dominant, optimizing kernels is gambling.

### Step 2 — Choose a lever using Amdahl

Don’t optimize what feels cool; optimize what moves end-to-end.

Practical heuristics from our own profiling (see [`kernel-optimization-guide.md`](../docs/kernel-optimization-guide.md)):
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
- [`experiments.md`](../b300/experiments.md)
- [`experiments.md`](../b200/experiments.md)

Promotion rule:
- if it becomes “how we should run by default”, update `session-state.md`
- if it changes how we measure/decide, update the runbook

---

## The Next Bottleneck (What Phase 1–2 Enabled Us To See)

Phase 1–2 explainers were necessary to make these statements confidently:

1. **KV-bias is not “just attention”**: backend choice can turn it into multiple attention calls + logaddexp combine.
   - Explainer #11: [`11-splitk-and-segment-combine.md`](11-splitk-and-segment-combine.md)

2. **FA4 `score_mod` reduces *functional overhead***: it keeps bias inside the main kernel instead of segmenting KV.
   - Explainer #3: [`03-score-mod.md`](03-score-mod.md)

3. **Once KV-bias is fixed, “other_in_self” dominates** (QKV, projections, copies, glue).
   - Evidence in B300 vision: [`optimization-vision.md`](../b300/optimization-vision.md)
   - Evidence in the broader perf story: [`kernel-optimization-guide.md`](../docs/kernel-optimization-guide.md)

So Phase 3 is mostly about attacking “other_in_self”, not endlessly re-tuning KV-bias.

---

## Phase 3 Roadmap (Concrete, Repo-Grounded)

This is the recommended ordering for “future worker wants progress fast.”

### A) Make sure you’re not benchmarking a fallback

On SM103, “it runs” can still mean “it fell back to something terrible.”

Checklist:
- confirm torch/CUDA stack (cu130 matters on B300): [`session-state.md`](../b300/session-state.md)
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
- [`session-state.md`](../b300/session-state.md)
- [`optimization-vision.md`](../b300/optimization-vision.md) (Option E)

### E) Level 5/6: bigger R&D bets (when the above is exhausted)

If you want to go deeper than “fix glue + GEMMs + compile,” the next tier is:
- Level 5: fuse adjacent work (RoPE-in-prologue style)
- Level 6: architecture-specific attention kernels (TMA/warp specialization)

Starting points:
- [`level5-level6-resources.md`](../b300/level5-level6-resources.md)
- ThunderKittens (Blackwell attention) is the canonical “Level 6” reference in our notes

---

## Blog-Derived Patterns (Translated into “What To Do Here”)

We already vendor the relevant blog notes under:
`notes/research/2025-12-24/incoming/perf/blogs/`

This section is a “translation layer”: what those posts imply for *our* work (and where in this repo it maps).

### 1) “Blackwell is a dataflow machine” (ThunderKittens)

Source notes:
- [`thunderkittens-blackwell.md`](../../research/2025-12-24/incoming/perf/blogs/thunderkittens-blackwell.md)

What to take away:
- The core framing is **pipeline bubbles**, not “one clever instruction”.
- On Blackwell, fully utilizing tensor cores often means **bigger tiles** (rule of thumb from TK: 128×128 systolic behavior) and **deeper load/compute/store overlap**.
- “Keep tensor cores hot” becomes a dataflow scheduling problem (producer/consumer orchestration, persistent scheduling, staged outputs).

Where it maps here:
- FA4/CuTe already looks like this: load warps + MMA warp(s) + epilogue warps + explicit pipelines.
  - `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`
  - `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_bwd_sm100.py`
  - `vendored/flash_attn_cute_score_mod/flash_attn/cute/pipeline.py`

How to use it in Phase 3:
- If you’re considering “Level 6 work”, phrase experiments as: “does this reduce bubbles / improve overlap?” (not “does this one micro-op run faster?”).
- When you change tiling/staging, always validate:
  - occupancy didn’t crater (register pressure / smem)
  - you didn’t accidentally disable persistence (e.g., path-specific conditions)

### 2) Warp specialization is now a compiler feature (Triton / PyTorch)

Source notes:
- [`warp-specialization.md`](../../research/2025-12-24/incoming/perf/blogs/warp-specialization.md)

What to take away:
- Warp specialization is a practical way to express “producer vs consumer” roles without manually interleaving instruction streams.
- In Triton, it’s enabled via autotune flags like `num_consumer_groups` and `num_buffers_warp_spec`.

Where it maps here:
- This is most relevant if/when we revive Triton kernel work on B200 (or if Triton improves on SM103).
- For CuTe kernels, warp specialization is already explicit (warp IDs and barriers), so this is more about **future Triton experiments**.

How to use it in Phase 3:
- Treat warp-spec as an experiment card, not a refactor: add the flags, benchmark, and record wins/losses.
- Don’t do warp-spec experiments on SM103 unless you have a “safe fallback” path; we’ve already seen tcgen05-related compile aborts in some modes.

### 3) TMA + mbarrier correctness is the price of admission (tcgen05)

Source notes:
- [`gau-nerst-tcgen05.md`](../../research/2025-12-24/incoming/perf/blogs/gau-nerst-tcgen05.md)

What to take away:
- Blackwell/Hopper-style kernels get their bandwidth via **TMA** (`cp.async.bulk.tensor`) and their correctness via **mbarrier phase accounting** (arrival count + expected bytes).
- When these go wrong, symptoms often look like: deadlocks / hangs / “mysterious wrong data” / “works on one GPU but not another”.

Where it maps here:
- CuTe kernels hide the PTX, but you still see the logic:
  - `cute.arch.mbarrier_arrive(...)`
  - `cute.arch.mbarrier_wait(...)`
  - “expect_tx” style barriers around loads
  - explicit “pipeline stage” objects

How to use it in Phase 3:
- If you’re editing CuTe load/stage code, treat barriers as *part of the algorithm*, not boilerplate.
- If you see an SM103-only hang: check toolchain first (#12), then assume a barrier/async-proxy mismatch before assuming “math bug”.

### 4) `mask_mod` vs `score_mod` is a performance choice (FlexAttention / CuTe DSL)

Source notes:
- [`flexattention_guide.md`](../../research/2025-12-24/incoming/perf/blogs/flexattention_guide.md)
- [`flexattn-for-inference.md`](../../research/2025-12-24/incoming/perf/blogs/flexattn-for-inference.md)

What to take away:
- `mask_mod` enables **block sparsity** (skip fully-masked blocks, fast-path fully-unmasked blocks).
- `score_mod` is more general but easier to make expensive (it runs on all elements that survive masking).

Where it maps here:
- Our “recompute” path uses block masks (great for sparsity); our KV-bias path uses score modification (best expressed as `score_mod`).
- Explainers to re-open when optimizing masks:
  - [`08-masking-and-mask_mod.md`](08-masking-and-mask_mod.md)
  - [`03-score-mod.md`](03-score-mod.md)

How to use it in Phase 3:
- Don’t accidentally re-encode a mask as a score_mod: you’ll lose the ability to skip work at the block level.
- If you’re trying to make a path faster, ask: “can any of this become block-sparse?” before writing a new kernel.

### 5) “Glue” is often memory-bound; treat it like bandwidth engineering (QuACK)

Source notes:
- [`getting-mem-bound-kernals-SOL.md`](../../research/2025-12-24/incoming/perf/blogs/getting-mem-bound-kernals-SOL.md)

What to take away:
- Once you’re in the memory-bound regime, the win is usually “fewer passes over memory” and “better reduction strategy”, not “more math tricks”.
- Reduction-heavy kernels (softmax/norm) can be bottlenecked by register pressure and extra loads; cluster-level strategies exist for very large reductions.

Where it maps here:
- After KV-bias is solved, we see meaningful time in copies, dtype conversions, and elementwise glue (`aten::copy_`, `aten::to`, etc.).
- That’s exactly the kind of workload where “speed-of-light” thinking applies: reduce redundant loads/stores and fuse where safe.

How to use it in Phase 3:
- When op-level profiling says “copies dominate”, treat it as a first-class optimization target, not a rounding error.
- Proposed experiments should aim to delete whole kernels (fuse/collapse conversions), not shave 1–2% off one kernel.

### 6) Compilation and host overhead are part of performance (Diffusers + Modal)

Source notes:
- [`torch-compile-and-diffusers.md`](../../research/2025-12-24/incoming/perf/blogs/torch-compile-and-diffusers.md)
- [`modal_host-overhead-inference-efficency.md`](../../research/2025-12-24/incoming/perf/blogs/modal_host-overhead-inference-efficency.md)

What to take away:
- The best `torch.compile` wins come from **regional compilation**, fewer graph breaks, and fewer recompiles.
- Host overhead shows up as **gaps in CUDA streams**; kernel launch count matters once kernels get short.

Where it maps here:
- We already treat CuTe calls as opaque to Dynamo for correctness.
- Our block-level profilers deliberately synchronize, so they are *breakdown truth*, not “peak overlap truth”.

How to use it in Phase 3:
- When compile is unstable on SM103, prefer small, controlled compile regions and record the exact mode/env that works.
- If/when we get the pipeline “fast enough”, expect launch overhead and host overhead to become visible again.

---

## Extending Phase 3

This playbook is meant to grow. The next Phase 3 explainer is:
- [`14-blog-patterns-to-experiments.md`](14-blog-patterns-to-experiments.md) (turn blog patterns into concrete experiment cards for this repo)

---

## How This Relates to the Explainers You Just Read

If you’re trying to decide “which explainer should I re-open right now?”, use this mapping:

- You are debugging **why KV-bias is slower on one backend** → #3 and #11
- You are debugging **why combine exists / how LSE should be used** → #7 and #11
- You are debugging **a weird SM103-only failure** → #12 + [`fa4-patches.md`](../b300/fa4-patches.md)
- You are extending **score_mod or adding new callbacks** → #3 + `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py`
- You are touching **forward scheduling / loads / pipelining** → #2, #5, #6
- You are trying to understand **why backward is expensive / structured the way it is** → #10

---

## References (Start Here)

- “You are here” map: [`optimization-map.md`](../optimization-map.md)
- Shareable perf story: [`kernel-optimization-guide.md`](../docs/kernel-optimization-guide.md)
- B300 protocol: [`investigation-runbook.md`](../b300/investigation-runbook.md)
- B300 current best-known config: [`session-state.md`](../b300/session-state.md)
- B300 forward-looking options: [`optimization-vision.md`](../b300/optimization-vision.md)
- Level 5/6 reading list: [`level5-level6-resources.md`](../b300/level5-level6-resources.md)
- Blog notes (local, vendored): `notes/research/2025-12-24/incoming/perf/blogs/`
- Phase 1–2 explainer index: [`README.md`](README.md)
