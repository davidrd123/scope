# B300 / SM103 Notes (Quick Index)

This folder is the “working set” for the B300 (SM103) optimization thread: how to reproduce, what’s currently true, and where to record new measurements.

## “At Hand” Navigation (How to Use This Folder)

Think of the docs as **four roles**:
- **Truth** (what works today): `session-state.md`
- **Protocol** (how we measure): `investigation-runbook.md`
- **Logbook** (what we tried): `experiments.md`
- **Strategy / R&D** (what we do next): `optimization-vision.md` + `development-plan.md` + Level 6 docs below

If you’re ever unsure what to read next, follow this loop:
1) reproduce baseline → 2) measure (block + op profile) → 3) decide next lever → 4) make one change → 5) log an experiment card → repeat.

## Where to Start

| If you want to… | Start here |
|---|---|
| Run the current “known-good” baseline | `session-state.md` |
| Measure correctly (and not lie to yourself) | `investigation-runbook.md` |
| Log a single change / result | `experiments.md` |
| See the longer narrative / hypotheses | `investigation.md` |
| See the roadmap / next levers | `optimization-vision.md` |
| See the learning ladder framing | `optimization-ladder.md` |
| External references / stack maturity links | `blackwell-docs.md` |
| Document layout contracts (blocks Level 6) | `layout-contracts.md` |
| Break down `other_in_self` with stacks | `other-in-self-breakdown.md` |
| Quick reference for Blackwell primitives | `blackwell-primitives-cheatsheet.md` |
| Map the vendored reference kernels (CuTe/CUTLASS/FA4) | `vendored-kernel-resources.md` |
| VAE decode graph + cuDNN planning notes | `vae-decode-architecture.md` |
| Kernel experiment gates (perf + quality) | `kernel-experiment-template.md` |

## Doc Map (By Workflow)

- **Reproduce / baseline**
  - `session-state.md` (blessed commands + pitfalls)
  - `setup-guide.md` (how to get onto the cu130 “fast stack”)
  - `fa4-patches.md` (FA4/CuTe setup + SM103 patch notes)
- **Measure**
  - `investigation-runbook.md` (block profiler + op profiler + decision points)
  - `scripts/profile_krea_pipeline_blocks.py` (block timing)
  - `scripts/profile_krea_pipeline_ops.py` (operator + stack attribution)
- **Decide next lever**
  - `optimization-vision.md` (ordered options; “what’s left”)
  - `development-plan.md` (workstreams + acceptance criteria)
- **Record**
  - `experiments.md` (one-change cards only)
- **Level 6 groundwork (only after measurement)**
  - `layout-contracts.md` (hard constraints at QKV/RoPE/cache/attn boundaries)
  - `other-in-self-breakdown.md` (fill once with real stack data; don’t guess)
  - `vendored-kernel-resources.md` (where to learn patterns from real kernels in-repo)
  - `kernel-experiment-template.md` (quality/perf gates for any kernel work)
  - `blackwell-primitives-cheatsheet.md` (TMA/mbarrier/tcgen05/TMEM reference)
  - `vae-decode-architecture.md` (decode-specific map + planning ideas)

## Ground Truth Conventions

- **Canonical perf comparisons:** `320x576`, 4 denoise steps, `kv_cache_attention_bias=0.3`, quality-preserving settings.
- **Quality gate:** FP8 quantization is currently **off-limits** on B300 (garbage output). Use `--quantization none` (BF16). Details: `session-state.md`.
- **One-change rule:** change exactly one thing per experiment card and write down:
  - command, env, baseline, result, artifacts, and the lesson.

## Upstream References (Shortlist)

If you’re trying to answer “is this a us-bug or an upstream landmine?”, this is the minimal set:

- `nvidia-cutlass-dsl` module shadowing `torch._inductor` cutlass utilities (can break `torch.compile` / `flex_attention`): see `notes/FA4/b300/setup-guide.md` and `notes/FA4/b300/investigation.md` (Issue 2).
- TorchAO FP8 + `torch.compile` + `aten.as_strided` hole: see the “Upstream refs” section in `session-state.md`.
- Conv3d BF16/FP16 regressions + “install cuDNN 9.15+”: PyTorch v2.9.1 release notes (linked in `session-state.md` and `blackwell-docs.md`).
- CUDAGraph “output overwritten” + step-marker spelling: `torch.compiler.cudagraph_mark_step_begin()` docs (linked in `session-state.md`); on SM103 we auto-ignore `SCOPE_TORCH_COMPILE_MODE=reduce-overhead` unless `SCOPE_ALLOW_REDUCE_OVERHEAD_SM103=1`.

## Deep Research Packets (Optional, but source-backed)

If you want the “receipts” (issue IDs / exact spellings) in one place:
- `notes/FA4/DeepResearch/2025-12-26/B300_optim_ladder/round02/claude_dr.md`
- `notes/FA4/DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/README.md`
- `notes/FA4/DeepResearch/2025-12-26/B300_step_back/doc_ref_guide/claude01.md`
