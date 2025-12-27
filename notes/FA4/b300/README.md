# B300 / SM103 Notes (Quick Index)

This folder is the “working set” for the B300 (SM103) optimization thread: how to reproduce, what’s currently true, and where to record new measurements.

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
