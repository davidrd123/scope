# Issue Note: Triton/Inductor tcgen05 LLVM abort on SM103 (B300/GB300)

**Status:** Fixed upstream in Triton **v3.5.1**, but still a local hazard until our environments upgrade.

## Symptom

During compilation (commonly via `torch.compile` paths that trigger Triton/TensorCore codegen on SM103), the process can hard-abort with an LLVM error like:

- `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st`

This is a *fatal abort*, not a Python exception we can catch/recover from.

## Upstream tracking (primary)

- `triton-lang/triton#8473` (opened 2025-10-17): **“Release 3.5 broke sm103 (GB300) support”**
  - Key detail from the issue body: Triton **3.4** supported SM103 (with `ptxas` from CUDA **12.9+**), a later change regressed SM103, and the fix on main involved an **LLVM bump**.
- `triton-lang/triton#8481` (closed): contains the exact `tcgen05.wait.st` LLVM abort text and links back to #8473.

## Update: fixed in Triton v3.5.1

Triton **v3.5.0** shipped between the regression and the fix. Triton **v3.5.1** includes the SM103 fix for the `tcgen05.wait.st` abort:

- Fix PR: https://github.com/triton-lang/triton/pull/8045
- Release notes: https://github.com/triton-lang/triton/releases/tag/v3.5.1

Practical implication for this repo:
- If your environment is on **`triton==3.5.0`**, assume SM103 tcgen05 compilation is **unsafe** and keep using the mitigations below.
- If we upgrade to **`triton==3.5.1`** (or a PyTorch build that vendors it), we should re-test the previously “known-bad” compile paths and update `notes/FA4/b300/session-state.md`.

### Verified in this repo (2025-12-27)

In an isolated cu130 env upgraded to `triton==3.5.1` (torch `2.9.0+cu130`), the previously-crashing compile mode is now stable:

- `SCOPE_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` **no longer aborts** on SM103.
- Steady-state FPS at our canonical settings was ~unchanged vs default compile mode, but warmup was longer (more autotuning).

Artifacts:
- Crash (triton 3.5.0): `outputs/b300_cu130_none_bias0.3_no_fuseproj_compile_mode_maxautotune_nocg_perf.log`
- Fixed (triton 3.5.1): `outputs/b300_cu130_triton351_compile_mode_maxautotune_nocg_perf2.log`

## Where it shows up in this repo

- We default to disabling FlexAttention compilation on B300 to avoid this abort:
  - `scripts/run_daydream_b300.sh` exports `DISABLE_FLEX_ATTENTION_COMPILE=1`
- Code comments reference the same failure family:
  - `src/scope/core/kernels/triton_attention.py` (tcgen05 intrinsic error context)
  - `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` (compile toggles + SM103 hazards)

## Mitigations we can actually use today

- Prefer non-Triton paths for attention on B300 (FA4/FlashAttention paths), and/or keep FlexAttention out of `torch.compile` on SM103:
  - `DISABLE_FLEX_ATTENTION_COMPILE=1`
- Ensure Triton sees a `ptxas` that supports `sm_103`:
  - set `TRITON_PTXAS_PATH` to CUDA **12.9+** (we already do this in the B300 scripts).

## Research TODOs (high value)

1. Validate `triton==3.5.1` (or a newer vendor bundle) in an *isolated* env and record:
   - whether the `tcgen05.wait.st` abort disappears,
   - whether any new failures appear (regressions),
   - and whether any “previously disabled” compile paths become safe enough to reconsider.
2. Capture a smallest-on-our-stack repro (ideally a `torch.compile` + Triton matmul/attention snippet) that triggers the abort on SM103.
3. Decide whether we want to:
   - pin an older/newer Triton in an *isolated* experimental env, or
   - wait for a PyTorch+Triton combo that includes the fix.
