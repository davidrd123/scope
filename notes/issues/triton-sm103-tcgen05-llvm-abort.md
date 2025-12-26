# Issue Note: Triton/Inductor tcgen05 LLVM abort on SM103 (B300/GB300)

**Status:** Upstream (not something we can “monkeypatch” in this repo).

## Symptom

During compilation (commonly via `torch.compile` paths that trigger Triton/TensorCore codegen on SM103), the process can hard-abort with an LLVM error like:

- `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st`

This is a *fatal abort*, not a Python exception we can catch/recover from.

## Upstream tracking (primary)

- `triton-lang/triton#8473` (open as of 2025-12-26): **“Release 3.5 broke sm103 (GB300) support”**
  - Key detail from the issue body: Triton **3.4** supported SM103 (with `ptxas` from CUDA **12.9+**), a later change regressed SM103, and the fix on main involved an **LLVM bump**.
- `triton-lang/triton#8481` (closed): contains the exact `tcgen05.wait.st` LLVM abort text and links back to #8473.

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

1. Identify the **first Triton release** that restores SM103 support after the LLVM bump (or a 3.5.x hotfix), and the **first PyTorch release** that vendors it.
2. Capture a smallest-on-our-stack repro (ideally a `torch.compile` + Triton matmul/attention snippet) that triggers the abort on SM103.
3. Decide whether we want to:
   - pin an older/newer Triton in an *isolated* experimental env, or
   - wait for a PyTorch+Triton combo that includes the fix.

