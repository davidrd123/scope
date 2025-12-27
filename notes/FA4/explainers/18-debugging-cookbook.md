# Debugging Cookbook: FA4 / KV-bias / SM103 “Why Is This Broken or Slow?”

> **Explainer #18** — A symptom→cause→fix guide for the failure modes we actually hit while wiring FA4/KV-bias into this repo (especially on SM103).
> **Updated:** 2025-12-27

---

## TL;DR

- Most “mystery perf regressions” are **backend selection** or **toolchain gating**, not math.
- **B300 quality gate:** FP8 output is currently garbage (gray/noise); use BF16 (`--quantization none`) when debugging correctness.
- Before you debug kernels: confirm **stack**, **backend**, and **ptxas**.
- Record failures like experiments: versions + env vars + one repro command.

---

## 0) 60-Second Preflight (Do This First)

Run these in the environment you’re benchmarking:

```bash
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0)); print('cap', torch.cuda.get_device_capability(0))"
python3 -c "import flash_attn; print('flash_attn ok', flash_attn.__version__)"
python3 -c "from flash_attn.cute.interface import _flash_attn_fwd; print('fa4 _flash_attn_fwd ok')"
echo "SCOPE_KV_BIAS_BACKEND=$SCOPE_KV_BIAS_BACKEND"
echo "TRITON_PTXAS_PATH=$TRITON_PTXAS_PATH"
```

If any of these fail, fix *that* before chasing kernel-level issues.

---

## 1) Symptom: “End-to-end FPS is ~1 FPS (or wildly slower than expected)”

Likely causes:

- `flash_attn` is missing in the current env, causing fallback to slower paths.
- KV-bias backend fell back to Triton/flex on SM103 (sometimes catastrophically slow).

What to check:

- `python3 -c "import flash_attn"` succeeds.
- Logs show which attention backend is used:
  - plain attention logs in `src/scope/core/pipelines/wan2_1/modules/attention.py`
  - KV-bias warnings in `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

What to do:

- On SM103, prefer `SCOPE_KV_BIAS_BACKEND=flash` as a “keep it working” baseline.
- Only enable `fa4` KV-bias when CuTe toolchain is known-good.

---

## 2) Symptom: “FA4/CuTe import fails” (or score_mod isn’t available)

Examples:

- `ImportError` from CUTLASS DSL / CuTe arch checks
- `_flash_attn_fwd(...)` exists but does not accept `score_mod` or `return_lse`

Likely causes:

- Your installed `nvidia-cutlass-dsl` / CuTe build doesn’t accept SM103 (`sm_103a`) yet.
- You’re importing CuTe modules from a wheel that doesn’t expose the feature you need.

What to check:

- `from flash_attn.cute.interface import _flash_attn_fwd` works.
- In this repo, the score_mod path requires vendored sources:
  - `vendored/flash_attn_cute_score_mod/flash_attn/cute/...`

What to do:

- Follow the SM103 patching notes:
  - [`fa4-patches.md`](../b300/fa4-patches.md)
- Keep a strict “don’t mix modules” stance (see #12, #17).

---

## 3) Symptom: “torch.compile / flex_attention blows up on SM103”

Examples:

- `NoValidChoicesError: target: flex_attention`
- `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st`

Likely causes:

- Inductor/Triton using a `ptxas` that doesn’t recognize `sm_103a`.
- A compile mode that triggers tcgen05 lowering that hard-aborts on this stack.

What to check:

- `TRITON_PTXAS_PATH` points to a CUDA install with SM103 support (often CUDA 12.9+ in our notes).
- Whether you set:
  - `DISABLE_FLEX_ATTENTION_COMPILE=1`

What to do:

- First try setting a newer `ptxas` via `TRITON_PTXAS_PATH`.
- If you hit `tcgen05.wait.st` hard-aborts on SM103, upgrade Triton to **v3.5.1** (SM103 fix) and re-test; see [`triton-sm103-tcgen05-llvm-abort.md`](../../issues/triton-sm103-tcgen05-llvm-abort.md).
- If the goal is “keep the system usable”, disable the problematic compile path and log it as a constraint.

References:
- [`12-sm103-notes.md`](12-sm103-notes.md)
- [`investigation.md`](../b300/investigation.md)

---

## 4) Symptom: “FA4 varlen + score_mod causes weird runtime DSL errors”

Example class of failures:

- `TileSchedulerArguments` mismatch / “wrong CuTe module” type errors

Likely cause:

- You’re mixing the wheel’s `flash_attn.cute` implementation with the vendored CuTe sources used for score_mod.

What to do:

- When `SCOPE_KV_BIAS_BACKEND=fa4`, keep FA4 varlen opt-in:
  - `SCOPE_ENABLE_FA4_VARLEN=1` (only if you know it works on your stack)
- Otherwise accept that “score_mod” and “varlen FA4 attention” may need to be toggled independently.

Where this is enforced:
- `src/scope/core/pipelines/wan2_1/modules/attention.py` (FA4 varlen gating)

---

## 5) Symptom: “FA4 KV-bias fails with ‘Can’t deduce the leading dimension…’”

Likely cause:

- CuTe tries to infer leading-dimension/stride metadata and gets confused when K/V are views into a larger cache tensor (especially with `B=1` slicing).

What to do:

- Prefer “normalize the view” over cloning:
  - in this repo we use a `q = q[0].unsqueeze(0)` pattern for `B=1` to avoid a costly copy.

Reference:
- [`03-score-mod.md`](03-score-mod.md)
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

---

## 6) Symptom: “torch.profiler is useless (CUPTI errors / device_time=0)”

Likely cause:

- CUPTI issues in the environment.

What to do:

- Prefer CUDA-event based probes:
  - `PROFILE_PIPELINE_BLOCKS=1` (block-level breakdown; synchronizes)
  - `PROFILE_ATTENTION=1` (attention breakdown; synchronizes)
- Or use external tools like `nsys` if available.

Reference:
- [`investigation-runbook.md`](../b300/investigation-runbook.md)

---

## 7) Symptom: “Profiler shows changes but end-to-end doesn’t move”

Likely cause:

- Your “win” is in a non-dominant block (Amdahl), or your profiler forced sync and changed overlap.

What to do:

- Keep two truths:
  - “breakdown truth” (profiling flags on)
  - “throughput truth” (profiling flags off)

Reference:
- [`13-optimization-bootstrapping.md`](13-optimization-bootstrapping.md)

---

## 8) What to Write Down (So Bugs Become Assets)

When something fails, capture:

- GPU + compute capability
- torch/CUDA versions
- `SCOPE_*` env vars and `TRITON_PTXAS_PATH`
- exact command + the first relevant stack trace / warning

Then append it to:

- [`session-state.md`](../b300/session-state.md) (if SM103-specific), or
- the relevant experiment card in [`experiments.md`](../b200/experiments.md) / [`experiments.md`](../b300/experiments.md).

---

## References

- Call path map: [`15-scope-to-fa4-call-path.md`](15-scope-to-fa4-call-path.md)
- Knobs map: [`17-backend-selection-and-knobs.md`](17-backend-selection-and-knobs.md)
- SM103 notes: [`12-sm103-notes.md`](12-sm103-notes.md)
- B300 runbook/session truth: [`investigation-runbook.md`](../b300/investigation-runbook.md), [`session-state.md`](../b300/session-state.md)
