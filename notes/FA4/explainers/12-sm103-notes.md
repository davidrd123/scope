# SM103 Notes (B300): What Carries Over from SM100, What Changes in Practice

> **Explainer #12** — SM103 (B300) runs “Blackwell-ish” code, but the *practical blockers* are toolchain/runtime/backends, not the math.
> This is a checklist-style bridge between the SM100 explainers and the B300 reality we’ve seen in this repo.
> **Updated:** 2025-12-25

---

## The Most Important Fact: Our Python Dispatch Treats SM103 as “10.x”

FA4’s CuTe interface uses **major-only** compute capability:
- `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py::_get_device_capability()` returns `torch.cuda.get_device_capability()[0]`

So:
- B200 (SM100) → major `10`
- B300 (SM103) → major `10`

In `vendored/flash_attn_cute_score_mod/flash_attn/cute/interface.py::_flash_attn_fwd(...)`, `compute_capability == 10` selects:
- `FlashAttentionForwardSm100(...)` from `vendored/flash_attn_cute_score_mod/flash_attn/cute/flash_fwd_sm100.py`

Meaning: **the SM100 explainers are still describing the code we run on SM103**. The gaps are mostly:
- “can we compile/dispatch it?” and
- “is the runtime stack actually SM103-native?”

---

## What Carries Over (Concepts)

If you understand the Phase 1 + Phase 2 explainers, you have the right mental model on SM103:

- **Online softmax math** still explains *why* LSE exists and how combine works (Explainer #7).
- **Warp specialization + pipelining** still explains the “producer/consumer” shape of the kernel (Explainer #2, #6, #10).
- **Split-K / segment combine** still uses the same `logaddexp`/weighting math (Explainer #11).
- **`score_mod` vs segment-combine bias** is still the key distinction for KV-bias (Explainer #3, #11).

---

## What Changes on SM103 (The Practical Blockers)

### 1) Toolchain gating: Cutlass-DSl / CuTe arch checks

On B300, the CuTe / tcgen05 stack often wants `sm_103a` (not `sm_100a/sm_100f`).
Older or unpatched `nvidia-cutlass-dsl` builds reject SM103, so FA4/CuTe can fail to import or compile.

Repo references:
- Patch guide: [`fa4-patches.md`](../b300/fa4-patches.md)
- Patch script: `scripts/patch_cutlass_sm103.sh`

Rule of thumb: if `from flash_attn.cute.interface import _flash_attn_fwd` fails on B300, don’t debug kernels yet — debug the toolchain.

### 2) `ptxas` must know `sm_103`

Even when you’re not writing Triton, parts of the stack can invoke toolchain components that need `ptxas` with SM103 support.
In our B300 work, using CUDA 12.9+ `ptxas` was a recurring requirement.

Repo reference:
- `_maybe_set_triton_ptxas_path()` in `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`

### 3) Backend selection is “correctness + perf critical” on SM103

On SM103, a backend can be:
- available but slow (forced fallback kernels), or
- unavailable (import-time failure), causing a “working but catastrophic” fallback path.

In this repo, the **KV-bias** backend defaults to “flash segment-combine” on SM103:
- `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py` sets `_KV_BIAS_BACKEND = ("flash" if _is_sm103() else "triton")`

Why that default exists: Triton KV-bias has been observed to be unusably slow on SM103 (Triton 3.5 scalar fallback).

Non-bias attention backend selection lives in:
- `src/scope/core/pipelines/wan2_1/modules/attention.py`

Notable SM103-specific guardrail:
- When `SCOPE_KV_BIAS_BACKEND=fa4`, we *disable FA4 varlen by default* to avoid mixing the wheel’s CuTe modules with the vendored score_mod-capable CuTe sources.
- Opt-in is `SCOPE_ENABLE_FA4_VARLEN=1`.

### 4) Runtime stack (cuDNN / CUDA) can dominate end-to-end on B300

This repo has strong evidence that the “B300 is slow” story can be mostly about decode/cuDNN, not attention:
- Default stack: ~8.8 FPS at `320x576`
- cu130 stack: ~15 FPS at `320x576` (decode gets dramatically faster; transformer becomes dominant)

If you’re optimizing kernels but running the wrong runtime stack, you can easily spend days chasing the wrong bottleneck.

Ground truth references:
- [`session-state.md`](../b300/session-state.md)
- [`investigation-runbook.md`](../b300/investigation-runbook.md)

---

## A Minimal “B300 Sanity Checklist” for Future Work

1. **Confirm stack**: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
2. **Confirm backend availability**:
   - `python -c "import flash_attn; print('flash_attn ok')"`
   - `python -c "from flash_attn.cute.interface import _flash_attn_fwd; print('FA4/CuTe ok')"` (only if you expect FA4)
3. **Confirm which KV-bias backend is active**: `echo $SCOPE_KV_BIAS_BACKEND`
4. **Confirm ptxas**: `echo $TRITON_PTXAS_PATH` (should point to a CUDA 12.9+ install on SM103)
5. **Use canonical measurements**: `320x576`, stable settings; see [`investigation-runbook.md`](../b300/investigation-runbook.md)

If any of those are “wrong”, fix them before micro-optimizing.

---

## How to Record SM103 Differences (So They’re Reproducible)

When you find an SM103-specific quirk, classify it:

1. **Dispatch difference**: wrong backend selected, missing feature gate, import-path mixing.
2. **Codegen/toolchain difference**: arch checks, `ptxas` mismatch, tcgen05 LLVM intrinsic errors.
3. **Runtime stack difference**: cuDNN kernel coverage, driver/toolkit mismatches.
4. **Workload/layout difference**: strides, B=1 slicing/layout assumptions, varlen edge cases.

And record (at minimum):
- GPU name + compute capability
- torch/CUDA/cuDNN versions
- exact env vars (`SCOPE_*`, `TRITON_PTXAS_PATH`)
- a single reproduction command and its output

---

## References

- SM100 mental model: [`02-blackwell-path.md`](02-blackwell-path.md)
- Segment combine + KV-bias backends: [`11-splitk-and-segment-combine.md`](11-splitk-and-segment-combine.md)
- B300 runbook: [`investigation-runbook.md`](../b300/investigation-runbook.md)
- B300 session truth: [`session-state.md`](../b300/session-state.md)
- B300 FA4 patching: [`fa4-patches.md`](../b300/fa4-patches.md)
- Runtime/backend code: `src/scope/core/pipelines/krea_realtime_video/modules/causal_model.py`
- Non-bias attention backend selection: `src/scope/core/pipelines/wan2_1/modules/attention.py`
